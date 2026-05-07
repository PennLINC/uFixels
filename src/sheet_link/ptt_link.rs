//! Linker C: PTT continuation across sheet boundaries.
//!
//! For each streamline endpoint of each sheet, run a fresh (unreferenced)
//! PTT propagation outward from that endpoint and watch which other sheets
//! the trajectory enters. The continuation direction is `−inward_tangent`,
//! i.e. the direction the streamline was *coming from* when it terminated —
//! so we're asking: if this u-fiber arc kept going past where we have data,
//! would it land in another sheet's territory?
//!
//! In practice, most cortical-endpoint extensions have very weak data support
//! (PTT stops within a step or two — there are no fixels in cortical grey
//! matter). But for sheets that were broken into pieces by a missing-streamline
//! gap, the PTT propagator naturally bridges the gap by following the residual
//! data support, and the continuation lands in the neighbouring sheet's fixels.

use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Mutex;

use odx_tractography::{arc_likelihood, ptt_walk, FixelIndex, PtfFrame, PttParams};

use super::{normalize, MergeEdge, MergeReason, SheetGraph};

#[derive(Debug, Clone, Copy)]
pub struct PttLinkParams {
    /// PTT probe-arc tunables. Re-use the same values you used for the
    /// connectivity filter (sharp curvature, short probe).
    pub probe: PttParams,
    /// Maximum continuation length per endpoint (mm). Total work scales
    /// linearly in this; 6mm covers typical inter-sheet gaps.
    pub continuation_mm: f32,
    /// Step size for PTT propagation.
    pub step_mm: f32,
    /// Stop continuing when the best-step arc-likelihood drops below this.
    pub min_step_likelihood: f32,
    /// Capture radius (mm) at each step for finding fixels along the path.
    pub capture_radius_mm: f32,
    /// `|d_fixel · trajectory_T|` floor for capture; same role as in
    /// `connectivity.rs`.
    pub cos_theta_capture: f32,
}

impl PttLinkParams {
    pub fn defaults() -> Self {
        Self {
            probe: PttParams {
                probe_length_mm: 3.0,
                probe_quality: 4,
                support_radius_mm: 2.5,
                angular_power: 4,
                k_max: 0.5,
                n_k_samples: 7,
            },
            continuation_mm: 6.0,
            step_mm: 1.0,
            min_step_likelihood: 0.01,
            capture_radius_mm: 1.7,
            cos_theta_capture: 0.5,
        }
    }
}

pub fn ptt_continuation_link(
    graph: &SheetGraph<'_>,
    fixel_index: &FixelIndex,
    fixel_sheet_id: &[u32],
    params: &PttLinkParams,
) -> Vec<MergeEdge> {
    // Aggregator: per-sheet-pair, the shortest reach_mm seen and its support.
    let best: Mutex<HashMap<(u32, u32), (f32, f32)>> = Mutex::new(HashMap::new());

    let n = graph.polylines.len();
    (0..n).into_par_iter().for_each(|i| {
        let source_sheet = graph.streamline_sheet_id[i];
        if source_sheet == 0 {
            return;
        }
        let ep = &graph.endpoints[i];

        // Two extensions per streamline: outward at start and at end.
        // start_tangent points INWARD (toward midpoint), so the outward
        // continuation direction is its negation.
        for (origin, inward) in [
            (ep.start, ep.start_tangent),
            (ep.end, ep.end_tangent),
        ] {
            let outward = [-inward[0], -inward[1], -inward[2]];
            if outward[0] == 0.0 && outward[1] == 0.0 && outward[2] == 0.0 {
                continue;
            }
            if let Some((target_sheet, reach, support)) = continue_until_cross_sheet(
                origin,
                outward,
                fixel_index,
                fixel_sheet_id,
                source_sheet,
                params,
            ) {
                let mut g = best.lock().unwrap();
                let key = if source_sheet < target_sheet {
                    (source_sheet, target_sheet)
                } else {
                    (target_sheet, source_sheet)
                };
                let entry = g.entry(key).or_insert((f32::INFINITY, 0.0));
                if reach < entry.0 {
                    *entry = (reach, support);
                }
            }
        }
    });

    let best = best.into_inner().unwrap();
    best.into_iter()
        .map(|((sa, sb), (reach, support))| {
            MergeEdge::new(
                sa,
                sb,
                reach,
                MergeReason::PttContinuation { reach_mm: reach, support },
            )
        })
        .collect()
}

/// Propagate PTT outward from `(origin, outward_dir)` for up to
/// `continuation_mm`. At each step, query the FixelIndex within
/// `capture_radius_mm`; if any captured fixel belongs to a sheet other than
/// `source_sheet`, return that sheet, the arc length to it, and the average
/// arc-likelihood up to that point.
fn continue_until_cross_sheet(
    origin: [f32; 3],
    outward_dir: [f32; 3],
    idx: &FixelIndex,
    fixel_sheet_id: &[u32],
    source_sheet: u32,
    params: &PttLinkParams,
) -> Option<(u32, f32, f32)> {
    let mut frame = PtfFrame {
        p: origin,
        f: build_frame_from_tangent(outward_dir),
    };
    let mut total = 0.0_f32;
    let mut lik_sum = 0.0_f32;
    let mut steps = 0_u32;

    while total < params.continuation_mm {
        // Sweep curvature grid for the best arc.
        let n_k = params.probe.n_k_samples.max(1);
        let dk = if n_k > 1 {
            2.0 * params.probe.k_max / (n_k as f32 - 1.0)
        } else {
            0.0
        };
        let mut best = (0.0_f32, 0.0_f32, -1.0_f32);
        if n_k == 1 {
            let lik = arc_likelihood(&frame, 0.0, 0.0, idx, &params.probe);
            best = (0.0, 0.0, lik);
        } else {
            for ii in 0..n_k {
                let k1 = -params.probe.k_max + dk * ii as f32;
                for jj in 0..n_k {
                    let k2 = -params.probe.k_max + dk * jj as f32;
                    let lik = arc_likelihood(&frame, k1, k2, idx, &params.probe);
                    if lik > best.2 {
                        best = (k1, k2, lik);
                    }
                }
            }
        }
        if best.2 < params.min_step_likelihood {
            return None;
        }
        ptt_walk(&mut frame, best.0, best.1, params.step_mm);
        total += params.step_mm;
        lik_sum += best.2;
        steps += 1;

        // Capture nearby fixels at the new position; find the first that
        // belongs to a different (assigned) sheet.
        let r2 = params.capture_radius_mm * params.capture_radius_mm;
        let candidates = idx.nearest_within(frame.p, r2.sqrt());
        for (fid, _d) in candidates {
            let handle = match idx.try_handle(fid) {
                Some(h) => h,
                None => continue,
            };
            // Direction-agreement check at this point.
            let t = frame.f[0];
            let dot = (handle.dir[0] * t[0] + handle.dir[1] * t[1] + handle.dir[2] * t[2]).abs();
            if dot < params.cos_theta_capture {
                continue;
            }
            let other = fixel_sheet_id[fid as usize];
            if other == 0 || other == source_sheet {
                continue;
            }
            let avg_lik = lik_sum / (steps as f32).max(1.0);
            return Some((other, total, avg_lik));
        }
    }
    None
}

fn build_frame_from_tangent(t: [f32; 3]) -> [[f32; 3]; 3] {
    let t = normalize(t);
    let seed = if t[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let n1 = normalize(cross(seed, t));
    let n2 = cross(t, n1);
    [t, n1, n2]
}

#[inline]
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
