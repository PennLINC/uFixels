//! Connectivity-based filter for u-fixel candidates.
//!
//! Per candidate fixel:
//! 1. Propagate a PTT trajectory (forward + backward) with sharp-curvature
//!    parameters tuned for u-fibers — short probe, wide curvature grid.
//! 2. Capture the visited fixels along the way and atomically bump
//!    `hit_count[v] += 1`.
//! 3. Test whether both polyline endpoints land near a cortical vertex.
//!
//! After all candidates have traced, a fixel survives iff
//! `hit_count[c] >= min_hits AND endpoints_near_cortex[c]`.
//!
//! No `Reference` template — we want u-fiber connectivity, not bundle-prior
//! gating. PTT here uses pure data support (`arc_likelihood`, the unreferenced
//! variant) and runs the same propagator math as
//! [`odx_tractography::propagate_ptt_one_way`] minus the reference gate.

use std::sync::atomic::{AtomicU16, Ordering};

use odx_tractography::{
    arc_likelihood, capture_visited_fixels, ptt_walk, FixelHandle, FixelId, FixelIndex, PtfFrame,
    PttParams, PttTrajectory,
};
use rayon::prelude::*;

use crate::cortex_kdtree::CortexKdTree;
use crate::no_end_mask::NoEndMask;

#[derive(Debug, Clone, Copy)]
pub struct ConnectivityConfig {
    /// PTT probe-arc tunables. For u-fibers we widen `k_max` to allow a tight
    /// (~2 mm radius) U-bend and shorten the probe so it doesn't overshoot.
    pub probe: PttParams,
    /// Step size (mm) for trajectory propagation.
    pub step_mm: f32,
    /// Maximum trajectory length (mm) per direction (so total length ≤ 2 × this).
    pub max_length_mm: f32,
    /// Stop propagation when the best candidate's data-support drops below this.
    pub min_step_likelihood: f32,
    /// Capture radius (mm): fixels within this distance of a trajectory point
    /// whose direction agrees with the local tangent are counted as "visited".
    pub capture_radius_mm: f32,
    /// `|d_fixel · trajectory_T|` threshold for capture.
    pub cos_theta_capture: f32,
    /// Maximum endpoint-to-cortex distance (mm) for the trajectory to count
    /// as "ends at cortex".
    pub endpoint_max_mm: f32,
    /// Minimum total polyline length (mm). Trajectories shorter than this
    /// (typically aborted-after-a-few-steps starts) don't count as u-fibers.
    pub min_total_length_mm: f32,
    /// Maximum total polyline length (mm). U-fibers are short; longer
    /// trajectories are likely traversing a long bundle.
    pub max_total_length_mm: f32,
    /// Minimum inbound hit count from *u-fiber-shaped* peer trajectories.
    pub min_hits: u16,
    /// Reject trajectories whose two endpoints land in different hemispheres
    /// (anatomically impossible for a u-fiber). Default `true`.
    pub reject_cross_hemisphere: bool,
    /// Minimum maximum-deviation-from-chord (mm). A real u-fiber arching ~5
    /// mm into WM has a deviation of ~5 mm; a near-straight bundle running
    /// parallel to cortex has ~0–1 mm. 2 mm cleanly rejects deep bundles
    /// (cingulum, IFOF tangent runs) without losing shallow u-fibers.
    pub min_arc_deviation_mm: f32,
    /// Maximum allowed distance from cortex (mm) at *any* point along the
    /// trajectory. Real u-fibers stay within ~5 mm of cortex throughout;
    /// a trajectory whose midpoint dives 15–20 mm into WM (cortico-callosal
    /// or corona-radiata) gets rejected. 0 disables this check.
    pub max_path_depth_mm: f32,
}

impl ConnectivityConfig {
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
            step_mm: 1.0,
            // Per-direction cap; total trajectory ≤ 2× this. 10mm matches
            // the typical depth of superficial u-fibers — short enough that
            // CC traces can't traverse the body and SLF traces can't reach
            // the next gyrus, but long enough that a real u-fiber from its
            // apex still touches cortex on both ends.
            max_length_mm: 10.0,
            min_step_likelihood: 0.01,
            capture_radius_mm: 1.7,
            cos_theta_capture: 0.5,
            endpoint_max_mm: 3.0,
            min_total_length_mm: 5.0,
            max_total_length_mm: 20.0,
            // K=3 means a candidate needs at least 3 *u-fiber-shaped peer
            // trajectories* to vote for it. With trajectory-shape gating this
            // is now a meaningful threshold (CC/SLF leaks fail the shape
            // check, so they don't get to vote for each other).
            min_hits: 3,
            reject_cross_hemisphere: true,
            min_arc_deviation_mm: 2.0,
            max_path_depth_mm: 8.0,
        }
    }
}

pub struct ConnectivityResult {
    /// Per-fixel inbound hit count from u-fiber-shaped peer trajectories.
    pub hit_count: Vec<u16>,
    /// Per-fixel boolean: candidate's own trajectory was u-fiber-shaped
    /// (endpoints near cortex AND total length within range).
    pub trajectory_pass: Vec<u8>,
    /// Final mask: 1 iff candidate AND `trajectory_pass` AND `hit_count >= min_hits`.
    pub keep: Vec<u8>,
    /// Per-candidate diagnostics for tuning. `0.0` for non-candidates.
    pub trajectory_length_mm: Vec<f32>,
    pub endpoint_min_dist_mm: Vec<f32>, // max(start_to_cortex, end_to_cortex)
    /// Surviving polylines (only populated when `keep_polylines = true`).
    pub polylines: Vec<Vec<[f32; 3]>>,
    /// FixelId that seeded each entry of `polylines` (parallel to it).
    pub polyline_fixel_id: Vec<FixelId>,
}

/// Run the connectivity filter over the candidate set.
///
/// `selected` is the Phase 1 selection mask (per-fixel `u8`, 1 = candidate).
///
/// **Hit-count semantics:** `hit_count[v]` is the number of *candidate* fixels
/// whose PTT trajectory visited `v`. We deliberately ignore visits from
/// non-candidates (i.e. we only count "tangent peers voting for me"), because
/// otherwise long dense bundles like SLF accumulate huge hit counts and crowd
/// out the sparser real u-fiber consensus we're trying to detect.
pub fn run_connectivity(
    config: &ConnectivityConfig,
    fixel_index: &FixelIndex,
    cortex: &CortexKdTree,
    no_end: Option<&NoEndMask>,
    selected: &[u8],
    keep_polylines: bool,
) -> ConnectivityResult {
    let nb_peaks = selected.len();
    let hit_count: Vec<AtomicU16> = (0..nb_peaks).map(|_| AtomicU16::new(0)).collect();
    let mut trajectory_pass = vec![0_u8; nb_peaks];
    let mut trajectory_length_mm = vec![0.0_f32; nb_peaks];
    let mut endpoint_min_dist_mm = vec![0.0_f32; nb_peaks];

    let candidates: Vec<FixelId> = (0..nb_peaks as u32)
        .filter(|&id| {
            (selected[id as usize] != 0) && fixel_index.try_handle(id).is_some()
        })
        .collect();

    // Per-candidate trace, in parallel. Trajectory-shape gating: a trace only
    // contributes hits if it is itself u-fiber-shaped (endpoints near cortex
    // AND total length within range). This is what makes the peer-vote signal
    // discriminating — CC/SLF leaks have non-u-shaped trajectories so they
    // don't get to vote for each other, and the dense peer consensus that was
    // drowning the signal goes away.
    let per_cand: Vec<(FixelId, bool, f32, f32, Option<Vec<[f32; 3]>>)> = candidates
        .par_iter()
        .map(|&fid| {
            let handle = fixel_index.handle(fid);
            let traj = propagate_unreferenced(handle, fixel_index, config);

            let total_len = polyline_length(&traj.points);
            let (start_d, start_hemi, end_d, end_hemi) = if traj.points.len() >= 2 {
                let (sd, sh) = cortex.nearest_dist_and_hemi(*traj.points.first().unwrap());
                let (ed, eh) = cortex.nearest_dist_and_hemi(*traj.points.last().unwrap());
                (sd, sh, ed, eh)
            } else {
                (f32::INFINITY, None, f32::INFINITY, None)
            };
            let endpoint_worst = start_d.max(end_d);
            let length_ok = total_len >= config.min_total_length_mm
                && total_len <= config.max_total_length_mm;
            let endpoint_ok = endpoint_worst <= config.endpoint_max_mm;
            let same_hemi_ok = if config.reject_cross_hemisphere {
                match (start_hemi, end_hemi) {
                    (Some(a), Some(b)) => a == b,
                    _ => false, // missing hemisphere info ⇒ reject conservatively
                }
            } else {
                true
            };
            let arc_dev = max_deviation_from_chord(&traj.points);
            let arc_ok = arc_dev >= config.min_arc_deviation_mm;
            let max_path_depth = if config.max_path_depth_mm > 0.0 {
                let mut max_d: f32 = 0.0;
                for p in &traj.points {
                    let d = cortex.nearest_dist(*p);
                    if d > max_d {
                        max_d = d;
                    }
                }
                max_d
            } else {
                0.0
            };
            let depth_ok = config.max_path_depth_mm <= 0.0
                || max_path_depth <= config.max_path_depth_mm;
            let no_end_ok = match no_end {
                Some(mask) if traj.points.len() >= 2 => {
                    !mask.is_no_end(*traj.points.first().unwrap())
                        && !mask.is_no_end(*traj.points.last().unwrap())
                }
                _ => true,
            };
            let trajectory_ok =
                length_ok && endpoint_ok && same_hemi_ok && arc_ok && depth_ok && no_end_ok;

            if trajectory_ok {
                let visited = capture_visited_fixels(
                    &traj,
                    fixel_index,
                    config.capture_radius_mm,
                    config.cos_theta_capture,
                );
                for v in visited {
                    if selected[v as usize] != 0 {
                        hit_count[v as usize].fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
            let polyline = if keep_polylines && trajectory_ok {
                Some(traj.points.clone())
            } else {
                None
            };
            (fid, trajectory_ok, total_len, endpoint_worst, polyline)
        })
        .collect();

    let mut polylines: Vec<Vec<[f32; 3]>> = Vec::new();
    let mut polyline_fixel_id: Vec<FixelId> = Vec::new();
    for (fid, ok, total_len, ep_worst, poly) in per_cand {
        if ok {
            trajectory_pass[fid as usize] = 1;
        }
        trajectory_length_mm[fid as usize] = total_len;
        endpoint_min_dist_mm[fid as usize] = ep_worst;
        if let Some(p) = poly {
            polylines.push(p);
            polyline_fixel_id.push(fid);
        }
    }

    let hit_count: Vec<u16> = hit_count
        .iter()
        .map(|a| a.load(Ordering::Relaxed))
        .collect();

    let mut keep = vec![0_u8; nb_peaks];
    for fid in 0..nb_peaks {
        if selected[fid] != 0
            && trajectory_pass[fid] != 0
            && hit_count[fid] >= config.min_hits
        {
            keep[fid] = 1;
        }
    }

    ConnectivityResult {
        hit_count,
        trajectory_pass,
        keep,
        trajectory_length_mm,
        endpoint_min_dist_mm,
        polylines,
        polyline_fixel_id,
    }
}

/// Maximum perpendicular deviation of any interior point from the chord
/// connecting `points.first()` and `points.last()`. Returns 0 if the
/// streamline has fewer than 3 points or zero-length chord.
fn max_deviation_from_chord(points: &[[f32; 3]]) -> f32 {
    if points.len() < 3 {
        return 0.0;
    }
    let a = points[0];
    let b = points[points.len() - 1];
    let abx = b[0] - a[0];
    let aby = b[1] - a[1];
    let abz = b[2] - a[2];
    let ab_len2 = abx * abx + aby * aby + abz * abz;
    if ab_len2 < 1.0e-9 {
        return 0.0;
    }
    let mut max_dev_sq = 0.0_f32;
    for p in &points[1..points.len() - 1] {
        let apx = p[0] - a[0];
        let apy = p[1] - a[1];
        let apz = p[2] - a[2];
        // Project AP onto AB.
        let t = (apx * abx + apy * aby + apz * abz) / ab_len2;
        let projx = a[0] + t * abx;
        let projy = a[1] + t * aby;
        let projz = a[2] + t * abz;
        let dx = p[0] - projx;
        let dy = p[1] - projy;
        let dz = p[2] - projz;
        let d2 = dx * dx + dy * dy + dz * dz;
        if d2 > max_dev_sq {
            max_dev_sq = d2;
        }
    }
    max_dev_sq.sqrt()
}

#[inline]
fn polyline_length(points: &[[f32; 3]]) -> f32 {
    let mut total = 0.0_f32;
    for w in points.windows(2) {
        let a = w[0];
        let b = w[1];
        let dx = b[0] - a[0];
        let dy = b[1] - a[1];
        let dz = b[2] - a[2];
        total += (dx * dx + dy * dy + dz * dz).sqrt();
    }
    total
}

// ---------------------------------------------------------------------------
// PTT propagation without a reference template.
// Mirrors odx_tractography::propagate_ptt_one_way / propagate_ptt but uses
// `arc_likelihood` (pure data support) instead of the reference-gated variant.
// ---------------------------------------------------------------------------

fn propagate_unreferenced(
    seed: &FixelHandle,
    idx: &FixelIndex,
    config: &ConnectivityConfig,
) -> PttTrajectory {
    let forward = propagate_one_way(seed.world_pos, seed.dir, idx, config);
    let backward = propagate_one_way(
        seed.world_pos,
        [-seed.dir[0], -seed.dir[1], -seed.dir[2]],
        idx,
        config,
    );

    // Concatenate: backward (reversed, drop the duplicated seed point) + forward.
    let mut points: Vec<[f32; 3]> = backward.points.iter().rev().skip(1).copied().collect();
    let mut tangents: Vec<[f32; 3]> = backward
        .tangents
        .iter()
        .rev()
        .skip(1)
        .map(|t| [-t[0], -t[1], -t[2]])
        .collect();
    let mut likelihoods: Vec<f32> = backward.likelihoods.iter().rev().copied().collect();
    points.extend(forward.points.iter().copied());
    tangents.extend(forward.tangents.iter().copied());
    likelihoods.extend(forward.likelihoods.iter().copied());
    PttTrajectory {
        points,
        tangents,
        likelihoods,
    }
}

fn propagate_one_way(
    seed_pos: [f32; 3],
    seed_dir: [f32; 3],
    idx: &FixelIndex,
    config: &ConnectivityConfig,
) -> PttTrajectory {
    let mut frame = PtfFrame {
        p: seed_pos,
        f: build_frame_from_tangent(seed_dir),
    };
    let mut traj = PttTrajectory {
        points: vec![seed_pos],
        tangents: vec![normalize(seed_dir)],
        likelihoods: Vec::new(),
    };
    let mut total = 0.0_f32;
    while total < config.max_length_mm {
        let (k1, k2, lik) = best_curvature_step(&frame, idx, config);
        if lik < config.min_step_likelihood {
            break;
        }
        ptt_walk(&mut frame, k1, k2, config.step_mm);
        traj.points.push(frame.p);
        traj.tangents.push(frame.f[0]);
        traj.likelihoods.push(lik);
        total += config.step_mm;
    }
    traj
}

fn best_curvature_step(
    frame: &PtfFrame,
    idx: &FixelIndex,
    config: &ConnectivityConfig,
) -> (f32, f32, f32) {
    let n_k = config.probe.n_k_samples.max(1);
    if n_k == 1 {
        let lik = arc_likelihood(frame, 0.0, 0.0, idx, &config.probe);
        return (0.0, 0.0, lik);
    }
    let dk = 2.0 * config.probe.k_max / (n_k as f32 - 1.0);
    let mut best = (0.0_f32, 0.0_f32, -1.0_f32);
    for i in 0..n_k {
        let k1 = -config.probe.k_max + dk * i as f32;
        for j in 0..n_k {
            let k2 = -config.probe.k_max + dk * j as f32;
            let lik = arc_likelihood(frame, k1, k2, idx, &config.probe);
            if lik > best.2 {
                best = (k1, k2, lik);
            }
        }
    }
    best
}


#[inline]
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

#[inline]
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n > 1e-9 {
        [v[0] / n, v[1] / n, v[2] / n]
    } else {
        v
    }
}
