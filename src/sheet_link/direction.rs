//! Linker A: direction-gated proximity merging.
//!
//! For each pair of sheets within range, find the closest cross-sheet
//! streamline pair. If that pair's max-pointwise-distance is `≤ eps_merge_mm`
//! AND the tangents at their contact point agree (`|cos| ≥ cos_min`), emit a
//! [`MergeEdge`] with `linkage_distance` = the max-pointwise-distance.
//!
//! The tangent veto is what differentiates this from raw clustering at a
//! larger ε: a chance contact between two unrelated bundles (e.g. left-right
//! hemisphere bridge) has crossing fibres, not aligned ones, so the cosine
//! check rejects it even when the spatial distance is small.

use rayon::prelude::*;
use std::collections::HashMap;

use kiddo::SquaredEuclidean;

use super::{
    pairwise_contact_metrics, MergeEdge, MergeReason, SheetGraph,
};

#[derive(Debug, Clone, Copy)]
pub struct DirectionGatedParams {
    /// Max-pointwise-distance ceiling (mm) for two streamlines to count as
    /// candidates for sheet merging.
    pub eps_merge_mm: f32,
    /// Min `|cos(tangent_a · tangent_b)|` at the contact sample. ~0.85 ⇒
    /// ≤ 32° angular disagreement.
    pub cos_min: f32,
}

impl Default for DirectionGatedParams {
    fn default() -> Self {
        Self {
            eps_merge_mm: 4.0,
            cos_min: 0.85,
        }
    }
}

pub fn direction_gated_link(
    graph: &SheetGraph<'_>,
    params: &DirectionGatedParams,
) -> Vec<MergeEdge> {
    let n = graph.resampled.len();
    let n_resample = graph.n_resample;
    let mid_idx = n_resample / 2;
    // Midpoint search radius: a generous over-estimate of how far apart
    // midpoints can be while still satisfying the pointwise-distance cutoff.
    let radius2 = (params.eps_merge_mm * 1.75).powi(2);
    let eps = params.eps_merge_mm;
    let cos_min = params.cos_min;

    // Per-sheet-pair best (smallest distance, contact cos at that pair).
    // Each thread builds its own HashMap, then we reduce.
    let per_thread: Vec<HashMap<(u32, u32), (f32, f32)>> = (0..n)
        .into_par_iter()
        .fold(
            HashMap::<(u32, u32), (f32, f32)>::new,
            |mut local, i| {
                let a = &graph.resampled[i];
                if a.len() != n_resample {
                    return local;
                }
                let sa = graph.streamline_sheet_id[i];
                if sa == 0 {
                    return local;
                }
                let cands = graph
                    .midpoint_tree
                    .within::<SquaredEuclidean>(&a[mid_idx], radius2);
                for c in cands {
                    let j = c.item as usize;
                    if j <= i {
                        continue;
                    }
                    let sb = graph.streamline_sheet_id[j];
                    if sb == 0 || sb == sa {
                        continue;
                    }
                    let b = &graph.resampled[j];
                    if b.len() != n_resample {
                        continue;
                    }
                    let (max_pw, _min_pw, contact_cos, _flip) =
                        pairwise_contact_metrics(a, b);
                    if max_pw > eps || contact_cos < cos_min {
                        continue;
                    }
                    let key = if sa < sb { (sa, sb) } else { (sb, sa) };
                    let entry = local.entry(key).or_insert((f32::INFINITY, 0.0));
                    if max_pw < entry.0 {
                        *entry = (max_pw, contact_cos);
                    }
                }
                local
            },
        )
        .collect();

    // Reduce thread-local maps.
    let mut best: HashMap<(u32, u32), (f32, f32)> = HashMap::new();
    for tl in per_thread {
        for (k, (d, c)) in tl {
            let entry = best.entry(k).or_insert((f32::INFINITY, 0.0));
            if d < entry.0 {
                *entry = (d, c);
            }
        }
    }

    best.into_iter()
        .map(|((sa, sb), (d, contact_cos))| {
            MergeEdge::new(
                sa,
                sb,
                d,
                MergeReason::DirectionGated {
                    contact_dist_mm: d,
                    contact_cos,
                },
            )
        })
        .collect()
}
