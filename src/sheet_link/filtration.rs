//! Linker B: percolation-veto filtration.
//!
//! Reuses the streamline-pair distance list from the original clustering
//! ([`crate::sheets::compute_neighbor_pairs`]). For each cross-sheet pair,
//! apply the same tangent-agreement veto as Linker A. Then for each unordered
//! sheet-pair, record the *smallest* qualifying inter-streamline distance —
//! exactly the level at which the two sheets would merge in a Kruskal
//! filtration. Emit one [`MergeEdge`] per qualifying sheet-pair with
//! `linkage_distance` = that filtration level.
//!
//! Conceptually equivalent to Linker A on the same data, but the linkage
//! semantics are explicitly dendrogram-cut: smaller values mean earlier
//! merging in the filtration. The threshold passed to [`super::merge_sheets`]
//! is the cut level on the dendrogram.

use std::collections::HashMap;

use crate::sheets::StreamlinePair;

use super::{pairwise_contact_metrics, MergeEdge, MergeReason, SheetGraph};

#[derive(Debug, Clone, Copy)]
pub struct FiltrationParams {
    /// Tangent-agreement floor at the contact pair. Same role as Linker A.
    pub cos_min: f32,
}

impl Default for FiltrationParams {
    fn default() -> Self {
        Self { cos_min: 0.85 }
    }
}

/// Aggregate cross-sheet pairs into sheet-level merge edges.
///
/// `pairs` is the all-pairs distance list (typically what
/// [`crate::sheets::compute_neighbor_pairs`] returned at `eps_max`). Each
/// entry must satisfy `d ≤ eps_max`, but otherwise the function tolerates
/// arbitrary order. The veto uses tangent agreement at the contact sample,
/// computed via [`super::pairwise_contact_metrics`] on the resampled
/// streamlines stored in the graph.
pub fn filtration_link(
    graph: &SheetGraph<'_>,
    pairs: &[StreamlinePair],
    params: &FiltrationParams,
) -> Vec<MergeEdge> {
    let mut best: HashMap<(u32, u32), (f32, f32)> = HashMap::new();
    let n_resample = graph.n_resample;

    for &(a_id, b_id, d) in pairs {
        let sa = graph.streamline_sheet_id[a_id as usize];
        let sb = graph.streamline_sheet_id[b_id as usize];
        if sa == 0 || sb == 0 || sa == sb {
            continue;
        }
        let a = &graph.resampled[a_id as usize];
        let b = &graph.resampled[b_id as usize];
        if a.len() != n_resample || b.len() != n_resample {
            continue;
        }
        let (_max_pw, _min_pw, contact_cos, _flip) = pairwise_contact_metrics(a, b);
        if contact_cos < params.cos_min {
            continue;
        }
        let key = if sa < sb { (sa, sb) } else { (sb, sa) };
        let entry = best.entry(key).or_insert((f32::INFINITY, 0.0));
        if d < entry.0 {
            *entry = (d, contact_cos);
        }
    }

    best.into_iter()
        .map(|((sa, sb), (d, contact_cos))| {
            MergeEdge::new(
                sa,
                sb,
                d,
                MergeReason::Filtration {
                    merge_eps_mm: d,
                    contact_cos,
                },
            )
        })
        .collect()
}
