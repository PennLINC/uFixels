//! Second-stage sheet linking: merge initially-clustered sheets into super-sheets
//! using anatomically-informed signals that resist percolation.
//!
//! All linkers in this module emit a common [`MergeEdge`] type, and the
//! [`merge_sheets`] resolver runs union-find over the pooled edges to produce
//! [`SuperSheetResult`]. Run any subset of linkers, concatenate their outputs,
//! pass to the resolver — they compose freely.
//!
//! See:
//! - [`direction::direction_gated_link`] (Option 2) — geometric proximity with a
//!   tangent-agreement veto.
//! - [`filtration::filtration_link`] (Option 5) — Kruskal-style filtration over
//!   the cross-sheet pair distance graph, again with the cosine veto.
//! - [`ptt_link::ptt_continuation_link`] (Option 6) — extend each streamline
//!   past its endpoint via PTT and link sheets the trajectory enters.

pub mod direction;
pub mod filtration;
pub mod ptt_link;

use kiddo::float::kdtree::KdTree;
use std::collections::HashMap;

use crate::sheets::{resample_all, SheetResult};

const KDTREE_BUCKET: usize = 32;

#[derive(Debug, Clone)]
pub enum MergeReason {
    DirectionGated { contact_dist_mm: f32, contact_cos: f32 },
    Filtration { merge_eps_mm: f32, contact_cos: f32 },
    PttContinuation { reach_mm: f32, support: f32 },
}

#[derive(Debug, Clone)]
pub struct MergeEdge {
    /// Lower-numbered sheet of the (a, b) pair (always a < b).
    pub a: u32,
    pub b: u32,
    /// Smaller = stronger linkage. Units depend on the linker (mm for
    /// geometric linkers, mm of PTT reach for the continuation linker).
    pub linkage_distance: f32,
    pub reason: MergeReason,
}

impl MergeEdge {
    pub fn new(s1: u32, s2: u32, linkage_distance: f32, reason: MergeReason) -> Self {
        let (a, b) = if s1 < s2 { (s1, s2) } else { (s2, s1) };
        Self {
            a,
            b,
            linkage_distance,
            reason,
        }
    }
}

/// Cached per-streamline geometry: endpoints, midpoint, inward-pointing
/// tangents at each endpoint. Built once and shared across linkers.
#[derive(Debug, Clone, Copy)]
pub struct StreamlineEndpoints {
    pub start: [f32; 3],
    pub end: [f32; 3],
    pub midpoint: [f32; 3],
    /// Unit vector at `start`, pointing toward the midpoint (inward).
    pub start_tangent: [f32; 3],
    /// Unit vector at `end`, pointing toward the midpoint (inward).
    pub end_tangent: [f32; 3],
}

/// Cached view over the sheet-clustering output, ready for second-stage
/// linker queries. All fields are precomputed so each linker just borrows.
pub struct SheetGraph<'a> {
    pub polylines: &'a [Vec<[f32; 3]>],
    pub polyline_fixel_id: &'a [u32],
    pub streamline_sheet_id: &'a [u32],
    /// Number of (1-indexed) sheets — `sheet_id ∈ 1..=n_sheets`. Sheet 0 is
    /// the unassigned bucket and is never merged into anything.
    pub n_sheets: u32,
    /// `sheet_streamlines[s]` lists the streamline ids belonging to sheet `s`
    /// (0 entry is for unassigned, intentionally non-empty so callers can
    /// inspect it but linkers don't merge into it).
    pub sheet_streamlines: Vec<Vec<u32>>,
    pub endpoints: Vec<StreamlineEndpoints>,
    pub resampled: Vec<Vec<[f32; 3]>>,
    pub n_resample: usize,
    pub midpoint_tree: KdTree<f32, u32, 3, KDTREE_BUCKET, u32>,
}

impl<'a> SheetGraph<'a> {
    pub fn build(
        polylines: &'a [Vec<[f32; 3]>],
        polyline_fixel_id: &'a [u32],
        streamline_sheet_id: &'a [u32],
        n_sheets: u32,
        n_resample: usize,
    ) -> Self {
        assert_eq!(polylines.len(), polyline_fixel_id.len());
        assert_eq!(polylines.len(), streamline_sheet_id.len());

        let resampled = resample_all(polylines, n_resample);
        let endpoints: Vec<StreamlineEndpoints> = polylines
            .iter()
            .map(|p| compute_endpoints(p))
            .collect();

        let mut sheet_streamlines: Vec<Vec<u32>> =
            vec![Vec::new(); (n_sheets + 1) as usize];
        for (i, &sid) in streamline_sheet_id.iter().enumerate() {
            if (sid as usize) < sheet_streamlines.len() {
                sheet_streamlines[sid as usize].push(i as u32);
            }
        }

        let mid_idx = n_resample / 2;
        let mut midpoint_tree: KdTree<f32, u32, 3, KDTREE_BUCKET, u32> = KdTree::new();
        for (i, r) in resampled.iter().enumerate() {
            if r.len() == n_resample {
                midpoint_tree.add(&r[mid_idx], i as u32);
            }
        }

        Self {
            polylines,
            polyline_fixel_id,
            streamline_sheet_id,
            n_sheets,
            sheet_streamlines,
            endpoints,
            resampled,
            n_resample,
            midpoint_tree,
        }
    }
}

pub struct SuperSheetResult {
    /// `sheet_to_supersheet[s]` = super-sheet id for sheet s (1-indexed; 0 if
    /// `s` is the unassigned bucket).
    pub sheet_to_supersheet: Vec<u32>,
    /// Per-fixel super-sheet id, parallel to `SheetResult::fixel_sheet_id`.
    pub fixel_supersheet_id: Vec<u32>,
    /// Streamline-count per super-sheet (entry 0 unused).
    pub supersheet_sizes: Vec<u32>,
    /// Edges that survived the resolver threshold and drove unioning.
    pub edges_used: Vec<MergeEdge>,
    /// Edges from linkers that were rejected by the resolver threshold.
    pub edges_considered: Vec<MergeEdge>,
}

/// Build super-sheets by union-find over the supplied edges, accepting only
/// those with `linkage_distance ≤ threshold`. The order of edges in the input
/// doesn't matter.
pub fn merge_sheets(
    sheets: &SheetResult,
    polyline_fixel_id: &[u32],
    nb_peaks: usize,
    edges: Vec<MergeEdge>,
    threshold: f32,
) -> SuperSheetResult {
    let n_sheets = sheets.sheet_sizes.len().saturating_sub(1) as u32;
    let mut uf = UnionFind::new((n_sheets + 1) as usize);
    let mut edges_used = Vec::new();
    let mut edges_considered = Vec::new();
    for e in edges {
        if e.linkage_distance <= threshold && e.a >= 1 && e.b >= 1 {
            uf.union(e.a as usize, e.b as usize);
            edges_used.push(e);
        } else {
            edges_considered.push(e);
        }
    }

    // Count streamlines per UF root, restricted to assigned sheets.
    let mut root_streamlines: HashMap<u32, u32> = HashMap::new();
    for &sid in &sheets.streamline_sheet_id {
        if sid == 0 {
            continue;
        }
        let root = uf.find(sid as usize) as u32;
        *root_streamlines.entry(root).or_insert(0) += 1;
    }

    // Assign super-sheet ids: largest first.
    let mut sorted: Vec<(u32, u32)> = root_streamlines.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

    let mut root_to_super: HashMap<u32, u32> = HashMap::new();
    let mut supersheet_sizes: Vec<u32> = vec![0];
    for (rank, (root, size)) in sorted.iter().enumerate() {
        root_to_super.insert(*root, (rank + 1) as u32);
        supersheet_sizes.push(*size);
    }

    let mut sheet_to_supersheet = vec![0_u32; (n_sheets + 1) as usize];
    for sid in 1..=n_sheets {
        let root = uf.find(sid as usize) as u32;
        if let Some(&super_id) = root_to_super.get(&root) {
            sheet_to_supersheet[sid as usize] = super_id;
        }
    }

    let mut fixel_supersheet_id = vec![0_u32; nb_peaks];
    for (i, &sid) in sheets.streamline_sheet_id.iter().enumerate() {
        if sid == 0 {
            continue;
        }
        let super_id = sheet_to_supersheet[sid as usize];
        if super_id == 0 {
            continue;
        }
        let fid = polyline_fixel_id[i] as usize;
        if fid < nb_peaks {
            fixel_supersheet_id[fid] = super_id;
        }
    }

    SuperSheetResult {
        sheet_to_supersheet,
        fixel_supersheet_id,
        supersheet_sizes,
        edges_used,
        edges_considered,
    }
}

// ---------------------------------------------------------------------------
// Helpers shared across linker implementations.

#[inline]
pub(crate) fn normalize(v: [f32; 3]) -> [f32; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n > 1e-9 {
        [v[0] / n, v[1] / n, v[2] / n]
    } else {
        v
    }
}

#[inline]
pub(crate) fn sq_dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

fn compute_endpoints(p: &[[f32; 3]]) -> StreamlineEndpoints {
    if p.len() < 2 {
        let zero = [0.0, 0.0, 0.0];
        return StreamlineEndpoints {
            start: zero,
            end: zero,
            midpoint: zero,
            start_tangent: zero,
            end_tangent: zero,
        };
    }
    let start = p[0];
    let end = p[p.len() - 1];
    let mid_idx = p.len() / 2;
    let midpoint = p[mid_idx];

    // Inward tangents — point from each endpoint toward the streamline's
    // interior. Smooth over the first/last 3 points to dampen jaggedness from
    // the PTT step granularity.
    let smooth_n = 3.min(p.len() - 1);
    let start_tangent = normalize([
        p[smooth_n][0] - p[0][0],
        p[smooth_n][1] - p[0][1],
        p[smooth_n][2] - p[0][2],
    ]);
    let last = p.len() - 1;
    let end_tangent = normalize([
        p[last - smooth_n][0] - p[last][0],
        p[last - smooth_n][1] - p[last][1],
        p[last - smooth_n][2] - p[last][2],
    ]);

    StreamlineEndpoints {
        start,
        end,
        midpoint,
        start_tangent,
        end_tangent,
    }
}

/// Returns the smaller of the max-pointwise distances over (A vs B) and
/// (A vs reverse(B)), and `(min_dist, contact_cos, flipped_b)`. Both
/// streamlines must have the same length (= `n_resample`).
pub(crate) fn pairwise_contact_metrics(
    a: &[[f32; 3]],
    b: &[[f32; 3]],
) -> (f32, f32, f32, bool) {
    let n = a.len();
    debug_assert_eq!(n, b.len());

    let mut max_fwd: f32 = 0.0;
    let mut max_rev: f32 = 0.0;
    let mut min_fwd2 = f32::INFINITY;
    let mut min_rev2 = f32::INFINITY;
    let mut min_i_fwd = 0;
    let mut min_i_rev = 0;
    for i in 0..n {
        let d_fwd2 = sq_dist(a[i], b[i]);
        let d_rev2 = sq_dist(a[i], b[n - 1 - i]);
        if d_fwd2.sqrt() > max_fwd {
            max_fwd = d_fwd2.sqrt();
        }
        if d_rev2.sqrt() > max_rev {
            max_rev = d_rev2.sqrt();
        }
        if d_fwd2 < min_fwd2 {
            min_fwd2 = d_fwd2;
            min_i_fwd = i;
        }
        if d_rev2 < min_rev2 {
            min_rev2 = d_rev2;
            min_i_rev = i;
        }
    }

    let (max_pointwise, min_pointwise, ia, ib, flipped) = if max_fwd <= max_rev {
        (max_fwd, min_fwd2.sqrt(), min_i_fwd, min_i_fwd, false)
    } else {
        (
            max_rev,
            min_rev2.sqrt(),
            min_i_rev,
            n - 1 - min_i_rev,
            true,
        )
    };

    // Tangent at the contact sample on each streamline.
    let ta = local_tangent(a, ia);
    let tb_raw = local_tangent(b, ib);
    let tb = if flipped {
        [-tb_raw[0], -tb_raw[1], -tb_raw[2]]
    } else {
        tb_raw
    };
    let cos_angle = (ta[0] * tb[0] + ta[1] * tb[1] + ta[2] * tb[2]).abs();
    (max_pointwise, min_pointwise, cos_angle, flipped)
}

fn local_tangent(p: &[[f32; 3]], i: usize) -> [f32; 3] {
    if p.len() < 2 {
        return [0.0, 0.0, 0.0];
    }
    if i + 1 < p.len() {
        normalize([
            p[i + 1][0] - p[i][0],
            p[i + 1][1] - p[i][1],
            p[i + 1][2] - p[i][2],
        ])
    } else {
        normalize([
            p[i][0] - p[i - 1][0],
            p[i][1] - p[i - 1][1],
            p[i][2] - p[i - 1][2],
        ])
    }
}

// ---------------------------------------------------------------------------

struct UnionFind {
    parent: Vec<u32>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x as u32 {
            self.parent[x] = self.parent[self.parent[x] as usize];
            x = self.parent[x] as usize;
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        match self.rank[ra].cmp(&self.rank[rb]) {
            std::cmp::Ordering::Less => self.parent[ra] = rb as u32,
            std::cmp::Ordering::Greater => self.parent[rb] = ra as u32,
            std::cmp::Ordering::Equal => {
                self.parent[rb] = ra as u32;
                self.rank[ra] += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sheets::SheetResult;

    fn fake_sheets(streamline_sheet_id: Vec<u32>, n_sheets: usize) -> SheetResult {
        let mut sheet_sizes = vec![0_u32; n_sheets + 1];
        for &sid in &streamline_sheet_id {
            if (sid as usize) < sheet_sizes.len() && sid > 0 {
                sheet_sizes[sid as usize] += 1;
            }
        }
        SheetResult {
            streamline_sheet_id,
            fixel_sheet_id: Vec::new(),
            sheet_sizes,
        }
    }

    #[test]
    fn merge_collapses_unioned_sheets() {
        let sheets = fake_sheets(vec![1, 1, 2, 2, 3, 3], 3);
        let polyline_fixel_id = vec![0_u32; 6];
        let edges = vec![MergeEdge::new(
            1,
            2,
            1.0,
            MergeReason::DirectionGated {
                contact_dist_mm: 1.0,
                contact_cos: 1.0,
            },
        )];
        let result = merge_sheets(&sheets, &polyline_fixel_id, 0, edges, 2.0);
        // Sheets 1 and 2 should now share a super-sheet, sheet 3 alone.
        assert_eq!(result.sheet_to_supersheet[1], result.sheet_to_supersheet[2]);
        assert_ne!(result.sheet_to_supersheet[1], result.sheet_to_supersheet[3]);
    }

    #[test]
    fn threshold_rejects_high_distance_edges() {
        let sheets = fake_sheets(vec![1, 1, 2, 2], 2);
        let edges = vec![MergeEdge::new(
            1,
            2,
            5.0,
            MergeReason::DirectionGated {
                contact_dist_mm: 5.0,
                contact_cos: 1.0,
            },
        )];
        let result = merge_sheets(&sheets, &[0_u32, 0, 0, 0], 0, edges, 3.0);
        assert_ne!(result.sheet_to_supersheet[1], result.sheet_to_supersheet[2]);
        assert_eq!(result.edges_used.len(), 0);
        assert_eq!(result.edges_considered.len(), 1);
    }

    #[test]
    fn supersheet_size_sorted_descending() {
        let sheets = fake_sheets(vec![1, 1, 1, 2, 2, 3, 3, 3, 3], 3);
        let edges = vec![MergeEdge::new(
            2,
            3,
            1.0,
            MergeReason::DirectionGated {
                contact_dist_mm: 1.0,
                contact_cos: 1.0,
            },
        )];
        let result = merge_sheets(&sheets, &[0_u32; 9], 0, edges, 2.0);
        // Super-sheet 1 should be {2,3} with 6 streamlines, super-sheet 2 = {1} with 3.
        assert_eq!(result.supersheet_sizes[1], 6);
        assert_eq!(result.supersheet_sizes[2], 3);
    }
}
