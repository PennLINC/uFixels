//! Discover discrete u-fiber "sheets" by clustering streamlines via a
//! pointwise-close-everywhere neighbor relation.
//!
//! Mental model: a sheet is like a flexible piece of paper bending to follow
//! the cortex. Two streamlines belong to the same sheet iff they lie within
//! `eps_mm` of each other *at every point along the arc*. A gap in the wall
//! of streamlines breaks the chain → distinct sheets. Two sheets that
//! spatially overlap but each have their own gap remain separate components.
//!
//! Algorithm:
//! 1. Resample each polyline to `n_resample` uniform arc-length points.
//! 2. Build a KD-tree over polyline midpoints to prefilter neighbour candidates.
//! 3. For each pair within `midpoint_radius_mm`, compute the smaller of
//!    `max_i ‖A[i] - B[i]‖` and `max_i ‖A[i] - reverse(B)[i]‖` (handle
//!    antipodal symmetry — u-fibers don't have a canonical direction).
//! 4. If ≤ `eps_mm`: union(A, B) in a union-find.
//! 5. Read out connected components, drop those smaller than `min_sheet_size`.

use kiddo::float::kdtree::KdTree;
use kiddo::SquaredEuclidean;
use rayon::prelude::*;

const KDTREE_BUCKET: usize = 32;

#[derive(Debug, Clone, Copy)]
pub struct SheetParams {
    /// Maximum allowed pointwise distance (mm) between two streamlines for
    /// them to count as sheet-neighbours.
    pub eps_mm: f32,
    /// Number of arc-length-uniform samples per streamline. 16 covers a 20 mm
    /// streamline at ~1.3 mm sample spacing.
    pub n_resample: usize,
    /// KD-tree midpoint search radius (mm) for the neighbour prefilter.
    /// Conservative upper bound — pairs whose midpoints are farther than this
    /// can't possibly satisfy the pointwise-distance cutoff.
    pub midpoint_radius_mm: f32,
    /// Drop sheets with fewer than this many streamlines (treat as noise).
    pub min_sheet_size: usize,
    /// Optional reclamation pass: after the main clustering, every still-
    /// unassigned streamline is merged into the *nearest assigned* sheet
    /// whose midpoint is within `Some(r)` mm. `None` skips the pass and
    /// leaves the unassigned bucket as-is. A value of ~5–8 mm closes most
    /// gaps without dragging in genuinely isolated singletons.
    pub reclaim_radius_mm: Option<f32>,
}

impl SheetParams {
    pub fn defaults() -> Self {
        Self {
            // Empirically: 2mm leaves ~80% of streamlines as singletons (the
            // surviving streamline density is sparser than 2mm pointwise);
            // 4mm percolates into hemisphere-spanning giants. 3mm is a clean
            // operating point where ~72% of streamlines assign and the biggest
            // sheet stays under 15% of survivors.
            eps_mm: 3.0,
            n_resample: 16,
            midpoint_radius_mm: 10.0,
            min_sheet_size: 5,
            reclaim_radius_mm: None,
        }
    }
}

pub struct SheetResult {
    /// Per-streamline sheet id (1-indexed; 0 = unassigned/noise).
    /// Length = `polylines.len()`.
    pub streamline_sheet_id: Vec<u32>,
    /// Per-fixel sheet id (1-indexed; 0 = no streamline seeded by this fixel
    /// or its sheet was below `min_sheet_size`). Length = `nb_peaks`.
    pub fixel_sheet_id: Vec<u32>,
    /// Streamline counts indexed by sheet id (entry 0 is unused noise bucket).
    pub sheet_sizes: Vec<u32>,
}

/// Snapshot of cluster statistics at one ε in a sweep, used for auto-selection.
#[derive(Debug, Clone, Copy)]
pub struct SheetEpsStats {
    pub eps_mm: f32,
    pub n_sheets: u32,
    pub n_assigned: u32,
    pub largest_sheet_size: u32,
    pub largest_sheet_fraction: f32,
}

/// Pair of streamline indices with their max-pointwise distance, ready for
/// threshold-driven union-find. Distance is the *smaller* of A↔B and A↔reverse(B).
/// Pair of streamline ids with their max-pointwise distance (smaller of A↔B
/// and A↔reverse(B)). Used internally by `cluster_from_pairs` and exposed for
/// the second-stage filtration linker.
pub type StreamlinePair = (u32, u32, f32);

pub fn cluster_sheets(
    polylines: &[Vec<[f32; 3]>],
    polyline_fixel_id: &[u32],
    nb_peaks: usize,
    params: &SheetParams,
) -> SheetResult {
    let n = polylines.len();
    if n == 0 {
        return empty_result(nb_peaks);
    }
    let resampled = resample_all(polylines, params.n_resample);
    let pairs = compute_neighbor_pairs(&resampled, params.n_resample, params.midpoint_radius_mm, params.eps_mm);
    let mut result = cluster_from_pairs(&pairs, params.eps_mm, params.min_sheet_size, n, polyline_fixel_id, nb_peaks);
    if let Some(r) = params.reclaim_radius_mm {
        apply_reclamation(&mut result, &resampled, params.n_resample, r, polyline_fixel_id);
    }
    result
}

/// Run a sweep of ε values, pick the largest ε at which the biggest sheet
/// stays at or below `giant_cap_fraction` of *all assigned* streamlines.
/// Returns the chosen ε, the per-ε statistics, and the cluster result at the
/// chosen ε (so the caller doesn't re-do the work).
pub fn auto_select_eps(
    polylines: &[Vec<[f32; 3]>],
    polyline_fixel_id: &[u32],
    nb_peaks: usize,
    eps_min: f32,
    eps_max: f32,
    eps_step: f32,
    giant_cap_fraction: f32,
    params: &SheetParams,
) -> (f32, Vec<SheetEpsStats>, SheetResult) {
    let n = polylines.len();
    if n == 0 {
        return (eps_min, Vec::new(), empty_result(nb_peaks));
    }
    let resampled = resample_all(polylines, params.n_resample);
    // Prefilter pairs once at eps_max + small slack so we don't recompute per ε.
    let pairs = compute_neighbor_pairs(&resampled, params.n_resample, params.midpoint_radius_mm, eps_max);

    let mut sweep: Vec<SheetEpsStats> = Vec::new();
    let mut eps = eps_min;
    let mut chosen_eps = eps_min;
    let mut last_pass = chosen_eps;
    while eps <= eps_max + 1e-6 {
        let result = cluster_from_pairs(&pairs, eps, params.min_sheet_size, n, polyline_fixel_id, nb_peaks);
        let n_assigned: u32 = result.streamline_sheet_id.iter().filter(|&&s| s > 0).count() as u32;
        let largest = result.sheet_sizes.iter().skip(1).copied().max().unwrap_or(0);
        let frac = if n_assigned > 0 { largest as f32 / n_assigned as f32 } else { 0.0 };
        sweep.push(SheetEpsStats {
            eps_mm: eps,
            n_sheets: result.sheet_sizes.len().saturating_sub(1) as u32,
            n_assigned,
            largest_sheet_size: largest,
            largest_sheet_fraction: frac,
        });
        if frac <= giant_cap_fraction {
            last_pass = eps;
        }
        eps += eps_step;
    }

    // Re-run UF at the chosen ε so we return the matching cluster result.
    // Reclamation (if requested) is applied only to the final result, not to
    // the per-ε sweep stats — otherwise reclamation would inflate the giant
    // component and confuse the percolation criterion.
    chosen_eps = last_pass;
    let mut result = cluster_from_pairs(&pairs, chosen_eps, params.min_sheet_size, n, polyline_fixel_id, nb_peaks);
    if let Some(r) = params.reclaim_radius_mm {
        apply_reclamation(&mut result, &resampled, params.n_resample, r, polyline_fixel_id);
    }
    (chosen_eps, sweep, result)
}

// ---------------------------------------------------------------------------

fn empty_result(nb_peaks: usize) -> SheetResult {
    SheetResult {
        streamline_sheet_id: Vec::new(),
        fixel_sheet_id: vec![0; nb_peaks],
        sheet_sizes: vec![0],
    }
}

pub fn resample_all(polylines: &[Vec<[f32; 3]>], n_resample: usize) -> Vec<Vec<[f32; 3]>> {
    polylines
        .par_iter()
        .map(|p| resample_arc_length(p, n_resample))
        .collect()
}

pub fn compute_neighbor_pairs(
    resampled: &[Vec<[f32; 3]>],
    n_resample: usize,
    midpoint_radius_mm: f32,
    eps_keep_mm: f32,
) -> Vec<StreamlinePair> {
    let n = resampled.len();
    let mid_idx = n_resample / 2;
    let mut tree: KdTree<f32, u64, 3, KDTREE_BUCKET, u32> = KdTree::new();
    for (i, r) in resampled.iter().enumerate() {
        if r.len() == n_resample {
            tree.add(&r[mid_idx], i as u64);
        }
    }
    let radius2 = midpoint_radius_mm * midpoint_radius_mm;
    (0..n)
        .into_par_iter()
        .flat_map_iter(|i| {
            let a = &resampled[i];
            let mut local: Vec<StreamlinePair> = Vec::new();
            if a.len() != n_resample {
                return local.into_iter();
            }
            let cands = tree.within::<SquaredEuclidean>(&a[mid_idx], radius2);
            for c in cands {
                let j = c.item as usize;
                if j <= i {
                    continue;
                }
                let b = &resampled[j];
                if b.len() != n_resample {
                    continue;
                }
                let d = max_pointwise_distance_with_flip(a, b);
                if d <= eps_keep_mm {
                    local.push((i as u32, j as u32, d));
                }
            }
            local.into_iter()
        })
        .collect()
}

fn cluster_from_pairs(
    pairs: &[StreamlinePair],
    eps_mm: f32,
    min_sheet_size: usize,
    n: usize,
    polyline_fixel_id: &[u32],
    nb_peaks: usize,
) -> SheetResult {
    let mut uf = UnionFind::new(n);
    for &(a, b, d) in pairs {
        if d <= eps_mm {
            uf.union(a as usize, b as usize);
        }
    }

    let mut root_size: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    for i in 0..n {
        *root_size.entry(uf.find(i) as u32).or_insert(0) += 1;
    }

    let mut kept_roots: Vec<(u32, u32)> = root_size
        .iter()
        .filter(|(_, &sz)| sz as usize >= min_sheet_size)
        .map(|(&r, &sz)| (r, sz))
        .collect();
    kept_roots.sort_by(|a, b| b.1.cmp(&a.1));

    let mut root_to_id: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    let mut sheet_sizes: Vec<u32> = vec![0];
    for (id_minus_1, (root, sz)) in kept_roots.iter().enumerate() {
        root_to_id.insert(*root, (id_minus_1 + 1) as u32);
        sheet_sizes.push(*sz);
    }

    let mut streamline_sheet_id = vec![0_u32; n];
    let mut fixel_sheet_id = vec![0_u32; nb_peaks];
    for i in 0..n {
        let root = uf.find(i) as u32;
        if let Some(&sid) = root_to_id.get(&root) {
            streamline_sheet_id[i] = sid;
            let fid = polyline_fixel_id[i] as usize;
            if fid < nb_peaks {
                fixel_sheet_id[fid] = sid;
            }
        }
    }

    SheetResult {
        streamline_sheet_id,
        fixel_sheet_id,
        sheet_sizes,
    }
}

/// For every streamline still at sheet_id == 0, find the *nearest* assigned
/// streamline (by midpoint distance) and inherit its sheet_id, provided the
/// midpoint is within `radius_mm`. Updates `streamline_sheet_id`,
/// `fixel_sheet_id`, and `sheet_sizes` in place.
fn apply_reclamation(
    result: &mut SheetResult,
    resampled: &[Vec<[f32; 3]>],
    n_resample: usize,
    radius_mm: f32,
    polyline_fixel_id: &[u32],
) {
    let n = result.streamline_sheet_id.len();
    if n == 0 || radius_mm <= 0.0 {
        return;
    }
    let mid_idx = n_resample / 2;

    // Build a KD-tree over midpoints of currently-assigned streamlines.
    let mut tree: KdTree<f32, u64, 3, KDTREE_BUCKET, u32> = KdTree::new();
    for i in 0..n {
        if result.streamline_sheet_id[i] != 0 && resampled[i].len() == n_resample {
            tree.add(&resampled[i][mid_idx], i as u64);
        }
    }
    if tree.size() == 0 {
        return;
    }

    let radius2 = radius_mm * radius_mm;
    let nb_peaks = result.fixel_sheet_id.len();
    for i in 0..n {
        if result.streamline_sheet_id[i] != 0 || resampled[i].len() != n_resample {
            continue;
        }
        let nn = tree.nearest_one::<SquaredEuclidean>(&resampled[i][mid_idx]);
        if nn.distance > radius2 {
            continue;
        }
        let owner = nn.item as usize;
        let sid = result.streamline_sheet_id[owner];
        result.streamline_sheet_id[i] = sid;
        if (sid as usize) < result.sheet_sizes.len() {
            result.sheet_sizes[sid as usize] += 1;
        }
        let fid = polyline_fixel_id[i] as usize;
        if fid < nb_peaks {
            result.fixel_sheet_id[fid] = sid;
        }
    }
}

// ---------------------------------------------------------------------------

pub fn resample_arc_length(points: &[[f32; 3]], n: usize) -> Vec<[f32; 3]> {
    if points.len() < 2 || n < 2 {
        return Vec::new();
    }
    // Cumulative arc length.
    let mut cum: Vec<f32> = Vec::with_capacity(points.len());
    cum.push(0.0);
    for w in points.windows(2) {
        let d = dist(w[0], w[1]);
        cum.push(cum.last().unwrap() + d);
    }
    let total = *cum.last().unwrap();
    if total < 1e-6 {
        return Vec::new();
    }
    let mut out: Vec<[f32; 3]> = Vec::with_capacity(n);
    for k in 0..n {
        let target = total * (k as f32) / (n as f32 - 1.0);
        // Find segment containing `target`.
        let mut seg = 0;
        while seg + 1 < cum.len() && cum[seg + 1] < target {
            seg += 1;
        }
        if seg + 1 >= cum.len() {
            out.push(points[points.len() - 1]);
            continue;
        }
        let span = cum[seg + 1] - cum[seg];
        let alpha = if span > 1e-9 {
            (target - cum[seg]) / span
        } else {
            0.0
        };
        let a = points[seg];
        let b = points[seg + 1];
        out.push([
            a[0] + alpha * (b[0] - a[0]),
            a[1] + alpha * (b[1] - a[1]),
            a[2] + alpha * (b[2] - a[2]),
        ]);
    }
    out
}

#[inline]
fn dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Smaller of (max pointwise A↔B) and (max pointwise A↔reverse(B)). Streamlines
/// from u-fibers don't have a canonical direction, so we always check both.
fn max_pointwise_distance_with_flip(a: &[[f32; 3]], b: &[[f32; 3]]) -> f32 {
    let n = a.len();
    let mut max_fwd = 0.0_f32;
    let mut max_rev = 0.0_f32;
    for i in 0..n {
        let d_fwd = dist(a[i], b[i]);
        let d_rev = dist(a[i], b[n - 1 - i]);
        if d_fwd > max_fwd {
            max_fwd = d_fwd;
        }
        if d_rev > max_rev {
            max_rev = d_rev;
        }
    }
    max_fwd.min(max_rev)
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
            // Path halving.
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
        let rank_a = self.rank[ra];
        let rank_b = self.rank[rb];
        match rank_a.cmp(&rank_b) {
            std::cmp::Ordering::Less => self.parent[ra] = rb as u32,
            std::cmp::Ordering::Greater => self.parent[rb] = ra as u32,
            std::cmp::Ordering::Equal => {
                self.parent[rb] = ra as u32;
                self.rank[ra] += 1;
            }
        }
    }
}
