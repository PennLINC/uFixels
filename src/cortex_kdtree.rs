//! Concatenated KD-tree over all four input cortical surfaces (wm L/R + pial
//! L/R), used by the connectivity filter to test whether a traced trajectory's
//! endpoints land near the cortex.
//!
//! Medial-wall vertices — where `wm[i] == pial[i]` (degenerate paired vertex,
//! no inward normal) — are excluded from the tree. They sit on the corpus
//! callosum / brainstem cut rather than on a real cortical termination zone,
//! so trajectories ending near them aren't actually ending at GM.

use kiddo::float::kdtree::KdTree;
use kiddo::SquaredEuclidean;

use crate::surface_io::LoadedSurface;

const KDTREE_BUCKET: usize = 32;
const PAIR_DEGEN_EPS_MM: f32 = 1.0e-3;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Hemi {
    Left,
    Right,
}

pub struct CortexKdTree {
    tree: KdTree<f32, u64, 3, KDTREE_BUCKET, u32>,
    /// Hemisphere of each KD-tree entry, indexed by the tree's u64 item id.
    hemi: Vec<Hemi>,
    n_added: u64,
    n_skipped_medial: u64,
}

impl CortexKdTree {
    /// Build a tree from all valid (non-medial-wall) vertices in the four
    /// paired surfaces. Each `(wm, pial)` pair contributes vertex `i` to the
    /// tree iff `wm[i] != pial[i]`.
    pub fn from_paired_surfaces(
        lh_wm: &LoadedSurface,
        lh_pial: &LoadedSurface,
        rh_wm: &LoadedSurface,
        rh_pial: &LoadedSurface,
    ) -> Self {
        let mut tree: KdTree<f32, u64, 3, KDTREE_BUCKET, u32> = KdTree::new();
        let mut hemi: Vec<Hemi> = Vec::new();
        let mut id: u64 = 0;
        let mut n_added: u64 = 0;
        let mut n_skipped: u64 = 0;
        for (wm, pial, h) in [
            (lh_wm, lh_pial, Hemi::Left),
            (rh_wm, rh_pial, Hemi::Right),
        ] {
            assert_eq!(wm.vertices.len(), pial.vertices.len(),
                "paired-vertex mismatch when building cortex KD-tree");
            for i in 0..wm.vertices.len() {
                let w = wm.vertices[i];
                let p = pial.vertices[i];
                if pair_degenerate(w, p) {
                    n_skipped += 1;
                    continue;
                }
                tree.add(&w, id); id += 1; n_added += 1; hemi.push(h);
                tree.add(&p, id); id += 1; n_added += 1; hemi.push(h);
            }
        }
        Self {
            tree,
            hemi,
            n_added,
            n_skipped_medial: n_skipped,
        }
    }

    pub fn n_vertices(&self) -> u64 { self.n_added }
    pub fn n_skipped_medial(&self) -> u64 { self.n_skipped_medial }

    /// Distance (mm) from `p` to the nearest non-medial cortical vertex.
    /// Returns `f32::INFINITY` if the tree is empty.
    pub fn nearest_dist(&self, p: [f32; 3]) -> f32 {
        self.nearest_dist_and_hemi(p).0
    }

    /// Distance plus the hemisphere of the nearest cortical vertex.
    pub fn nearest_dist_and_hemi(&self, p: [f32; 3]) -> (f32, Option<Hemi>) {
        if self.tree.size() == 0 {
            return (f32::INFINITY, None);
        }
        let nn = self.tree.nearest_one::<SquaredEuclidean>(&p);
        let h = self.hemi.get(nn.item as usize).copied();
        (nn.distance.sqrt(), h)
    }
}

#[inline]
fn pair_degenerate(a: [f32; 3], b: [f32; 3]) -> bool {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt() < PAIR_DEGEN_EPS_MM
}
