//! Surface-normal-driven detection of u-fiber-compatible fixels.
//!
//! See `README.md` for the algorithm and design notes. Top-level entry point:
//! [`run_hemisphere`].

#![warn(rust_2018_idioms)]

pub mod end_condition;
pub mod normal;
pub mod surface_io;
pub mod surface_mask;
pub mod traversal;
pub mod ufixel;

use std::sync::atomic::AtomicU8;

pub use end_condition::EndCondition;
pub use surface_mask::SurfaceVoxelSet;
pub use traversal::VoxelGrid;
pub use ufixel::{run_hemisphere, HemisphereOutputs};

/// Algorithm-wide tunables, in degrees and millimetres.
#[derive(Debug, Clone, Copy)]
pub struct UfixelsConfig {
    /// Target angle (deg) between fixel and surface normal. 90 = fixel parallel to cortex.
    pub compatible_angle_deg: f32,
    /// A fixel is "compatible" if `|fixel_to_normal_angle - compatible_angle|` is within this many degrees.
    pub max_angle_diff_deg: f32,
    /// Hard cap on inward walk distance (mm).
    pub max_depth_mm: f32,
    /// Tiny initial offset (mm) so we don't sample the seed voxel.
    pub min_step_eps_mm: f32,
}

impl Default for UfixelsConfig {
    fn default() -> Self {
        Self {
            compatible_angle_deg: 90.0,
            max_angle_diff_deg: 20.0,
            max_depth_mm: 20.0,
            min_step_eps_mm: 0.1,
        }
    }
}

/// Shared, lock-free per-fixel selection mask. Indexed by *global* fixel index
/// (`offsets[compact] + within_voxel_idx`).
pub type SelectedFixels = Vec<AtomicU8>;

/// Build a fresh selection mask of the right length for an ODX dataset.
pub fn new_selection_mask(nb_peaks: usize) -> SelectedFixels {
    (0..nb_peaks).map(|_| AtomicU8::new(0)).collect()
}

/// Snapshot the atomic selection mask to a plain `Vec<u8>` (suitable for
/// [`odx_rs::OdxDataset::insert_dpf_u8`]).
pub fn snapshot_selection_mask(mask: &SelectedFixels) -> Vec<u8> {
    mask.iter()
        .map(|a| a.load(std::sync::atomic::Ordering::Relaxed))
        .collect()
}
