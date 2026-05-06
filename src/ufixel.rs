//! Per-vertex inward walk. Produces the three output maps and, as a side
//! effect, marks selected fixels in a shared atomic mask.

use std::sync::atomic::Ordering;

use odx_rs::OdxDataset;
use rayon::prelude::*;

use crate::end_condition::EndCondition;
use crate::normal::inward_normals;
use crate::surface_io::LoadedSurface;
use crate::surface_mask::SurfaceVoxelSet;
use crate::traversal::{RayWalker, VoxelGrid};
use crate::{SelectedFixels, UfixelsConfig};

pub struct HemisphereOutputs {
    /// Closest-to-target diff (deg) found in the first masked voxel after stepping into WM.
    /// `NaN` for invalid vertices and voxels with zero fixels.
    pub first_voxel_diff: Vec<f32>,
    /// Walk depth in mm while compatibility held.
    pub compatible_depth: Vec<f32>,
    /// One [`EndCondition`] per vertex, encoded as `f32` (NaN for invalid).
    pub end_condition: Vec<f32>,
}

/// Run the inward-walk algorithm for one hemisphere.
///
/// `selected` is a shared atomic mask of length `dataset.nb_peaks()` — the same
/// instance is typically passed for both hemispheres so a single DPF mask can
/// be written at the end.
pub fn run_hemisphere(
    config: &UfixelsConfig,
    dataset: &OdxDataset,
    grid: &VoxelGrid,
    surface_voxels: &SurfaceVoxelSet,
    wm: &LoadedSurface,
    pial: &LoadedSurface,
    selected: &SelectedFixels,
) -> anyhow::Result<HemisphereOutputs> {
    let normals = inward_normals(&wm.vertices, &pial.vertices)
        .map_err(|e| anyhow::anyhow!("paired-vertex check failed: {e}"))?;

    let n = wm.vertices.len();
    let directions = dataset.directions();
    let offsets = dataset.offsets();

    // Precompute angular thresholds in cosine-space to avoid acos() in the inner loop.
    // Under antipodal symmetry the angle between a fixel and the normal lives in
    // [0, π/2], so we clamp the (target ± max_diff) interval to that range and then
    // convert. cos is monotonically decreasing on [0, π/2], so the smaller angle
    // gives the larger cosine.
    let target = config.compatible_angle_deg.to_radians().clamp(0.0, std::f32::consts::FRAC_PI_2);
    let max_diff = config.max_angle_diff_deg.to_radians().max(0.0);
    let angle_lo = (target - max_diff).max(0.0);
    let angle_hi = (target + max_diff).min(std::f32::consts::FRAC_PI_2);
    let cos_hi = angle_lo.cos();
    let cos_lo = angle_hi.cos();

    let mut first_voxel_diff = vec![f32::NAN; n];
    let mut compatible_depth = vec![0.0_f32; n];
    let mut end_condition = vec![f32::NAN; n];

    // Process vertices in parallel. Each thread mutates only its own slot of
    // the per-vertex output vectors plus the shared atomic mask.
    let outputs: Vec<(f32, f32, f32)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let normal = match normals[i] {
                Some(n) => n,
                None => return (f32::NAN, 0.0, f32::NAN),
            };
            let seed = wm.vertices[i];
            let seed_voxel = grid.voxel_of(seed);

            // Start a hair into the brain so we don't sample the seed voxel.
            let start = [
                seed[0] + config.min_step_eps_mm * normal[0],
                seed[1] + config.min_step_eps_mm * normal[1],
                seed[2] + config.min_step_eps_mm * normal[2],
            ];
            let walker = RayWalker::new(grid, start, normal);

            let mut entered_mask = false;
            let mut depth_mm = 0.0_f32;
            let mut comp_depth = 0.0_f32;
            let mut first_diff = f32::NAN;
            let mut end = EndCondition::LeftMask;

            for (ijk, segment_mm) in walker {
                // Out-of-grid → bail.
                let flat = match grid.flat_index(ijk) {
                    Some(f) => f,
                    None => {
                        end = EndCondition::LeftMask;
                        break;
                    }
                };
                let compact_signed = grid.flat_to_compact[flat];

                // Off-mask handling: tolerate a partial-volume gap before WM,
                // but treat any return-to-off-mask after entering WM as exiting.
                if compact_signed < 0 {
                    if !entered_mask {
                        depth_mm += segment_mm;
                        if depth_mm >= config.max_depth_mm {
                            end = EndCondition::LeftMask;
                            break;
                        }
                        continue;
                    } else {
                        end = EndCondition::LeftMask;
                        break;
                    }
                }

                // Surface-stop test, with a 1-voxel exclusion ball around the seed.
                let near_seed = (ijk[0] - seed_voxel[0]).abs() <= 1
                    && (ijk[1] - seed_voxel[1]).abs() <= 1
                    && (ijk[2] - seed_voxel[2]).abs() <= 1;
                if !near_seed && surface_voxels.contains(ijk) {
                    end = EndCondition::HitSurface;
                    break;
                }

                // Fixels at this voxel.
                let compact = compact_signed as usize;
                let fix_start = offsets[compact] as usize;
                let fix_end = offsets[compact + 1] as usize;
                let voxel_fixels = &directions[fix_start..fix_end];

                // Find the fixel whose angle to the normal is closest to the target,
                // working in |cos| space so we can drop acos in the inner loop.
                let target_cos = target.cos();
                let mut best_abs_cos = f32::NAN;
                let mut best_local_idx: i32 = -1;
                for (li, fx) in voxel_fixels.iter().enumerate() {
                    let dot = fx[0] * normal[0] + fx[1] * normal[1] + fx[2] * normal[2];
                    let abs_cos = dot.abs();
                    let cur_diff = (abs_cos - target_cos).abs();
                    let prev_diff = if best_abs_cos.is_nan() {
                        f32::INFINITY
                    } else {
                        (best_abs_cos - target_cos).abs()
                    };
                    if cur_diff < prev_diff {
                        best_abs_cos = abs_cos;
                        best_local_idx = li as i32;
                    }
                }

                // Convert the best abs-cosine into a degree-difference for output.
                let voxel_best_diff_deg = if best_local_idx >= 0 {
                    let angle = best_abs_cos.clamp(0.0, 1.0).acos();
                    (angle - target).abs().to_degrees()
                } else {
                    f32::NAN
                };

                // Record the first masked voxel's best diff exactly once.
                if !entered_mask {
                    first_diff = voxel_best_diff_deg;
                    entered_mask = true;
                }

                // Compatibility test: best |cos| inside [cos_lo, cos_hi]?
                let compatible =
                    best_local_idx >= 0 && best_abs_cos >= cos_lo && best_abs_cos <= cos_hi;
                if !compatible {
                    end = EndCondition::NoCompatibleFixel;
                    break;
                }

                // Mark winning fixel as selected.
                let global_fixel_idx = fix_start + best_local_idx as usize;
                selected[global_fixel_idx].store(1, Ordering::Relaxed);

                depth_mm += segment_mm;
                comp_depth = depth_mm;
                if depth_mm >= config.max_depth_mm {
                    end = EndCondition::MaxDepth;
                    break;
                }
            }

            (first_diff, comp_depth, end.as_f32())
        })
        .collect();

    for (i, (a, b, c)) in outputs.into_iter().enumerate() {
        first_voxel_diff[i] = a;
        compatible_depth[i] = b;
        end_condition[i] = c;
    }

    Ok(HemisphereOutputs {
        first_voxel_diff,
        compatible_depth,
        end_condition,
    })
}
