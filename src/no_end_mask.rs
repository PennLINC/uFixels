//! Per-voxel "no-end" mask, ported from DSI-Studio's autotrack idea (and the
//! same machinery TRXViz uses in its Hausdorff plan-prep).
//!
//! Logic: for each masked voxel, take the *primary* peak's QA (`dpf/amplitude`
//! per the ODX SPEC). Compute Otsu's threshold over that distribution; voxels
//! whose primary-peak QA exceeds `factor × otsu` are flagged as forbidden
//! endpoints. A trajectory ending in such a voxel is sitting in dense fibrous
//! WM, not at cortex — i.e. its u-fibre-shape is illusory and the trace is
//! just clipped by the length cap.

use anyhow::Result;
use nalgebra::Vector4;

use odx_rs::OdxDataset;

use crate::traversal::VoxelGrid;

pub struct NoEndMask {
    /// Flat voxel mask (1 = forbidden as endpoint), C-order.
    data: Vec<u8>,
    dims: [usize; 3],
    /// For world→voxel lookup.
    world_to_voxel: nalgebra::Matrix4<f64>,
    /// Otsu threshold value used to derive `data`. Reported for diagnostics.
    pub otsu_threshold: f32,
    /// Number of voxels marked as no-end.
    pub n_forbidden: u32,
}

impl NoEndMask {
    /// Build the mask. `factor` scales Otsu's threshold — DSI-Studio's autotrack
    /// uses 0.6 by default ("not-end fixel-otsu factor"), so a voxel's primary
    /// peak must exceed 60% of the Otsu threshold to be flagged.
    pub fn build(dataset: &OdxDataset, grid: &VoxelGrid, factor: f32) -> Result<Self> {
        let amplitudes = dataset
            .scalar_dpf_f32("amplitude")
            .map_err(|_| anyhow::anyhow!("ODX has no `dpf/amplitude` — can't build no-end mask"))?;
        let offsets = dataset.offsets();
        let nb_voxels = dataset.nb_voxels();

        // Primary peak per masked voxel = max QA across that voxel's fixels.
        // (Our connectivity filter has already implicitly used this convention
        // — peak is "primary" if it has the largest amplitude.)
        let mut primary_qa: Vec<f32> = Vec::with_capacity(nb_voxels);
        for v in 0..nb_voxels {
            let s = offsets[v] as usize;
            let e = offsets[v + 1] as usize;
            let mut m = 0.0_f32;
            for i in s..e {
                if amplitudes[i] > m {
                    m = amplitudes[i];
                }
            }
            primary_qa.push(m);
        }

        let otsu = compute_otsu(&primary_qa);
        let threshold = otsu * factor;

        let total = grid.dims[0] * grid.dims[1] * grid.dims[2];
        let mut data = vec![0_u8; total];
        let ijks = dataset.compact_to_ijk();
        let mut n_forbidden: u32 = 0;
        for (compact, ijk) in ijks.iter().enumerate() {
            if primary_qa[compact] > threshold {
                let flat = (ijk[0] as usize) * grid.dims[1] * grid.dims[2]
                    + (ijk[1] as usize) * grid.dims[2]
                    + (ijk[2] as usize);
                data[flat] = 1;
                n_forbidden += 1;
            }
        }

        Ok(Self {
            data,
            dims: grid.dims,
            world_to_voxel: grid.world_to_voxel,
            otsu_threshold: otsu,
            n_forbidden,
        })
    }

    /// Is the voxel containing `world_pos` flagged as a no-end voxel?
    /// Returns false if `world_pos` is outside the grid.
    pub fn is_no_end(&self, world_pos: [f32; 3]) -> bool {
        let v = self.world_to_voxel
            * Vector4::new(
                world_pos[0] as f64,
                world_pos[1] as f64,
                world_pos[2] as f64,
                1.0,
            );
        let i = (v.x + 0.5).floor() as i32;
        let j = (v.y + 0.5).floor() as i32;
        let k = (v.z + 0.5).floor() as i32;
        if i < 0
            || j < 0
            || k < 0
            || i as usize >= self.dims[0]
            || j as usize >= self.dims[1]
            || k as usize >= self.dims[2]
        {
            return false;
        }
        let flat = (i as usize) * self.dims[1] * self.dims[2]
            + (j as usize) * self.dims[2]
            + (k as usize);
        self.data[flat] != 0
    }
}

/// Standard Otsu's method on a positive-valued sample. Returns 0.0 if the
/// sample is empty or has no positive values.
fn compute_otsu(values: &[f32]) -> f32 {
    let pos: Vec<f32> = values.iter().copied().filter(|&v| v > 0.0).collect();
    if pos.is_empty() {
        return 0.0;
    }
    let max = pos.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if max <= 0.0 {
        return 0.0;
    }

    const NBINS: usize = 256;
    let bin_w = max / NBINS as f32;
    let mut hist = [0_u64; NBINS];
    for &v in &pos {
        let mut b = (v / bin_w) as usize;
        if b >= NBINS {
            b = NBINS - 1;
        }
        hist[b] += 1;
    }

    let total: u64 = pos.len() as u64;
    let sum: f64 = (0..NBINS).map(|i| (i as f64 + 0.5) * bin_w as f64 * hist[i] as f64).sum();

    let mut w_b: u64 = 0;
    let mut sum_b: f64 = 0.0;
    let mut max_var: f64 = 0.0;
    let mut best_t: f32 = 0.0;
    for i in 0..NBINS {
        w_b += hist[i];
        if w_b == 0 {
            continue;
        }
        let w_f = total - w_b;
        if w_f == 0 {
            break;
        }
        sum_b += (i as f64 + 0.5) * bin_w as f64 * hist[i] as f64;
        let mu_b = sum_b / w_b as f64;
        let mu_f = (sum - sum_b) / w_f as f64;
        let var = (w_b as f64) * (w_f as f64) * (mu_b - mu_f).powi(2);
        if var > max_var {
            max_var = var;
            best_t = (i as f32 + 1.0) * bin_w; // upper edge of the bin
        }
    }
    best_t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn otsu_separates_bimodal() {
        // Two clusters: low values around 0.1, high values around 1.0.
        let mut v = Vec::new();
        for _ in 0..100 {
            v.push(0.1);
        }
        for _ in 0..100 {
            v.push(1.0);
        }
        let t = compute_otsu(&v);
        assert!(t > 0.1 && t < 1.0, "expected threshold between modes, got {t}");
    }

    #[test]
    fn otsu_handles_empty() {
        assert_eq!(compute_otsu(&[]), 0.0);
    }
}
