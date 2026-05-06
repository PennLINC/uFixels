//! Voxel grid utilities: world↔voxel transforms, fast `flat→compact` lookup,
//! and Amanatides–Woo voxel-by-voxel ray traversal.

use anyhow::{anyhow, Result};
use nalgebra::{Matrix4, Vector3, Vector4};
use odx_rs::OdxDataset;

/// Pre-computed grid metadata derived from an [`OdxDataset`].
pub struct VoxelGrid {
    pub dims: [usize; 3],
    pub voxel_to_world: Matrix4<f64>,
    pub world_to_voxel: Matrix4<f64>,
    /// Flat C-order (i-slowest) index → compact (mask-only) index, or -1 if off-mask.
    pub flat_to_compact: Vec<i32>,
}

impl VoxelGrid {
    pub fn from_dataset(ds: &OdxDataset) -> Result<Self> {
        let h = ds.header();
        let dims = [
            h.dimensions[0] as usize,
            h.dimensions[1] as usize,
            h.dimensions[2] as usize,
        ];
        let total: usize = dims[0] * dims[1] * dims[2];

        let a = h.voxel_to_rasmm;
        let voxel_to_world = Matrix4::new(
            a[0][0], a[0][1], a[0][2], a[0][3], a[1][0], a[1][1], a[1][2], a[1][3], a[2][0],
            a[2][1], a[2][2], a[2][3], a[3][0], a[3][1], a[3][2], a[3][3],
        );
        let world_to_voxel = voxel_to_world
            .try_inverse()
            .ok_or_else(|| anyhow!("voxel_to_rasmm affine is singular"))?;

        let mask = ds.mask();
        if mask.len() != total {
            return Err(anyhow!(
                "mask length {} does not match dims product {}",
                mask.len(),
                total
            ));
        }
        let mut flat_to_compact = vec![-1_i32; total];
        let mut compact = 0_i32;
        for (i, &m) in mask.iter().enumerate() {
            if m != 0 {
                flat_to_compact[i] = compact;
                compact += 1;
            }
        }
        if compact as usize != ds.nb_voxels() {
            return Err(anyhow!(
                "mask non-zero count {} != header nb_voxels {}",
                compact,
                ds.nb_voxels()
            ));
        }

        Ok(Self {
            dims,
            voxel_to_world,
            world_to_voxel,
            flat_to_compact,
        })
    }

    #[inline]
    pub fn flat_index(&self, ijk: [i32; 3]) -> Option<usize> {
        let [i, j, k] = ijk;
        if i < 0
            || j < 0
            || k < 0
            || i as usize >= self.dims[0]
            || j as usize >= self.dims[1]
            || k as usize >= self.dims[2]
        {
            return None;
        }
        Some((i as usize) * self.dims[1] * self.dims[2] + (j as usize) * self.dims[2] + (k as usize))
    }

    /// Compact (mask-only) index for an `ijk` cell, or `None` if off-mask / off-grid.
    #[inline]
    pub fn compact_index(&self, ijk: [i32; 3]) -> Option<usize> {
        let flat = self.flat_index(ijk)?;
        let c = self.flat_to_compact[flat];
        if c < 0 {
            None
        } else {
            Some(c as usize)
        }
    }

    /// World-space (RAS+mm) → fractional voxel index. Voxel centers are at integer ijk.
    #[inline]
    pub fn world_to_voxel_frac(&self, p: [f32; 3]) -> [f64; 3] {
        let v = self.world_to_voxel
            * Vector4::new(p[0] as f64, p[1] as f64, p[2] as f64, 1.0);
        [v.x, v.y, v.z]
    }

    /// Floor a fractional ijk to the integer voxel containing the point. The
    /// "voxel centered at integer ijk" convention used here means the cell with
    /// integer index `i` spans `[i - 0.5, i + 0.5)` along that axis.
    #[inline]
    pub fn voxel_of(&self, p: [f32; 3]) -> [i32; 3] {
        let f = self.world_to_voxel_frac(p);
        [
            (f[0] + 0.5).floor() as i32,
            (f[1] + 0.5).floor() as i32,
            (f[2] + 0.5).floor() as i32,
        ]
    }
}

/// Amanatides–Woo voxel-by-voxel ray walker, working in fractional voxel space
/// where each cell `[i, j, k]` spans `[i - 0.5, i + 0.5)` along each axis.
///
/// Yields `(ijk, segment_length_world_mm)` pairs where `segment_length` is the
/// world-space distance the ray travels *while inside* `ijk`.
pub struct RayWalker {
    /// World-mm per unit of voxel-space ray parameter `t` (so `step * world_per_t = mm`).
    world_per_t: f64,
    step: [i32; 3],
    t_max: [f64; 3],
    t_delta: [f64; 3],
    current: [i32; 3],
    t_accum: f64,
}

impl RayWalker {
    pub fn new(grid: &VoxelGrid, origin_world: [f32; 3], dir_world: [f32; 3]) -> Self {
        // Convert origin and direction into voxel space using the affine.
        // Origin: full transform. Direction: rotation/scale only (no translation).
        let pos_vox4 = grid.world_to_voxel
            * Vector4::new(
                origin_world[0] as f64,
                origin_world[1] as f64,
                origin_world[2] as f64,
                1.0,
            );
        let dir_vox4 = grid.world_to_voxel
            * Vector4::new(
                dir_world[0] as f64,
                dir_world[1] as f64,
                dir_world[2] as f64,
                0.0,
            );
        let pos_vox = [pos_vox4.x, pos_vox4.y, pos_vox4.z];
        let dir_vox = [dir_vox4.x, dir_vox4.y, dir_vox4.z];

        // World-space length of one unit of voxel-space ray parameter:
        // we want to know mm per unit of `dir_vox` magnitude. Because we built
        // dir_vox by mapping dir_world (a unit vector) through world_to_voxel,
        // walking a parameter step `dt` along dir_vox in voxel space corresponds
        // to walking `dt` mm in world space (the rotation factor stays as the
        // input vector's world-mm length, which is 1).
        let dir_world_len = (dir_world[0] * dir_world[0]
            + dir_world[1] * dir_world[1]
            + dir_world[2] * dir_world[2])
            .sqrt() as f64;
        let world_per_t = dir_world_len.max(f64::EPSILON);

        // Setup Amanatides–Woo state in fractional voxel space (cells span [i-0.5, i+0.5)).
        let current = [
            (pos_vox[0] + 0.5).floor() as i32,
            (pos_vox[1] + 0.5).floor() as i32,
            (pos_vox[2] + 0.5).floor() as i32,
        ];
        let mut step = [0_i32; 3];
        let mut t_max = [f64::INFINITY; 3];
        let mut t_delta = [f64::INFINITY; 3];
        for axis in 0..3 {
            let d = dir_vox[axis];
            if d.abs() < 1.0e-12 {
                step[axis] = 0;
                t_max[axis] = f64::INFINITY;
                t_delta[axis] = f64::INFINITY;
                continue;
            }
            if d > 0.0 {
                step[axis] = 1;
                let next_boundary = current[axis] as f64 + 0.5;
                t_max[axis] = (next_boundary - pos_vox[axis]) / d;
            } else {
                step[axis] = -1;
                let prev_boundary = current[axis] as f64 - 0.5;
                t_max[axis] = (prev_boundary - pos_vox[axis]) / d;
            }
            t_delta[axis] = (1.0 / d).abs();
        }

        Self {
            world_per_t,
            step,
            t_max,
            t_delta,
            current,
            t_accum: 0.0,
        }
    }
}

impl Iterator for RayWalker {
    type Item = ([i32; 3], f32);

    fn next(&mut self) -> Option<Self::Item> {
        // Find which axis hits a boundary first.
        let axis = if self.t_max[0] < self.t_max[1] {
            if self.t_max[0] < self.t_max[2] {
                0
            } else {
                2
            }
        } else if self.t_max[1] < self.t_max[2] {
            1
        } else {
            2
        };

        let exiting = self.current;
        let t_exit = self.t_max[axis];
        let segment_t = (t_exit - self.t_accum).max(0.0);
        let segment_mm = (segment_t * self.world_per_t) as f32;

        // Advance.
        self.t_accum = t_exit;
        self.current[axis] += self.step[axis];
        self.t_max[axis] += self.t_delta[axis];

        // Guard: degenerate (zero) direction yields infinite segments.
        if !segment_mm.is_finite() {
            return None;
        }

        Some((exiting, segment_mm))
    }
}

/// Cosine of the angle between two 3-vectors (returns 0 if either is degenerate).
#[inline]
pub fn cos_angle(a: [f32; 3], b: [f32; 3]) -> f32 {
    let na = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    let nb = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
    if na < 1.0e-8 || nb < 1.0e-8 {
        return 0.0;
    }
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    dot / (na * nb)
}

#[inline]
pub fn vec3(v: Vector3<f32>) -> [f32; 3] {
    [v.x, v.y, v.z]
}
