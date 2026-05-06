//! Voxelize triangle meshes into a flat boolean grid for "did we hit a surface?" tests.

use crate::traversal::VoxelGrid;

/// Bit-packed boolean over `dims[0] * dims[1] * dims[2]` voxels.
pub struct SurfaceVoxelSet {
    bits: Vec<u8>,
    dims: [usize; 3],
}

impl SurfaceVoxelSet {
    pub fn empty(grid: &VoxelGrid) -> Self {
        let total = grid.dims[0] * grid.dims[1] * grid.dims[2];
        Self {
            bits: vec![0_u8; total.div_ceil(8)],
            dims: grid.dims,
        }
    }

    #[inline]
    pub fn contains(&self, ijk: [i32; 3]) -> bool {
        if let Some(flat) = flat_index(self.dims, ijk) {
            (self.bits[flat / 8] >> (flat & 7)) & 1 != 0
        } else {
            false
        }
    }

    fn set(&mut self, flat: usize) {
        self.bits[flat / 8] |= 1 << (flat & 7);
    }

    /// Rasterize all triangles of a mesh by barycentric sampling. Sample density
    /// is set so adjacent samples are no more than half a voxel apart along the
    /// triangle's longest edge.
    pub fn add_triangles(&mut self, grid: &VoxelGrid, vertices: &[[f32; 3]], triangles: &[[u32; 3]]) {
        for tri in triangles {
            let v0 = vertices[tri[0] as usize];
            let v1 = vertices[tri[1] as usize];
            let v2 = vertices[tri[2] as usize];

            // Mark vertices themselves.
            for v in [&v0, &v1, &v2] {
                let ijk = grid.voxel_of(*v);
                if let Some(flat) = grid.flat_index(ijk) {
                    self.set(flat);
                }
            }

            // Determine sample density from the longest edge in *voxel* space.
            let f0 = grid.world_to_voxel_frac(v0);
            let f1 = grid.world_to_voxel_frac(v1);
            let f2 = grid.world_to_voxel_frac(v2);
            let e01 = dist(f0, f1);
            let e12 = dist(f1, f2);
            let e20 = dist(f2, f0);
            let max_edge = e01.max(e12).max(e20);
            // Two samples per voxel along the longest edge → 0.5-voxel spacing.
            let n = (max_edge * 2.0).ceil().max(1.0) as u32;

            for s in 0..=n {
                for t in 0..=n.saturating_sub(s) {
                    let u = (n - s - t) as f64;
                    let inv = 1.0 / n as f64;
                    let bu = u * inv;
                    let bs = s as f64 * inv;
                    let bt = t as f64 * inv;
                    let p = [
                        (bu * v0[0] as f64 + bs * v1[0] as f64 + bt * v2[0] as f64) as f32,
                        (bu * v0[1] as f64 + bs * v1[1] as f64 + bt * v2[1] as f64) as f32,
                        (bu * v0[2] as f64 + bs * v1[2] as f64 + bt * v2[2] as f64) as f32,
                    ];
                    let ijk = grid.voxel_of(p);
                    if let Some(flat) = grid.flat_index(ijk) {
                        self.set(flat);
                    }
                }
            }
        }
    }
}

#[inline]
fn flat_index(dims: [usize; 3], ijk: [i32; 3]) -> Option<usize> {
    let [i, j, k] = ijk;
    if i < 0
        || j < 0
        || k < 0
        || i as usize >= dims[0]
        || j as usize >= dims[1]
        || k as usize >= dims[2]
    {
        return None;
    }
    Some((i as usize) * dims[1] * dims[2] + (j as usize) * dims[2] + (k as usize))
}

#[inline]
fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}
