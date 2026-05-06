//! Synthetic correctness check for the per-vertex inward walk.
//!
//! Build a tiny 1×1×5 ODX along z with hand-crafted fixels and a single seed
//! vertex with normal = +z. Verify each end-condition path.

use odx_rs::{OdxBuilder, OdxDataset};
use ufixels::{
    new_selection_mask, run_hemisphere, snapshot_selection_mask,
    surface_io::LoadedSurface, surface_mask::SurfaceVoxelSet,
    traversal::VoxelGrid, EndCondition, UfixelsConfig,
};

const PERPENDICULAR_X: [f32; 3] = [1.0, 0.0, 0.0];
const PARALLEL_Z: [f32; 3] = [0.0, 0.0, 1.0];

fn identity_affine() -> [[f64; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// 1×1×5 grid with all voxels in mask. Returns a builder pre-filled with the
/// given per-voxel fixel lists (`peaks_per_voxel.len()` must equal 5).
fn build_dataset(peaks_per_voxel: &[Vec<[f32; 3]>]) -> OdxDataset {
    assert_eq!(peaks_per_voxel.len(), 5);
    let mask = vec![1_u8; 5];
    let mut builder = OdxBuilder::new(identity_affine(), [1, 1, 5], mask);
    for peaks in peaks_per_voxel {
        builder.push_voxel_peaks(peaks);
    }
    builder.finalize().expect("builder finalize")
}

fn run_one_vertex(dataset: &OdxDataset, config: &UfixelsConfig) -> (f32, f32, f32, Vec<u8>) {
    let grid = VoxelGrid::from_dataset(dataset).expect("grid");
    let surface_voxels = SurfaceVoxelSet::empty(&grid); // No surface stops in synthetic test.

    // Single paired vertex: wm at (0,0,0), pial at (0,0,-1) → inward normal = +z.
    let wm = LoadedSurface {
        vertices: vec![[0.0, 0.0, 0.0]],
        triangles: vec![],
    };
    let pial = LoadedSurface {
        vertices: vec![[0.0, 0.0, -1.0]],
        triangles: vec![],
    };

    let selected = new_selection_mask(dataset.nb_peaks());
    let outs = run_hemisphere(config, dataset, &grid, &surface_voxels, &wm, &pial, &selected)
        .expect("run_hemisphere");
    (
        outs.first_voxel_diff[0],
        outs.compatible_depth[0],
        outs.end_condition[0],
        snapshot_selection_mask(&selected),
    )
}

#[test]
fn no_compatible_fixel_path() {
    // Voxels along z: [0]=perp, [1]=perp, [2]=parallel (incompatible), [3]=perp, [4]=perp.
    // Walk should cover voxels 0 and 1, fail at 2, end with NO_COMPATIBLE_FIXEL.
    let peaks = vec![
        vec![PERPENDICULAR_X],
        vec![PERPENDICULAR_X],
        vec![PARALLEL_Z],
        vec![PERPENDICULAR_X],
        vec![PERPENDICULAR_X],
    ];
    let dataset = build_dataset(&peaks);
    let config = UfixelsConfig::default();
    let (first_diff, compat_depth, end, selected) = run_one_vertex(&dataset, &config);

    assert!(
        first_diff.abs() < 1e-3,
        "first-voxel diff should be ~0 (perpendicular fixel matches 90° target), got {first_diff}"
    );
    // Seed (0,0,0) lives on the boundary of voxel [0,0,0] (cells span [i-0.5, i+0.5)).
    // After the +z eps step, walker covers [0,0,0] (segment 0.4mm), then [0,0,1]
    // (segment 1.0mm), then [0,0,2] which has the parallel-z (incompatible) fixel.
    // Surface-set is empty so no surface-stop interference. compatible_depth = 1.4mm.
    assert!(
        (1.39..=1.41).contains(&compat_depth),
        "compatible_depth should be ~1.4mm (segments through [0,0,0] and [0,0,1]), got {compat_depth}"
    );
    assert_eq!(end as i32, EndCondition::NoCompatibleFixel as i32);

    // Two fixels (in voxels 0 and 1) should be marked selected. Voxels 3 and 4 are
    // perpendicular but never reached.
    let n_selected: u32 = selected.iter().map(|&b| b as u32).sum();
    assert_eq!(n_selected, 2);
    assert_eq!(selected[0], 1);
    assert_eq!(selected[1], 1);
}

#[test]
fn max_depth_path() {
    // All five voxels perpendicular → walk should hit max_depth.
    let peaks = vec![
        vec![PERPENDICULAR_X],
        vec![PERPENDICULAR_X],
        vec![PERPENDICULAR_X],
        vec![PERPENDICULAR_X],
        vec![PERPENDICULAR_X],
    ];
    let dataset = build_dataset(&peaks);
    let config = UfixelsConfig {
        max_depth_mm: 2.0, // small cap so we hit it before running out of voxels
        ..Default::default()
    };
    let (_first, compat_depth, end, _selected) = run_one_vertex(&dataset, &config);
    assert!(
        compat_depth >= 2.0 - 1e-3,
        "compatible_depth should reach max_depth (2.0mm), got {compat_depth}"
    );
    assert_eq!(end as i32, EndCondition::MaxDepth as i32);
}

#[test]
fn left_mask_path() {
    // Mask only voxels 0 and 1 → walking off the end should produce LEFT_MASK.
    let mut mask = vec![1_u8; 5];
    mask[2] = 0;
    mask[3] = 0;
    mask[4] = 0;
    let mut builder = OdxBuilder::new(identity_affine(), [1, 1, 5], mask);
    builder.push_voxel_peaks(&[PERPENDICULAR_X]);
    builder.push_voxel_peaks(&[PERPENDICULAR_X]);
    let dataset = builder.finalize().unwrap();

    let config = UfixelsConfig::default();
    let (_first, _compat, end, _sel) = run_one_vertex(&dataset, &config);
    assert_eq!(end as i32, EndCondition::LeftMask as i32);
}

#[test]
fn empty_first_voxel_records_nan_diff_and_stops() {
    // First masked voxel has zero fixels → not compatible, walk ends with
    // NO_COMPATIBLE_FIXEL and first_voxel_diff is NaN.
    let peaks = vec![
        vec![],                  // [0,0,0] — first voxel walker enters
        vec![PERPENDICULAR_X],
        vec![PERPENDICULAR_X],
        vec![PERPENDICULAR_X],
        vec![PERPENDICULAR_X],
    ];
    let dataset = build_dataset(&peaks);
    let config = UfixelsConfig::default();
    let (first_diff, compat_depth, end, selected) = run_one_vertex(&dataset, &config);

    assert!(
        first_diff.is_nan(),
        "first_voxel_diff should be NaN when first masked voxel has no fixels, got {first_diff}"
    );
    assert!(compat_depth < 1e-3, "compatible_depth should be ~0, got {compat_depth}");
    assert_eq!(end as i32, EndCondition::NoCompatibleFixel as i32);
    assert!(selected.iter().all(|&b| b == 0));
}
