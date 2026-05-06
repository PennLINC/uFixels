use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use odx_rs::OdxDataset;
use ufixels::{
    new_selection_mask, run_hemisphere, snapshot_selection_mask,
    surface_io::{read_surface, write_shape},
    surface_mask::SurfaceVoxelSet,
    traversal::VoxelGrid,
    UfixelsConfig,
};

#[derive(Parser, Debug)]
#[command(version, about = "Detect u-fiber-compatible fixels under cortical surfaces.")]
struct Cli {
    /// Left-hemisphere white matter surface (.surf.gii).
    #[arg(long)]
    lh_wm: PathBuf,
    /// Left-hemisphere pial surface (.surf.gii).
    #[arg(long)]
    lh_pial: PathBuf,
    /// Right-hemisphere white matter surface (.surf.gii).
    #[arg(long)]
    rh_wm: PathBuf,
    /// Right-hemisphere pial surface (.surf.gii).
    #[arg(long)]
    rh_pial: PathBuf,
    /// ODX file (directory or .odx archive) with fixels.
    #[arg(long)]
    odx: PathBuf,

    /// Prefix for output files. The CLI appends `lh.<map>.shape.gii` etc.
    #[arg(long)]
    output_prefix: PathBuf,

    /// Optional path to write a copy of the ODX with a `dpf/ufixel_selected`
    /// mask attached, for visualisation in trxviz.
    #[arg(long)]
    write_selected_odx: Option<PathBuf>,

    /// Target angle (deg) between fixel and surface normal. 90 = parallel to cortex.
    #[arg(long, default_value_t = 90.0)]
    compatible_angle: f32,
    /// Tolerance (deg) around the target angle.
    #[arg(long, default_value_t = 20.0)]
    max_angle_diff: f32,
    /// Maximum inward walk distance (mm).
    #[arg(long, default_value_t = 20.0)]
    max_depth: f32,
    /// Initial step (mm) into white matter to avoid sampling the seed voxel.
    #[arg(long, default_value_t = 0.1)]
    min_step_eps: f32,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = UfixelsConfig {
        compatible_angle_deg: cli.compatible_angle,
        max_angle_diff_deg: cli.max_angle_diff,
        max_depth_mm: cli.max_depth,
        min_step_eps_mm: cli.min_step_eps,
    };

    eprintln!("loading ODX: {}", cli.odx.display());
    let mut dataset = OdxDataset::open(&cli.odx)
        .with_context(|| format!("opening ODX {}", cli.odx.display()))?;

    eprintln!("loading surfaces");
    let lh_wm = read_surface(&cli.lh_wm)?;
    let lh_pial = read_surface(&cli.lh_pial)?;
    let rh_wm = read_surface(&cli.rh_wm)?;
    let rh_pial = read_surface(&cli.rh_pial)?;

    eprintln!("building voxel grid + flat→compact lookup");
    let grid = VoxelGrid::from_dataset(&dataset)?;

    eprintln!("voxelizing surface meshes");
    let mut surface_voxels = SurfaceVoxelSet::empty(&grid);
    for surf in [&lh_wm, &lh_pial, &rh_wm, &rh_pial] {
        surface_voxels.add_triangles(&grid, &surf.vertices, &surf.triangles);
    }

    let selected = new_selection_mask(dataset.nb_peaks());

    eprintln!("running lh ({} vertices)", lh_wm.vertices.len());
    let lh = run_hemisphere(
        &config,
        &dataset,
        &grid,
        &surface_voxels,
        &lh_wm,
        &lh_pial,
        &selected,
    )?;
    eprintln!("running rh ({} vertices)", rh_wm.vertices.len());
    let rh = run_hemisphere(
        &config,
        &dataset,
        &grid,
        &surface_voxels,
        &rh_wm,
        &rh_pial,
        &selected,
    )?;

    write_outputs(&cli.output_prefix, "lh", &lh)?;
    write_outputs(&cli.output_prefix, "rh", &rh)?;

    if let Some(out_odx) = cli.write_selected_odx.as_ref() {
        let mask_bytes = snapshot_selection_mask(&selected);
        let n_selected: u32 = mask_bytes.iter().map(|&b| b as u32).sum();
        eprintln!(
            "writing selected-fixel ODX ({}/{} fixels selected): {}",
            n_selected,
            mask_bytes.len(),
            out_odx.display()
        );
        dataset
            .insert_dpf_u8("ufixel_selected", mask_bytes)
            .with_context(|| "attaching ufixel_selected DPF")?;
        dataset
            .save(out_odx)
            .with_context(|| format!("saving ODX {}", out_odx.display()))?;
    }

    Ok(())
}

fn write_outputs(
    prefix: &std::path::Path,
    hemi: &str,
    outs: &ufixels::HemisphereOutputs,
) -> Result<()> {
    let prefix_str = prefix.as_os_str().to_string_lossy();
    let path = |suffix: &str| -> PathBuf {
        PathBuf::from(format!("{prefix_str}{hemi}.{suffix}.shape.gii"))
    };
    write_shape(&path("first_voxel_diff"), &outs.first_voxel_diff)?;
    write_shape(&path("compatible_depth"), &outs.compatible_depth)?;
    write_shape(&path("end_condition"), &outs.end_condition)?;
    Ok(())
}
