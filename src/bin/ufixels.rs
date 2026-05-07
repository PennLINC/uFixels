use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum TrxGrouping {
    /// `super` if `--link-*` ran, else `sheet`.
    Auto,
    /// Emit sheet_NNNN groups (and sheet_unassigned).
    Sheet,
    /// Emit super_NNNN groups (and sheet_unassigned).
    Super,
}
use odx_tractography::{FixelIndex, PttParams, Tractogram, TrxDType, TrxHeader};
use odx_rs::OdxDataset;
use ufixels::{
    auto_select_eps, cluster_sheets, direction_gated_link, filtration_link, merge_sheets,
    new_selection_mask, ptt_continuation_link, run_connectivity, run_hemisphere,
    snapshot_selection_mask,
    sheet_link::{
        direction::DirectionGatedParams, filtration::FiltrationParams, ptt_link::PttLinkParams,
        SheetGraph,
    },
    sheets::compute_neighbor_pairs,
    surface_io::{read_surface, write_shape},
    surface_mask::SurfaceVoxelSet,
    traversal::VoxelGrid,
    ConnectivityConfig, CortexKdTree, MergeEdge, NoEndMask, SheetParams, UfixelsConfig,
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

    /// Append the per-fixel ufixels DPFs (selection masks, hit_count, sheet_id)
    /// to the ODX at this path **in place**. If you want to keep the source
    /// ODX pristine, copy it first (`cp source.odx target.odx`) and pass the
    /// copy here. The full ODX is NOT rewritten — only the DPFs are appended,
    /// so this adds only a few hundred KB regardless of the ODX size.
    #[arg(long)]
    append_dpfs_to_odx: Option<PathBuf>,

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

    /// Run the connectivity filter (PTT trace + hit count + endpoint check)
    /// over the geometric candidate set. Roughly doubles runtime; off by default.
    #[arg(long)]
    connectivity_filter: bool,
    /// Minimum number of u-fiber-shaped peer trajectories that must vote for a
    /// candidate. Only trajectories that themselves pass the endpoint AND
    /// length checks get to contribute hits, so this is a meaningful threshold.
    #[arg(long, default_value_t = 3)]
    min_hits: u16,
    /// Minimum total trajectory length (mm) for it to be a u-fiber candidate.
    #[arg(long, default_value_t = 5.0)]
    min_total_length_mm: f32,
    /// Maximum total trajectory length (mm). U-fibers are short.
    #[arg(long, default_value_t = 35.0)]
    max_total_length_mm: f32,
    /// PTT max curvature (1/mm). 0.5 ⇒ min radius ~2 mm (sharp U-bend).
    #[arg(long, default_value_t = 0.5)]
    ptt_k_max: f32,
    /// PTT probe arc length (mm). Shorter → can navigate tighter bends.
    #[arg(long, default_value_t = 3.0)]
    ptt_probe_mm: f32,
    /// Maximum PTT trajectory length per direction (mm). Total trajectory ≤
    /// 2× this. 10 mm matches superficial-u-fiber depth — long enough that a
    /// genuine u-fiber from its apex reaches cortex on both ends; short enough
    /// that CC/SLF traces stay in deep WM and fail the endpoint check.
    #[arg(long, default_value_t = 10.0)]
    max_trace_length_mm: f32,
    /// Endpoint distance threshold (mm) to nearest cortical vertex.
    #[arg(long, default_value_t = 3.0)]
    endpoint_max_mm: f32,
    /// Reject trajectories whose endpoints land in different hemispheres
    /// (anatomically impossible for a u-fiber). Default true.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    reject_cross_hemisphere: bool,
    /// Minimum max-deviation-from-chord (mm). Filters out near-straight
    /// trajectories (cingulum, tangent runs of long bundles) that satisfy
    /// length+endpoint criteria but don't actually arc. 0 disables.
    #[arg(long, default_value_t = 2.0)]
    min_arc_deviation_mm: f32,
    /// Maximum allowed distance from cortex (mm) at any point along the
    /// trajectory. Real u-fibers stay shallow throughout; cortico-callosal
    /// projection arcs dive 15+ mm into WM. 0 disables.
    #[arg(long, default_value_t = 8.0)]
    max_path_depth_mm: f32,
    /// Reject trajectories whose endpoints land in high-QA WM voxels
    /// (DSI-Studio autotrack "no-end" idea). A primary-peak QA above
    /// `factor × Otsu` flags the voxel as forbidden — real u-fibers
    /// terminate at low-QA cortex, while CC/longitudinal-bundle traces
    /// hit the length cap inside fibrous WM. DSI-Studio uses 0.6 for
    /// general tractography; for u-fibre detection we want only the very
    /// densest bundles (CC, IFOF body) flagged, so 2.0 is the right
    /// operating point. Set to 0 to disable.
    #[arg(long, default_value_t = 2.0)]
    no_end_otsu_factor: f32,
    /// Optional TRX file to dump the surviving u-fiber-shaped trajectories
    /// for visualisation in trxviz.
    #[arg(long)]
    debug_trx: Option<PathBuf>,
    /// Which group set to emit in the TRX. TRXViz colours by *all* groups in
    /// last-wins order, and TRX zip iteration is nondeterministic, so emitting
    /// both `sheet_*` and `super_*` makes "color by group" arbitrary. Pick one.
    /// `auto` = `super` when sheet linking ran, else `sheet`.
    #[arg(long, value_enum, default_value_t = TrxGrouping::Auto)]
    trx_group_by: TrxGrouping,

    /// Cluster surviving streamlines into discrete u-fiber sheets via
    /// pointwise-distance connected components (requires --connectivity-filter).
    #[arg(long)]
    cluster_sheets: bool,
    /// Max pointwise distance (mm) between two streamlines for them to count
    /// as sheet-neighbours. Smaller = more sheets / sharper splits. Empirical
    /// sweet spot ≈ 3 mm; ε=4 percolates into hemisphere-spanning giants.
    #[arg(long, default_value_t = 3.0)]
    sheet_eps_mm: f32,
    /// Drop sheets with fewer than this many streamlines.
    #[arg(long, default_value_t = 5)]
    min_sheet_size: usize,

    /// Auto-select ε by sweeping and picking the largest ε that keeps the
    /// biggest sheet at or below `--giant-cap-fraction` of all assigned
    /// streamlines (i.e. just below the percolation transition). Overrides
    /// `--sheet-eps-mm` when set.
    #[arg(long)]
    auto_sheet_eps: bool,
    /// Lower bound for the auto-eps sweep (mm).
    #[arg(long, default_value_t = 1.5)]
    auto_eps_min_mm: f32,
    /// Upper bound for the auto-eps sweep (mm).
    #[arg(long, default_value_t = 5.0)]
    auto_eps_max_mm: f32,
    /// Step (mm) for the auto-eps sweep.
    #[arg(long, default_value_t = 0.5)]
    auto_eps_step_mm: f32,
    /// Maximum allowed largest-sheet fraction. Auto-eps picks the largest ε
    /// that keeps `largest_sheet / total_assigned ≤ this`. 0.20 sits well
    /// below the typical percolation jump.
    #[arg(long, default_value_t = 0.20)]
    giant_cap_fraction: f32,

    /// After clustering, merge each still-unassigned (or sub-min-size)
    /// streamline into its nearest assigned sheet whose midpoint is within
    /// this many mm. 0 = off (the unassigned bucket stays as-is). 5–8 mm
    /// closes most gaps without dragging in genuinely isolated singletons.
    #[arg(long, default_value_t = 0.0)]
    reclaim_radius_mm: f32,

    /// Delete unassigned streamlines (sheet_id == 0) entirely from the
    /// outputs: skipped from the TRX, their seed fixels removed from the
    /// `ufixel_selected_connectivity` DPF. Combine with `--min-sheet-size`
    /// to also remove sub-threshold "tiny" sheets (those become unassigned
    /// before this filter runs).
    #[arg(long)]
    drop_unassigned: bool,

    /// Run Linker A (direction-gated proximity merge) over the sheet set.
    #[arg(long)]
    link_direction_gated: bool,
    /// Linker A: max-pointwise-distance ceiling (mm) for cross-sheet candidate pairs.
    #[arg(long, default_value_t = 4.0)]
    merge_eps_mm: f32,
    /// Linker A & B: tangent-agreement floor at the contact pair (cosine).
    #[arg(long, default_value_t = 0.85)]
    merge_cos_min: f32,

    /// Run Linker B (filtration over the cross-sheet pair distance graph).
    /// Reuses the pair list cached at the auto-eps upper bound.
    #[arg(long)]
    link_filtration: bool,
    /// Linker B: dendrogram cut-level (mm). Sheet pairs whose first
    /// veto-passing pair is at distance ≤ this get linked.
    #[arg(long, default_value_t = 4.0)]
    filtration_cut_mm: f32,

    /// Run Linker C (PTT continuation across sheet boundaries).
    #[arg(long)]
    link_ptt: bool,
    /// Linker C: maximum continuation length (mm) per endpoint.
    #[arg(long, default_value_t = 6.0)]
    ptt_continuation_mm: f32,

    /// Final resolver threshold on linkage_distance (mm for all three
    /// linkers). Edges with `linkage_distance > this` are dropped before
    /// union-find. The percolation cliff sits between 1.5 and 2.0 mm
    /// once the no-end filter is on (filtered data is sparser, so the
    /// linker edges pile up at smaller distances). Sweep manually if your
    /// data has unusual fixel density. Watch the `top sizes` output: if
    /// the top super-sheet exceeds ~20% of all assigned streamlines,
    /// you've crossed the percolation transition and the merge isn't
    /// anatomically meaningful.
    #[arg(long, default_value_t = 1.5)]
    supersheet_threshold: f32,
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
    let dataset = OdxDataset::open(&cli.odx)
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

    let mask_bytes = snapshot_selection_mask(&selected);
    let n_selected: u32 = mask_bytes.iter().map(|&b| b as u32).sum();
    eprintln!(
        "phase 1 selection: {}/{} fixels ({:.2}%)",
        n_selected,
        mask_bytes.len(),
        100.0 * n_selected as f32 / mask_bytes.len() as f32
    );

    let mut conn_result: Option<ufixels::ConnectivityResult> = None;
    let mut sheet_result: Option<ufixels::SheetResult> = None;
    let mut supersheet_result: Option<ufixels::SuperSheetResult> = None;
    if cli.connectivity_filter {
        eprintln!("building FixelIndex");
        let fixel_index = FixelIndex::build(&dataset);
        eprintln!("building cortex KD-tree (excluding medial-wall vertices)");
        let cortex = CortexKdTree::from_paired_surfaces(&lh_wm, &lh_pial, &rh_wm, &rh_pial);
        eprintln!(
            "  cortex KD-tree: {} vertices, {} medial-wall vertices skipped",
            cortex.n_vertices(),
            cortex.n_skipped_medial()
        );

        let no_end_mask: Option<NoEndMask> = if cli.no_end_otsu_factor > 0.0 {
            match NoEndMask::build(&dataset, &grid, cli.no_end_otsu_factor) {
                Ok(m) => {
                    eprintln!(
                        "  no-end mask: Otsu={:.4}, factor={}, {} voxels forbidden as endpoints",
                        m.otsu_threshold, cli.no_end_otsu_factor, m.n_forbidden
                    );
                    Some(m)
                }
                Err(e) => {
                    eprintln!("  no-end mask disabled: {}", e);
                    None
                }
            }
        } else {
            eprintln!("  no-end mask disabled (--no-end-otsu-factor 0)");
            None
        };

        let conn_config = ConnectivityConfig {
            probe: PttParams {
                probe_length_mm: cli.ptt_probe_mm,
                k_max: cli.ptt_k_max,
                ..ConnectivityConfig::defaults().probe
            },
            max_length_mm: cli.max_trace_length_mm,
            endpoint_max_mm: cli.endpoint_max_mm,
            min_total_length_mm: cli.min_total_length_mm,
            max_total_length_mm: cli.max_total_length_mm,
            min_hits: cli.min_hits,
            reject_cross_hemisphere: cli.reject_cross_hemisphere,
            min_arc_deviation_mm: cli.min_arc_deviation_mm,
            max_path_depth_mm: cli.max_path_depth_mm,
            ..ConnectivityConfig::defaults()
        };
        eprintln!(
            "running connectivity filter (k_max={}, probe={}mm, dir_max={}mm, total_len=[{},{}]mm, min_hits={}, cross_hemi_reject={}, min_arc_dev={}mm, max_path_depth={}mm)",
            conn_config.probe.k_max,
            conn_config.probe.probe_length_mm,
            conn_config.max_length_mm,
            conn_config.min_total_length_mm,
            conn_config.max_total_length_mm,
            conn_config.min_hits,
            conn_config.reject_cross_hemisphere,
            conn_config.min_arc_deviation_mm,
            conn_config.max_path_depth_mm
        );
        let need_polylines = cli.debug_trx.is_some() || cli.cluster_sheets;
        let mut result = run_connectivity(
            &conn_config,
            &fixel_index,
            &cortex,
            no_end_mask.as_ref(),
            &mask_bytes,
            need_polylines,
        );
        let n_kept: u32 = result.keep.iter().map(|&b| b as u32).sum();
        let n_traj_pass: u32 = result.trajectory_pass.iter().map(|&b| b as u32).sum();
        let n_hits_meet: usize = result.hit_count.iter().filter(|&&h| h >= cli.min_hits).count();
        // Diagnostics: trajectory length / endpoint distance distributions over candidates.
        let mut lengths: Vec<f32> = (0..result.trajectory_length_mm.len())
            .filter(|&i| mask_bytes[i] != 0)
            .map(|i| result.trajectory_length_mm[i])
            .collect();
        lengths.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut endpts: Vec<f32> = (0..result.endpoint_min_dist_mm.len())
            .filter(|&i| mask_bytes[i] != 0)
            .map(|i| result.endpoint_min_dist_mm[i])
            .collect();
        endpts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let pct = |v: &Vec<f32>, q: f32| -> f32 {
            if v.is_empty() { return 0.0; }
            let i = ((v.len() - 1) as f32 * q / 100.0).round() as usize;
            v[i]
        };
        eprintln!(
            "  trace length (mm) p25/p50/p75/p95: {:.2} / {:.2} / {:.2} / {:.2}",
            pct(&lengths, 25.0), pct(&lengths, 50.0), pct(&lengths, 75.0), pct(&lengths, 95.0)
        );
        eprintln!(
            "  endpoint→cortex (mm worst-of-pair) p25/p50/p75/p95: {:.2} / {:.2} / {:.2} / {:.2}",
            pct(&endpts, 25.0), pct(&endpts, 50.0), pct(&endpts, 75.0), pct(&endpts, 95.0)
        );
        eprintln!(
            "  trajectory_pass (own trace u-fiber-shaped):  {} ({:.2}% of phase 1)",
            n_traj_pass,
            100.0 * n_traj_pass as f32 / n_selected.max(1) as f32
        );
        eprintln!(
            "  hit_count >= {}:                              {}",
            cli.min_hits, n_hits_meet
        );
        eprintln!(
            "  final keep (trajectory_pass AND hits >= K):  {} ({:.2}% of phase 1)",
            n_kept,
            100.0 * n_kept as f32 / n_selected.max(1) as f32
        );

        if cli.cluster_sheets {
            let sp = SheetParams {
                eps_mm: cli.sheet_eps_mm,
                min_sheet_size: cli.min_sheet_size,
                reclaim_radius_mm: if cli.reclaim_radius_mm > 0.0 {
                    Some(cli.reclaim_radius_mm)
                } else {
                    None
                },
                ..SheetParams::defaults()
            };

            let sr = if cli.auto_sheet_eps {
                eprintln!(
                    "auto-selecting sheet ε by sweep [{}, {}] step {}mm, giant_cap={:.2}",
                    cli.auto_eps_min_mm, cli.auto_eps_max_mm, cli.auto_eps_step_mm,
                    cli.giant_cap_fraction
                );
                let (chosen, sweep, sr) = auto_select_eps(
                    &result.polylines,
                    &result.polyline_fixel_id,
                    mask_bytes.len(),
                    cli.auto_eps_min_mm,
                    cli.auto_eps_max_mm,
                    cli.auto_eps_step_mm,
                    cli.giant_cap_fraction,
                    &sp,
                );
                eprintln!("  ε     sheets  assigned        largest        largest_frac");
                for s in &sweep {
                    let marker = if (s.eps_mm - chosen).abs() < 1e-3 { " <- picked" } else { "" };
                    eprintln!(
                        "  {:>4.2}  {:>6}  {:>8}        {:>7}        {:>5.3}{}",
                        s.eps_mm, s.n_sheets, s.n_assigned, s.largest_sheet_size, s.largest_sheet_fraction, marker
                    );
                }
                eprintln!("  selected ε = {}mm", chosen);
                sr
            } else {
                eprintln!(
                    "clustering sheets (eps={}mm, min_size={})",
                    sp.eps_mm, sp.min_sheet_size
                );
                cluster_sheets(
                    &result.polylines,
                    &result.polyline_fixel_id,
                    mask_bytes.len(),
                    &sp,
                )
            };

            let n_sheets = sr.sheet_sizes.len().saturating_sub(1);
            let n_assigned: u32 = sr.streamline_sheet_id.iter().filter(|&&s| s > 0).count() as u32;
            let mut top: Vec<u32> = sr.sheet_sizes[1..].to_vec();
            top.sort_by(|a, b| b.cmp(a));
            let top_str: String = top
                .iter()
                .take(10)
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            eprintln!(
                "  {} sheets, {} streamlines assigned ({:.2}% of survivors); top sizes: {}",
                n_sheets,
                n_assigned,
                100.0 * n_assigned as f32 / result.polylines.len().max(1) as f32,
                top_str
            );
            sheet_result = Some(sr);
        }

        // Second-stage sheet linking. Any combination of the three linkers can
        // run; their edges get pooled and resolved by union-find.
        let want_link = cli.link_direction_gated || cli.link_filtration || cli.link_ptt;
        if want_link {
            let sr = sheet_result.as_ref().expect("sheet linking requires --cluster-sheets");
            let n_sheets = sr.sheet_sizes.len().saturating_sub(1) as u32;
            eprintln!("building SheetGraph for second-stage linking");
            let graph = SheetGraph::build(
                &result.polylines,
                &result.polyline_fixel_id,
                &sr.streamline_sheet_id,
                n_sheets,
                SheetParams::defaults().n_resample,
            );

            let mut all_edges: Vec<MergeEdge> = Vec::new();

            if cli.link_direction_gated {
                let p = DirectionGatedParams {
                    eps_merge_mm: cli.merge_eps_mm,
                    cos_min: cli.merge_cos_min,
                };
                eprintln!(
                    "running Linker A (direction-gated): eps={}mm cos_min={}",
                    p.eps_merge_mm, p.cos_min
                );
                let edges = direction_gated_link(&graph, &p);
                eprintln!("  Linker A emitted {} edges", edges.len());
                all_edges.extend(edges);
            }

            if cli.link_filtration {
                let p = FiltrationParams { cos_min: cli.merge_cos_min };
                eprintln!(
                    "running Linker B (filtration): cos_min={} cut={}mm",
                    p.cos_min, cli.filtration_cut_mm
                );
                // Re-extract pairs at the auto-eps upper bound. Slack so the
                // dendrogram has room past the chosen ε.
                let pair_eps = cli.auto_eps_max_mm.max(cli.filtration_cut_mm + 1.0);
                let pairs = {
                    use ufixels::sheets::resample_all;
                    let resampled = resample_all(
                        &result.polylines,
                        SheetParams::defaults().n_resample,
                    );
                    compute_neighbor_pairs(
                        &resampled,
                        SheetParams::defaults().n_resample,
                        SheetParams::defaults().midpoint_radius_mm.max(pair_eps * 1.5),
                        pair_eps,
                    )
                };
                let edges = filtration_link(&graph, &pairs, &p);
                let edges_below_cut: Vec<_> = edges
                    .into_iter()
                    .filter(|e| e.linkage_distance <= cli.filtration_cut_mm)
                    .collect();
                eprintln!("  Linker B emitted {} edges (below cut)", edges_below_cut.len());
                all_edges.extend(edges_below_cut);
            }

            if cli.link_ptt {
                let p = PttLinkParams {
                    continuation_mm: cli.ptt_continuation_mm,
                    ..PttLinkParams::defaults()
                };
                eprintln!(
                    "running Linker C (PTT continuation): continuation={}mm",
                    p.continuation_mm
                );
                // Need a FixelIndex; build it the same way as the connectivity
                // filter (the CONNECTIVITY block above already built one but
                // didn't expose it; rebuild rather than thread the lifetime).
                let fixel_index = odx_tractography::FixelIndex::build(&dataset);
                let edges = ptt_continuation_link(
                    &graph,
                    &fixel_index,
                    &sr.fixel_sheet_id,
                    &p,
                );
                eprintln!("  Linker C emitted {} edges", edges.len());
                all_edges.extend(edges);
            }

            eprintln!(
                "merging sheets via union-find: {} pooled edges, threshold={}",
                all_edges.len(),
                cli.supersheet_threshold
            );
            let ssr = merge_sheets(
                sr,
                &result.polyline_fixel_id,
                mask_bytes.len(),
                all_edges,
                cli.supersheet_threshold,
            );
            let n_super = ssr.supersheet_sizes.len().saturating_sub(1);
            let n_solo = ssr.supersheet_sizes.iter().skip(1).filter(|&&s| s > 0).count();
            let mut top: Vec<u32> = ssr.supersheet_sizes[1..].to_vec();
            top.sort_by(|a, b| b.cmp(a));
            let top_str: String = top
                .iter()
                .take(10)
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            eprintln!(
                "  {} super-sheets ({} non-empty); top sizes: {}",
                n_super, n_solo, top_str
            );
            eprintln!(
                "  edges_used={} edges_considered={}",
                ssr.edges_used.len(),
                ssr.edges_considered.len()
            );
            supersheet_result = Some(ssr);
        }

        // --drop-unassigned: zero the seed fixel's connectivity-mask bit for any
        // streamline that didn't end up in a sheet. The polyline list is left
        // intact; the TRX writer below skips dropped streamlines per its own
        // mask. Only meaningful when sheets were clustered (sheet_result set).
        if cli.drop_unassigned {
            if let Some(sr) = sheet_result.as_ref() {
                let mut n_dropped = 0_usize;
                for (i, &sid) in sr.streamline_sheet_id.iter().enumerate() {
                    if sid == 0 {
                        let fid = result.polyline_fixel_id[i] as usize;
                        if fid < result.keep.len() && result.keep[fid] != 0 {
                            result.keep[fid] = 0;
                            n_dropped += 1;
                        }
                    }
                }
                eprintln!(
                    "  --drop-unassigned: removed {} fixels from connectivity mask",
                    n_dropped
                );
            } else {
                eprintln!(
                    "  --drop-unassigned: ignored (requires --cluster-sheets)"
                );
            }
        }

        if let Some(trx_path) = cli.debug_trx.as_ref() {
            let h = dataset.header();
            let header = TrxHeader {
                voxel_to_rasmm: h.voxel_to_rasmm,
                dimensions: h.dimensions,
                nb_streamlines: 0,
                nb_vertices: 0,
                extra: Default::default(),
            };
            let mut trx = Tractogram::with_header(header);

            // Per-input-streamline → TRX-streamline-id (or u32::MAX if dropped).
            // When --drop-unassigned is set, sheet_id == 0 streamlines aren't
            // emitted at all and their slots are u32::MAX. Otherwise every
            // streamline gets emitted.
            let drop_unassigned_in_trx = cli.drop_unassigned && sheet_result.is_some();
            let sheet_ids_opt = sheet_result.as_ref().map(|sr| &sr.streamline_sheet_id);
            let mut trx_idx: Vec<u32> = Vec::with_capacity(result.polylines.len());
            let mut next_trx_id: u32 = 0;
            for (i, poly) in result.polylines.iter().enumerate() {
                let drop = drop_unassigned_in_trx
                    && sheet_ids_opt.is_some_and(|sids| sids[i] == 0);
                if drop {
                    trx_idx.push(u32::MAX);
                } else {
                    trx.push_streamline(poly)
                        .map_err(|e| anyhow::anyhow!("push_streamline: {e}"))?;
                    trx_idx.push(next_trx_id);
                    next_trx_id += 1;
                }
            }

            // Pick exactly one group set so TRXViz's last-wins coloring (over
            // a nondeterministic-iteration HashMap) is unambiguous.
            let want_super = match cli.trx_group_by {
                TrxGrouping::Super => supersheet_result.is_some(),
                TrxGrouping::Sheet => false,
                TrxGrouping::Auto => supersheet_result.is_some(),
            };

            if want_super {
                let sr = sheet_result.as_ref().unwrap();
                let ssr = supersheet_result.as_ref().unwrap();
                let n_super = ssr.supersheet_sizes.len().saturating_sub(1);
                for super_id in 1..=n_super as u32 {
                    let group_indices: Vec<u32> = sr
                        .streamline_sheet_id
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &sid)| {
                            if sid == 0 || trx_idx[i] == u32::MAX {
                                return None;
                            }
                            if ssr.sheet_to_supersheet.get(sid as usize).copied()
                                == Some(super_id)
                            {
                                Some(trx_idx[i])
                            } else {
                                None
                            }
                        })
                        .collect();
                    if !group_indices.is_empty() {
                        trx.insert_group(format!("super_{:04}", super_id), group_indices);
                    }
                }
                eprintln!("  TRX grouped by super-sheet ({} super_NNNN groups)", n_super);
            } else if let Some(sr) = sheet_result.as_ref() {
                let n_sheets = sr.sheet_sizes.len().saturating_sub(1);
                for sid in 1..=n_sheets as u32 {
                    let group_indices: Vec<u32> = sr
                        .streamline_sheet_id
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &s)| {
                            if s == sid && trx_idx[i] != u32::MAX {
                                Some(trx_idx[i])
                            } else {
                                None
                            }
                        })
                        .collect();
                    if !group_indices.is_empty() {
                        trx.insert_group(format!("sheet_{:04}", sid), group_indices);
                    }
                }
                eprintln!("  TRX grouped by sheet ({} sheet_NNNN groups)", n_sheets);
            }

            // Always emit the unassigned bucket if any, regardless of which
            // grouping level we chose — it's the same set of streamlines
            // (sheet_id == 0 ⇔ supersheet_id == 0 by construction).
            if let Some(sr) = sheet_result.as_ref() {
                if !drop_unassigned_in_trx {
                    let unassigned: Vec<u32> = sr
                        .streamline_sheet_id
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &s)| {
                            if s == 0 && trx_idx[i] != u32::MAX {
                                Some(trx_idx[i])
                            } else {
                                None
                            }
                        })
                        .collect();
                    if !unassigned.is_empty() {
                        trx.insert_group("sheet_unassigned".to_string(), unassigned);
                    }
                }
            }

            let n_written = trx.nb_streamlines();
            let any_trx = trx
                .to_trx(TrxDType::Float32)
                .map_err(|e| anyhow::anyhow!("materializing TRX: {e}"))?;
            any_trx
                .save(trx_path)
                .with_context(|| format!("writing TRX {}", trx_path.display()))?;
            eprintln!(
                "  wrote {} trajectories to {}{}",
                n_written,
                trx_path.display(),
                if drop_unassigned_in_trx {
                    format!(
                        " (dropped {} unassigned)",
                        result.polylines.len() - n_written
                    )
                } else {
                    String::new()
                }
            );
        }

        conn_result = Some(result);
    }

    if let Some(out_odx) = cli.append_dpfs_to_odx.as_ref() {
        use odx_tractography::writeback;
        eprintln!("appending ufixels DPFs in place to: {}", out_odx.display());
        writeback::write_dpf_u8(out_odx, "ufixel_selected", &mask_bytes)
            .with_context(|| "appending ufixel_selected")?;
        if let Some(result) = conn_result.as_ref() {
            writeback::write_dpf_u8(out_odx, "ufixel_selected_connectivity", &result.keep)
                .with_context(|| "appending ufixel_selected_connectivity")?;
            writeback::write_dpf_u16(out_odx, "ufixel_hit_count", &result.hit_count)
                .with_context(|| "appending ufixel_hit_count")?;
        }
        if let Some(sr) = sheet_result.as_ref() {
            writeback::write_dpf_u32(out_odx, "ufixel_sheet_id", &sr.fixel_sheet_id)
                .with_context(|| "appending ufixel_sheet_id")?;
        }
        if let Some(ssr) = supersheet_result.as_ref() {
            writeback::write_dpf_u32(out_odx, "ufixel_supersheet_id", &ssr.fixel_supersheet_id)
                .with_context(|| "appending ufixel_supersheet_id")?;
        }
        eprintln!("  done");
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
