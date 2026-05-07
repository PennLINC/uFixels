# ufixels methods

A chronicle of the methodology, dead ends, and parameter sweeps that led to
the current ufixels pipeline. The crate's user-facing intro is in
[`README.md`](README.md); this document is for posterity and for tuning.

---

## Pipeline overview (current best)

Three phases, applied sequentially. Each phase consumes the output of the
previous one and either rejects or refines candidates.

```
Phase 1 — Geometric criterion (per cortical vertex)
  walk inward along the surface normal, find fixels tangent to cortex
       │
       ▼  selected fixels (~89k on a 392k-fixel adult ODX)
Phase 2 — Connectivity filter (per candidate fixel)
  PTT trace, multi-criterion gate, peer-vote hit count
       │
       ▼  surviving candidates (~41k of 89k = 46%)
Phase 3 — Sheet clustering (per surviving streamline)
  pointwise-distance neighbour graph, union-find connected components,
  auto-ε at percolation safe point, reclamation of singletons
       │
       ▼  discrete sheets (~664 sheets, top sheet 5,253 streamlines)
```

Outputs:

| File | Content |
| --- | --- |
| `*lh.first_voxel_diff.shape.gii` | per-vertex degrees from target tangent angle in first WM voxel |
| `*lh.compatible_depth.shape.gii` | per-vertex inward walk depth (mm) |
| `*lh.end_condition.shape.gii` | per-vertex enum: 0=NoCompatibleFixel, 1=HitSurface, 2=MaxDepth, 3=LeftMask, NaN=invalid |
| `*rh.*` | same three for right hemisphere |
| `dpf/ufixel_selected` | uint8 mask, Phase 1 candidates |
| `dpf/ufixel_selected_connectivity` | uint8 mask, Phase 2 survivors |
| `dpf/ufixel_hit_count` | uint16 peer-vote count |
| `dpf/ufixel_sheet_id` | uint32 sheet membership |
| `--debug-trx` TRX | one streamline per Phase 2 survivor, grouped by sheet |

---

## Phase 1 — Geometric criterion

**Inputs:** four GIFTI surfaces (lh.wm, lh.pial, rh.wm, rh.pial) with paired
vertex correspondence; one ODX dataset.

**Algorithm.** For each cortical vertex `i`:

1. Compute inward normal `n_i = normalize(wm[i] - pial[i])`.
2. Position `p = wm[i] + min_step_eps · n_i`. Walk Amanatides–Woo voxel-by-voxel.
3. At each voxel, look up the fixels and compute angles to `n_i` (with
   antipodal symmetry: `angle = acos(|d · n|)`).
4. **Compatibility test:** `|angle − target| ≤ tolerance`. Default
   `target = 90°` (fixel tangent to cortex), `tolerance = 20°`.
5. End conditions: `NoCompatibleFixel`, `HitSurface` (voxelised
   surface-mesh shell, with one-voxel seed exclusion), `MaxDepth` (default
   20 mm), `LeftMask`.

**Bug found and fixed during testing.** Initially I computed compatibility
via `|cos(angle)| ∈ [cos_lo, cos_hi]`. This collapses when `target = 90°`
because `|cos|` folds back: `|cos(90°+x)| = |cos(90°−x)|`. Fix: clamp the
angle interval to `[0, π/2]` first, then convert to cosine. ([ufixel.rs:51](src/ufixel.rs))

---

## Phase 2 — Connectivity filter

**The motivation.** Phase 1 selects ~22.7% of all fixels (89k of 392k on
the test data). Visual inspection of trxviz shows clear contamination:
SLF, IFOF body, callosal radiations passing tangent under cortex
("tangent leakage"). Phase 2's job is to reject these without losing
real u-fibers.

**The mechanism.** For each Phase 1 candidate fixel, we run a fresh
unreferenced PTT trace from the fixel's position with the fixel's
direction as the initial tangent. Forward + backward propagation,
~10 mm per direction, sharp-curvature parameters (`k_max = 0.5`,
`n_k_samples = 7`, `probe_length = 3 mm`).

The trace either looks u-fibre-shaped (passes a battery of checks) or
doesn't. Surviving fixels also accumulate **peer hits**: each candidate's
trace contributes a vote to every other candidate fixel it visits.

### The seven gates a trajectory must pass

A trace is u-fibre-shaped iff *all* of these hold:

1. **Length** in `[min_total_length_mm, max_total_length_mm]` (default 5–35 mm).
2. **Both endpoints near cortex** (default ≤ 3 mm from nearest pial/wm
   vertex, medial-wall vertices excluded).
3. **Same hemisphere** at both endpoints. Anatomically u-fibres can't
   cross the midline.
4. **Bow from chord** ≥ `min_arc_deviation_mm` (default 2 mm). Filters
   out near-straight trajectories: cingulum, IFOF tangent runs.
5. **Max depth from cortex** ≤ `max_path_depth_mm` (default 8 mm).
   Real u-fibres stay shallow throughout.
6. **No-end mask:** neither endpoint sits in a voxel whose primary-peak
   QA exceeds `factor × Otsu_threshold` (default factor=2.0). Borrowed
   from DSI-Studio autotrack — fibres that "end" in dense WM didn't
   actually terminate, they just hit the trace cap.
7. **Peer hits** ≥ `min_hits` (default 3). Counted *only* from other
   candidates whose trace was itself u-fibre-shaped.

### Why hit count needed trajectory-shape gating

The first hit-count implementation counted every cross-fixel visit. With
80k candidates each tracing 50 fixels worth of trajectory, the median
candidate ended up with 41 hits and a CC-tangent SLF segment had MORE
hits than a real u-fibre. Hit count became a percolation-edge measure of
fixel density, not connectivity quality.

The fix: each candidate's vote only counts if **its own trajectory was
u-fibre-shaped** — endpoints near cortex, length within range, etc.
A CC-tangent fixel's trace dives into deep WM, fails the shape gate,
and silently doesn't get to vote for its CC-neighbour peers.
([connectivity.rs:181-228](src/connectivity.rs))

### Why DSI-Studio's autotrack factor of 0.6 didn't work

DSI-Studio uses `0.6 × Otsu` as the no-end threshold for general
tractography — flag voxels where primary-peak QA exceeds 60% of the
Otsu split. On this data, Otsu = 0.172, so `0.6 × Otsu` flags 127,258
voxels as forbidden. That's most of the WM.

The reason: Otsu is computed over masked-voxel QA, which separates "real
WM" from "low-QA WM edges". The factor 0.6 lands deep into the WM class.
For u-fibre detection we want only the *very* densest bundles flagged.
Factor 2.0 (= 1.2 × Otsu) flags 8,524 voxels — CC body, IFOF core,
brainstem cores. ([no_end_mask.rs](src/no_end_mask.rs))

---

## Phase 3 — Sheet clustering

**Why streamlines, not fixels.** Initially I considered clustering fixels
by spatial+directional connected components. But under gyral crowns,
fixels from many u-fibres converge into the same WM voxel — fixel-CC
clustering would either merge unrelated sheets or splinter at the
crowns. A streamline preserves endpoint-pair information, so two
streamlines crossing through the same crown voxel can still belong to
different sheets if their endpoints differ.

**The metric.** Two streamlines `A`, `B` are sheet-neighbours iff
`max_i ‖A[i] − B[i]‖ ≤ ε` after arc-length resampling to 16 points,
with antipodal flip (`min` over forward and reverse alignments).

This is **max** pointwise distance, not mean. Mean blurs gaps; max
enforces them. A bending sheet stays one component because each
successive streamline is within ε of its neighbour — chains of small
steps allow arbitrary curvature. A gap (no streamline within ε at every
point) breaks the chain. ([sheets.rs](src/sheets.rs))

**Auto-ε via percolation cap.** Sweep ε from 1.5 to 5 mm in 0.5 steps.
At each ε, run union-find and measure `largest_sheet_size /
total_assigned`. Pick the largest ε at which that ratio stays
≤ `giant_cap_fraction` (default 0.20).

The percolation transition is sharp:

```
eps=2 mm   1101 sheets    21% assigned   top sizes: 372, 337, 148, 137, 128
eps=3 mm    909 sheets    72% assigned   top sizes: 9343, 5700, 4083, 2525, 1911   ← in-cap
eps=4 mm    476 sheets    86% assigned   top sizes: 23876, 23184, 181, ...        ← post-percolation
eps=5 mm   ...
```

(Numbers from a pre-no-end run; after no-end the auto-ε shifts to ~4 mm
because the data thinned.)

**Reclamation.** After clustering, ~50% of streamlines are singletons or
in sub-`min_sheet_size` components. Reclamation: build a KD-tree over
midpoints of *assigned* streamlines; each unassigned streamline inherits
the sheet-id of its nearest assigned neighbour, provided the midpoint is
within `--reclaim-radius-mm` (default 5 mm). Coverage jumps from ~53%
to ~99%. The 1% that remain unassigned are genuinely isolated.

---

## Sweeps that informed defaults

### Phase 1 — `--max-angle-diff` choice

`first_voxel_diff` distribution on the test data:

```
p25 = 7°, p50 = 17°, p75 = 37°
```

A 20° tolerance bisects right at the median. Tightening to 10° loses
half the candidates including obvious real u-fibres at gyral folds.
Loosening past 30° starts admitting clearly off-axis fixels. 20° kept.

### Phase 2 — `--min-hits` (peer-only counting)

| K | candidates ≥ K | endpoint-pass rate at this K |
|---|---|---|
| 1 | 89,074 (100%) | 59.2% |
| 3 | 87,724 (98.5%) | 60.1% |
| 5 | 85,591 | 59.8% |
| 10 | 79,807 | 58.9% |
| 20 | 67,746 | 57.0% |
| 50 | 37,536 | 53.9% |
| 100 | 10,699 | 54.5% |
| 150 | 2,382 | 56.1% |

Endpoint-pass rate is essentially flat across K — peer hit count and
endpoint quality are independent. K=3 is mostly cosmetic; the real
filtering comes from endpoint + length + arc-deviation + depth +
no-end. Default kept at 3 anyway because the few candidates that *do*
fail K=3 are genuinely isolated and worth dropping.

### Phase 2 — `--max-trace-length-mm` (per direction)

Initially set to 20 mm/direction (40 mm total). A CC body fibre at this
length traverses commissural fibres into contralateral cortex on both
sides → both endpoints land in cortex → endpoint check passes
spuriously.

At 10 mm/direction (20 mm total), CC traces don't have enough length to
reach contralateral cortex; they end in deep WM and fail the endpoint
check. Real u-fibres seeded at 5 mm depth still touch cortex on both
ends (10 mm of trace from depth 5 mm reaches the surface).

```
20 mm/dir: 53,662 trajectory_pass (60% of phase 1)
10 mm/dir: 47,682 trajectory_pass (53% of phase 1) — the right scale
 8 mm/dir: too short, real u-fibres start failing endpoint check
```

### Phase 2 — `--max-path-depth-mm`

Sweep at the 10 mm/direction setting:

```
threshold=0 (off):    47,682 trajectories
threshold=4 mm:       46,932  (drops 750)
threshold=6 mm:       47,665  (drops 17)
threshold=8 mm:       47,682  (drops 0)
threshold=12 mm:      47,682  (drops 0)
```

After length cap + endpoint check + arc-deviation, basically *no*
trajectory has any point deeper than 8 mm from cortex. The depth filter
is largely a safety check rather than a primary discriminator on this
data. Kept at 8 mm anyway — cheap (~2 ms added) and might bite on
weirder ODXs.

### Phase 2 — `--no-end-otsu-factor`

DSI-Studio's default 0.6 was way too aggressive:

| factor | forbidden voxels | trajectory_pass | % phase 1 |
|---|---|---|---|
| 0.0 (off) | 0 | 47,682 | 53.5% |
| **0.6 (DSI-Studio default)** | **127,258** | **1,577** | **1.8%** |
| 1.0 | 61,805 | 7,794 | 8.8% |
| 1.5 | 25,954 | 25,967 | 29.2% |
| **2.0 (ufixels default)** | **8,524** | **41,277** | **46.3%** |
| 3.0 | 743 | 47,434 | 53.2% |

Otsu split sits between WM and non-WM, so `0.6 × Otsu` flags most of WM.
For u-fibre detection we want only the densest cores flagged → factor
2.0.

### Phase 3 — `--sheet-eps-mm` (auto)

```
eps=2.0 mm  1101 sheets    21% assigned   top sizes: 372, 337, 148
eps=2.5 mm  1301 sheets    53% assigned   top sizes: 809, 752, 717   ← auto-pick (cap 0.20)
eps=3.0 mm   909 sheets    72% assigned   top sizes: 9343, 5700, 4083 (above cap)
eps=3.5 mm   632 sheets    91% assigned   top size 21,044            ← percolation
eps=4.0 mm   476 sheets    96% assigned   top sizes: 23876, 23184    ← collapsed
```

Auto-pick chose ε=2.5 because `top_size / total_assigned = 0.025` ≤ 0.20.
At ε=3.0 the ratio jumps to 0.21 and gets rejected. The percolation
cliff is between 3.0 and 3.5.

After the no-end filter this shifts: the data is sparser, so auto-ε
moves to 4.0 mm with the same percolation criterion.

### Phase 3 — `--reclaim-radius-mm`

```
radius=0 (off):    32,653 streamlines assigned (53%)
radius=5 mm:       60,702 streamlines assigned (99%)
```

5 mm reclamation closes ~all gaps without merging genuinely isolated
streamlines.

### Sheet-linking experiments (`sheet_link/`)

I implemented three "second-stage" linkers to merge adjacent sheets into
super-sheets:

- **Linker A — direction-gated proximity**: for each sheet pair within
  range, find the closest cross-sheet streamline pair; merge if the
  pointwise distance is small AND the tangents at the contact point
  agree (cosine ≥ 0.85). ([sheet_link/direction.rs](src/sheet_link/direction.rs))
- **Linker B — filtration with veto**: same as A but reuses the cached
  pair list from clustering, framed as a Kruskal dendrogram cut.
  ([sheet_link/filtration.rs](src/sheet_link/filtration.rs))
- **Linker C — PTT continuation**: extend each streamline past its
  endpoint by additional PTT propagation; if the continuation enters
  another sheet's fixels, link them. ([sheet_link/ptt_link.rs](src/sheet_link/ptt_link.rs))

Sweep of the resolver threshold with Linker A (no other linkers):

```
threshold=1.5  1286 super-sheets   top size 1126   (almost no merging)
threshold=2.0   708 super-sheets   top size 4385   (sweet spot pre-no-end)
threshold=2.5   256 super-sheets   top size 54,294 (giant emerges)
threshold=3.0    22 super-sheets   top size 60,455 (full collapse)
```

After the no-end filter is added, the percolation cliff for super-sheet
linking shifts down too:

```
threshold=1.5  661 super-sheets   top size 5253   (3 merges)
threshold=2.0  582 super-sheets   top size 13,078 (collapsed)
```

**Conclusion: super-sheets aren't useful with the current upstream
filter stack.** The sheets that survive Phase 2 are already at the right
granularity. Linker A at threshold 1.5 makes 3 merges out of 661
sheets — decorative. The infrastructure stays in the codebase as
`sheet_link/` (5 files, 3 unit tests) for future use cases (looser
filtering, broken sheets from tracking gaps).

---

## Lessons learned / dead ends

1. **Sheet-level clustering on fixels was wrong from the start.** Initial
   plan was spatial+directional connected components on fixels. Under
   gyral crowns, this would either merge unrelated sheets or splinter
   them. Streamlines preserve endpoint-pair info; fixels lose it.

2. **Mean MDF (QuickBundles) doesn't work for sheets.** QuickBundles is
   designed for tube-shaped bundles. On wide u-fibre fans, two
   streamlines on opposite edges of the fan have MDF ≈ fan width. To
   merge the whole fan you need a threshold large enough to also fuse
   unrelated bundles. **Max** pointwise distance + connected components
   is the right primitive.

3. **DSI-Studio's autotrack factor is dataset-dependent.** Their 0.6
   default is calibrated for general-purpose tractography. For u-fibre
   detection we want a much higher factor (2.0) because we'er actually
   excluding streamlines based on ending in it.

4. **Length cap matters more than I expected.** A 20 mm-per-direction
   cap (40 mm total) lets CC fibres reach contralateral cortex on both
   sides via commissural fibres → endpoint check spuriously passes.
   Halving to 10 mm/direction was the single most impactful filter
   change.

5. **Hit count needed gating.** Naive "count all visits" was a measure
   of fixel density, not consensus. Trajectory-shape-gated voting
   ("only u-fibre-shaped traces vote") fixed the signal.

6. **Endpoint-distance check is not enough on its own.** A trajectory can
   land near a cortical vertex without anatomically terminating *into*
   that cortex — the cingulum runs parallel to and within 3 mm of the
   cingulate cortex throughout. The bow-from-chord check (≥ 2 mm)
   catches these because they're near-straight even though their
   endpoints satisfy distance. Might want to revisit this so we can
   keep the interesting within-wall SWM.

7. **Super-sheets weren't worth keeping enabled by default.** With the
   tight upstream filters, the discrete sheets are already pretty good.
   Super-sheets at safe thresholds make 3 merges out of 660+ — pure noise.
   They'd be useful with looser upstream filtering.

8. **TRXViz palette was visibly bad for sheets.** With 1300+ groups and
   only 8 hash-fallback colours, "color by group" looked monochrome.
   Replaced the fallback with golden-angle HSV (in
   `trxviz_core::palette::distinct_color_index/hash`); now hundreds of
   groups stay visually distinct.

9. **TRX last-wins coloring + HashMap iteration nondeterminism.** When
   the TRX has both `sheet_NNNN` and `super_NNNN` groups, TRXViz
   iterates them in HashMap order (nondeterministic) and the last
   group's colour wins. Fix: emit only one grouping level per TRX,
   controlled by `--trx-group-by`.

10. **In-place ODX append > full save.** Initial `--write-selected-odx`
    used `OdxDataset::save()` which rewrites the entire 290 MB
    archive. Switching to `odx_tractography::writeback::write_dpf_*`
    (in-place zip append) reduced the cost to ~228 KB regardless of
    source ODX size.

---

## Final default settings (production)

```sh
ufixels \
  --lh-wm   lh.white.surf.gii   --lh-pial lh.pial.surf.gii \
  --rh-wm   rh.white.surf.gii   --rh-pial rh.pial.surf.gii \
  --odx     subject.odx \
  --output-prefix subject_ \
  --append-dpfs-to-odx subject_with-ufixels.odx \
  --connectivity-filter \
  --cluster-sheets --auto-sheet-eps --reclaim-radius-mm 5 \
  --debug-trx subject_ufixels.trx
```

All other parameters default to their tuned values:

```
Phase 1:
  --compatible-angle 90    target: tangent to cortex
  --max-angle-diff 20      tolerance
  --max-depth 20           max inward walk (mm)

Phase 2 (--connectivity-filter):
  --max-trace-length-mm 10 per direction
  --ptt-k-max 0.5          allows ~2 mm radius U-bend
  --ptt-probe-mm 3         shorter probe for tight bends
  --endpoint-max-mm 3      endpoint near cortex
  --reject-cross-hemisphere true
  --min-arc-deviation-mm 2 trajectory must bow
  --max-path-depth-mm 8    no point deeper than 8 mm
  --no-end-otsu-factor 2.0 only flag densest WM
  --min-hits 3             peer-vote threshold
  --min-total-length-mm 5
  --max-total-length-mm 35

Phase 3 (--cluster-sheets --auto-sheet-eps):
  --giant-cap-fraction 0.20 percolation safety
  --auto-eps-min/max 1.5–5  sweep range
  --reclaim-radius-mm 5     close most gaps
  --min-sheet-size 5
```

On a typical 392k-fixel adult ODX with 257k cortical vertices, runtime
is ~50 s wall-clock on an 8-core laptop, peak memory ~5 GB.

---

## Anatomy/algorithm crosswalk

For each *anatomical* error mode I observed in trxviz, here's the filter
that addresses it:

| Visible problem | Filter that catches it | Default |
| --- | --- | --- |
| SLF tangent run under cortex | bow-from-chord (cingulum-style straight) | 2 mm |
| CC body crossing midline | cross-hemisphere | on |
| Cortico-callosal projection arc | max path depth | 8 mm |
| CC fibres "ending" in cingulate cortex | no-end mask + path depth | factor 2.0 |
| Cingulum running parallel to cortex | bow-from-chord | 2 mm |
| Lone-tangent SLF candidate | peer hit count (gated) | K=3 |
| 80mm-trace through deep WM | length cap | 20 mm total |
| Tracking gap-induced "split sheets" | reclamation | r=5 mm |
| Single bad fixel as a singleton "sheet" | min_sheet_size | 5 |
