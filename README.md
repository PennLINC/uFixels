# ufixels

Surface-normal-driven detection of u-fiber-compatible fixels in superficial
white matter.

For each paired vertex of a `wm.surf.gii` / `pial.surf.gii` cortical surface,
`ufixels` walks inward along the surface normal voxel by voxel through an
[ODX](https://github.com/PennLINC/odx-rs) fixel image. At each voxel it asks:
is there a fixel oriented roughly tangent to the cortex? The depth at which
tangent fixels run out — or the ray hits the opposite gyral wall — is a useful
per-vertex marker of superficial white-matter organisation.

## Outputs

Per hemisphere:

- `*.first_voxel_diff.shape.gii` — degrees from the target tangent angle in the
  first masked voxel under the cortex (smaller = more tangent).
- `*.compatible_depth.shape.gii` — millimetres walked through compatible-fixel
  territory before the walk terminated.
- `*.end_condition.shape.gii` — why the walk stopped (encoded as `f32`):
  `0` no compatible fixel, `1` hit opposite surface, `2` reached max depth,
  `3` left the WM mask, `NaN` invalid (degenerate paired vertex).

Optionally, `--write-selected-odx <PATH>` writes a copy of the input ODX with
a `dpf/ufixel_selected` per-fixel `uint8` mask attached, for visualisation in
[trxviz](https://github.com/PennLINC/TRXViz).

## Usage

```sh
ufixels \
  --lh-wm    lh.white.surf.gii   --lh-pial lh.pial.surf.gii \
  --rh-wm    rh.white.surf.gii   --rh-pial rh.pial.surf.gii \
  --odx      dataset.odx \
  --output-prefix subject01_ \
  --write-selected-odx subject01_with-ufixels.odx
```

Tunables (all degrees / mm):

| Flag | Default | Meaning |
| --- | --- | --- |
| `--compatible-angle` | 90 | Target angle between fixel and surface normal. 90° = parallel to cortex. |
| `--max-angle-diff` | 20 | Tolerance around the target angle. |
| `--max-depth` | 20 | Hard cap on inward walk distance. |
| `--min-step-eps` | 0.1 | Initial offset (mm) into white matter to avoid the seed voxel. |

The four input GIFTI surfaces and the ODX must already be in the same
RAS+mm coordinate system (typically subject ACPC space or a common template).

## License

Dual-licensed under MIT or Apache-2.0 at your option.
