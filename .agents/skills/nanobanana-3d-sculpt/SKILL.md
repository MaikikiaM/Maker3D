---
name: nanobanana-3d-sculpt
description: >-
  Use this skill when designing, sculpting, or upgrading imaginative 3D models,
  stylized Catan terrain tiles, organic miniatures, props, architectural reliefs,
  or painterly PBR textures in Maker3D. Combines Antigravity's Nano Banana
  image-generator subagent (concept art, orthographic depth/heightmaps, and
  geometry-locked painterly albedo passes) with pure NumPy/SciPy watertight CAD
  projection and tolerance enforcement.
---

# Nano Banana 3D Concept-to-CAD Workflow

Bridge artistic imagination (Nano Banana via `image-generator`) with strict
engineering tolerances (pure `numpy` + `scipy` watertight manifolds). Never rely
on blind pure-math guesswork for complex artistic shapes or textures when a 10-second
visual blueprint unlocks cohesive silhouette and painterly transitions.

## Core Principle: Imagination First, Math Second, Tolerances Always

1. **Nano Banana (`image-generator` subagent)** provides silhouette invention,
   sculpted proportions, organic/figurative details, and continuous painterly blending.
2. **Pure NumPy/SciPy (`Maker3D` CAD bridge)** enforces exact dimensions, road/pad
   clearances, magnet pockets, multi-LOD watertight topology (`watertight=True`,
   `chi=2`), and crisp physical printability.

## 4-Stage Production Workflow

### Stage 1: Hexagonal 6-View Concept & Proportion Blueprint
Before writing geometry code for a new asset (terrain tile, miniature, helmet,
building, diorama prop):
1. Call `invoke_subagent` with `TypeName="image-generator"` using the **6-View
   Hexagonal Turnaround** prompt template in [prompt-recipes.md](./references/prompt-recipes.md)
   to generate a clean **3×2 equidistant 6-angle studio sheet** (`0°, 60°, 120°,
   180°, 240°, 300°`—one elevated 3/4 view facing each of the 6 hexagonal sides)
   for true 360-degree tabletop readability with zero "backstage" dead wall.
2. Inspect the returned image via `read_file` to verify silhouette clarity across
   all 6 faces, 3D printability (avoid fragile floating overhangs), realistic
   biome relief ratio, and color harmony.

### Stage 2: Orthographic Relief & Geometry-Locked Albedo
Choose the branch matching the geometry class:

- **Branch A — Reliefs, Bas-Reliefs & Catan Hex Tiles (`2.5D` heightfields)**
  1. Generate a **strictly top-down orthographic grayscale heightmap** via
     `image-generator` (`pure white = peak`, `pure black = 0 mm margin/collar`,
     zero perspective tilt, zero cast shadows).
  2. **(Recommended for painterly realism)** First project the heightmap + procedural
     micro-crags onto the canonical hex domain and save a crisp top-down **shaded
     clay relief preview** (matching `[-R, R] × [-R, R]` UV bounds). Then invoke
     `image-generator` again in **image-edit / reference mode** passing that
     clay preview so Nano Banana paints seamless, natural PBR albedo **pixel-locked
     to your exact 3D ridges, gullies, and hex border**!
  3. Inspect both heightmap and albedo via `read_file` before meshing. Check that
     albedo uses flat ambient studio lighting (no baked directional cast shadows).

- **Branch B — Full-3D Figurines, Props, Helmets & Architecture (`3D SDF / Loft`)**
  1. Generate aligned **orthographic Front, Side, and Top views** via
     `image-generator` (equal scale, clean dark background silhouette).
  2. Extract binary silhouettes / contour profiles in Python (`PIL` + `scipy.ndimage`)
     to carve an exact **Visual Hull / SDF intersection** or spline loft, then fuse
     mechanical CAD flanges, chamfers, and sockets in pure `numpy`/`scipy`.

- **Branch C — Single-Perspective Hero-to-360° Completion (Clay + Partial-Texture Nano Banana Loop)**
  Use whenever starting from **only one hero 3/4 reference (`0°`)** (e.g. a user-supplied concept render) where blind angles (`60°, 120°, 180°, 240°, 300°`) would otherwise degrade into blurry procedural Voronoi patches:
  1. **Workhorse Camera & Clay Lock (Mandatory Gate):** Calibrate `hero_cam_params` directly against the hero reference's hex/plinth vertices (`MAE <= 1.5 px`, matching real pitch and FOV — never project from an uncalibrated or overly shallow grazing angle). Sculpt and visually audit `clay_vs_hero` and `clay_6views` via `read_file` until every canyon, cirque bowl, and jagged ridge is sharp, open, and artifact-free across all 360° **before running any texturing**. If a validated hero geometry already covers the hex cleanly (e.g., `Z_geom_v4.npy` + `hero_nanobanana.jpg` scoring `9/10` at `0°`), **lock and reuse that exact geometry** rather than replacing it with an inferior procedural mountain or leaving a flat outer lawn.
  2. **Project `0°` & Render Clay+Partial-Texture Guides:** Project the de-lit `0°` hero photo onto its cleanly visible faces (`ndotv` unoccluded — **never** delete bright snow/highlights on slopes with luminance-rejection hacks). Next, render the missing views (`180°`, `60°`, `120°`, `240°`, `300°`, and `90°` zenithal if bowls/overhangs remain occluded) showing **already-projected `0°` colors on visible margins + shaded neutral 3D clay on blind areas** (`partial_view_XXX.png`).
  3. **Multi-View Nano Banana Paint-Over + Workhorse Audit:** Invoke `image-generator` per uncovered angle passing **both** `partial_view_XXX.png` (geometry + color-continuity lock) and the original `0°` hero reference, using Recipe 5 in [prompt-recipes.md](./references/prompt-recipes.md). Inspect **every** returned image with `read_file`: verify silhouette alignment against the clay mask, reject/regenerate if Nano Banana hallucinates extra peaks outside the clay or bakes harsh directional shadows, and align sub-pixel to the clay silhouette.
  4. **Winner-Takes-Most 360° Blending (Zero Halos, Zero Top-Down Cliff Stretching):** Erode each view's foreground mask by `2–3 px` inside its clay silhouette before projection so **zero** studio background or plinth bevel ever ghosts onto rock/meadow edges. Harmonize exposure/color across adjacent views and blend with steep normal-facing `ndotv^p` winner-takes-most weights (keeping crisp photographic micro-detail from whichever camera faces each facet straightest, instead of blurring across views or replacing large surfaces with Voronoi quilting). **Crucial for steep terrain (`>= 55°` scarps):** do **NOT** flatten the final 6-view blend solely into a `2.5D` top-down albedo PNG (`top-down` stretches vertical cliffs `5×–8×`). Instead, pass the validated base mesh + the 6 de-lit Nano Banana camera renders + calibrated cameras to `blender-sculpt-bake` for **Smart UV 3D Cycles Camera-Projection Baking** (aislando UV de zócalo para aprovechar `>= 80%` del atlas `4K` en el terreno, con super-resolución / micro-detalle mineral en las `6` vistas si las fuentes son `1024×1024` y sombras `PCFSoftShadowMap` en el visor) and **3D Normal Displace** bilineal on back faces, then inspect all 6 PBR + Clay views side-by-side against the `0°` hero reference.

### Stage 3: Tolerance Projection & Watertight Export
Run or adapt the Catan bridge script [nanobanana_to_catan_hex.py](./scripts/nanobanana_to_catan_hex.py)
for hexagonal tiles, or mirror its geometric invariants in custom asset builders:

```powershell
python .agents/skills/nanobanana-3d-sculpt/scripts/nanobanana_to_catan_hex.py `
  --heightmap <path_to_heightmap.png> `
  --albedo <path_to_albedo.png> `
  --out-dir Tableros-Catan/<Tile_Dir> `
  --prefix catan_<tile> `
  --relief-mm 58.0 `
  --nx 420 `
  --tex-size 2048
```

Verify script invariants:
- `Y-up` PNG alignment matches UV coordinates (`v=0` at `Y=-R`, `v=1` at `Y=+R`).
- Road margin (`3.50 mm`) and settlement corner pads (`Ø20.00 mm`) clamp cleanly to `Z_BASE=25.00 mm`.
- Bottom magnet pocket (`Ø10.20 × 5.20 mm`) remains open downward while closing `chi=2`.
- Output directory stays clean: `watertight=True`, `chi=2`, zero degenerate triangles.

### Stage 4: Hexagonal 6-View Visual Audit
Always render a **6-view equidistant hexagonal turnaround** (`0°, 60°, 120°, 180°,
240°, 300°` in a 3×2 studio panel) from the actual compiled STL/DEM + PBR textures,
a **side-by-side zoomed 2D-Albedo vs. Untextured 3D-Clay parity check** across key
terrain regions, AND verify the live WebGL viewer in Chromium (`0` console errors, checking both `Hiperreal PBR` and untextured `Arcilla CAD`), comparing all 6 angles against the Stage 1 Nano Banana concept sheet via `read_file`.
Refine slope opening (`slope_gamma`), snow/moisture altitude masks (`0%` snow on
`>=44°`), or run a targeted `image-generator` paint-over until every hexagonal
face matches the artistic standard of the concept.

## Hard-Won 3D Sculpting, Triangulation & Viewer Rules

Apply these engineering invariants on all high-detail Maker3D terrain/asset builds (`build_catan_biome.py`, `terrain_mesh_utils.py`, `assemble_viewer_v2.py`):

1. **Never Diagnose Faceting by Triangle Count Alone — Eliminate `dFdx/dFdy` + Macro Normal Double-Tilting (`--tangent-norm`):**
   - When 3D micro-relief is already physically sculpted into the mesh (`--sculpt-from-albedo`), do **not** bake macro DEM gradients (`np.gradient(Z)`) into the `normalMap`, and do **not** omit per-vertex `tangent` attributes in WebGL.
   - Without explicit `tangent` attributes (`vertexTangents: true`, `USE_TANGENT` define in Three.js `r128`), `MeshStandardMaterial` computes screen-space `dFdx/dFdy(vUv)` per flat triangle; squashed overhead UVs on steep walls cause alternating bright/dark diamond zigzags even at `>2M` triangles.
   - Always pair **pure high-frequency tangent residual `normalMap`s** (centered near `(128,128,255)`) with smooth per-vertex analytic tangents and tight shadow bias (`bias = -0.00025`, `normalBias = 0.08`).

2. **Uniform 3D Density on Steep Cliffs & Bicubic Resampling (`order=3` + Red-Green + Taubin):**
   - Uniform XY grids stretch triangles `2×–3×` vertically on `≥45°` scarps. Always use **bicubic (`order=3`)** DEM resampling in `resample_canonical_xy` (never `order=1` linear), warp toward slopes, align diagonals (`contour_aligned_edge_flip`), and refine stretched slopes/peaks (`≥38°`) using conformal **Red-Green 3D subdivision** (`--subdivide-passes 1 --edge-len-max 0.265`) evaluating new midpoints on the bicubic `2048×2048` DEM.
   - Follow with **zero-shrinkage Taubin smoothing** (`--taubin-passes 4`, `λ=0.34, μ=-0.35`) on interior `Z` only to remove triangle-edge zigzags while keeping `~99.9%` peak height and volume intact (`watertight=True`, `chi=2`, HTML `< 50 MB`).

3. **Never Flatten Sculpted Relief Amplitude to Hide Triangles:**
   - Do **not** dampen walls (`wall_damp`) or apply broad spatial blurs (`~0.34 mm`) to the bump field—that turns sculpted rocks back into flat 2D caricatures. Preserve full physical relief (`--sculpt-from-albedo 0.95`), rounding only sub-2-triangle pebble needle tips (`--sculpt-crown-mm 0.115`) while fixing visual facets via smooth tangents + adaptive 3D subdivision + Taubin.

4. **High-Res Quadrant Albedo (`2048×2048`) & Mineral De-Inking (`deink_mineral_albedo`):**
   - For intricate ground/rock textures (gravel, cobblestones, mineral veins), generate zoomed `image-generator` quadrants + center bridge stitched into `2048×2048` and run `deink_mineral_albedo` to convert 2D dark comic ink lines into natural warm mineral cavities before physical 3D displacement.

5. **Choose `2.5D` vs. True `3D` CSG by Overhang Geometry + Triplanar Cliffs:**
   - Use `2.5D` heightfields + adaptive subdivision + GLSL 3D triplanar rock/clay shaders (`--triplanar-mode rock|clay`) for canyons, terraces, strata, and sand dunes.
   - Use standalone 3D solids fused via `manifold3d` (`build_bosque_3d.py` pattern) over an inpainted base terrain for trees, overhanging foliage, buildings, or undercut organic miniatures that melt in `2.5D`.

6. **Local Numbered Version Snapshots (`versiones/v01/`, `v02/`, …) & Decoupled Master STL Resolution (`--stl-nx`):**
   - Since `*.stl` files stay strictly local (never committed to GitHub, decoupled from the `< 50 MB` HTML WebGL viewer limit), compile the printable Master STL at ultra-high density via `--stl-nx` / `--stl-edge-len-max` (e.g. `~7.5M–9.5M` triangles, **hard ceiling `<= 10,000,000 Δ`** to keep slicing/optimization tractable) while keeping the embedded WebGL LOD mesh at `--nx 408` (`~2.80M Δ`, `< 50 MB` HTML).
   - Before major geometry/shader changes on hero assets, archive `<dir>/versiones/vNN_<label>/` locally (HTML viewer, STL, 6-view PNG, `master_metrics_v2.json`, exact CLI `build_info.txt`) excluded from Git via `.gitignore` so every visual state stays A/B-comparable without bloating the Git repository.

7. **100% 2D-to-3D Geometry Parity — Every Painted Detail MUST Exist in Physical 3D (Zero Flat Painted Surfaces):**
   - Never ship a model where visual elements in the 2D albedo (submerged river stones, waterfall foam cascades, cobble bank margins, dark basalt cracks/pores, gold crystal clusters, terrace grooves) sit painted on a smooth flat slab. If you toggle **`Arcilla CAD` (untextured clay)** in the viewer, **every recognizable 2D texture feature must already be sculpted in real 3D geometry**:
     - **Normalize dark rocks (`lum_eq`) before sculpting:** Dark basalt/slate has `3×–4×` narrower RGB luminance range than light sandstone/limestone. Always normalize local contrast (`lum / max(local_mean, 0.14)`) inside `sculpt_albedo_microrelief` so dark volcanic cracks, fractures, and pores carve at the same physical millimeter depth as light stone.
     - **Never zero out water relief blindly:** Preserve strictly monotonic downhill water hydraulics (`N → S`, zero transverse foam walls blocking flow), while still sculpting submerged boulder/pebble domes and downstream-aligned zero-mean waterfall flow ribs (`~0.25–0.45 mm`) inside water corridors so rivers and cascades read as physical 3D relief in Clay.
     - **Sculpt steep cliff walls laterally along 3D vertex normals (`--cliff-sculpt`) & project triplanar wall detail:** Pure `Z(x,y)` displacement and top-down `vUv` cannot carve perpendicular relief on `≥45°` vertical cliffs and stretch shoreline/pebble texels into vertical smears. Pair normal-direction `(nx,ny,nz)` cliff rock strata/cracks (`sculpt_cliff_walls_3d` with Jacobian foldover damping) with upslope rock-hue probing + synchronized 3D object-space (`vObjPos`) triplanar strata/normals (`assemble_viewer_v2.py`) while rounding river-cobble crowns (`rock_pos` blur without dulling `rock_neg` cracks).
     - **Do not over-flatten edges with huge aprons or post-blurs:** Keep perimeter hex aprons compact (`~5.0–5.5 mm`, keeping `road=3.5 mm` and `Ø20` pads flat at `Z_BASE`) so outer rock outcrops retain steep sculptural relief, and keep post-bump smoothing ultra-tight (`≤ 0.05 mm`) so tiny stones and rock pores are not erased prior to meshing.
     - **Mandatory Clay-vs-Albedo Verification:** Always generate and inspect side-by-side zoomed crops (`2D Albedo` vs. `Untextured 3D Clay Z` + live WebGL `Arcilla CAD` toggle) before closing any hero sculpt.
