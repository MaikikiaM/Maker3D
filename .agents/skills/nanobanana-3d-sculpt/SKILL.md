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
240°, 300°` in a 3×2 studio panel) from the actual compiled STL/DEM + PBR textures
AND verify the live WebGL viewer in Chromium (`0` console errors), comparing all
6 angles side-by-side against the Stage 1 Nano Banana concept sheet via `read_file`.
Refine slope opening (`slope_gamma`), snow/moisture altitude masks (`0%` snow on
`>=44°`), or run a targeted `image-generator` paint-over until every hexagonal
face matches the artistic standard of the concept.
