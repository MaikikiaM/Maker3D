# Nano Banana Prompt Recipes for Maker3D

Use these structured prompt templates when invoking `invoke_subagent(TypeName="image-generator", ...)`.
Keep prompts concrete about projection (isometric vs strictly orthographic), lighting
(flat diffuse vs studio render), contrast bounds, and 3D printability.

---

## 1. Hexagonal 6-View Equidistant Concept Sheet (Stage 1 — Preferred for Catan & 360° Tiles)
> Generates all 6 equidistant faces every 60° around the hexagon (`0°, 60°, 120°, 180°, 240°, 300°`)
> so the sculpt reads monumentally from every seat around the tabletop with zero dead backface.

```text
Studio miniature turnaround presentation sheet arranged in a clean 3x2 grid on a
soft neutral light-gray studio backdrop, showing 6 equidistant elevated 3/4 views
(0°, 60°, 120°, 180°, 240°, 300°—one camera facing each of the 6 hexagonal sides)
of the EXACT SAME 3D-printable Catan tabletop miniature hex tile for [BIOME].
Target vertical proportions: [RELIEF DESCRIPTION, e.g., balanced alpine peaks
rising ~35-40% of the hex width / dense layered pine canopies / terraced clay
mesas]. Ensure true 360-degree radial interest on EVERY view (zero flat back
wall), crisp dark charcoal hexagonal architectural plinth with clean flat outer
road lane and 6 circular corner settlement pads, museum-grade oil-painted
photorealistic PBR surfaces where colors melt naturally into one another, and
crisp readable forms suitable for FDM/full-color 3D printing.
```

---

## 2. Strict Orthographic Grayscale Heightmap (Stage 2A — Geometry)
> Use to synthesize sculptural relief that projects cleanly into a DEM.

```text
Strictly top-down 90-degree orthographic grayscale depth heightmap (Z-depth pass)
of [SUBJECT: e.g., jagged alpine S-curve mountain cordillera with 5 dramatic sharp
peaks and glacial cirque bowls] centered inside a flat-topped regular hexagon
(pointy-top orientation) with a generous smooth black border around the outer edges
and corners. Pure white (#FFFFFF) = highest summits, smooth mid-grays = ridges,
ravines and foothills, pure black (#000000) = completely flat base level near the
outer hexagon perimeter and 6 corner pads. Zero perspective distortion, zero text,
zero cast shadows, zero horizon, crisp continuous 16-bit-style tonal gradients.
```

---

## 3. Geometry-Locked Painterly Albedo Edit-Pass (Stage 2A — Texture)
> Pass a top-down shaded clay render generated from your actual NumPy DEM as the
> reference image to edit, so every ridge, gully, and corner pad stays 100% aligned.

```text
Paint a photorealistic miniature-scale top-down diffuse albedo texture directly
over the provided orthographic grayscale terrain render, strictly preserving every
ridge, peak, valley, cliff edge, flat 3.5mm perimeter collar, and 6 corner pads
in exact pixel alignment. Replace gray shading with seamlessly blended natural oil-
painted colors: sunlit weathered granite crags, warm ochre scree talus, alpine moss
ravines, sunlit olive meadow foothills, and a clean dark charcoal slate collar at
the outer border. Completely flat uniform overhead diffuse lighting with ZERO
directional cast shadows, ZERO glossy specular highlights, ZERO perspective tilt,
and ZERO snow on vertical rock cliffs.
```

---

## 4. Full-3D Orthographic Silhouette Views (Stage 2B — Figurines & Props)
> Use when carving volumetric 3D SDFs / Visual Hulls / spline lofts in Python.

```text
Three cleanly separated orthographic technical character/prop turnaround views
(Front elevation, Right side elevation, Top plan view) of [SUBJECT] aligned on a
shared baseline, solid matte mid-gray clay sculpt on a pure pitch-black background
with subtle studio rim light and zero perspective foreshortening. Strong readable
silhouette, chunky 3D-printable proportions (no hair-thin antennas or floating
parts), sharp beveled armor/architectural panel lines, uniform scale across all
three views.
```
