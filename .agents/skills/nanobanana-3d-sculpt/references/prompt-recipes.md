# Nano Banana Prompt Recipes for Maker3D

Use these structured prompt templates when invoking `invoke_subagent(TypeName="image-generator", ...)`.
Keep prompts concrete about projection (isometric vs strictly orthographic), lighting
(flat diffuse vs studio render), contrast bounds, and 3D printability.

---

## 1. Tabletop Hero Concept & Turnaround Sheet (Stage 1)
> Ideal for exploring creative silhouettes before coding any geometry.

```text
High-detail physical studio sculpture concept sheet for a 3D-printable tabletop
miniature [ASSET TYPE: e.g., Catan mountain range hex tile / medieval wheat
farmstead hex / stylized sci-fi helmet]. Clean neutral light-gray studio
background. Left half: dramatic 3/4 isometric hero view showing bold sculpted
silhouette, layered secondary details, overhangs <= 45 degrees suitable for
3D printing, and rich natural hand-painted/photoreal material transitions.
Right half: matching orthographic top-down plan view and side elevation profile
at uniform scale. Matte miniature paint finish, crisp readable forms at tabletop
viewing distance.
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
