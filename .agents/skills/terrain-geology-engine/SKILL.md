---
name: terrain-geology-engine
description: >-
  Simulación física de erosión fluvial, cañones, cuencas hidrográficas y máscaras
  geológicas con Landlab para losetas Catán y dioramas 3D. Garantiza drenaje
  monótono descendente real sin lagos artificiales y 0.0000% de nieve en
  pendientes >= 44 grados.
---

# Motor Geológico e Hidrológico (`Landlab` — `hydraulic_terrain_gen.py`)

Utiliza esta skill para generar **relieve geológico coherente por leyes físicas de erosión fluvial (`FastscapeEroder`) y difusión de laderas (`LinearDiffuser`)**, en lugar de depender de heightmaps planos o ruidos puramente aleatorios que rompen cauces o generan artefactos en valles.

## 1. Presets Geomorfológicos Disponibles

El script `tools/terrain/hydraulic_terrain_gen.py` soporta 4 biomas hidráulicos parametrizables:
- **`alpine_canyon`**: Garganta fluvial profunda N→S flanqueada por escarpes escalonados y cumbres alpinas; excelente para montañas y valles dramáticos.
- **`cordillera`**: Cadena montañosa oblicua con dos vertientes escarpadas y quebradas erosivas divergentes.
- **`terrace_hills`**: Lomas sedimentarias onduladas y terrazas amplias ideales para praderas, bosques o pastos con drenaje suave.
- **`river_delta`**: Llanura aluvial meandriforme bifurcada con cuencas amplias y bancos de grava de bajo relieve.

## 2. Invocación Estándar

```powershell
python tools/terrain/hydraulic_terrain_gen.py `
  --preset alpine_canyon `
  --nx 384 `
  --max-height 32.0 `
  --steps 35 `
  --seed 42 `
  --out-dir <dir_salida>/terrain
```

## 3. Garantías Matemáticas y de Ingeniería

1. **Respeto exacto al carril Catán:**
   - Todo píxel dentro del carril exterior de caminos (`ROAD_MARGIN = 3.50 mm`) y dentro de los 6 pads circulares de poblado (`Ø20.00 mm`, `PAD_RADIUS = 10.00 mm`) se clampa a **`0.000 mm`** de elevación relativa sobre `Z_BASE = 25.00 mm`, con transición suave fuera de esos bordes.
2. **Drenaje hidrológico real sin cuencas trampa (`Sink Filling + FlowAccumulator`):**
   - Usa `FlowAccumulator` (`D8`), `DepressionFinderAndRouter`, `FastscapeEroder` y `LinearDiffuser` sobre `RasterModelGrid` seguido de enrutamiento de caudal hacia la salida frontal (`N → S`) asegurando descenso monótono en la línea de talweg.
3. **Máscara física de nieve (`0.0000%` en `>= 44.0°`):**
   - La máscara `snow_mask.png` y el albedo base se calculan tras medir la pendiente real `slope_deg = degrees(arctan(|grad Z|))`. Toda pared con `slope_deg >= 44.0°` queda forzada a **cero nieve** (`rock` expuesto).
4. **Artefactos exportados en `--out-dir`:**
   - `dem_float32.npy` (elevación relativa en mm respecto a `Z_BASE`, Y-up)
   - `dem_16bit.png` (heightmap limpio de 16 bits sin banding)
   - `slope_deg.npy`, `flow_mask.png`, `snow_mask.png`, `rock_mask.png`, `meadow_mask.png`
   - `albedo_base.png` (paleta PBR coherente sin sombras arrojadas)
   - `terrain_metrics.json` (verificación numérica de altura máx., % nieve en `>=44°`, descenso de cauce y tolerancia de pads/caminos).

## 4. Paso Siguiente Obligatorio

Pasa `dem_float32.npy`, `slope_deg.npy` y `albedo_base.png` directamente a `tools/blender/sculpt_fuse_bake_render.py` (`blender-sculpt-bake`) para añadir esculpido 3D lateral en acantilados, fusión booleana estanca con la base `build123d` y props 3D.
