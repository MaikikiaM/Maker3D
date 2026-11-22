---
name: blender-sculpt-bake
description: >-
  Esculpido 3D real en Blender 5.3 headless (estratos laterales en normales de
  acantilados >= 45 grados), fusión estanca entre relieve/props 3D y zócalo CAD
  B-Rep preservando pads/caminos/imán, proyección UV/PBR y renderizado de
  auditoría 6 vistas PBR + 6 vistas Arcilla CAD + ortográfico.
---

# Esculpido 3D, Fusión Booleana y Renderizado en Blender 5.3

Esta skill resuelve los dos límites principales del pipeline antiguo en NumPy:
1. **Esculpido 3D verdadero (`(nx, ny, nz)`):** En escarpes verticales (`>= 45°`), el desplazamiento se aplica **a lo largo de las normales 3D de superficie**, creando estratos de roca y relieves laterales perceptibles en modo **Arcilla CAD**.
2. **Fusión 3D estanca real:** Árboles, rocas colgantes, edificios o miniaturas (`--props-glb` o `--demo-trees N`) se unen geométricamente de forma sólida al relieve y a la base B-Rep sin fundir pads, carril de carreteras ni el alojamiento inferior de imán.

Consulta las particularidades verificadas de `bpy` en Blender 5.3.0 en [blender5-bpy-guide.md](./references/blender5-bpy-guide.md).

## 1. Ejecución Headless Principal (`sculpt_fuse_bake_render.py`)

```powershell
blender.exe --background --python tools/blender/sculpt_fuse_bake_render.py -- `
  --base-stl <dir>/catan_base_brep.stl `
  --dem-npy <dir>/terrain/dem_float32.npy `
  --albedo-png <dir>/terrain/albedo_base.png `
  --slope-npy <dir>/terrain/slope_deg.npy `
  --demo-trees 5 `
  --out-dir <dir>/blender
```
*(Sustituir `--demo-trees` o combinar con `--props-glb <asset1.glb>` cuando existan modelos generados vía `cloud_3d_bridge.py`).*

## 2. Garantías Geométricas y de Auditoría

- **Protección estricta de interfaz Catán:**
  - El perímetro exterior hexagonal (`110.8513 × 128.0000 mm`), la altura exacta de base (`25.0000 mm`), el borde plano de carreteras (`3.50 mm`), los 6 pads circulares (`Ø20.00 mm`) y la cavidad inferior del imán (`Ø10.20 × 5.20 mm`) proceden de la base B-Rep (`build123d`) o se refuerzan mediante operación booleana limpia, evitando que la textura o el remesh invadan las zonas funcionales.
- **Validación topológica en `bmesh` antes de exportar:**
  - Verifica `non_manifold_edges == 0` y `boundary_edges == 0`.
  - Exporta `tile_master.stl` (impresión) y `tile_viewer.glb` (visor WebGL con materiales PBR y albedo embebido).
- **Renderizado automático de auditoría (3 paneles):**
  1. `audit_6views_pbr.png` (mosaico 3×2 a 6 ángulos hexagonales + vista inferior/isométrica mostrando el hueco de imán, con materiales PBR e iluminación estudio 3 puntos).
  2. `audit_6views_clay.png` (mosaico 3×2 idéntico con shader **arcilla mate uniforme sin texturas**, diseñado para detectar al instante cualquier detalle 2D que no exista en 3D).
  3. `audit_ortho.png` (vista cenital ortográfica).

## 3. Regla Obligatoria Post-Render y Uso del MCP `blender`

- **Siempre** abre e inspecciona `audit_6views_pbr.png`, `audit_6views_clay.png` y `audit_ortho.png` con `read_file` tras ejecutar el script.
- Si el usuario prefiere sesión interactiva con ventana de Blender abierta, recuerda que el addon **`blender_mcp`** está instalado y activado en Blender 5.3 (`Interface: MCP for Blender`), conectado al servidor MCP `"blender"` (`mcp-for-blender.exe`). Para pipelines automatizados y reproducibles en segundo plano sin GUI, utiliza `blender.exe --background`.

## 4. Proyección Multi-Cámara 3D (`Hero + Vistas Nano Banana` → `9.5/10` en 360°)

Cuando una loseta combina vista frontal heroica (`0°`) con vistas completadas por `image-generator` (`60°, 120°, 180°, 240°, 300°`):
1. **Conservar geometría de alta cobertura y densidad isotrópica en pared vertical:** partir de base densa (`nx_mesh >= 768`) y subdividir adaptativamente aristas largas del relieve (`bmesh.ops.subdivide_edges` en `> 0.24 mm`, dejando intacto el zócalo B-Rep) hasta `~1.8M–2.5M+` triángulos isotrópicos; aplicar `poly.use_smooth = True` **solo** sobre caras orgánicas (`slot 1`), manteniendo planas/nítidas las aristas del zócalo, carril y pads.
2. **Aislamiento UV Zócalo vs Relieve (`>= 80%` aprovechamiento 4K):** no meter el zócalo dentro del `Smart UV Project` principal del relieve. Asignar al zócalo un material PBR independiente (gunmetal pizarra oscuro `#1E2228` con anillos de poblado y biseles maquinados procedurales o mapa `512 px`) y ejecutar `Smart UV Project` únicamente sobre las caras seleccionadas del relieve orgánico para multiplicar `> 3×` la densidad de píxeles/mm².
3. **Super-resolución previa, proyección Cycles sin lavado y esculpido bilineal (`Normal Displace`):** reescalar/nitidez o enriquecer las vistas alineadas a `2K–4K` reales (micro-grano de roca/canchal/musgo y filtrado de alta frecuencia) antes de proyectar con `ndotv^6`, sin suavizar pesos `W_cams` sobre la vista frontal `0°`. Al esculpir en 3D por normal en frente y espalda (`120°–240°`), usar **interpolación bilineal exacta** sobre el albedo y conectar micro-normal de alta frecuencia independiente de la compresión. En bordes pradera-zócalo, evitar estiramientos radiales amplios (`rim_pull`), filtrando el zócalo metálico en máscara y terminando el musgo/roca real limpiamente en la frontera `ROAD_MARGIN`.

## 5. Refinado Macro High-Pass (`8K` + `Slot 2` CC0 + Anclaje 3 Imágenes)

Para elevar renders provisionales de relieve (`passA` `8K`) mediante parches macro `"posición humana"` (`--stage render-macros` → `macro_patch_align.py` → `--stage passB`):
1. **Prompting sin frenos conservadores ni descuido del fondo:** evitar instrucciones como *"exact same pixels / subtle / do NOT exaggerate"* que congelan la pintura borrosa o generan solo ruido superficial sobre valles planos. Exigir reemplazo integral de pintura plana en fondos/valles por vegetación o pedregal 3D denso y, tras validar un parche maestro (ej. `B2_150`), pasarlo como **Imagen 3 (referencia obligatoria de acabado)** junto a Imagen 1 (`_pbr`) e Imagen 2 (`_clay`) en sesión limpia.
2. **Puertas de alineación High-Pass (`tools/generative/macro_patch_align.py`):** como `passB` inyecta solo `log_nb - gaussian_blur(sigma=6 px)`, armonizar primero ganancia paso bajo (`gain_l`) antes de evaluar `SSIM_low >= 0.80` (`mean_flow_px <= 4.0 px`) y separar croma (`chroma_deltaE <= 6.0`) de exposición (`deltaE_low <= 11.0`), combinando la comprobación métrica (`hp_std ~ 0.12–0.14`) con inspección visual frente al parche de referencia.
3. **Protección de rocas/props CC0 (`Slot 2`) en `passB`:** al combinar relieve (`poly_mat == 1`) con pedregal CC0 real (`poly_mat == 2`), excluir explícitamente `~is_slot2_v` tanto en `sculpt_3d_from_highpass_bandlimited` como en `sculpt_3d_from_baked_albedo` (`tools/blender/catan_hero360_bake.py`) para preservar intacta la geometría escaneada de los bloques.
