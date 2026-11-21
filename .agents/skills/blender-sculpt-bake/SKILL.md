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
