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

## 5. Refinado Macro `8K` (`--projection-mode full`, ECC Sobel y Compuerta Anti-Cráter)

Para elevar renders provisionales (`passA` `8K`) mediante parches macro `"posición humana"` (`--stage render-macros` → `macro_patch_align.py` → `prep-local` → `gpu-bake-only` → `finalize-local`):
1. **Prompting integral sin frenos conservadores:** exigir reemplazo completo de pintura plana en fondos/valles por vegetación o pedregal 3D denso y anclar un parche maestro validado como **Imagen 3 (referencia obligatoria)** junto a `_pbr` y `_clay`.
2. **Alineación robusta ante vegetación nueva (`macro_patch_align.py`):** ejecutar ECC sobre magnitud de gradiente **Sobel** (insensible al cambio de color al añadir musgo nuevo), acotar Farneback ponderado por gradiente en crestas (`max_flow_px=2.5`) para no deformar cojines planos y exportar tanto `aligned_rgb/` (`sigma=180 px` exposición) como `aligned_hp/` (`<< 8` cero-byte bajo).
3. **Proyección cromática completa vs `High-Pass` (`--projection-mode full`):** usar por defecto `--projection-mode full` (`project_full_albedo_and_ao`, `ndotv^10`, refuerzo `1.35×` en `0°`, fallback pizarra-oliva neutro en grietas ocluidas) para conservar el color real `2K`; purgar siempre capas UV ajenas a `"UVMap"` al abrir `passA_scene.blend` para evitar el límite `MAX_MTFACE=8` de Blender.
4. **Protección de props `Slot 2` y compuerta anti-cráter (`catan_hero360_bake.py`):** excluir `~is_slot2_v` al esculpir y aplicar `cliff_gate` (`slope >= 42°`) en `sculpt_3d_from_baked_albedo` (`0.000 mm` de excavación negativa en tierra/musgo planos; solo ondulación positiva suave `0..+0.10 mm` en musgo verde).

## 6. Cómputo GPU Remoto Zero-Idle en Google Colab Pro (`tools/colab/`)

1. **Verificar alcance Drive (`drive.file`) antes de lanzar GPU:** `python tools/colab/colab_api_runner.py check-drive` (si devuelve `False`, ejecutar `python tools/colab/colab_api_runner.py auth` 1 vez).
2. **Ejecutar orquestador en 4 fases (`run_v05_passb_autonomous.py` + `live_monitor_server.py` en `:8799`):**
   - **Fase 1 (`prep-local` PC, `0` créditos):** genera `gpu_job_scene.blend` y caché exacta `macro_vis_cache.npz` (`BVHTree` + `np.packbits`, recarga `< 0.05 s`).
   - **Fase 2 (`drive_staging.py`, GPU apagada = `0` créditos):** subida resumable a `Maker3D_Colab_Staging`.
   - **Fase 3 (`--stage gpu-bake-only`, `L4 OptiX` `~2–2.5 min`):** sonda `__KERNEL_OK__` `< 25s`, telemetría `[TELEM]` en `t=0s`, `line_buffering=True` (`flush=True`), autorrefresco OAuth `>1h`, solo PNGs hacia Drive y **`DeleteRuntime` inmediato antes de descargar**.
   - **Fase 4 (`--stage finalize-local` PC, `0` créditos):** descarga Drive + borrado temporal Drive + `sculpt_3d` anti-cráter + `STL`/`GLB` + auditoría `0.20 mm`.
