# ESTADO VIVO DEL PROYECTO — Maker3D (`C:\Users\migue\Antigravity\Maker3D`)

## Misión Actual
Suite modular CAD/Mesh/Escultura 3D con IA (**Maker3D 2.0**) para piezas paramétricas exactas (B-Rep / OpenSCAD+BOSL2), losetas modulares (`Tableros-Catan`), miniaturas orgánicas (Image-to-3D + Blender 5.3) y visores WebGL móviles (`Visores-HTML`).
**Último hito completado (`25/11/2026` — `montana_hero360 v07-shrubs8k` Entregado · `0` créditos Colab):** Corregido el desenfoque/liso del valle mediante proyección multi-banda Burt-Adelson `(S/Smax)^32` (`Laplacian std` `0.0433` → `0.10811`), `Quilt 2K` perimetral y esculpido botánico `3D` (`908` mini-arbustos en cojín + `1,200` guijarros facetados, `0` cráteres).

## Arquitectura y Estado de Motores (OPERATIVA)
- **Hardware Objetivo FDM:** **InfiMech MX Pro** (`8` cabezales independientes, OrcaSlicer), **exclusivamente boquillas `0.20 mm` + `PLA` con técnicas Full Color**.
- **Motores Locales:** `Blender 5.3.0`, `OrcaSlicer 2.5.2` (`%LOCALAPPDATA%\Programs\OrcaSlicer\orca-slicer.exe`), `PrusaSlicer 2.9.7` (`INDX` + `ColorMix`), `OpenSCAD Nightly 2026.11.21` (`Manifold + BOSL2`), `Bambu Studio 02.08.02.61`, `Node v24` + `@gltf-transform/cli 4.5.1`, Python 3.12 (`build123d`, `pymeshlab`, `landlab`, `lib3mf`, `trimesh`, `manifold3d`, `opencv-python`). `python tools/env_check.py` -> `PASS`.
- **Cómputo Remoto Colab Pro (`Zero-Idle GPU 4 Fases`):**
  - `drive_staging.py`: pre-subida a Google Drive (`Maker3D_Colab_Staging`, scope `drive.file`) **con GPU apagada (`0` créditos)** y descarga tras `DeleteRuntime`.
  - `colab_api_runner.py`: `SCOPES = colaboratory + drive.file`, `timeout=(10,60)`, 5 reintentos kernel, sonda `__KERNEL_OK__` (`<25s`) y `check-drive`.
  - `run_v05_passb_autonomous.py`: separación `prep-local` (caché `BVHTree` `np.packbits` `~1.2 MB`) → `gpu-bake-only` (`~2–2.5 min` `L4 OPTIX` puros, `flush=True`, `[TELEM]` en `t=0s`) → `finalize-local` (CPU local).
- **MCPs Verificados:** `build123d`, `openscad`, `blender`.

## Estado `Tableros-Catan` (Reordenado) + `montana v07-shrubs8k`
- **Estructura limpia:** `Tableros-Catan/losetas/<bioma>/work/` (`montana`, `bosque`, `colina`, `pasto`, `campo`, `desierto`, `oro`, `oceano`), `Tableros-Catan/docs/`, `Tableros-Catan/assets/`, `Tableros-Catan/legacy/` y `Tableros-Catan/archivo/` (scripts `tools/` desacoplados con `--tile <bioma>` / `--work-dir`).
- **`losetas/montana` (`v07-shrubs8k`):** `6,149,730 Δ`, `watertight=true`, `χ=2`, `degen=0`, `thin_wall=0.0%`, `overhangs=0.162%`, `6/6` pads `25.011 mm`, `catan_montana_hero360_v07.3mf` (`99.50 MB`). Visores `v07-shrubs8k.html` (`75.81 MB` en `.gitignore`) y `...-raw.html` (`575.11 MB`) en `Sol Alpino` con `0` errores WebGL.
- **`losetas/campo` (`v09-pbr` Clay-Locked 8K · Entregado · Colab T4-HIGHMEM OPTIX Zero-Idle):**
  - Geometría base `v07` (`--wheat-style ploughed` `0.88 mm`, `3,319,632 Δ`, `watertight=true`, `χ=2`, `degen=0`, `repaired=false`, `overhangs=0.417%`, `thin_wall=0.0%` a `0.20 mm`, `6/6` pads `25.0 mm`, imán `5.2 mm`, `0` props descartados).
  - Sustitución de proyección Nano Banana (`v08`) por textura `4096×4096` (`~32 px/mm`) sintetizada en `tools/blender/build_campo_clay_locked_4k.py` directamente desde `campo_semantic_fields.npz` y `parcel_angles_deg` (bloqueo `0.000 mm` a cada cresta pajiza, entresurco tierra, camino/roderas, claro agrícola y bordes verdes) + sombreado continuo suave por vértice por slot semántico (`2..14`, `0` parches triangulares planos) horneado en Colab `Tesla T4 SHAPE_HIGHMEM OPTIX` (`361.7 s`, `DeleteRuntime` previo a descarga).
  - Visor WebGL final: `Visores-HTML/catan-bioma-campo-v09-pbr.html` (`34.148 MiB` `< 50 MB`, `83.78%` ahorro `meshopt+webp 4K`, `0` errores WebGL), enlazado como `Bioma 5K` en `Visores-HTML/index.html` (conservando `v01`–`v08`) y archivado en `Tableros-Catan/losetas/campo/versiones/v09_pbr_clay_locked/`.
- **Trazabilidad:** `tools/blender/build_v07_shrubs.py --tile montana`, `tools/blender/build_campo_clay.py`, `tools/blender/build_campo_clay_locked_4k.py`, `tools/blender/campo_pbr_bake.py`, `tools/colab/run_campo_pbr_colab.py`, `Tableros-Catan/losetas/montana/versiones/v07_shrubs8k/`, `Tableros-Catan/losetas/campo/versiones/v09_pbr_clay_locked/`, [montana_hero360_v07_showcase.md](file:///C:/Users/migue/.gemini/antigravity/brain/34342c68-2385-41c9-a165-20883e62734d/montana_hero360_v07_showcase.md) y `TELEMETRIA_CUOTAS.md`.

## Próximas Tareas Prioritarias
1. Revisión del usuario del visor final `Visores-HTML/catan-bioma-campo-v09-pbr.html`.
2. Resto de biomas (`bosque`, `colina`, `pasto`, `desierto`, `oro`, `oceano`) con el estándar `Maker3D 2.0`.
