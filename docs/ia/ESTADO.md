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
- **`losetas/campo` (`v03-clay`, paso pre-pintado):** Campiña baja (`Z_suelo ≈ 26–31 mm`, surcos `0.94 mm`) + árboles botánicos de alta resolución (`robles ~87.5k Δ/ud` con corteza estriada + 5 ramas + follaje Voronoi V-crease, `cipreses ~61.8k Δ/ud`, arbustos `subdiv=3`, `Z_max = 44.435 mm`), `3,088,706 Δ`, `watertight=true`, `χ=2`, `degen=0`, `repaired=false`, `thin_wall=0.0%` (`0.20 mm`), `overhangs=0.19%`. Visor `Visores-HTML/catan-bioma-campo-v03-clay.html` (`15.31 MiB` / `16.05 MB`, `meshopt+webp`, `0` errores WebGL) en `index.html` (conservando `v01`/`v02`) y snapshot en `Tableros-Catan/losetas/campo/versiones/v03_clay_pre_paint/`.
- **Trazabilidad:** `tools/blender/build_v07_shrubs.py --tile montana`, `tools/blender/build_campo_clay.py`, `Tableros-Catan/losetas/montana/versiones/v07_shrubs8k/`, [montana_hero360_v07_showcase.md](file:///C:/Users/migue/.gemini/antigravity/brain/34342c68-2385-41c9-a165-20883e62734d/montana_hero360_v07_showcase.md) y `TELEMETRIA_CUOTAS.md`.

## Próximas Tareas Prioritarias
1. Revisión del usuario del visor 3D `Visores-HTML/catan-bioma-campo-v03-clay.html` y, tras reset de cuota `HTTP 429`, regeneración de las vistas pintadas adaptadas a `v03` antes de proyectar/hornear `8K` en GPU Colab (`L4` bajo demanda).
2. Revisión del visor `catan-bioma-montana-hero360-v07-shrubs8k.html` y resto de biomas (`bosque`, `colina`, `pasto`, `desierto`, `oro`, `oceano`).
