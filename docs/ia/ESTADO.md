# ESTADO VIVO DEL PROYECTO — Maker3D (`C:\Users\migue\Antigravity\Maker3D`)

## Misión Actual
Suite modular CAD/Mesh/Escultura 3D con IA (**Maker3D 2.0**) para piezas paramétricas exactas (B-Rep / OpenSCAD+BOSL2), losetas modulares (`Tableros-Catan`), miniaturas orgánicas (Image-to-3D + Blender 5.3) y visores WebGL móviles (`Visores-HTML`).
**Último hito completado (`25/11/2026` — `montana_hero360 v06-full8k` Entregado):** Ejecutado de punta a punta en flujo **Zero-Idle GPU 4 Fases** (`~14.2 min` runtime `L4` con `DeleteRuntime` previo a descarga, `0` runtimes activos, `0%` cuota Gemini), eliminados al 100% los mini-cráteres de valle y preservada la proyección cromática íntegra `2K -> 8K` (`6.15M Δ`, `χ=2`, `degen=0`).

## Arquitectura y Estado de Motores (OPERATIVA)
- **Motores Locales:** `Blender 5.3.0`, `OpenSCAD Nightly 2026.11.21` (`Manifold + BOSL2`), `Bambu Studio 02.08.02.61` (perfiles `0.20 mm`), `Node v24` + `@gltf-transform/cli 4.5.1`, Python 3.12 (`build123d`, `pymeshlab`, `landlab`, `lib3mf`, `trimesh`, `manifold3d`, `opencv-python`). `python tools/env_check.py` -> `PASS`.
- **Cómputo Remoto Colab Pro (`Zero-Idle GPU 4 Fases`):**
  - `drive_staging.py`: pre-subida a Google Drive (`Maker3D_Colab_Staging`, scope `drive.file`) **con GPU apagada (`0` créditos)** y descarga tras `DeleteRuntime`.
  - `colab_api_runner.py`: `SCOPES = colaboratory + drive.file`, `timeout=(10,60)`, 5 reintentos kernel, sonda `__KERNEL_OK__` (`<25s`) y `check-drive`.
  - `run_v05_passb_autonomous.py`: separación `prep-local` (caché `BVHTree` `np.packbits` `~1.2 MB`) → `gpu-bake-only` (`~2–2.5 min` `L4 OPTIX` puros, `flush=True`, `[TELEM]` en `t=0s`) → `finalize-local` (CPU local).
- **MCPs Verificados:** `build123d`, `openscad`, `blender`.

## Estado `montana_hero360` (`v06-full8k` Completado · `v03/v04/v05` Intactos)
- **`v06-full8k`:** `6,149,730 Δ`, `watertight=true`, `χ=2`, `degen=0`, `6/6` pads `25.0 mm`, `catan_montana_hero360_v06.3mf` (`98.41 MB`). Visores `v06-full8k.html` (`68.93 MB` en `.gitignore`) y `...-raw.html` (`542.27 MB`) en `Sol Alpino` con `0` errores WebGL.
- **Trazabilidad:** Bitácoras en `versiones/v06_full8k/`, [montana_hero360_v06_showcase.md](file:///C:/Users/migue/.gemini/antigravity/brain/34342c68-2385-41c9-a165-20883e62734d/montana_hero360_v06_showcase.md) y `TELEMETRIA_CUOTAS.md`.

## Próximas Tareas Prioritarias
1. Revisión visual final por el usuario del visor `catan-bioma-montana-hero360-v06-full8k.html` (o laminado `--bambu-slice` si pide tiempos/G-code).
2. Siguiente bioma o loseta modular Catán (enrutando por `.agents/skills/maker3d-router/SKILL.md`).
