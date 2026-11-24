# ESTADO VIVO DEL PROYECTO — Maker3D (`C:\Users\migue\Antigravity\Maker3D`)

## Misión Actual
Suite modular CAD/Mesh/Escultura 3D con IA (**Maker3D 2.0**) para piezas paramétricas exactas (B-Rep / OpenSCAD+BOSL2), losetas modulares (`Tableros-Catan`), miniaturas orgánicas (Image-to-3D + Blender 5.3) y visores WebGL móviles (`Visores-HTML`).
**Último hito completado (`25/11/2026` — Arquitectura Zero-Idle GPU + Fixes `v06`):** Tras entregar `montana_hero360 v05-macro8k` (`F0–F10`) e identificar mini-cráteres en tierra, lavado por High-Pass 6px y tiempo muerto GPU en Colab, quedó implementada la canalización `v06` (**`0` créditos gastados** aún; lista para relanzar con las `16` fotos `2K` existentes tras autorizar `drive.file`).

## Arquitectura y Estado de Motores (OPERATIVA)
- **Motores Locales:** `Blender 5.3.0`, `OpenSCAD Nightly 2026.11.21` (`Manifold + BOSL2`), `Bambu Studio 02.08.02.61` (perfiles `0.20 mm`), `Node v24` + `@gltf-transform/cli 4.5.1`, Python 3.12 (`build123d`, `pymeshlab`, `landlab`, `lib3mf`, `trimesh`, `manifold3d`, `opencv-python`). `python tools/env_check.py` -> `PASS`.
- **Cómputo Remoto Colab Pro (`Zero-Idle GPU 4 Fases`):**
  - `drive_staging.py`: pre-subida a Google Drive (`Maker3D_Colab_Staging`, scope `drive.file`) **con GPU apagada (`0` créditos)** y descarga tras `DeleteRuntime`.
  - `colab_api_runner.py`: `SCOPES = colaboratory + drive.file`, `timeout=(10,60)`, 5 reintentos kernel, sonda `__KERNEL_OK__` (`<25s`) y `check-drive`.
  - `run_v05_passb_autonomous.py`: separación `prep-local` (caché `BVHTree` `np.packbits` `~1.2 MB`) → `gpu-bake-only` (`~2–2.5 min` `L4 OPTIX` puros, `flush=True`, `[TELEM]` en `t=0s`) → `finalize-local` (CPU local).
- **MCPs Verificados:** `build123d`, `openscad`, `blender`.

## Estado `montana_hero360` (`v05` Entregado · Código `v06` Listo)
- **`v05-macro8k` (`dbe6955`):** `6.15M Δ`, `χ=2`, `degen=0`, `0` errores WebGL (`75.45 MB` / `532.21 MB`), `0` runtimes activos.
- **Correcciones `v06` ya programadas:** (1) Compuerta anti-cráter `cliff_gate` (`slope >= 42°`) + musgo positivo suave (`0..+0.10 mm`) en `catan_hero360_bake.py`; (2) Proyección cromática completa `2K` (`project_full_albedo_and_ao`) conservando musgo, rocas y nieve de Nano Banana + fallback pizarra neutro; (3) Luz por defecto `Sol Alpino` más viva en `optimize_and_build_viewer.py`.

## Próximas Tareas Prioritarias
1. Ejecutar 1 vez `python tools/colab/colab_api_runner.py auth` para añadir el scope `drive.file` al token local.
2. Cuando el usuario lo ordene, lanzar `v06` con las 16 fotos `2K` ya guardadas (`0%` cuota Gemini, `~2 min` `L4 OptiX`): `python -u tools/colab/run_v05_passb_autonomous.py` (usando `--projection-mode full` por defecto).
