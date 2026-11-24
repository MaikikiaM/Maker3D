# ESTADO VIVO DEL PROYECTO — Maker3D (`C:\Users\migue\Antigravity\Maker3D`)

## Misión Actual
Suite modular CAD/Mesh/Escultura 3D con IA (**Maker3D 2.0**) para piezas paramétricas exactas (B-Rep / OpenSCAD+BOSL2), losetas modulares (`Tableros-Catan`), miniaturas orgánicas (Image-to-3D + Blender 5.3) y visores WebGL móviles (`Visores-HTML`).
**Última entrega completada (`montana_hero360 v05-macro8k`, boquilla `0.20 mm`):** Pipeline `F0 → F10` cerrado íntegramente vía API oficial de **Google Colab Pro (`NVIDIA L4` `OPTIX`)**, auditoría `0.20 mm` estanca (`6.15M Δ`, `χ=2`, `degen=0`), `.3mf` verificado y doble visor HTML (`8K Meshopt+WebP` y `RAW`) sin errores en Chromium.

## Arquitectura y Estado de Motores (OPERATIVA)
- **Motores Locales:** `Blender 5.3.0`, `OpenSCAD Nightly 2026.11.21` (`Manifold + BOSL2`), `Bambu Studio 02.08.02.61` (perfiles `0.20 mm` `a1_n02 / p1s_n02 / x1c_n02`), `Node v24` + `@gltf-transform/cli 4.5.1`, Python 3.12 (`build123d`, `pymeshlab`, `landlab`, `lib3mf`, `trimesh`, `manifold3d`, `opencv-python`). `python tools/env_check.py` -> `PASS`.
- **Cómputo Remoto Colab Pro (`tools/colab/` — OPERATIVO 100% DESATENDIDO):** `colab_api_runner.py` (autorrefresco OAuth2 >1h + retries kernel/WS), `run_v05_passb_autonomous.py` (transferencia paralela `8` hilos) y `live_monitor_server.py` (`http://localhost:8799`). Validado en `NVIDIA L4` (`22.49 GiB` VRAM, `52.96 GiB` RAM, `OPTIX`) con `DeleteRuntime` automático (`0` runtimes residuales). Ver [TELEMETRIA_CUOTAS.md](docs/ia/TELEMETRIA_CUOTAS.md).
- **MCPs Verificados:** `build123d`, `openscad`, `blender`.
- **Skills Maker3D 2.0:** `maker3d-router`, `parametric-brep-cad`, `generative-3d-mesh`, `terrain-geology-engine`, `blender-sculpt-bake`, `print-audit-and-webgl` (+ `nanobanana-3d-sculpt` legacy).

## Entrega `montana_hero360 v05-macro8k` (`COMPLETADA F0–F10`)
- **F5.3 (`16/16` OK):** Alineación ECC sobre gradiente Sobel + Farneback ponderado (`max_flow=2.5 px`) + High-Pass `16-bit` (`ssim_low 0.648–0.806`, `ridge_recall_5px 0.868–0.983`).
- **F6 (`Colab NVIDIA L4 OptiX` — `19m 03s`):** `baked_albedo.png` (`8192×8192`, `64.01 MB`), `baked_normal.png` (`8192×8192`, `74.18 MB`), `baked_roughness.png` (`4096×4096`, `0.68 MB`), `ao_cavity_8k.png`, `tile_master.stl` (`293.24 MB`), `tile_viewer.glb` (`399.14 MB`), `audit_6views_pbr/clay.png`, `audit_ortho.png`.
- **F7 (`PASS` sin reparación):** `6,149,730` triángulos, `is_watertight=true`, `euler_number=2`, `degenerate_faces=0`, `repaired=false`, `overhang=0.058%`, `6/6` pads `25.0 mm`, `catan_montana_hero360_v05.3mf` (`100.04 MB`, `lib3mf` OK).
- **F8–F10 (`PASS`):** `catan-bioma-montana-hero360-v05-macro8k.html` (`75.45 MB`, Meshopt+WebP `8K`, en `.gitignore` por `>50 MB`) y `...-raw.html` (`532.21 MB`) verificados con `0` errores en Chromium; snapshot en `versiones/v05_macro8k/` y `v03/v04` intactos.

## Próximas Tareas / Mejoras Opcionales
1. Laminar `catan_montana_hero360_v05.3mf` con `--bambu-slice --printer a1_n02` solo cuando se desee generar G-code `0.20 mm`.
2. Para una futura `v06`: iluminar preset inicial de `optimize_and_build_viewer.py` (`Sol Alpino` más brillante) y desaturar subsuelo marrón `passA` bajo el musgo verde.
