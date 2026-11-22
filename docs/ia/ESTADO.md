# ESTADO VIVO DEL PROYECTO — Maker3D (`C:\Users\migue\Antigravity\Maker3D`)

## Misión Actual
Suite modular CAD/Mesh/Escultura 3D con IA (**Maker3D 2.0**) para piezas paramétricas exactas (B-Rep / OpenSCAD+BOSL2), losetas modulares (`Tableros-Catan`), miniaturas orgánicas (Image-to-3D + Blender 5.3) y visores WebGL móviles (`Visores-HTML`).
**En curso (`montana_hero360 v05-macro8k`, boquilla `0.20 mm` / capa `0.04–0.06 mm`):** Pausa controlada en **F5.2/F5.3** por agotamiento de cuota en API/app Gemini. Todo el cómputo pesado previo (`F0–F5.1`) está preservado en `Tableros-Catan/pipeline-biomas/montana_hero360/work_v05/` y listo para retomar desde F5.3.

## Arquitectura y Estado de Motores (OPERATIVA)
- **Motores Locales:** `Blender 5.3.0`, `OpenSCAD Nightly 2026.11.21` (`Manifold + BOSL2`), `Bambu Studio 02.08.02.61` (incluidos perfiles `a1_n02 / p1s_n02 / x1c_n02` de `0.20 mm`), `Node v24` + `@gltf-transform/cli 4.5.1`, Python 3.12 (`build123d`, `pymeshlab`, `landlab`, `lib3mf`, `trimesh`, `manifold3d`, `opencv-python-headless 5.1.0`). `python tools/env_check.py` -> `PASS`.
- **MCPs Verificados:** `build123d`, `openscad`, `blender`.
- **Skills Maker3D 2.0:** `maker3d-router`, `parametric-brep-cad`, `generative-3d-mesh`, `terrain-geology-engine`, `blender-sculpt-bake`, `print-audit-and-webgl` (+ `nanobanana-3d-sculpt` legacy).

## Avance `montana_hero360 v05-macro8k` (`work_v05/` — NO SOBRESCRIBIR)
- **F0–F1 (`PASS`):** Perfiles `0.20 mm` activos (`tool_paths.py`, `mesh_and_print_audit.py`); `optimize_and_build_viewer.py` con soporte HTML RAW chunked verificado en Chromium sobre `v04` (`134.76 MB`, `0` errores).
- **F2–F3 (`PASS`, `χ=2`):** `tile_base.stl` (`1.13M Δ`), `tile_geology.stl` (`5.80M` caras, micro-fracturas graníticas anisotrópicas `geology_micro_detail.py`), `tile_base_talus.stl` (`6.15M` caras, **125 rocas Poly Haven CC0** enterradas 45–55% vía `Manifold3D`, `talus_manifest.json` + `.rock_faces.npy`).
- **F4–F5.1 (`PASS`):** `passA_albedo_8k.png` (`67.6%` UV activo en `8192×8192`), `passA_scene.blend`, `16` tríos `2048×2048` en `macro/renders/` (`85.7%` cobertura BVH). Aplicado parche en `catan_hero360_bake.py` (`is_slot2_v`) excluyendo rocas CC0 de `sculpt_3d_from_baked_albedo`.
- **F5.2–F5.3 (PAUSA AQUÍ):** `10/16` en `macro_report.json`; **`B2_150`** (`nb/patch_B2_150_nb.jpg`, `hp_std=0.13063`) aprobada como referencia ("B2 prime"). Faltan `6` (`B3_210, B5_330, C0_peak, C1_cirqueW, C2_cirqueE, D0_valleyS`) + mejorar `9` v1 (`A0..A5, B0, B1, B4`) usando modo **3 imágenes** (`_pbr` + `_clay` + `patch_B2_150_nb.jpg`) descrito en `work_v05/Estado_Continuacion_Montana_Hero360_v05.pdf`.

## Entregas Anteriores y Legacy Preservados
- **8/8 biomas Catán** y `montana_hero360 v03-meshopt` (`18.79 MB`) / `v04-raw` (`134.76 MB`) en `Visores-HTML/` y `versiones/`.

## Próximas Tareas Prioritarias (Retomar en F5.3 → F10)
1. Generar/recibir las 15 vistas macro al nivel de `patch_B2_150_nb.jpg` y alinear con `tools/generative/macro_patch_align.py --macro-dir .../work_v05/macro --sigma-hp 6.0`.
2. Ejecutar Blender `--stage passB --bake-res 8192 --normal-res 8192 --rough-res 4096` -> `tile_master.stl` + `tile_viewer.glb` + auditoría `--nozzle-mm 0.2 --catan-check` (`watertight=true`, `χ=2`, `degen=0`, sin `--bambu-slice`).
3. Generar y verificar en Chromium los visores `...-v05-macro8k.html` y `...-v05-macro8k-raw.html`, QA lado a lado frente a `V4`, archivar en `versiones/v05_macro8k/` y sincronizar con GitHub.
