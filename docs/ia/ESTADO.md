# ESTADO VIVO DEL PROYECTO — Maker3D (`C:\Users\migue\Antigravity\Maker3D`)

## Misión Actual
Suite modular CAD/Mesh/Escultura 3D con IA (**Maker3D 2.0**) para piezas paramétricas exactas (B-Rep / OpenSCAD+BOSL2), losetas modulares (`Tableros-Catan`), miniaturas orgánicas (Image-to-3D + Blender 5.3) y visores WebGL móviles (`Visores-HTML`).

## Arquitectura Maker3D 2.0 (OPERATIVA Y VERIFICADA)
- **Motores y Binarios Locales:** `Blender 5.3.0` (portable `--background` + `CYCLES` CPU + addon `blender_mcp` activado), `OpenSCAD Nightly 2026.11.21` (`--backend=Manifold` + `BOSL2`), `Bambu Studio 02.08.02.61` CLI (perfiles A1/P1S/X1C), `uv/uvx 0.13.9`, `Node v24` + `@gltf-transform/cli 4.5.1` (`gltf-transform.cmd` Meshopt + WebP).
- **Stack Python 3.12:** `build123d 0.13.0`, `pymeshlab 2025.7.post1`, `landlab 2.11.0`, `lib3mf 2.5.0`, `trimesh 5.1.1`, `manifold3d 3.5.4`, `gradio-client 2.7.2`, `fal-client 1.0.3`, `replicate 1.0.7`, `python-dotenv 1.2.4`, `numpy`, `scipy`, `pillow`.
- **Servidores MCP (`~/.gemini/config/mcp_config.json` — Handshake JSON-RPC verificado):**
  - `build123d` (`build123d-mcp v0.3.91`, 48 tools B-Rep/STEP/validación).
  - `openscad` (`OpenSCAD MCP Server v4.2.4`, 12 tools + `BOSL2` + `Manifold`).
  - `blender` (`MCP for Blender v1.30.0`, 10 tools + addon persistente en Blender 5.3).
- **Suite de 6 Skills + Regla (`.agents/skills/` y `.agents/rules/maker3d-architecture.md`):**
  1. `maker3d-router` (orquestador principal para cualquier nueva petición 3D).
  2. `parametric-brep-cad` (`build123d` STEP+STL y `OpenSCAD + BOSL2`, tabla de holguras FDM).
  3. `generative-3d-mesh` (Nano Banana → `cloud_3d_bridge.py`: HF ZeroGPU `TRELLIS.2`/`Hunyuan3D`/`Hi3DGen` o `fal`/`replicate`).
  4. `terrain-geology-engine` (`Landlab` erosión fluvial, drenaje monótono y `0%` nieve en `>=44°`).
  5. `blender-sculpt-bake` (estratos 3D en normales `>=45°`, fusión booleana estanca + 6 vistas PBR/Arcilla + orto).
  6. `print-audit-and-webgl` (`trimesh`/`PyMeshLab`/`Manifold3D`/`lib3mf`, Bambu CLI solo bajo petición + visor HTML `Meshopt+WebP <20MB`).
- **Scripts de Producción (`tools/`):** `tool_paths.py`, `env_check.py`, `catan_kit/build_catan_base_brep.py` (`--export-cutter`), `terrain/hydraulic_terrain_gen.py`, `blender/sculpt_fuse_bake_render.py`, `blender/build_hero_mountain_local.py`, `blender/fuse_generative_catan_tile.py`, `blender/catan_3d_factory_csg.py`, `blender/catan_3d_factory_bake.py`, `generative/cloud_3d_bridge.py`, `audit/mesh_and_print_audit.py`, `webgl/optimize_and_build_viewer.py`.
- **Smoke Test End-to-End (`tools/smoke_test_out/` — 100% PASS):** Base OCCT `110.8513×128.0000×25.0000 mm` + Landlab `alpine_canyon` (`0.0%` nieve `>=44°`, talweg monótono `314` pts) + Blender unión booleana 5 pinos 3D (`non_manifold=0`) + `lib3mf` + Bambu A1 (`14h 28m 59s`, `282` capas) + Visor Meshopt+WebP (`1.08 MB`, `0` errores).
- **Prueba Real `montana_maker3d2/` (100% Local — PASS):** Recreación de `Montaña Centrada` con B-Rep `build123d` + Blender `5.3.0` OpenVDB `0.24 mm` + 176 props 3D booleanos (`245,774 V / 491,544 Δ`, `repaired=false`, `is_watertight=true`, `χ=2`, `6/6` pads `25.0 mm`); STL `23.44 MB`, `3MF` `8.24 MB`, Bambu A1 (`12h 55m 17s`), visor `Visores-HTML/catan-bioma-montana-maker3d2.html` (`13.04 MB`, `0` errores).
- **Prueba Real `montana_trellis2/` (Híbrida `TRELLIS.2` + `build123d` — PASS):** Reparado `cloud_3d_bridge.py` (`int(...)`) y ejecutado `microsoft/TRELLIS.2` (`1024`, `300k Δ`, PBR `2048`) + solidificación `OpenVDB 0.25 mm` + booleano B-Rep + bake `EMIT 2K` (`710,851 V / 1,422,182 Δ`, `is_watertight=true`, `genus=121` vía `--allow-high-genus`, `Manifold3D: Error.NoError`, `non_manifold=0`, `1` componente, `6/6` pads `25.0 mm`, `270.115 cm³`); STL `67.82 MB`, `3MF` `22.26 MB` (sin laminar), visor `Visores-HTML/catan-bioma-montana-trellis2.html` (`43.76 MB`, `Meshopt+WebP 2048`, 360° frente/trasera verificados en Chromium con `0` errores).
- **Fábrica Modular 3D `catan_3d_factory` (`montana_factory_v8` — Obsoleto `2/10` visual):** Pipeline geométrico validado en `3MF`/`WebGL` (`3.955 MB`), superado estéticamente por el pipeline Hero 360°.
- **Pipeline Hero 360° `catan_hero360_bake.py` (`montana_hero360` — PASS 360°):** Conserva el skyline y geometría V4 (`Z_geom_v4.npy` + `hero_nanobanana.jpg` en `0°`), completa 3 contrafuertes y 2 circos glaciares norte (`Z_hero360.npy`) alineados `pixel-a-pixel` con 5 vistas `Nano Banana` (`IoU 0.919–0.942`), proyecta `BVH + dot(N,V)^4` sobre `Smart UV Project 3D`, esculpe grietas 3D dual-banda por normal real (`[-0.689, +0.319] mm`) y hornea PBR `4K DIFFUSE / 2K NORMAL+ROUGHNESS`. `125,571 V / 251,138 Δ`, `repaired=false`, `watertight=true`, `χ=2`, `non_manifold=0`, `thin_wall=0.0%`, `overhang=0.013%`, `6/6` pads `25.0 mm`, `346.283 cm³`, `catan_montana_hero360.3mf` (`4.16 MB`), visor `Visores-HTML/catan-bioma-montana-hero360.html` (`8.11 MB` Meshopt+WebP, `84.18%` ahorro, `0` errores WebGL en `0°`/`~180°`/`Arcilla CAD`).

## Activos Legacy Preservados (`Tableros-Catan/`)
- Los **8/8 biomas Catán** (`Montaña V4/DEF/Centrada V3`, `Bosque 3D`, `Colinas`, `Pastos`, `Campos`, `Desierto`, `Río de Oro v08`, `Océano`), visores en `Visores-HTML/` y la skill `.agents/skills/nanobanana-3d-sculpt/` permanecen intactos y plenamente operativos.

## Próximas Tareas Prioritarias
1. Revisar con el usuario la nota visual de `montana_hero360` (frente, espalda `180°` y `Arcilla CAD`) y, tras validación, replicar `montana_hero360_prep.py` + `tools/blender/catan_hero360_bake.py` al resto de biomas Catán.
2. Opcional para el usuario: añadir `FAL_KEY` o `REPLICATE_API_TOKEN` en `.env` solo si desea superar el límite diario gratuito de ZeroGPU (`300 s/día`, ya conectado con `HF_TOKEN`).
