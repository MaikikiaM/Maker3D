# ESTADO VIVO DEL PROYECTO — Maker3D (`C:\Users\migue\Antigravity\Maker3D`)

## Misión Actual
Suite CAD/Mesh profesional de modelado procedimental e impresión 3D avanzada (FDM/CMYKWOG/Full-Color) para piezas de ingeniería y tableros modulares (`Tableros-Catan`) junto con visores interactivos WebGL/Android (`Visores-HTML`).

## Estado del Componente Activo: `Tableros-Catan / Loseta Montaña 3D (Cordillera V3 PBR)`
- **Pipeline V3 (`pipeline-montana/prototype_v3_cordillera.py` + `build_catan_mountain_master_v2.py`):** 100% Pure NumPy + SciPy Delaunay + PIL + zipfile (sin `trimesh`). Muestreo canónico FFT para alineación bit-idéntica entre todos los LODs y texturas UV 4K/2K.
- **Geometría y Tolerancias Catan:** Hexágono regular exacto (`R=64.00mm`, `L=64.00mm`, `W=110.8513mm`, `D=128.00mm`, área `106.42 cm²`), zócalo 100% sólido (`Z=0..25.00mm`), cumbre en `Z=83.000mm` (`58.00mm` de relieve), cordillera estilizada de 5 agujas puntiagudas conectadas en S + 4 circos glaciares, carril plano `3.50mm`, 6 bases de poblado/ciudad `Ø20.00mm`, cavidad magnética inferior `Ø10.20×5.20mm`, sin hueco de ficha.
- **Malla Maestra y LODs Estancos (`watertight=True`, $\chi=2$):**
  - Master: `V=2,590,913`, `F=5,181,822` triángulos (`Vol=438.72 cm³`) en STL (`247.09 MB`), PLY (`101.31 MB`), OBJ+MTL (`361.75 MB`) y 3MF (`129.39 MB`).
  - LODs: `lod1_1m` (`1,021,686 Δ`), `lod2_250k` (`256,934 Δ`), `lod3_50k` (`59,870 Δ`), todos verificados `chi=2`.
- **Texturizado Hiperrealista Pintado PBR (`4096×4096` / `2048×2048`):**
  - UV `Y-up` alineado y verificado round-trip (1,000 vértices exactos contra PNG), incluyendo **`catan_mountain_roughness_4k.png`** junto a Diffuse, Normal, Winter y Autumn 4K.
  - Gradiente mineral continuo con advección por pendiente (`slope_advect_field`), escorrentía `D-inf` (`flow_accumulation`), 16 direcciones HBAO, transición de escarcha y transición suave al zócalo.
  - **Cero nieve en escarpes $\ge 44.0^\circ$** (`MaxSnow>=44° = 0.000000`), cobertura `Snow=6.59%`, `Meadow=2.98%`, `Rock=90.43%`, `Steep>=44°=61.81%`.
- **Auditoría CAD (`renders_cad/`) y Visor Android AAA PBR (`4.56 MB`):**
  - 11 renders/animaciones regenerados con especular PBR (`render_audit_suite_v2.py`) y espectro tonal continuo verificado.
  - Visor WebGL Three.js (`226,230 Δ`, zócalo hexagonal con normales planas limpias, Diffuse+Normal+Roughness 2K, HDR IBL por `PMREMGenerator`, sombras suaves `PCFSoftShadowMap`, 3 atmósferas, corte sólido stencil Z y drawer responsive móvil) desplegado en las 5 rutas HTML y verificado en Chromium (`0` errores de consola, `<1.2s`).

## Skills y Workflows del Workspace
- **`nanobanana-3d-sculpt` (`.agents/skills/nanobanana-3d-sculpt/SKILL.md`):** Integra generación imaginativa de Nano Banana (`image-generator`) —turnarounds conceptuales, mapas de altura cenitales y texturizado pictórico alineado al DEM— con el puente CAD puro NumPy/SciPy `scripts/nanobanana_to_catan_hex.py` (`watertight=True`, $\chi=2$, cotas Catan exactas).
- **Prototipo Experimental V4 Nano Banana (`pipeline-montana/build_v4_nanobanana_preview.py` -> `Visores-HTML/preview-montana-v4-nanobanana.html`):** Refinado tras auditoría visual (`slope_gamma=0.72`, `relief=60mm` -> `Z=85mm`, recorte físico de nieve en `>=44°` con relleno mineral FFT `max_snow_ge44=0.000000%`). Malla estanca `V=106,929 / F=213,854` ($\chi=2$, Vol `474.86 cm³`) en visor preview `3.78 MB` (sin sobrescribir V3).

## Próximas Tareas Prioritarias
1. Ajustar curva gamma del *heightmap* Nano Banana y filtro físico de nieve $\ge 44^\circ$ si el usuario decide promover el prototipo V4 Nano Banana a versión maestra `5M+ Δ`.
2. Validar corte de impresión multicolor real (`infimech_cmykwog_mountain_slicing_guide.md`) con `catan_mountain_fullcolor.3mf` V2 o derivados LOD en OrcaSlicer/PrusaSlicer.
3. Continuar con el resto de losetas modulares del tablero Catan 127 losetas siguiendo la arquitectura V2 pure NumPy/SciPy.
