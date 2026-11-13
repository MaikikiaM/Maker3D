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
- **`nanobanana-3d-sculpt` (`.agents/skills/nanobanana-3d-sculpt/SKILL.md`):** Actualizada a **6 vistas equidistantes hexagonales (`0°–300°`)** cada 60°, combinando conceptos y *heightmaps* cenitales de `image-generator` con [build_catan_biome.py](../../Tableros-Catan/pipeline-biomas/build_catan_biome.py) (`watertight=True`, $\chi=2$, faldón suavizado `16.5mm`).
- **Avance Biomas Catan 3D (`4/8` completados — pausa por cuota `image-generator`):**
  - **Bioma 1 (Montaña equilibrada):** `Z=65.0mm` (`relief=40mm`), `V=95,697 / F=191,390 Δ` ($\chi=2$, `427.51 cm³`), visor `Visores-HTML/catan-bioma-montana.html` (`3.43 MB`).
  - **Bioma 2 (Bosque / Madera):** `Z=63.0mm` (`relief=38mm`), `V=95,697 / F=191,390 Δ` ($\chi=2$, `399.76 cm³`), visor `Visores-HTML/catan-bioma-bosque.html` (`3.33 MB`).
  - **Bioma 3 (Colinas / Arcilla):** `Z=53.0mm` (`relief=28mm`), `V=95,697 / F=191,390 Δ` ($\chi=2$, `380.92 cm³`), visor `Visores-HTML/catan-bioma-colina.html` (`3.11 MB`).
  - **Bioma 4 (Pastos / Lana):** `Z=38.0mm` (`relief=13mm`), `V=95,697 / F=191,390 Δ` ($\chi=2$, `319.42 cm³`), pradera esmeralda + muro seco de caliza + sendero arenoso, visor `Visores-HTML/catan-bioma-pasto.html` (`2.94 MB`).
  - **Bioma 5 (Campos / Trigo):** lámina conceptual 6 vistas (`catan_fields_turnaround_1794609034842.jpg`) generada; *heightmap* cenital y **Biomas 6–8** (`Desierto`, `Río de Oro`, `Océano`) detenidos limpiamente por `429 RESOURCE_EXHAUSTED` de `gemini-3.1-flash-image` hasta `2026-11-14T03:17:43Z` (`04:17:43` CET).

## Próximas Tareas Prioritarias
1. Reanudar **Bioma 5–8** (`campo` desde su lámina 6 vistas ya guardada, `desierto`, `oro`, `oceano`) a partir de las `04:17:43` CET (`2026-11-14T03:17:43Z`) tras el reseteo de `gemini-3.1-flash-image`.
2. Validar corte de impresión multicolor real (`infimech_cmykwog_mountain_slicing_guide.md`) con `catan_mountain_fullcolor.3mf` V2 o derivados LOD en OrcaSlicer/PrusaSlicer.
3. Continuar con el resto de losetas modulares del tablero Catan 127 losetas siguiendo la arquitectura V2 pure NumPy/SciPy.
