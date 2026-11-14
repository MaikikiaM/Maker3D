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
- **Avance Biomas Catan 3D (`4/8` completados y mejorados a `291,546 Δ` PBR — Biomas 5–8 pendientes de cuota):**
  - **Bioma 1 (Montaña equilibrada V2):** `Z=65.0mm` (`relief=40mm`), `V=145,775 / F=291,546 Δ` ($\chi=2$, `383.29 cm³`), soldado morfológico (`ridge-clean 0.85`), paleta alpina cálida granito-musgo-cascajo-nieve (`<44°`) y borde limpio, visor `Visores-HTML/catan-bioma-montana.html` (`4.92 MB`).
  - **Bioma 2 (Bosque / Madera 3D Real V3):** `Z=57.46mm` (`V=281,229 / F=416,246 Δ` en `build_bosque_3d.py`, `299.55 cm³`), 15 coníferas 3D auténticas (tronco curvado + contrafuertes radiales + 9–11 coronas festoneadas salientes) unidas por booleana exacta `manifold3d` (`genus=41`, 0 aristas abiertas) sobre suelo inpainted limpio con `resample_canonical_xy` y peñas biseladas, visor `Visores-HTML/catan-bioma-bosque.html` (`10.46 MB`).
  - **Bioma 3 (Colinas / Arcilla V3 — Mallado por Contorno + Estratos 3D):** `Z=53.0mm` (`relief=28mm`), `V=451,953 / F=903,902 Δ` ($\chi=2$, `watertight=True`, `385.88 cm³`), malla `nx=740` deformada por gradiente y con flips de arista alineados a curvas de nivel (`1,145,189` flips vía `terrain_mesh_utils.py`), faldón/pads de esquina pizarra limpios y shader triplanar GLSL `clay` sin estrías verticales, visor `Visores-HTML/catan-bioma-colina.html` (`15.27 MB`).
  - **Bioma 4 (Pastos / Lana V2):** `Z=38.0mm` (`relief=13mm`), `V=145,775 / F=291,546 Δ` ($\chi=2$, `318.97 cm³`), micro-normal en sillares de caliza + pradera esmeralda limpia hasta el carril, visor `Visores-HTML/catan-bioma-pasto.html` (`4.30 MB`).
  - **Bioma 5 (Campos / Trigo V3):** `Z=40.0mm` (`relief=15mm`), `V=338,097 / F=676,190 Δ` ($\chi=2$, `watertight=True`, `331.86 cm³`), malla `--nx 640 --contour-align` (`760,307` flips), terrazas doradas de cultivo, muros de mampostería, era circular y dos almiares, visor `Visores-HTML/catan-bioma-campo.html` (`12.22 MB`).
  - **Bioma 6 (Desierto y Oasis V3):** `Z=47.0mm` (`relief=22mm`), `V=338,097 / F=676,190 Δ` ($\chi=2$, `watertight=True`, `364.47 cm³`), malla `--nx 640 --contour-align` (`803,843` flips), duna barján afilada con ondas eólicas, afloramiento de arenisca estratificada y poza turquesa con lajas calizas, visor `Visores-HTML/catan-bioma-desierto.html` (`12.10 MB`).
  - **Bioma 7 (Río de Oro V3):** `Z=51.0mm` (`relief=26mm`), `V=381,965 / F=763,926 Δ` ($\chi=2$, `watertight=True`, `374.53 cm³`), terrazas basálticas soldadas (`--ridge-clean 0.80`), shader triplanar mineral 3D (`--triplanar-mode rock`, `--keep-dark-rock`), vetas de cuarzo aurífero 24K y cascada turquesa, visor `Visores-HTML/catan-bioma-oro.html` (`13.63 MB`).
  - **Bioma 8 (`Océano`):** **completado** con malla alineada por contorno (`nx=660`, `851,219` flips, `V=359,697`, `F=719,390`, `chi=2`, `wt=True`, `wnd=True`, `degen=0`, `Vol=315.45 cm³`, `Z=41.00 mm`, `--keep-dark-rock`), arrecife de basalto escalonado, laguna turquesa y olas en espiral con visor [catan-bioma-oceano.html](../../Visores-HTML/catan-bioma-oceano.html) (`13.12 MB`). Total completado: **8/8 biomas Catan**.

## Próximas Tareas Prioritarias
1. Revisión visual por parte del usuario de los 8 biomas Catan en [Visores-HTML/index.html](../../Visores-HTML/index.html) para ajustes estéticos o preparación de placas de impresión 3D.
2. Validar corte de impresión multicolor real (`infimech_cmykwog_mountain_slicing_guide.md`) con `catan_mountain_fullcolor.3mf` V2 o derivados LOD en OrcaSlicer/PrusaSlicer.
3. Continuar con el resto de losetas modulares del tablero Catan 127 losetas siguiendo la arquitectura V2 pure NumPy/SciPy.
