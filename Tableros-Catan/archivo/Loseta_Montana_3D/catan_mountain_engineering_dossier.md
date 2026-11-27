# Dossier de Ingeniería CAD — Loseta de Montaña Catan 3D (Cordillera V3 PBR)

## 1. Especificaciones Geométricas y Tolerancias Mecánicas
- **Hexágono Regular Estándar Catan 3D:**
  - Radio circunscrito ($R$ / Lado $L$): **64.0000 mm**
  - Apotema ($h = R\cos 30^\circ$): **55.4256 mm** (6 triángulos equiláteros de `64 × 64 × 64 mm`)
  - Ancho cara a cara ($W = 2 \cdot \text{Apotema}$): **110.8513 mm**
  - Vértice a vértice ($D = 2R$): **128.0000 mm**
  - Área de base: **10,641.72 mm²** (`106.42 cm²`)
- **Zócalo Sólido y Alzado:**
  - Zócalo perimetral completamente sólido (`Z = 0.0000` a `25.0000 mm`), **cero vaciados internos**, sin necesidad de soportes FDM.
  - Cota máxima de cumbre: **83.0000 mm** (`58.0000 mm` de relieve alpino sobre la plataforma `Z = 25.00 mm`).
  - **Carril de carreteras:** Margen perimetral estrictamente plano de **3.50 mm** en los 6 bordes a `Z = 25.0000 mm`.
  - **Plataformas de poblados/ciudades (*Ciudades y Caballeros*):** 6 bases circulares estrictamente planas de **Ø 20.00 mm** (`r = 10.00 mm`) centradas en cada vértice a `Z = 25.0000 mm`.
  - **Alojamiento magnético inferior:** Cavidad cilíndrica abierta inferior de **Ø 10.20 mm × 5.20 mm** centrada en `(0, 0, 0..5.20)` para imán de neodimio N52 `Ø10×5 mm`.
  - **Superficie alpina pura:** Sin rebaje ni hueco para ficha numérica.

## 2. Auditoría Topológica de Malla y LODs (Pure NumPy + SciPy Delaunay)
- **Malla Maestra (`catan_mountain_solid.stl`, `.ply`, `.obj`, `.3mf`):**
  - Vértices (`V`): **2,590,913** | Aristas (`E`): **7,772,733** | Triángulos (`F`): **5,181,822** (>5.0M requeridos)
  - Característica de Euler: **$\chi = V - E + F = 2$** (100% 2-manifold estanco `watertight=True`, cero aristas sueltas o degeneradas)
  - Área mínima de triángulo: **0.001963 mm²** (>0 estricto)
  - Volumen sólido calculado: **438.72 cm³**
- **LODs Estancos Verificados (`watertight=True`, $\chi=2$):**
  - `catan_mountain_lod1_1m.stl`: **1,021,686** triángulos (`438.73 cm³`)
  - `catan_mountain_lod2_250k.stl`: **256,934** triángulos (`438.74 cm³`)
  - `catan_mountain_lod3_50k.stl`: **59,870** triángulos (`438.73 cm³`)
- **Visor WebGL Táctil AAA PBR (`catan_mountain_viewer.html`):**
  - Malla interactiva fluida: **113,117** vértices / **226,230** triángulos con normales planas arquitectónicas separadas en el zócalo hexagonal, texturas 2K **Diffuse + Normal + Roughness**, iluminación HDR IBL (`THREE.PMREMGenerator`), sombras direccionales suaves `PCFSoftShadowMap` (`2048×2048`) y 3 atmósferas (`Sol Alpino`, `Atardecer`, `Nublado`) en **4.56 MB**.

## 3. Geomorfología Escultórica V3 y Texturizado Hiperrealista Pintado
- **Cordillera Estilizada V3 (`prototype_v3_cordillera.py`):** 5 agujas alpinas esbeltas en curva S (cumbre principal en `(-3.5, +4.5) mm` a `Z = 83.0000 mm`, sub-cumbres a `47.0 / 41.5 / 30.5 / 29.5 mm` de relieve), aristas sinuosas con cornisas, 4 circos glaciares esculpidos y cuencas glaciares colgantes sobre valles protegidos sin escalones perimetrales.
- **Pipeline Pictórico Continuo e Hidrológico (`D-inf` + LIC):** Escorrentía hidráulica `flow_accumulation`, estriaciones minerales advecidas por pendiente (`slope_advect_field`), 16 direcciones HBAO y muestreo canónico FFT bit-idéntico entre todos los LODs y texturas UV.
- **Texturas PBR 4K Alineadas (`Y-up` estándar WebGL / OBJ / 3MF):**
  - `catan_mountain_diffuse_4k.png` / `2k.png`, `catan_mountain_normal_4k.png`, **`catan_mountain_roughness_4k.png`** + variantes estacionales `winter_4k` y `autumn_4k`.
  - Verificación round-trip UV automática (1,000 vértices aleatorios): coincidencia binaria exacta contra PNG en disco.
- **Cobertura Física:**
  - **Nieve / Escarcha Glacial:** **6.59%** | **Pradera / Musgo de Valle:** **2.98%** | **Roca Granítica / Canchal:** **90.43%**
  - **Paredes escarpadas ($\ge 44.0^\circ$):** **61.81%** del relieve con **0.000000** nieve en pendientes $\ge 44^\circ$.
