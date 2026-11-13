# Dossier de Ingeniería CAD — Loseta de Montaña Catan 3D (V2 Master)

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
  - Volumen sólido calculado: **433.55 cm³** (`435.50 cm³` en LOD1)
- **LODs Estancos Verificados (`watertight=True`, $\chi=2$):**
  - `catan_mountain_lod1_1m.stl`: **1,021,686** triángulos
  - `catan_mountain_lod2_250k.stl`: **256,934** triángulos
  - `catan_mountain_lod3_50k.stl`: **59,870** triángulos
- **Visor WebGL Táctil (`catan_mountain_viewer.html`):**
  - Malla interactiva fluida: **95,457** vértices / **190,910** triángulos + texturas Diffuse/Normal 2K (`3.40 MB`).

## 3. Geomorfología Escultórica V2 y Física de Texturizado
- **Geomorfología V2.22:** Aguja piramidal asimétrica tipo Matterhorn en `(-1.5, +2.5) mm` con aristas conectadas sin uniones radiales artificiales ni "efecto panettone"; 4 circos glaciares escarpados en la cara norte; estratos tectónicos inclinados y micro-relieve FFT multibanda calibrado en milímetros.
- **Texturas UV Alineadas (`Y-up` estándar WebGL / OBJ / 3MF):**
  - `catan_mountain_diffuse_4k.png` / `2k.png` + `catan_mountain_normal_4k.png` + variantes estacionales `winter_4k` y `autumn_4k`.
  - Verificación round-trip UV automática (1,000 vértices aleatorios): coincidencia binaria exacta contra PNG en disco.
- **Cobertura Física y Uso Completo de Paletas:**
  - **Nieve:** **10.66%** | **Tundra / Praderas:** **10.21%** | **Roca alpina / Granito:** **79.13%**
  - **Paredes escarpadas ($\ge 44.0^\circ$):** **56.59%** del relieve con **0.000000** nieve en pendientes $\ge 44^\circ$.
  - **Paleta 45 Grises (`G00..G44`):** **45 / 45 activos** (mínimo **84 px** en `G44`).
  - **Paleta 15 Verdes (`V00..V14`):** **15 / 15 activos** (mínimo **20,014 px** en `V13`), concentrados en valles bajos alrededor de los 6 bordes con franja amortiguadora de roca pura antes de la cota de nieve.
