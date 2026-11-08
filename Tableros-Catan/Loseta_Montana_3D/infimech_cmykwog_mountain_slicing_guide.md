# Guía Técnica de Laminado y Producción Full-Color: Montaña 3D Catan

## 1. Resumen de la Suite de Archivos Maestros Full-Color

Se ha generado una suite completa de formatos estándar para garantizar compatibilidad universal con cualquier laminador (**OrcaSlicer**, **Bambu Studio**, **PrusaSlicer**, **Infimech Studio**, **Cura**, **Materialise Magics**) y software de renderizado:

| Archivo | Formato | Tipo de Color | Propósito Principal |
| :--- | :--- | :--- | :--- |
| [`catan_mountain_fullcolor.3mf`](file:///c:/Users/migue/Proyectos%20IA%20varios/Miscelanea/Tableros%20Catan/Loseta_Montana_3D/catan_mountain_fullcolor.3mf) | 3MF Oficial (ISO/IEC 23000-22) | Textura 2K Embebida (`<m:texture2d>`) | Laminadores modernos (OrcaSlicer / Bambu Studio / PrusaSlicer) con detección automática de textura a color. |
| [`catan_mountain_fullcolor.obj`](file:///c:/Users/migue/Proyectos%20IA%20varios/Miscelanea/Tableros%20Catan/Loseta_Montana_3D/catan_mountain_fullcolor.obj) | Wavefront OBJ + [`catan_mountain.mtl`](file:///c:/Users/migue/Proyectos%20IA%20varios/Miscelanea/Tableros%20Catan/Loseta_Montana_3D/catan_mountain.mtl) | Mapeado UV Planar + Textura Difusa 4K | Máxima fidelidad de textura ($4096 \times 4096$, $31.2\text{ \mu m}$/píxel) para renderizado y laminado fotográfico. |
| [`catan_mountain_diffuse_4k.png`](file:///c:/Users/migue/Proyectos%20IA%20varios/Miscelanea/Tableros%20Catan/Loseta_Montana_3D/catan_mountain_diffuse_4k.png) | Imagen PNG 4096&times;4096 (24-bit RGB) | Mapa Difuso Espectral Continuo | Mapa de color maestro fotorrealista continuo (roca, nieve, firn, líquenes). |
| [`catan_mountain_diffuse_2k.png`](file:///c:/Users/migue/Proyectos%20IA%20varios/Miscelanea/Tableros%20Catan/Loseta_Montana_3D/catan_mountain_diffuse_2k.png) | Imagen PNG 2048&times;2048 (24-bit RGB) | Textura Optimizada para Visor WebGL | Embebida en el visor 3D interactivo para carga ultrarrápida. |
| [`catan_mountain_fullcolor.ply`](file:///c:/Users/migue/Proyectos%20IA%20varios/Miscelanea/Tableros%20Catan/Loseta_Montana_3D/catan_mountain_fullcolor.ply) | Stanford PLY | Color por Vértice (24-bit RGB, 3,410 tonos únicos) | Laminadores que procesan color directamente por atributos de vértice (Gouraud shading). |
| [`catan_mountain_solid.stl`](file:///c:/Users/migue/Proyectos%20IA%20varios/Miscelanea/Tableros%20Catan/Loseta_Montana_3D/catan_mountain_solid.stl) | STL Estándar | Geometría Monolítica Pura | 100% Watertight Solid ($394.27\text{ cm}^3$, 138,620 caras), con cavidad para imán $\varnothing 10.2 \times 5.2\text{ mm}$. |
| [`catan_mountain_viewer.html`](file:///c:/Users/migue/Proyectos%20IA%20varios/Miscelanea/Tableros%20Catan/Loseta_Montana_3D/catan_mountain_viewer.html) | Aplicación WebGL Three.js | Textura 2K + Stencil Solid Capping | Inspección visual 3D interactiva en cualquier navegador (corte Z sólido y presets de zoom). |

---

## 2. Parámetros de Geometría y Tolerancias Mecánicas

- **Dimensiones Exteriores:**
  - Ancho de cara a cara (eje X): $2 \times \text{apotema} = 110.851\text{ mm}$.
  - Largo de punta a punta (eje Y): $2 \times R = 128.000\text{ mm}$.
  - Altura total (eje Z): $83.000\text{ mm}$ ($25.000\text{ mm}$ zócalo $+ 58.000\text{ mm}$ relieve alpino).
- **Zócalo y Carriles de Juego:**
  - Cota de la mesa de juego: $Z = 25.00\text{ mm}$ (perfectamente horizontal y calibrada con las losetas adyacentes del tablero de 127 hexágonos).
  - Carril perimetral de carreteras: $3.50\text{ mm}$ de anchura plana a lo largo de las 6 aristas hexagonales.
  - Asientos para poblados y ciudades: 6 plataformas circulares de $\varnothing 20.00\text{ mm}$ centradas en los vértices del hexágono a $Z = 25.00\text{ mm}$.
- **Alojamiento Inferior para Imán:**
  - Cavidad cilíndrica de $\varnothing 10.20\text{ mm} \times 5.20\text{ mm}$ centrada en la base $(X=0, Y=0)$.
  - Diseñada para imanes de neodimio N52 de $\varnothing 10\text{ mm} \times 5\text{ mm}$.
  - Tolerancia diametral: $+0.20\text{ mm}$ para inserción limpia sin holgura excesiva.

---

## 3. Instrucciones de Laminación en Bambu Studio / OrcaSlicer / Infimech Studio

### Método A: Importación de 3MF Texturizado Oficial (`catan_mountain_fullcolor.3mf`)
1. Abrir OrcaSlicer / Bambu Studio.
2. Arrastrar y soltar `catan_mountain_fullcolor.3mf`.
3. El laminador detectará el archivo como un modelo texturizado con coordenadas UV y material difuso.
4. En el panel de **Pintura de Color** (*Color Painting*), seleccionar la herramienta de conversión de textura a filamentos (*Texture to Filaments / Halftone Dithering*).
5. Asignar los filamentos cargados en la Infimech MX Pro o AMS:
   - **Filamento 1 (Cian):** Asignado a tonos de firn/hielo.
   - **Filamento 2 (Magenta):** Mezcla sustractiva para tonos cálidos y sombras basálticas.
   - **Filamento 3 (Amarillo):** Mezcla para líquenes dorados y pátina de granito.
   - **Filamento 4 (Negro / Key):** Zócalo base ($Z \le 25\text{ mm}$) y fisuras escarpadas.
   - **Filamento 5 (Blanco puro):** Cumbre Matterhorn, aristas y neveros.
   - **Filamento 6 (Verde / Tundra):** Acelerador de verde para saturación orgánica de líquenes sin requerir mezcla excesiva Cyan+Yellow.

### Método B: Importación de Wavefront OBJ + Textura 4K (`catan_mountain_fullcolor.obj`)
1. Asegurarse de que `catan_mountain_fullcolor.obj`, `catan_mountain.mtl` y `catan_mountain_diffuse_4k.png` estén en la misma carpeta.
2. Importar `catan_mountain_fullcolor.obj` en el laminador.
3. La textura 4K se proyecta automáticamente sobre los 138,620 triángulos con resolución de $31.25\text{ \mu m}$ por píxel.

---

## 4. Configuración de Calidad de Impresión FDM

- **Altura de Capa Adaptativa (*Adaptive Layer Height*):**
  - Base y zócalo ($Z = 0$ a $25\text{ mm}$): $0.20\text{ mm}$ para rapidez de deposición.
  - Macizo rocoso intermedio ($Z = 25$ a $65\text{ mm}$): $0.12\text{ mm}$.
  - Cumbre, aristas cimeras y neveros ($Z = 65$ a $83\text{ mm}$): **$0.08\text{ mm}$** para eliminar las líneas de capa en pendientes pronunciadas.
- **Paredes y Relleno:**
  - **Bucles de pared perimetral:** 4 bucles ($1.6\text{ mm}$ de grosor total).
  - **Capas superiores / inferiores:** 6 capas.
  - **Densidad de relleno:** 20% a 25% Gyroid (Giroide) para excelente rigidez omnidireccional y sensación sólida en mano.
- **Inserción del Imán de Neodimio:**
  - **Opción Encastre Posterior (Recomendada):** Imprimir la loseta normalmente; una vez despegada de la base, aplicar una gota de cianoacrilato en la cavidad inferior e insertar a presión el imán $\varnothing 10 \times 5\text{ mm}$.
  - **Opción Encapsulado Oculto:** Introducir una pausa programada (`M600` o `M0`) en la capa correspondiente a $Z = 5.2\text{ mm}$, insertar el imán en la cavidad y reanudar la impresión para que la siguiente capa de plástico cubra por completo el imán.
