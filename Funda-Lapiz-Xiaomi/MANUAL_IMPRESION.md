# 📐 Estuche Definitivo V8.2 para Xiaomi Smart Pen 2
### Sistema de Suspensión Elástica por Patitas en T Rectas, Slide-to-Release Real, Uñero hasta el Suelo e Imanes Hexagonales

Diseño optimizado de alta precisión en **3 piezas independientes (PLA/PETG + TPU 85A)** con chasis ensanchado anti-constricción (**$230.0 \times 39.2 \times 18.6\text{ mm}$**):

---

## 🚀 Innovaciones Clave de la Versión V8.2

1. **Suspensión Elástica en T Recta (4 Patitas):**
   - El inserto de TPU ya no queda encajonado ni comprimido lateralmente por las paredes de PETG.
   - **Posición Equidistante ($X = \pm 51.00\text{ mm}$):** Amplia libertad de apertura en la zona central para coger y soltar el lápiz con suavidad.
   - Sobresalen $h/2 = 7.80\text{ mm}$ creando un colchón de aire lateral continuo.
   - **Barra transversal de apoyo 100% recta ($26.40\text{ mm}$ de longitud, $2.40\text{ mm}$ de grosor):** Sin rampitas ni aletas separadas de la pared; toda la cara exterior de la T apoya firmemente contra el PETG.
   - **Nervio de unión fino ($2.40\text{ mm}$):** Mantiene exactamente el mismo grosor en el vástago que une las paredes paralelas del TPU para no trasladar fuerzas excesivas al lápiz.
   - Altura continua de $12.50\text{ mm}$ desde $Z=0$ apoyada plana en la cama (**0% soportes necesarios**).
2. **Simetría Geométrica Absoluta respecto al Centro de la Holgura ($X = 0.00\text{ mm}$):**
   - El semicírculo ergonómico de agarre se sitúa en el centro matemático ($X = 0.00\text{ mm}$).
   - Distancia hacia el frontal (punta) idéntica a la trasera (culata): **$102.00\text{ mm}$ a cada lado** (longitud total TPU = $204.00\text{ mm}$).
3. **Mecanismo *Slide-to-Release* Real (Embocadura hasta $X = -92.0\text{ mm}$):**
   - La embocadura abierta llega de $X = +71.0\text{ mm}$ a $X = -92.0\text{ mm}$ ($163.0\text{ mm}$ abiertos) y la transición cónica conecta suavemente con la cámara de la punta ($-95.0\text{ mm}$, domo a $-98.0\text{ mm}$).
   - Al deslizar el lápiz hacia atrás contra $+71.0\text{ mm}$, la punta sale de la capucha superior ($-92.0$ a $-98.0\text{ mm}$) para extraerlo hacia arriba sin obstáculos.
4. **Suelo de TPU Ligero ($2.00\text{ mm}$):**
   - Eliminado el exceso macizo: suelo elástico de **$2.00\text{ mm}$** (altura TPU = $12.50\text{ mm}$, altura inserto rígido = $12.30\text{ mm}$, altura cerrada del estuche = $18.60\text{ mm}$).
5. **Uñero del TPU hasta el Suelo con Curvatura Conservada ($R = 32.0\text{ mm}$):**
   - La muesca central del TPU baja hasta el suelo del canal ($Z = 2.00\text{ mm}$), dejando mucho más espacio para los dedos alrededor del lápiz mientras conserva una curvatura suave gracias a su circunferencia ampliada ($R = 32.0\text{ mm}$).
6. **Relleno Reducido a 20% en TPU y Envolvencia Suavizada ($7.20\text{ mm}$):**
   - Mayor suavidad de absorción de impactos y apertura sedosa de las paredes.
7. **Cavidades de Repuesto 15% Más Amplias:**
   - Alojamiento de las 2 puntas de repuesto aumentado un 15% en todos los diámetros y longitudes para extracción instantánea con la yema del dedo.
8. **Chasis Rígido de PETG Blindado y Cavidades Hexagonales para Imanes:**
   - Cajera ensanchada a $31.40\text{ mm}$ y labio continuo de $1.70\text{ mm}$ en los 4 lados con encaje *snap-fit* hueco funcional y precarga suave sobre TPU ($0.15\text{ mm}$).
   - **8 cavidades hexagonales ($6.20\text{ mm}$ entre caras planas)** para imanes cilíndricos D6x3 mm en las 4 esquinas ($X = \pm 109.50\text{ mm}, Y = \pm 14.40\text{ mm}$): los 6 vértices dejan escapar el aire atrapado y el exceso de pegamento sin presionar las paredes ($\ge 1.52\text{ mm}$ interiores y $\ge 1.92\text{ mm}$ al exterior).
   - Uñera ergonómica centrada en $X = 0.00\text{ mm}$ y teléfono `644 727 688` grabado en la tapa.

---

## ⏱️ Guía de Pausas en el Laminador (*Pause at Height* para Capa de 0.20 mm)

> [!IMPORTANT]
> Configura las pausas por **altura $Z$** (o al inicio de la capa indicada si tu laminador numera desde la capa 1 a $0.20\text{ mm}$):

### A. Para la Base ([`estuche_xiaomi_pen2_base_PLA.stl`](file:///c:/Users/migue/Antigravity/Maker3D/Funda-Lapiz-Xiaomi/Modelos-3D/estuche_xiaomi_pen2_base_PLA.stl))
1. **Orientación:** Suelo apoyado plano en la cama de impresión ($Z = 0$).
2. **Pausa en Slicer:** A **$Z = 9.20\text{ mm}$** (inicio de la **Capa 47** a $0.20\text{ mm}$).
   - Hasta $Z = 9.20\text{ mm}$ (Capa 46 completada), las 4 cavidades hexagonales quedan abiertas y listas para recibir los imanes.
   - Al reanudar en $Z = 9.20\text{ mm}$ (Capa 47–48), el cabezal teje la membrana continua de $0.40\text{ mm}$ sellando los imanes herméticamente.
3. **Acción:** Insertar los 4 imanes D6x3 mm (con una gota de adhesivo si se desea; el sobrante escapa por los vértices del hexágono) en las 4 esquinas ($X = \pm 109.5\text{ mm}, Y = \pm 14.4\text{ mm}$).
4. **Reanudar:** La impresora terminará el labio de encaje hasta $Z = 15.00\text{ mm}$.

### B. Para la Tapa ([`estuche_xiaomi_pen2_tapa_PLA.stl`](file:///c:/Users/migue/Antigravity/Maker3D/Funda-Lapiz-Xiaomi/Modelos-3D/estuche_xiaomi_pen2_tapa_PLA.stl) o lisa)
1. **Orientación:** Viene exportada ya **invertida** (techo plano con texto grabado apoyado en la cama a $Z = 0$).
2. **Pausa en Slicer:** A **$Z = 8.60\text{ mm}$** (inicio de la **Capa 44** a $0.20\text{ mm}$).
   - Hasta $Z' = 8.60\text{ mm}$ (Capa 43 completada), las 4 cavidades hexagonales quedan abiertas.
   - Al reanudar en $Z' = 8.60\text{ mm}$ (Capa 44–45), la boquilla sella los imanes bajo el puente plástico de $0.40\text{ mm}$.
3. **Acción:** Insertar los 4 imanes D6x3 mm comprobando la atracción diagonal con la base.
4. **Reanudar:** La impresora termina la costura a $9.00\text{ mm}$ y continúa subiendo para tejer la cerradura semicircular hasta $Z = 11.20\text{ mm}$ (sin ningún soporte).

---

## 🧭 Esquema Poka-Yoke de Polaridad Diagonal

```
VISTA SUPERIOR DE LA BASE (Mirando el interior abierto):
           Punta del Lápiz                     Culata y Repuestos
               (X = -109.5)                        (X = +109.5)
          +--------------------------------------------+
(Y = +14.4)|  [ NORTE ]                      [ SUR ]   |
          |                                            |
          |          Cajera Suspensión TPU             |
          |                                            |
(Y = -14.4)|  [  SUR  ]                      [ NORTE ]  |
          +--------------------------------------------+

VISTA CARA DE CONTACTO DE LA TAPA (Para atraer a la base):
          +--------------------------------------------+
(Y = +14.4)|  [  SUR  ]                      [ NORTE ]  |
          |                                            |
          |                                            |
(Y = -14.4)|  [ NORTE ]                      [  SUR  ]  |
          +--------------------------------------------+
```

* **A 0° (Orientación correcta):** Todos los polos opuestos se atraen $\rightarrow$ Cierre magnético nítido y firme.
* **A 180° (Al revés):** Todos los polos se repelen $\rightarrow$ La tapa es expulsada físicamente, impidiendo colocarla mal.

---

## 📂 Archivos STL V8.2 Listos para Imprimir

Ubicación principal: [`Modelos-3D/`](file:///c:/Users/migue/Antigravity/Maker3D/Funda-Lapiz-Xiaomi/Modelos-3D)

| Archivo STL | Filamento | Dimensiones ($X \times Y \times Z$) | Estanqueidad | Descripción |
| :--- | :--- | :--- | :--- | :--- |
| [`estuche_xiaomi_pen2_base_PLA.stl`](file:///c:/Users/migue/Antigravity/Maker3D/Funda-Lapiz-Xiaomi/Modelos-3D/estuche_xiaomi_pen2_base_PLA.stl) | PETG / PLA | **$230.0 \times 39.2 \times 15.0\text{ mm}$** | **Watertight: True** | Base $9.6\text{ mm}$, labio $5.4\text{ mm}$ ($1.7\text{ mm}$ en 4 lados), imanes hexagonales |
| [`estuche_xiaomi_pen2_tapa_PLA.stl`](file:///c:/Users/migue/Antigravity/Maker3D/Funda-Lapiz-Xiaomi/Modelos-3D/estuche_xiaomi_pen2_tapa_PLA.stl) | PETG / PLA | **$230.0 \times 39.2 \times 11.2\text{ mm}$** | **Watertight: True** | Tapa $9.0\text{ mm}$, click hueco, imanes hexagonales, teléfono grabado |
| [`estuche_xiaomi_pen2_tapa_lisa_PLA.stl`](file:///c:/Users/migue/Antigravity/Maker3D/Funda-Lapiz-Xiaomi/Modelos-3D/estuche_xiaomi_pen2_tapa_lisa_PLA.stl) | PETG / PLA | **$230.0 \times 39.2 \times 11.2\text{ mm}$** | **Watertight: True** | Versión de tapa completamente lisa sin grabado |
| [`estuche_xiaomi_pen2_inserto_TPU.stl`](file:///c:/Users/migue/Antigravity/Maker3D/Funda-Lapiz-Xiaomi/Modelos-3D/estuche_xiaomi_pen2_inserto_TPU.stl) | TPU 85A | **$204.0 \times 31.2 \times 12.5\text{ mm}$** | **Watertight: True** | **Versión V8.2:** patitas en T rectas, suelo $2.0\text{ mm}$, uñero al suelo $R=32\text{ mm}$, slide-to-release real |
| [`estuche_xiaomi_pen2_inserto_rigido_sin_envolturas.stl`](file:///c:/Users/migue/Antigravity/Maker3D/Funda-Lapiz-Xiaomi/Modelos-3D/estuche_xiaomi_pen2_inserto_rigido_sin_envolturas.stl) | PLA / PETG | **$204.0 \times 31.2 \times 12.3\text{ mm}$** | **Watertight: True** | Inserto alternativo rígido con cuna en U sin clips (holgura de cierre limpia) |
| [`estuche_xiaomi_pen2_ensamblado_cerrado.stl`](file:///c:/Users/migue/Antigravity/Maker3D/Funda-Lapiz-Xiaomi/Modelos-3D/estuche_xiaomi_pen2_ensamblado_cerrado.stl) | Ensamblaje | **$230.0 \times 39.2 \times 18.6\text{ mm}$** | **Watertight: True** | Modelo cerrado completo para verificación CAD |
| [`estuche_xiaomi_pen2_placa_impresion.stl`](file:///c:/Users/migue/Antigravity/Maker3D/Funda-Lapiz-Xiaomi/Modelos-3D/estuche_xiaomi_pen2_placa_impresion.stl) | Placa | **$230.0 \times 135.2 \times 15.0\text{ mm}$** | **Watertight: True** | Las 3 piezas separadas sin solapes (usar *Split to Objects* y laminar TPU aparte) |
| [`generar_estuche.py`](file:///c:/Users/migue/Antigravity/Maker3D/Funda-Lapiz-Xiaomi/generar_estuche.py) | Script | - | - | Generador paramétrico Manifold3D completo |

---

## 🖨️ Recomendaciones de Laminado para V8.2

* **Motor de Paredes:** Usa **Arachne** para máxima precisión en las patitas elásticas en T rectas, las cavidades hexagonales y el grabado del teléfono.
* **Ajustes para el TPU 85A:**
  - **Relleno:** $20\%$ (Giroide o Cúbico).
  - **Paredes:** 2 perímetros ($0.80\text{ mm}$ total).
  - **Capas superiores e inferiores:** 3 capas.
  - Esto proporciona la elasticidad lateral ideal para que las paredes abran con suavidad y las 4 patitas absorban cualquier caída.
* **Ajustes para el PETG/PLA (Base y Tapa):**
  - **Relleno:** $25\%$ a $30\%$.
  - **Paredes:** 3 a 4 perímetros ($1.20 - 1.60\text{ mm}$) para que el labio de encaje sea 100% macizo y robusto.
