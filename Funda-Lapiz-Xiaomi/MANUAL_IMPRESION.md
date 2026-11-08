# 📐 Estuche Definitivo V8 para Xiaomi Smart Pen 2
### Sistema de Suspensión Elástica por Patitas Flotantes, Slide-to-Release (+4.0 mm), Simetría Total y Suelo Reforzado

Diseño optimizado de alta precisión en **3 piezas independientes (PLA/PETG + TPU 85A)** con chasis ensanchado anti-constricción (**$223.7 \times 39.2 \times 20.6\text{ mm}$**):

---

## 🚀 Innovaciones Clave de la Versión V8

1. **Suspensión Elástica Flotante Equidistante (4 Patitas):**
   - El inserto de TPU ya no queda encajonado ni comprimido lateralmente por las paredes de PETG.
   - **Posición Equidistante ($X = \pm 51.00\text{ mm}$):** Las patitas se sitúan exactamente a la misma distancia de la holgura central ($51.0\text{ mm}$) que del final del calcetín ($51.0\text{ mm}$), ampliando el vano libre central a **$102.0\text{ mm}$** para que el calcetín se abra con máxima suavidad con los dedos.
   - Sobresalen $h/2 = 7.80\text{ mm}$ creando un colchón de aire lateral continuo.
   - **Pie con apoyo plano continuo de $10.40\text{ mm}$** ($4.0\text{ mm}$ de pata lisa a cada lado del nervio) eliminando el pico para un agarre firme contra la pared de PETG.
   - **Aletas curvas alargadas que se meten hacia dentro** ($1.50\text{ mm}$ de retroceso) en los extremos, alcanzando una longitud de pie de $26.40\text{ mm}$.
   - Altura continua de $14.50\text{ mm}$ desde $Z=0$ apoyada plana en la cama (**0% soportes necesarios**).
2. **Simetría Geométrica Absoluta respecto al Centro de la Holgura ($X = 0.00\text{ mm}$):**
   - El semicírculo ergonómico de agarre se sitúa en el centro matemático ($X = 0.00\text{ mm}$).
   - Distancia hacia el frontal (punta) idéntica a la trasera (culata): **$102.00\text{ mm}$ a cada lado** (longitud total TPU = $204.00\text{ mm}$).
   - **Parachoques frontal macizo de goma de $11.0\text{ mm}$** protegiendo la punta del lápiz ante caídas directas.
3. **Mecanismo *Slide-to-Release* (+4.0 mm de carrera):**
   - La punta queda resguardada bajo una capucha superior de goma de $8.0\text{ mm}$.
   - Con la carrera longitudinal libre de **$4.00\text{ mm}$**, basta empujar el lápiz hacia atrás con el pulgar para destapar la punta y levantar el lápiz limpiamente hacia arriba en vertical, sin doblar la punta ni forzar el TPU.
4. **Suelo Macizo Reforzado (+2.0 mm):**
   - El suelo de goma bajo el lápiz se duplica de $2.00\text{ mm}$ a **$4.00\text{ mm}$ macizos** (altura total del TPU = $14.50\text{ mm}$).
5. **Relleno Reducido a 20% en TPU:**
   - Mayor suavidad de absorción de impactos y apertura sedosa de las paredes.
6. **Envolvencia Suavizada un 10%:**
   - Garganta ensanchada a $7.20\text{ mm}$ para un tacto suave sin resistencia excesiva.
7. **Cavidades de Repuesto 15% Más Amplias:**
   - Alojamiento de las 2 puntas de repuesto aumentado un 15% en todos los diámetros y longitudes para extracción instantánea con la yema del dedo.
8. **Chasis Rígido de PETG Blindado y Estable:**
   - Cajera ensanchada a $31.40\text{ mm}$ para albergar la suspensión.
   - Cierre magnético por 8 imanes D6x3 mm encapsulados en las 4 esquinas ($X = \pm 106.00\text{ mm}, Y = \pm 16.20\text{ mm}$).
   - Uñera ergonómica centrada en $X = 0.00\text{ mm}$.
   - Grabado en bajorrelieve del teléfono `644 727 688` en la tapa.

---

## ⏱️ Guía de Pausas en el Laminador (*Pause at Height* para Capa de 0.20 mm)

> [!IMPORTANT]
> Configura las pausas exactas según la pieza que estés laminando:

### A. Para la Base ([`estuche_xiaomi_pen2_base_PLA.stl`](file:///c:/Users/migue/Proyectos%20IA%20varios/Miscelanea/Dise%C3%B1os%203D/02_Funda_Lapiz_Xiaomi/01_Modelos_3D/estuche_xiaomi_pen2_base_PLA.stl))
1. **Orientación:** Suelo apoyado plano en la cama de impresión ($Z = 0$).
2. **Pausa en Slicer:** A **$Z = 10.40\text{ mm}$** (**Capa 52** a 0.20 mm).
   - En la Capa 51 ($Z = 10.20\text{ mm}$), las 4 cavidades quedan abiertas y listas para recibir los imanes.
   - En la Capa 52 ($Z = 10.40\text{ mm}$), el cabezal teje la membrana continua de $0.40\text{ mm}$ sellando los imanes herméticamente.
3. **Acción:** Insertar los 4 imanes D6x3 mm en las 4 esquinas ($X = \pm 106.0\text{ mm}, Y = \pm 16.2\text{ mm}$).
4. **Reanudar:** La impresora terminará el labio de encaje hasta $Z = 16.00\text{ mm}$.

### B. Para la Tapa ([`estuche_xiaomi_pen2_tapa_PLA.stl`](file:///c:/Users/migue/Proyectos%20IA%20varios/Miscelanea/Dise%C3%B1os%203D/02_Funda_Lapiz_Xiaomi/01_Modelos_3D/estuche_xiaomi_pen2_tapa_PLA.stl) o lisa)
1. **Orientación:** Viene exportada ya **invertida** (techo plano con texto grabado apoyado en la cama a $Z = 0$).
2. **Pausa en Slicer:** A **$Z = 9.80\text{ mm}$** (**Capa 49** a 0.20 mm).
   - En la Capa 48 ($Z' = 9.60\text{ mm}$), las 4 cavidades quedan abiertas.
   - En la Capa 49 ($Z' = 9.80\text{ mm}$), la boquilla sella los imanes bajo el puente plástico de $0.40\text{ mm}$.
3. **Acción:** Insertar los 4 imanes D6x3 mm comprobando la atracción diagonal con la base.
4. **Reanudar:** La impresora termina la costura a $10.00\text{ mm}$ y continúa subiendo para tejer la cerradura semicircular hasta $Z = 13.20\text{ mm}$ (sin ningún soporte).

---

## 🧭 Esquema Poka-Yoke de Polaridad Diagonal

```
VISTA SUPERIOR DE LA BASE (Mirando el interior abierto):
           Punta del Lápiz                     Culata y Repuestos
               (X = -106.0)                        (X = +106.0)
          +--------------------------------------------+
(Y = +16.2)|  [ NORTE ]                      [ SUR ]   |
          |                                            |
          |          Cajera Suspensión TPU             |
          |                                            |
(Y = -16.2)|  [  SUR  ]                      [ NORTE ]  |
          +--------------------------------------------+

VISTA CARA DE CONTACTO DE LA TAPA (Para atraer a la base):
          +--------------------------------------------+
(Y = +16.2)|  [  SUR  ]                      [ NORTE ]  |
          |                                            |
          |                                            |
(Y = -16.2)|  [ NORTE ]                      [  SUR  ]  |
          +--------------------------------------------+
```

* **A 0° (Orientación correcta):** Todos los polos opuestos se atraen $\rightarrow$ Cierre magnético nítido y firme.
* **A 180° (Al revés):** Todos los polos se repelen $\rightarrow$ La tapa es expulsada físicamente, impidiendo colocarla mal.

---

## 📂 Archivos STL V8 Listos para Imprimir

Ubicación principal: [`01_Modelos_3D/`](file:///c:/Users/migue/Proyectos%20IA%20varios/Miscelanea/Dise%C3%B1os%203D/02_Funda_Lapiz_Xiaomi/01_Modelos_3D)

| Archivo STL | Filamento | Dimensiones ($X \times Y \times Z$) | Estanqueidad | Descripción |
| :--- | :--- | :--- | :--- | :--- |
| [`estuche_xiaomi_pen2_base_PLA.stl`](file:///c:/Users/migue/Proyectos%20IA%20varios/Miscelanea/Dise%C3%B1os%203D/02_Funda_Lapiz_Xiaomi/01_Modelos_3D/estuche_xiaomi_pen2_base_PLA.stl) | PETG / PLA | **$223.7 \times 39.2 \times 16.0\text{ mm}$** | **Watertight: True** | Base $10.6\text{ mm}$ con labio $5.4\text{ mm}$, cajera de $31.4\text{ mm}$ y uñera $X=0$ |
| [`estuche_xiaomi_pen2_tapa_PLA.stl`](file:///c:/Users/migue/Proyectos%20IA%20varios/Miscelanea/Dise%C3%B1os%203D/02_Funda_Lapiz_Xiaomi/01_Modelos_3D/estuche_xiaomi_pen2_tapa_PLA.stl) | PETG / PLA | **$223.7 \times 39.2 \times 13.2\text{ mm}$** | **Watertight: True** | Tapa $10.0\text{ mm}$, cerradura semicircular, teléfono grabado |
| [`estuche_xiaomi_pen2_tapa_lisa_PLA.stl`](file:///c:/Users/migue/Proyectos%20IA%20varios/Miscelanea/Dise%C3%B1os%203D/02_Funda_Lapiz_Xiaomi/01_Modelos_3D/estuche_xiaomi_pen2_tapa_lisa_PLA.stl) | PETG / PLA | **$223.7 \times 39.2 \times 13.2\text{ mm}$** | **Watertight: True** | Versión de tapa completamente lisa sin grabado |
| [`estuche_xiaomi_pen2_inserto_TPU.stl`](file:///c:/Users/migue/Proyectos%20IA%20varios/Miscelanea/Dise%C3%B1os%203D/02_Funda_Lapiz_Xiaomi/01_Modelos_3D/estuche_xiaomi_pen2_inserto_TPU.stl) | TPU 85A | **$204.0 \times 31.2 \times 14.5\text{ mm}$** | **Watertight: True** | **Versión V8:** 4 patitas elásticas, suelo $4.0\text{ mm}$, slide $+4.0\text{ mm}$, repuestos $+15\%$ |
| [`estuche_xiaomi_pen2_inserto_rigido_sin_envolturas.stl`](file:///c:/Users/migue/Proyectos%20IA%20varios/Miscelanea/Dise%C3%B1os%203D/02_Funda_Lapiz_Xiaomi/01_Modelos_3D/estuche_xiaomi_pen2_inserto_rigido_sin_envolturas.stl) | PLA / PETG | **$204.0 \times 31.2 \times 14.5\text{ mm}$** | **Watertight: True** | Inserto alternativo con cuna en U vertical sin clips |
| [`estuche_xiaomi_pen2_ensamblado_cerrado.stl`](file:///c:/Users/migue/Proyectos%20IA%20varios/Miscelanea/Dise%C3%B1os%203D/02_Funda_Lapiz_Xiaomi/01_Modelos_3D/estuche_xiaomi_pen2_ensamblado_cerrado.stl) | Ensamblaje | **$223.7 \times 39.2 \times 20.6\text{ mm}$** | **Watertight: True** | Modelo cerrado completo para verificación CAD |
| [`estuche_xiaomi_pen2_placa_impresion.stl`](file:///c:/Users/migue/Proyectos%20IA%20varios/Miscelanea/Dise%C3%B1os%203D/02_Funda_Lapiz_Xiaomi/01_Modelos_3D/estuche_xiaomi_pen2_placa_impresion.stl) | Placa | **$223.7 \times 102.2 \times 16.0\text{ mm}$** | **Watertight: True** | Placa de impresión con las 3 piezas orientadas |
| [`generar_estuche.py`](file:///c:/Users/migue/Proyectos%20IA%20varios/Miscelanea/Dise%C3%B1os%203D/02_Funda_Lapiz_Xiaomi/generar_estuche.py) | Script | - | - | Generador paramétrico Manifold3D completo |

---

## 🖨️ Recomendaciones de Laminado para V8

* **Motor de Paredes:** Usa **Arachne** para máxima precisión en las patitas elásticas curvas y en el grabado del teléfono.
* **Ajustes para el TPU 85A:**
  - **Relleno:** $20\%$ (Giroide o Cúbico).
  - **Paredes:** 2 perímetros ($0.80\text{ mm}$ total).
  - **Capas superiores e inferiores:** 3 capas.
  - Esto proporciona la elasticidad lateral ideal para que las paredes abran con suavidad y las 4 patitas absorban cualquier caída.
* **Ajustes para el PETG/PLA (Base y Tapa):**
  - **Relleno:** $25\%$ a $30\%$.
  - **Paredes:** 3 a 4 perímetros ($1.20 - 1.60\text{ mm}$) para que el labio de encaje sea 100% macizo y robusto.
