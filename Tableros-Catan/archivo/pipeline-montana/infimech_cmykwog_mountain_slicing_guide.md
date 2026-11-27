# Guía Maestra de Laminado y Fabricación en Infimech MX Pro (8 Cabezales)
## Loseta de Montaña Épica Catán (CMYKWOG - B)

Esta guía detalla los parámetros de laminación (*slicing*) en **OrcaSlicer**, **Bambu Studio** o **PrusaSlicer** para fabricar la loseta de montaña con la arquitectura de 8 herramientas independientes de tu **Infimech MX Pro**.

---

### 1. Asignación de Herramientas y Parámetros Térmicos

| Herramienta | Filamento | Tonalidad | Diámetro Boquilla | Temp. Impresión | Temp. Espera (Dock) | Rol en la Pieza |
| :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| **T0** | PLA Estructural | Negro / Gris Antracita | **0,4 mm o 0,6 mm** | 215 °C | 150 °C | Zócalo base ($0-25\text{ mm}$) y relleno masivo interno |
| **T1** | PLA Estándar | Cian (Cyan) | **0,4 mm** | 210 °C | 150 °C | Matices de hielo glaciar y lagunas |
| **T2** | PLA Estándar | Magenta | **0,4 mm** | 210 °C | 150 °C | Subtonos minerales y sombras violáceas |
| **T3** | PLA Estándar | Gris Granito / Piedra | **0,4 mm** | 215 °C | 150 °C | Paredes de roca, aristas y cresterías |
| **T4** | PLA Estándar | Negro Puro (Key) | **0,4 mm** | 215 °C | 150 °C | Grietas profundas y oquedades de cantera |
| **T5** | PLA Estándar | Verde Esmeralda (Green) | **0,4 mm** | 210 °C | 150 °C | Musgo alpino y faldas de vegetación baja |
| **T6** | PLA Estándar | Naranja Vivo (Orange) | **0,4 mm** | 210 °C | 150 °C | Cantera de mineral de hierro, óxidos y arcilla |
| **T7** | PLA Estándar | Blanco Nieve Puro (White) | **0,2 mm** | 205 °C | 145 °C | Ventisqueros, picos nevados y seracs |

---

### 2. Altura de Capa Variable Adaptativa (*Adaptive Layer Height*)

* **Zona Zócalo Base ($Z = 0,00\text{ a }25,00\text{ mm}$):**
  * Altura de capa: **$0,24\text{ mm}$** (o $0,28\text{ mm}$ si T0 es boquilla de 0,6 mm).
  * 1 sola herramienta activa (**T0**).
  * **Cero cambios de herramienta** durante los primeros 25 mm.
* **Zona Transición y Cantera ($Z = 25,00\text{ a }50,00\text{ mm}$):**
  * Altura de capa: **$0,12\text{ mm}$**.
  * Entran en acción T3 (Roca), T6 (Hierro), T5 (Musgo).
* **Zona Alta Montaña y Nieve ($Z = 50,00\text{ a }83,00\text{ mm}$):**
  * Altura de capa: **$0,08\text{ mm}$**.
  * Se activa la herramienta **T7 (Boquilla de 0,2 mm)** para esculpir las aristas del Matterhorn y la nieve con precisión de joyería.

---

### 3. Configuración de Retracción y Docking en la Infimech

Para evitar cualquier hilo o goteo al desatracar:
1. **Retracción previa al aparcamiento (*Toolchange Retract*):**
   * Longitud: **$2,5\text{ mm}$**.
   * Velocidad de retracción: **$45\text{ mm/s}$**.
2. **Torre de Purga (*Prime Tower / Wipe Tower*):**
   * En un cambiador de herramientas independiente **no hace falta una torre de purga gigante**, solo un pequeño pilar de cebado (*Priming Volume* de **$15 - 20\text{ mm}^3$**) para estabilizar la presión antes de entrar a la pared exterior.
3. **Generador de Perímetros:**
   * Seleccionar **Arachne** para permitir anchos de línea dinámicos de $0,26\text{ a }0,50\text{ mm}$ en las crestas rocosas afiladas.

---

### 4. Tiempo Total Estimado de Impresión por Loseta

* **Tiempo zócalo base ($0-25\text{ mm}$ a 0,24 mm):** $\approx 1\text{ hora } 15\text{ min}$.
* **Tiempo relieve topográfico ($25-83\text{ mm}$ a 0,12 / 0,08 mm):** $\approx 6\text{ horas } 30\text{ min}$.
* **Tiempo de cambios mecánicos de cabezal (docking):** $\approx 1\text{ hora } 10\text{ min}$.
* **Tiempo total por loseta de montaña:** **$\approx 8\text{ horas } 55\text{ min}$** (frente a las 38 horas que llevaría con 0,2 mm monomaterial).
* **Consumo de filamento:** $\approx 145\text{ g}$ de PLA en total por loseta (con relleno giroide al 12% y 3 paredes).
