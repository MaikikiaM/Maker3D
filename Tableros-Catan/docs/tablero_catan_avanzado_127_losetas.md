# Especificación Técnica: Catán Épico 3D (127 Losetas) en Marco Cuadrado 146 × 146 cm

> [!NOTE]
> Configuración de **Gran Formato** para 6 jugadores simultáneos. Se inscribe un **hexágono regular de 127 losetas (13 filas)** dentro de un **marco cuadrado exterior de 146 × 146 cm (1460 mm)**. Cada loseta cuenta con un radio $R = 64,0\text{ mm}$ (ancho cara a cara $W = 110,85\text{ mm}$, escala +40,4% lineal y +97% en área respecto al Catán estándar).

---

## 1. Geometría del Marco Cuadrado de 146 × 146 cm y Hexágono Inscrito

| Parámetro Geométrico | Cota Milimétrica | Equivalencia en cm | Función en Mesa / CAD |
| :--- | :---: | :---: | :--- |
| **Lado del Marco Cuadrado Exterior** | **1460,0 mm** | **146,0 cm** | Límite exterior total de la mesa de juego |
| **Anchura del Hexágono Inscrito (X)** | **1441,1 mm** | **144,1 cm** | 13 losetas de ancho en la fila ecuatorial |
| **Fondo del Hexágono Inscrito (Y)** | **1280,0 mm** | **128,0 cm** | 13 filas de losetas de vértice a vértice |
| **Margen lateral (Izquierda / Derecha)** | **9,5 mm** | **~0,95 cm** | Borde de contención perimetral del marco |
| **Margen frontal (Arriba / Abajo)** | **90,0 mm** | **9,0 cm** | Espacio para marcadores de puntos y reglas |
| **4 Esquinas Triangulares Libres** | **~420 × 420 mm** | **~42 × 42 cm** | **Consolas de Jugador (Player Hubs) integradas** |

---

## 2. Dimensiones de la Loseta Individual (Escala +40,4% / Área +97%)

* **Radio circunscrito / Lado de arista ($R = L$):** **$64,00\text{ mm}$** (Catán normal: 45,6 mm).
* **Distancia entre caras opuestas ($W$):** **$110,85\text{ mm}$** (Catán normal: 79,0 mm).
* **Diámetro vértice a vértice ($D$):** **$128,00\text{ mm}$** (Catán normal: 91,2 mm).
* **Apotema ($a$):** **$55,43\text{ mm}$** ($W/2$).
* **Superficie unitaria:** **$106,42\text{ cm}^2$** (prácticamente el **DOBLE** de una loseta de Catán estándar de 54 cm²).
* **Zócalo base común (Grosor Mínimo):** **$25,00\text{ mm}$ ($2,5\text{ cm}$)** de altura uniforme en todas las losetas.
* **Cota Z máxima de relieve (Montañas):** **$83,00\text{ mm}$** ($25\text{ mm base} + 58\text{ mm relieve}$).
* **Chaflán plano en vértices:** **$\varnothing 20,00\text{ mm}$ a cota $Z=25,00\text{ mm}$** (asiento para murallas, ciudades y metrópolis triples sin colisión).
* **Ranura machihembrada para caminos/barcos:** **$3,00\text{ mm (ancho)} \times 1,20\text{ mm (profundidad)}$** en las 6 aristas superiores.

---

## 3. Balance de Piezas del Tablero (127 Losetas)

Se conserva intacto el balance económico de 39 tierras y se expande el océano a 88 losetas de agua marina:

### A. Tierra Firme (39 Losetas Productivas y Estratégicas)
* 🌲 **Madera / Bosque (7):** 5 en Continente + 1 en Isla de Oro + 1 en Atolón Norte *(Z total: 63 mm)*.
* 🧱 **Arcilla / Colinas (6):** 5 en Continente + 1 en Isla de Oro *(Z total: 53 mm)*.
* 🐑 **Lana / Pastos (7):** 5 en Continente + 1 en Isla de Oro + 1 en Atolón Sur *(Z total: 38 mm)*.
* 🌾 **Trigo / Cultivos (7):** 5 en Continente + 1 en Isla de Oro + 1 en Atolón Norte *(Z total: 35 mm)*.
* ⛰️ **Mineral / Montañas (6):** 5 en Continente + 1 en Isla de Oro *(Z total: 83 mm)*.
* ✨ **Río de Oro (3):** 2 en Isla de Oro + 1 en Atolón Norte *(Z total: 48 mm)*.
* 🏜️ **Desiertos / Yermos (3):** 1 en Isla de Oro (Cráter) + 2 en Atolón Sur (Guarida Pirata) *(Z total: 31 mm)*.

### B. Masa Oceánica (88 Losetas de Agua)
* **Anillo Oceánico Perimetral (36 losetas):** Corona exterior continua que rodea todo el tablero a cota plana $Z = 25,00\text{ mm}$.
* **Aguas Interiores y Gran Estrecho (52 losetas):** Canales navegables a cota plana $Z = 25,00\text{ mm}$.

---

## 4. Presupuesto y Cálculo de Imanes de 10 mm (BOM para 127 Losetas)

* **Total de losetas:** 127 hexágonos.
* **Caras por hexágono:** 6 caras.
* **Total de imanes necesarios:** $127 \times 6 = \mathbf{762\text{ imanes}}$.
* **Margen de rotura/extravío (+15%):** **Comprar 875 a 900 imanes**.
* **Tipo de imán recomendado:** **Neodimio N52 disco $\varnothing 10 \times 3\text{ mm}$** (o $\varnothing 10 \times 2\text{ mm}$).
  * *Área polar activa:* $78,54\text{ mm}^2$ (4 veces más área de contacto que un imán de 5 mm).
  * *Fuerza de atracción por cara:* $\approx 2,10\text{ kg}$ (contacto metal-metal directo).
  * *Estabilidad angular:* Con $10\text{ mm}$ de diámetro en la pared de $25\text{ mm}$, anula por completo el cabeceo o pivoteo en las aristas de $64\text{ mm}$.
* **Bolsillos en CAD:** Cilíndricos de $\varnothing 10,20\text{ mm} \times 3,10\text{ mm}$ de profundidad, centrados a $Z = 12,50\text{ mm}$ de la base (la mitad exacta del zócalo de $25\text{ mm}$).
  * Quedan **$7,50\text{ mm}$ de pared sólida por arriba** y **$7,50\text{ mm}$ por debajo**, garantizando una robustez estructural masiva.
* **Polaridad:** Alternada N-S-N-S-N-S mediante plantilla maestra (Jig) para acoplamiento universal en cualquier rotación.

---

## 5. Distribución de las 13 Filas del Hexágono Regular (127 Losetas)

El hexágono regular centrado ($n=7$) se compone de 13 filas horizontales escalonadas:

```text
Fila 00 (07 losetas):               [ ~ ] [ ~ ] [ ~ ] [ ~ ] [ ~ ] [ ~ ] [ ~ ]
Fila 01 (08 losetas):             [ ~ ] [ ~ ] [ ~ ] [ ~ ] [ ~ ] [ ~ ] [ ~ ] [ ~ ]
Fila 02 (09 losetas):           [ ~ ] [ ~ ] [ ~ ] [ ~ ] [ ~ ] [Oro] [ ~ ] [ ~ ] [ ~ ]
Fila 03 (10 losetas):         [ ~ ] [Mad] [Tri] [Lan] [ ~ ] [ ~ ] [Mad] [Tri] [ ~ ] [ ~ ]
Fila 04 (11 losetas):       [ ~ ] [Arc] [Min] [Mad] [Tri] [ ~ ] [ ~ ] [ ~ ] [ ~ ] [ ~ ] [ ~ ]
Fila 05 (12 losetas):     [ ~ ] [Lan] [Tri] [Arc] [Min] [ ~ ] [ ~ ] [ ~ ] [Lan] [Oro] [ ~ ] [ ~ ]
Fila 06 (13 losetas):   [ ~ ] [Mad] [Arc] [Lan] [Min] [Tri] [ ~ ] [ ~ ] [Mad] [Des] [Min] [ ~ ] [ ~ ]  <- ECUADOR
Fila 07 (12 losetas):     [ ~ ] [Lan] [Min] [Mad] [Arc] [ ~ ] [ ~ ] [ ~ ] [Tri] [Oro] [Arc] [ ~ ]
Fila 08 (11 losetas):       [ ~ ] [Tri] [Arc] [Min] [ ~ ] [ ~ ] [ ~ ] [ ~ ] [ ~ ] [ ~ ] [ ~ ]
Fila 09 (10 losetas):         [ ~ ] [Mad] [Lan] [ ~ ] [ ~ ] [ ~ ] [Des] [Lan] [ ~ ] [ ~ ]
Fila 10 (09 losetas):           [ ~ ] [ ~ ] [ ~ ] [ ~ ] [ ~ ] [Des] [ ~ ] [ ~ ] [ ~ ]
Fila 11 (08 losetas):             [ ~ ] [ ~ ] [ ~ ] [ ~ ] [ ~ ] [ ~ ] [ ~ ] [ ~ ]
Fila 12 (07 losetas):               [ ~ ] [ ~ ] [ ~ ] [ ~ ] [ ~ ] [ ~ ] [ ~ ]
```

---

## 6. Inventario Maestro Celda por Celda (127 Losetas)

| ID (r, c) | Zona | Terreno / Bioma | Ficha Numérica | Pips | Cota Z | Puerto Asociado |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| `(0, 0)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(0, 1)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(0, 2)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(0, 3)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(0, 4)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(0, 5)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(0, 6)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(1, 0)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(1, 1)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(1, 2)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(1, 3)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(1, 4)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(1, 5)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(1, 6)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(1, 7)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(2, 0)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(2, 1)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(2, 2)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(2, 3)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(2, 4)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(2, 5)` | Atolón Norte | Río de Oro | **6** | ••••• | 48 mm | - |
| `(2, 6)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(2, 7)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(2, 8)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(3, 0)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(3, 1)` | Continente | Bosque (Madera) | **8** | ••••• | 63 mm | ⚓ **Puerto Genérico 3:1** (Norte) |
| `(3, 2)` | Continente | Cultivos (Trigo) | 10 | ••• | 35 mm | - |
| `(3, 3)` | Continente | Pastos (Lana) | **6** | ••••• | 38 mm | ⚓ **Puerto Lana 2:1** (Noreste) |
| `(3, 4)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(3, 5)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(3, 6)` | Atolón Norte | Bosque (Madera) | 5 | •••• | 63 mm | ⚓ **Puerto Madera 2:1** (Norte) |
| `(3, 7)` | Atolón Norte | Cultivos (Trigo) | 9 | •••• | 35 mm | ⚓ **Puerto Trigo 2:1** (Noreste) |
| `(3, 8)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(3, 9)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(4, 0)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(4, 1)` | Continente | Colinas (Arcilla) | 12 | • | 53 mm | ⚓ **Puerto Arcilla 2:1** (Noroeste) |
| `(4, 2)` | Continente | Montañas (Mineral) | 4 | ••• | 83 mm | - |
| `(4, 3)` | Continente | Bosque (Madera) | 3 | •• | 63 mm | - |
| `(4, 4)` | Continente | Cultivos (Trigo) | 10 | ••• | 35 mm | ⚓ **Puerto Trigo 2:1** (Este) |
| `(4, 5)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(4, 6)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(4, 7)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(4, 8)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(4, 9)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(4, 10)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(5, 0)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(5, 1)` | Continente | Pastos (Lana) | **6** | ••••• | 38 mm | ⚓ **Puerto Genérico 3:1** (Oeste) |
| `(5, 2)` | Continente | Cultivos (Trigo) | 4 | ••• | 35 mm | - |
| `(5, 3)` | Continente | Colinas (Arcilla) | 5 | •••• | 53 mm | - |
| `(5, 4)` | Continente | Montañas (Mineral) | **8** | ••••• | 83 mm | ⚓ **Puerto Mineral 2:1** (Este) |
| `(5, 5)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(5, 6)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(5, 7)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(5, 8)` | Isla de Oro | Pastos (Lana) | 10 | ••• | 38 mm | ⚓ **Puerto Lana 2:1** (Noroeste) |
| `(5, 9)` | Isla de Oro | Río de Oro | **8** | ••••• | 48 mm | - |
| `(5, 10)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(5, 11)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(6, 0)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(6, 1)` | Continente | Bosque (Madera) | 12 | • | 63 mm | ⚓ **Puerto Madera 2:1** (Oeste) |
| `(6, 2)` | Continente | Colinas (Arcilla) | 11 | •• | 53 mm | - |
| `(6, 3)` | Continente | Pastos (Lana) | 2 | • | 38 mm | - |
| `(6, 4)` | Continente | Montañas (Mineral) | 5 | •••• | 83 mm | - |
| `(6, 5)` | Continente | Cultivos (Trigo) | 9 | •••• | 35 mm | ⚓ **Puerto Genérico 3:1** (Este) |
| `(6, 6)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(6, 7)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(6, 8)` | Isla de Oro | Bosque (Madera) | 4 | ••• | 63 mm | - |
| `(6, 9)` | Isla de Oro | Desierto / Guarida | - | - | 31 mm | - |
| `(6, 10)` | Isla de Oro | Montañas (Mineral) | 5 | •••• | 83 mm | ⚓ **Puerto Mineral 2:1** (Este) |
| `(6, 11)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(6, 12)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(7, 0)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(7, 1)` | Continente | Pastos (Lana) | 11 | •• | 38 mm | ⚓ **Puerto Genérico 3:1** (Suroeste) |
| `(7, 2)` | Continente | Montañas (Mineral) | 3 | •• | 83 mm | - |
| `(7, 3)` | Continente | Bosque (Madera) | 9 | •••• | 63 mm | - |
| `(7, 4)` | Continente | Colinas (Arcilla) | **6** | ••••• | 53 mm | - |
| `(7, 5)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(7, 6)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(7, 7)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(7, 8)` | Isla de Oro | Cultivos (Trigo) | 9 | •••• | 35 mm | - |
| `(7, 9)` | Isla de Oro | Río de Oro | 3 | •• | 48 mm | - |
| `(7, 10)` | Isla de Oro | Colinas (Arcilla) | 11 | •• | 53 mm | ⚓ **Puerto Arcilla 2:1** (Sureste) |
| `(7, 11)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(8, 0)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(8, 1)` | Continente | Cultivos (Trigo) | 10 | ••• | 35 mm | - |
| `(8, 2)` | Continente | Colinas (Arcilla) | 3 | •• | 53 mm | - |
| `(8, 3)` | Continente | Montañas (Mineral) | 4 | ••• | 83 mm | ⚓ **Puerto Genérico 3:1** (Sureste) |
| `(8, 4)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(8, 5)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(8, 6)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(8, 7)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(8, 8)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(8, 9)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(8, 10)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(9, 0)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(9, 1)` | Continente | Bosque (Madera) | 11 | •• | 63 mm | - |
| `(9, 2)` | Continente | Pastos (Lana) | **8** | ••••• | 38 mm | ⚓ **Puerto Genérico 3:1** (Sur) |
| `(9, 3)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(9, 4)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(9, 5)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(9, 6)` | Atolón Sur | Desierto / Guarida | - | - | 31 mm | - |
| `(9, 7)` | Atolón Sur | Pastos (Lana) | 2 | • | 38 mm | - |
| `(9, 8)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(9, 9)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(10, 0)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(10, 1)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(10, 2)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(10, 3)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(10, 4)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(10, 5)` | Atolón Sur | Desierto / Guarida | - | - | 31 mm | - |
| `(10, 6)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(10, 7)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(10, 8)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(11, 0)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(11, 1)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(11, 2)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(11, 3)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(11, 4)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(11, 5)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(11, 6)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(11, 7)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(12, 0)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(12, 1)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(12, 2)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(12, 3)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(12, 4)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(12, 5)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
| `(12, 6)` | Océano Abierto | Océano / Mar | - | - | 25 mm | - |
---

## 7. Análisis CAD del Estilo Esculpido (Deconstrucción de las Fotos de Referencia)

Las imágenes de referencia proporcionadas definen con absoluta claridad el lenguaje formal del proyecto: **las losetas no son prismas extruidos macizos, sino dioramas topográficos orgánicos esculpidos sobre un zócalo común funcional**:

```text
======================= ANATOMÍA CAD DE UNA LOSETA ESCULPIDA =======================
                  ▲ Pico de Montaña (Z = 45 mm total)
                 / \
                /   \
               /     \    [Repisa Ficha Numérica Ø26mm]
              /       \___|---|___
             /                     \
[Chaflán Vértice Ø20mm]             \               [Chaflán Vértice Ø20mm]
    +-------------+                 +-------------------+  <- Cota Z = 12.0 mm (Zócalo)
    |  (IMÁN N52) |   ZÓCALO COMÚN  |     (IMÁN N52)    |  <- Centrado a Z = 6.0 mm
    +-------------+-----------------+-------------------+  <- Cota Z = 0.0 mm (Base Mesa)
```

### 7.1. Tratamiento Volumétrico y Alturas por Bioma

| Bioma | Altura de Relieve | Cota Z Total | Características Esculpidas Específicas |
| :--- | :---: | :---: | :--- |
| 🌾 **Trigo / Cultivos** | **+3 a 6 mm** | **15 a 18 mm** | Terreno bajo y transitable. Dividido en bancales o parcelas agrícolas geométricas con textura de surcos arados, pequeños pajares y granjas en miniatura. |
| 🐑 **Pastos / Lana** | **+6 a 10 mm** | **18 a 22 mm** | Colinas verdes onduladas y suaves. Incorpora un cercado de madera para el ganado, ovejas blancas en miniatura y un pequeño meandro de agua/riachuelo. |
| 🏜️ **Desierto / Yermo** | **+2 a 4 mm** | **14 a 16 mm** | Prácticamente plano. Esculpido con delicadas dunas y ondas de viento en la arena, detalles de fósiles/rocas y un pedestal plano central para el Ladrón. |
| 🧱 **Arcilla / Colinas** | **+10 a 16 mm** | **22 a 28 mm** | Cantera a cielo abierto escalonada. Mesetas aterrazadas en gradas concéntricas (tipo cañón de Colorado) con curvas de nivel geológicas en arcilla roja. |
| 🌲 **Bosque / Madera** | **+14 a 18 mm** | **26 a 30 mm** | Arboleda densa. Copas esféricas de árboles redondeados dispuestas en forma de herradura o anillo perimetral, dejando un claro central para el número. |
| ⛰️ **Montaña / Mineral** | **+25 a 35 mm** | **37 a 47 mm** | **Pico central escarpado**: gran masa cónica/piramidal que se eleva en el medio de la casilla con crestas angulares y nieve en la cima (o volcán con cráter). A media ladera incorpora una meseta cilíndrica plana para la ficha. |
| 🌊 **Agua / Océano** | **+2 a 3 mm** | **14 a 15 mm** | Superficie marina plana con textura de oleaje suave y micro-crestas de espuma. |

### 7.2. Los 3 Elementos Críticos de Interfaz CAD y Juego

1. **Cavidad Circular para la Ficha Numérica (Number Chit Pocket):**
   * En lugar de colocar la ficha suelta sobre el terreno (lo que provocaría que resbale al tirar los dados), cada loseta incorpora una **cavidad circular plana de $\varnothing 26,0\text{ mm} \times 1,8\text{ mm}$ de profundidad**.
   * La ficha de madera/acrílico encaja a ras de la superficie de asiento, quedando fija e inamovible.
2. **Marco Perimetral Plano Libre (2,5 mm):**
   * A lo largo de las 6 aristas de 64 mm, la loseta mantiene un rebaje horizontal de $2,5\text{ mm}$ de ancho a cota base ($Z = 12\text{ mm}$) con la **ranura machihembrada de $3,0 \times 1,2\text{ mm}$**.
   * Esto garantiza que los caminos y barcos apoyan con total estabilidad sin colisionar con las copas de los árboles ni con las rocas de las montañas.
3. **Chaflanes Circulares en Vértices ($\varnothing 20\text{ mm}$):**
   * En cada una de las 6 esquinas, el relieve topográfico se retranquea para dejar un plato circular horizontal de **$20\text{ mm}$ de diámetro** a cota $Z=12\text{ mm}$.
   * Sobre este plato plano asientan sin interferencias la **Muralla base**, la **Ciudad**, la **Metrópoli encajada** y la peana giratoria del **Caballero**.
