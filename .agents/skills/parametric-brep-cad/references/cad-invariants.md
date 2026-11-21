# Referencia de Invariantes CAD y Tolerancias FDM (Maker3D 2.0)

Valores de diseño recomendados para boquilla `0.40 mm` (PLA / PETG) en Bambu Lab A1 / P1S / X1C y tolerancias canónicas de Catán.

## 1. Tabla de Holguras y Encajes FDM

| Tipo de Ajuste | Holgura por Cara | Holgura Diametral Total | Uso Típico |
|---|---|---|---|
| **Press-fit fuerte (imán/rodamiento)** | `+0.10 mm` | `+0.20 mm` | Imanes Ø10×5 → Ø10.20×5.20, ejes metálicos |
| **Ajuste deslizante justo** | `+0.20 mm` | `+0.40 mm` | Tapas que no bailan, ensamblaje manual firme |
| **Ajuste holgado / piezas largas** | `+0.25 a +0.30 mm` | `+0.50 a +0.60 mm` | Piezas que encajan horizontalmente, guías, raíles |
| **Bisagra print-in-place** | `+0.35 a +0.40 mm` | `+0.70 a +0.80 mm` | Bisagras articuladas impresas en una pieza |
| **Compensación pata de elefante** | Chaflán `0.6–0.8 mm` a `45°` | — | Toda cara en `Z = 0` con tolerancia crítica |

## 2. Reglas Geométricas de Imprimibilidad

- **Espesor mínimo de pared:** `>= 0.80 mm` (`2 × 0.40 mm`); estructural portante `>= 1.20 mm` (3 líneas) o `>= 1.60 mm` (4 líneas).
- **Salientes y pilares aislados:** diámetro mínimo `>= 1.60 mm`; evitar relación altura/diámetro `> 8:1` sin chaflán en la raíz.
- **Ángulo máximo sin soporte:** `45°` respecto a la vertical (hasta `50°` con buena ventilación de capa en PLA).
- **Agujeros horizontales (en eje X/Y):** usar perfil de lágrima (`teardrop`) en la parte superior para diámetros `>= 5 mm` si no se quieren soportes internos.
- **Chaflanes vs. Filetes en vertical:** en caras orientadas a `Z=0` o voladizos inferiores usar **`chamfer` (`45°`)**, nunca `fillet` circular (el `fillet` inferior arranca a casi `90°` de voladizo y cae mal).

## 3. Especificación Canónica Loseta Catán Maker3D

- **Tipo:** Hexágono regular **pointy-top**.
- **Radio exterior (centro a vértice):** `R = 64.0000 mm`.
- **Apotema (centro a lado):** `APOTHEM = R * cos(30°) = 55.425626 mm`.
- **Ancho total (lado a lado):** `W = 2 * APOTHEM = 110.851252 mm`.
- **Largo total (vértice a vértice):** `D = 2 * R = 128.000000 mm`.
- **Altura del zócalo:** `Z_BASE = 25.0000 mm`.
- **Carril de carreteras:** banda plana perimetral interior de `ROAD_MARGIN = 3.50 mm` a `Z = 25.00 mm`.
- **Pads de poblado (6 vértices):** discos de radio `PAD_RADIUS = 10.00 mm` (`Ø20.00 mm`) centrados en los 6 vértices, estrictamente planos a `Z = 25.00 mm`.
- **Hueco de imán inferior:** centrado en `(0, 0, 0)`, cilindro `Ø10.20 × 5.20 mm` (o hexágono equivalente con canales laterales de escape de pegamento).
- **Biseles del zócalo:** chaflán inferior `0.80 mm` en `Z = 0`.
