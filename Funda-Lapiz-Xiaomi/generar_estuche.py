"""
=============================================================================
GENERADOR PARAMÉTRICO DE ESTUCHE DEFINITIVO (V8.1) PARA XIAOMI SMART PEN 2
=============================================================================
Diseño optimizado: Sistema de Suspensión Elástica por Patitas en T Rectas,
Simetría Total respecto al Centro de la Holgura, Mecanismo Slide-to-Release Real,
Uñero de TPU hasta el Suelo y Carcasa Blindada de PETG.

Novedades clave de la versión V8.1:
  1. Suspensión Elástica por 4 Patitas en T Rectas:
     - El cuerpo central de TPU (anchura h = 15.6 mm) ya no está constreñido por
       las paredes rígidas de PETG.
     - 4 patitas en X = ±51.0 mm con nervio fino (2.40 mm) para no trasladar
       fuerzas excesivas sobre el lápiz.
     - Pie transversal de apoyo 100% recto (26.40 mm, sin rampitas) y más grueso
       (2.40 mm solo en la sección que toca el PETG).
  2. Simetría Total respecto al Centro de la Holgura (X = 0.00 mm):
     - El semicírculo ergonómico de agarre es el centro geométrico absoluto.
     - Distancia hacia el frontal y trasera idéntica (102.0 mm, TPU = 204.0 mm).
  3. Mecanismo Slide-to-Release Real (embocadura abierta hasta X = -92.0 mm):
     - Cono de transición suave sin estrangulamientos (-75 a -89 mm) hasta la
       cámara de punta (-95 mm, domo -98 mm).
     - En reposo adelantado, ~4 mm de la punta quedan protegidos bajo la capucha.
     - Al empujar hacia atrás contra +71.0 mm (163.0 mm abiertos), cualquier lápiz
       de 152-158 mm libera completamente la punta para levantarlo sin obstáculos.
  4. Suelo de TPU Ligero (2.00 mm):
     - Eliminado el aumento macizo de suelo: suelo de 2.00 mm, altura TPU = 12.50 mm
       y altura total cerrada = 18.60 mm.
  5. Uñero de TPU hasta el Suelo con Curvatura Conservada (R = 32.0 mm):
     - La muesca del TPU baja hasta Z = 2.00 mm (suelo del canal) aumentando la
       circunferencia completa para conservar una curvatura suave.
  6. Envolvencia Suavizada (garganta 7.20 mm) y Repuestos +15% Más Amplios.
  7. Carcasa Blindada de PETG (39.20 x 18.60 mm cerrada, 8 imanes D6x3 mm).
  8. 100% Libre de Soportes para Impresión 3D FDM.
=============================================================================
"""

import os
import sys
import numpy as np
from fontTools.ttLib import TTFont
from fontTools.pens.basePen import BasePen
import manifold3d as m3d
import trimesh

# ---------------------------------------------------------------------------
# 1. PARÁMETROS GEOMÉTRICOS DEL ESTUCHE Y LÁPIZ (V8)
# ---------------------------------------------------------------------------
PHONE_NUMBER = "644 727 688"   # Número de teléfono grabado en la tapa
PHONE_TEXT_HEIGHT = 7.50       # Altura del texto grabado
ENGRAVE_DEPTH = 0.50           # Profundidad de grabado

# Geometría del lápiz (Xiaomi Smart Pen 2 / Focus Pen: 152 a 158 mm)
PEN_DIA = 8.8                  # Diámetro del cuerpo (mm)
PEN_RADIUS = PEN_DIA / 2.0     # 4.40 mm
CHANNEL_R = 4.85               # Radio del canal en TPU (holgura suave)

# Apertura y cerradura semicircular para agarre del lápiz (Centrada en el origen):
SCOOP_R = 15.0                 # Radio del arco semicircular (mm)
SCOOP_X = 0.00                 # Centro longitudinal X = 0.00 mm (SIMETRÍA ABSOLUTA)
SCOOP_CLEARANCE = 0.10         # Holgura funcional para cierre sin fricción (mm)

# Uñeras exteriores ergonómicas simétricas en la base del semicírculo:
PRY_NOTCH_LEN = 12.00          # Longitud ergonómica de la uñera (mm)
PRY_NOTCH_H = 1.00             # Altura en cada mitad (2.00 mm total entre base y tapa)
PRY_NOTCH_DEPTH = 1.10         # Profundidad hacia adentro (pared de 2.0 mm -> 0.90 mm macizo continuo)
PRY_NOTCH_R = 0.90             # Radio de redondeo de esquinas (mm)

# Inserto blando (TPU 85A a 20% de relleno) V8.1:
TPU_LEN = 204.00               # Longitud exterior total del TPU (102.0 mm a cada lado del centro)
TPU_WIDTH = 15.60              # Anchura 'h' del cuerpo principal del TPU (mm)
TPU_FLOOR = 2.00               # Suelo elástico fino (2.00 mm bajo el lápiz, sin exceso macizo)
TPU_H = 12.50                  # Altura completa del TPU (mm)
TPU_CORNER_R = 1.20            # Radio de esquinas del inserto (mm)
TPU_SCOOP_R = 32.00            # Radio ampliado del uñero del TPU para llegar al suelo conservando curvatura

# 4 Patitas de suspensión amortiguadoras en T recta:
# Ubicadas en los cuartos de la longitud: X = ± TPU_LEN / 4 = ± 51.0 mm
LEG_X_OFFSETS = [-TPU_LEN / 4.0, +TPU_LEN / 4.0]  # [-51.0, +51.0] mm (Equidistante: misma distancia de la holgura que del final)
LEG_STEM_LEN = TPU_WIDTH / 2.0                     # h/2 = 7.80 mm de separación libre a cada lado
LEG_STEM_W = 2.40                                  # Grosor del nervio de la patita (mm) — sin aumentar para no transmitir fuerza al lápiz
LEG_FLAT_PER_SIDE = 4.00                           # Tramo base a cada lado del nervio (mm)
LEG_WING_LEN = 8.00                                # Tramo exterior a cada lado (mm)
LEG_PAD_THICK = 2.40                               # Grosor reforzado únicamente en el pie recto que toca el PETG (mm)
LEG_BOW = 0.00                                     # Sin rampitas: pie en T totalmente recto en contacto completo con el PETG
LEG_PAD_LEN = 2 * (LEG_STEM_W / 2.0 + LEG_FLAT_PER_SIDE + LEG_WING_LEN) # 26.40 mm total recto

# Anchura total a través de las patitas:
TOTAL_TPU_WIDTH = TPU_WIDTH + 2 * LEG_STEM_LEN    # 15.60 + 2 * 7.80 = 31.20 mm

# Compresión elástica de la tapa sobre los hombros del TPU:
COMPRESSION = 0.40             # 0.40 mm de apriete elástico

# Dimensiones interiores de la cajera rígida de PETG:
POCKET_WIDTH = TOTAL_TPU_WIDTH + 0.20             # 31.40 mm (aloja las patitas con holgura suave)
POCKET_LEN = TPU_LEN + 0.40                       # 204.40 mm
POCKET_R = 1.20

# Pieza 1: Base rígida (PLA / PETG) - 9.60 mm:
BASE_FLOOR = 1.60              # Suelo rígido inferior (mm)
BASE_BODY_H = 9.60             # Altura exterior de la base hasta la costura (9.60 mm)
LIP_H = 5.40                   # Altura del labio vertical reforzado (5.40 mm)
LIP_THICKNESS = 1.70           # Grosor macizo del labio (1.70 mm continuo)

# Dimensiones exteriores del labio:
lip_width = POCKET_WIDTH + 2 * LIP_THICKNESS      # 31.40 + 3.40 = 34.80 mm
lip_len = POCKET_LEN + 1.00                       # 205.40 mm
lip_corner_r = 1.00

# Pieza 2: Tapa rígida (PLA / PETG) - 9.00 mm:
LID_BODY_H = 9.00              # Altura exterior neta de la tapa sobre la costura (9.00 mm)
TOTAL_CLOSED_H = BASE_BODY_H + LID_BODY_H         # 18.60 mm total cerrado
LID_CEILING = 2.40             # Techo sólido de la tapa (mm)

LID_WALL = 2.00                # Grosor de pared exterior de la tapa (2.00 mm macizos)
SLIDE_TOL = 0.20               # Holgura radial de deslizamiento (mm)
LIP_SETBACK = LID_WALL + SLIDE_TOL  # 2.20 mm de escalón

# Cierre magnético por 8 imanes D6x3 mm en las 4 esquinas:
MAG_DIA = 6.00                    # Diámetro nominal del imán (mm)
MAG_HEIGHT = 3.00                 # Altura nominal del imán (mm)
MAG_HOLE_D = 6.20                 # Diámetro de cavidad con holgura (mm)
MAG_HOLE_H = 3.40                 # Altura de cavidad (+0.40 mm sobre imán para holgura de boquilla)
MAG_MEMBRANE = 0.40               # Membrana de plástico en la cara de contacto (0.40 mm = 2 capas de 0.20 mm)
MAG_Y_OFFSET = (POCKET_WIDTH / 2.0) + LIP_THICKNESS - 1.20 # 16.20 mm (adaptado al nuevo ancho)
MAG_X_OFFSET = 106.00             # Desplazamiento longitudinal X en esquinas (mm)

# Dimensiones exteriores totales del estuche blindado:
CASE_WIDTH = lip_width + 2 * SLIDE_TOL + 2 * LID_WALL  # 39.20 mm ("armatoste" confortable y ergonómico)
CASE_LEN = 2 * (MAG_X_OFFSET + MAG_HOLE_D / 2.0 + 2.75) # 223.70 mm
CORNER_R = 2.50                   # Radio de curvatura vertical exterior (mm)


# ---------------------------------------------------------------------------
# 2. CONVERTIDOR DE TEXTO VECTORIAL TTF A MANIFOLD3D
# ---------------------------------------------------------------------------
class PolygonPen(BasePen):
    def __init__(self, glyph_set, num_steps=8):
        super().__init__(glyph_set)
        self.num_steps = num_steps
        self.contours = []
        self.current_contour = []
        self.current_pt = (0, 0)

    def _moveTo(self, pt):
        if self.current_contour:
            self.contours.append(np.array(self.current_contour, dtype=np.float64))
            self.current_contour = []
        self.current_pt = pt
        self.current_contour.append(pt)

    def _lineTo(self, pt):
        self.current_pt = pt
        self.current_contour.append(pt)

    def _curveToOne(self, pt1, pt2, pt3):
        p0 = np.array(self.current_pt, dtype=np.float64)
        p1 = np.array(pt1, dtype=np.float64)
        p2 = np.array(pt2, dtype=np.float64)
        p3 = np.array(pt3, dtype=np.float64)
        for i in range(1, self.num_steps + 1):
            t = i / self.num_steps
            b_pt = (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3
            self.current_contour.append((b_pt[0], b_pt[1]))
        self.current_pt = pt3

    def _qCurveToOne(self, pt1, pt2):
        p0 = np.array(self.current_pt, dtype=np.float64)
        p1 = np.array(pt1, dtype=np.float64)
        p2 = np.array(pt2, dtype=np.float64)
        for i in range(1, self.num_steps + 1):
            t = i / self.num_steps
            b_pt = (1 - t)**2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2
            self.current_contour.append((b_pt[0], b_pt[1]))
        self.current_pt = pt2

    def _closePath(self):
        if self.current_contour:
            self.contours.append(np.array(self.current_contour, dtype=np.float64))
            self.current_contour = []
        self.current_pt = (0, 0)

    def _endPath(self):
        self._closePath()


def text_to_cross_section(text, target_height=PHONE_TEXT_HEIGHT, font_path=None):
    if font_path is None:
        font_candidates = [
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/segoeuib.ttf",
            "C:/Windows/Fonts/calibrib.ttf",
            "C:/Windows/Fonts/arial.ttf"
        ]
        for f in font_candidates:
            if os.path.exists(f):
                font_path = f
                break
    font = TTFont(font_path)
    glyph_set = font.getGlyphSet()
    cmap = font.getBestCmap()
    units_per_em = font['head'].unitsPerEm

    hmtx = font['hmtx']
    pen = PolygonPen(glyph_set)
    cursor_x = 0
    y_min, y_max = float('inf'), float('-inf')

    for char in text:
        glyph_name = cmap.get(ord(char), None)
        if glyph_name is None:
            continue
        glyph = glyph_set[glyph_name]
        start_contour_idx = len(pen.contours)
        glyph.draw(pen)
        adv, lsb = hmtx[glyph_name]
        for i in range(start_contour_idx, len(pen.contours)):
            pen.contours[i][:, 0] += cursor_x
            y_min = min(y_min, pen.contours[i][:, 1].min())
            y_max = max(y_max, pen.contours[i][:, 1].max())
        cursor_x += adv

    if y_max == float('-inf'):
        y_min, y_max = 0, units_per_em
    raw_h = y_max - y_min
    scale = target_height / raw_h

    scaled_contours = []
    for c in pen.contours:
        c_scaled = (c - np.array([cursor_x / 2.0, y_min])) * scale
        c_scaled[:, 1] -= target_height / 2.0
        scaled_contours.append(c_scaled)

    all_polys = []
    for sc in scaled_contours:
        all_polys.append(sc)

    cs = m3d.CrossSection(all_polys, fillrule=m3d.FillRule.EvenOdd)
    return cs, cursor_x * scale


# ---------------------------------------------------------------------------
# 3. PRIMITIVAS Y AYUDAS GEOMÉTRICAS 3D
# ---------------------------------------------------------------------------
def rounded_box(length, width, height, r):
    rx = max(0.1, length / 2.0 - r)
    ry = max(0.1, width / 2.0 - r)
    c = m3d.CrossSection.circle(r, circular_segments=32)
    pts = [c.translate([rx, ry]), c.translate([-rx, ry]), c.translate([-rx, -ry]), c.translate([rx, -ry])]
    cs = m3d.CrossSection.compose(pts).hull()
    return m3d.Manifold.extrude(cs, height)

def make_cylinder_x(length, radius, segments=32):
    cyl = m3d.Manifold.cylinder(length, radius, radius, circular_segments=segments, center=True)
    return cyl.rotate([0, 90, 0])

def make_cylinder_y(length, radius, segments=32):
    cyl = m3d.Manifold.cylinder(length, radius, radius, circular_segments=segments, center=True)
    return cyl.rotate([90, 0, 0])

def make_cone_x(length, r_left, r_right, segments=32):
    cone = m3d.Manifold.cylinder(length, r_left, r_right, circular_segments=segments, center=True)
    return cone.rotate([0, 90, 0])

def rounded_slot_3d(length, height, depth, r=1.0):
    rx = max(0.1, length / 2.0 - r)
    rz = max(0.1, height / 2.0 - r)
    c = m3d.CrossSection.circle(r, circular_segments=24)
    pts = [c.translate([rx, rz]), c.translate([-rx, rz]), c.translate([-rx, -rz]), c.translate([rx, -rz])]
    cs = m3d.CrossSection.compose(pts).hull()
    solid = m3d.Manifold.extrude(cs, depth * 2).translate([0, 0, -depth])
    solid = solid.rotate([90, 0, 0])
    return solid

def build_suspension_leg(stem_len=LEG_STEM_LEN, stem_w=LEG_STEM_W, flat_per_side=LEG_FLAT_PER_SIDE, wing_len=LEG_WING_LEN, pad_thick=LEG_PAD_THICK, bow=LEG_BOW, height=TPU_H):
    """
    Genera una patita elástica de suspensión en T totalmente recta:
    - Nervio de unión fino (stem_w = 2.40 mm) para no trasladar fuerzas excesivas al lápiz.
    - Barra transversal de apoyo totalmente recta (bow = 0.00 mm, 26.40 mm de longitud)
      sin rampitas laterales, apoyando de punta a punta contra la pared de PETG.
    - Grosor reforzado únicamente en el pie de apoyo (pad_thick = 2.40 mm).
    - Misma altura que todo el TPU (height = 12.50 mm) desde la cama Z=0: 100% libre de soportes.
    """
    x_flat = stem_w / 2.0 + flat_per_side   # 1.2 + 4.0 = 5.2 mm
    x_tip = x_flat + wing_len               # 5.2 + 8.0 = 13.2 mm
    if bow == 0.0:
        pts = [
            [stem_w / 2.0, -0.5],
            [stem_w / 2.0, stem_len - pad_thick],
            [x_tip, stem_len - pad_thick],
            [x_tip, stem_len],
            [-x_tip, stem_len],
            [-x_tip, stem_len - pad_thick],
            [-stem_w / 2.0, stem_len - pad_thick],
            [-stem_w / 2.0, -0.5],
        ]
    else:
        pts = [
            [stem_w / 2.0, -0.5],
            [stem_w / 2.0, stem_len - pad_thick],
            [x_flat, stem_len - pad_thick],
            [x_tip, stem_len - pad_thick - bow],
            [x_tip, stem_len - bow],
            [x_flat, stem_len],
            [-x_flat, stem_len],
            [-x_tip, stem_len - bow],
            [-x_tip, stem_len - pad_thick - bow],
            [-x_flat, stem_len - pad_thick],
            [-stem_w / 2.0, stem_len - pad_thick],
            [-stem_w / 2.0, -0.5],
        ]
    cs = m3d.CrossSection([pts])
    cs = cs.offset(-0.4, join_type=m3d.JoinType.Round).offset(0.4, join_type=m3d.JoinType.Round)
    return m3d.Manifold.extrude(cs, height)


# ---------------------------------------------------------------------------
# 4. GENERADORES DE COMPONENTES V8
# ---------------------------------------------------------------------------
def build_tpu_insert(enveloping=True):
    """
    Pieza 3: Inserto protector en TPU (TPU 85A a 20% infill) V8.
    - 4 Patitas de suspensión amortiguadoras en X = ±34.0 mm con separación h/2 = 7.8 mm.
    - Simetría 100% respecto al centro de la holgura (X = 0.00 mm):
      * Distancia del centro a la culata: 102.0 mm.
      * Distancia del centro a la punta: 102.0 mm (11.0 mm de parachoques macizo frontal).
    - Mecanismo Slide-to-Release de 4.0 mm:
      * El lápiz puede desplazarse 4.0 mm hacia atrás para liberar el plumín de la visera.
    - Suelo macizo aumentado a 4.00 mm bajo el lápiz.
    - Envolvencia suavizada al 10% (garganta 7.20 mm).
    - 2 hendiduras de puntas de repuesto ampliadas un 15%.
    """
    body = rounded_box(TPU_LEN, TPU_WIDTH, TPU_H, TPU_CORNER_R)
    
    # Adición de las 4 patitas elásticas de suspensión:
    for lx in LEG_X_OFFSETS:
        leg_pos = build_suspension_leg().translate([lx, TPU_WIDTH / 2.0, 0])
        leg_neg = build_suspension_leg().rotate([0, 0, 180]).translate([lx, -TPU_WIDTH / 2.0, 0])
        body = body + leg_pos + leg_neg

    pen_r = 4.50  # radio del canal cilíndrico (Ø 9.0 mm para lápiz de 8.8 mm)
    pen_z = TPU_FLOOR + PEN_RADIUS  # 2.00 + 4.40 = 6.40 mm
    
    # 1. Canal cilíndrico principal:
    # Carrera desde X = +71.0 mm (pared trasera) hasta X = -75.0 mm (longitud 146.0 mm)
    pen_back_x = 71.0
    cyl_front_x = -75.0
    cyl_len = pen_back_x - cyl_front_x
    cyl_center_x = (pen_back_x + cyl_front_x) / 2.0  # -2.0 mm
    main_cyl = make_cylinder_x(cyl_len + 0.2, pen_r, segments=48).translate([cyl_center_x - 0.1, 0, pen_z])
    
    # 2. Cono de transición para el cuello de la punta (suave sin estrangulamientos):
    cone_end_x = -89.0
    cone_len = cyl_front_x - cone_end_x  # 14.0 mm
    cone_center_x = (cyl_front_x + cone_end_x) / 2.0  # -82.0 mm
    cone = make_cone_x(cone_len + 0.2, 3.0, pen_r, segments=48).translate([cone_center_x, 0, pen_z])
    
    # 3. Cámara de desahogo de la punta para Slide-to-Release real:
    # Llega hasta X = -95.0 mm (esfera hasta -98.0 mm, dejando 4.0 mm de parachoques frontal hasta -102.0).
    tip_end_x = -95.0
    tip_len = cone_end_x - tip_end_x  # 6.0 mm
    tip_center_x = (cone_end_x + tip_end_x) / 2.0  # -92.0 mm
    nib_cyl = make_cylinder_x(tip_len, 3.0, segments=32).translate([tip_center_x, 0, pen_z])
    nib_dome = m3d.Manifold.sphere(3.0, circular_segments=32).translate([tip_end_x, 0, pen_z])
    
    # 4. Embocadura superior (open_slot):
    # Llega hasta X = -92.0 mm (longitud abierta 163.0 mm desde +71.0 mm).
    # En reposo adelantado (-96.0 mm), ~4 mm de la punta quedan cubiertos bajo la capucha (-92..-98).
    # Al deslizar hacia atrás (+71.0), lápices de 152-158 mm quedan con 5-11 mm libres por delante de -92.0.
    slot_front_x = -92.0
    slot_len = pen_back_x - slot_front_x  # 163.0 mm
    slot_center_x = (pen_back_x + slot_front_x) / 2.0  # -10.5 mm
    
    if enveloping:
        # Garganta 7.20 mm (-10% de envolvencia sobre V7 para extracción sedosa)
        throat_w = 7.20
        pts_throat = [
            [0.0, pen_z - 1.0],
            [+3.5, pen_z - 1.0],
            [+pen_r * 0.95, pen_z + 1.8],
            [+throat_w / 2.0, pen_z + 3.4],
            [+throat_w / 2.0, pen_z + 4.6],
            [+4.80, TPU_H + 2.0],
            [-4.80, TPU_H + 2.0],
            [-throat_w / 2.0, pen_z + 4.6],
            [-throat_w / 2.0, pen_z + 3.4],
            [-pen_r * 0.95, pen_z + 1.8],
            [-3.5, pen_z - 1.0]
        ]
    else:
        # Cuna en U vertical pura para filamentos rígidos
        pts_throat = [
            [0.0, pen_z - 1.0],
            [+pen_r, pen_z - 1.0],
            [+pen_r, TPU_H + 2.0],
            [-pen_r, TPU_H + 2.0],
            [-pen_r, pen_z - 1.0]
        ]
        
    cs_throat = m3d.CrossSection([pts_throat])
    open_slot = m3d.Manifold.extrude(cs_throat, slot_len).translate([0, 0, -slot_len / 2.0])
    open_slot = open_slot.rotate([90, 0, 90]).translate([slot_center_x, 0, 0])
    
    # 5. Muescas suaves para dedos en el centro absoluto (X = 0.00 mm):
    # Llega hasta el suelo del canal (Z = TPU_FLOOR = 2.00 mm) con circunferencia ampliada (TPU_SCOOP_R = 32.0 mm)
    # para conservar la curvatura suave aunque abarque más longitud del lápiz.
    scoop_z = TPU_FLOOR + TPU_SCOOP_R  # 34.00 mm
    scoop_cyl = make_cylinder_y(TPU_WIDTH + 6.0, TPU_SCOOP_R, segments=64).translate([0, 0, scoop_z])
    scoop_box = m3d.Manifold.cube([TPU_SCOOP_R * 2, TPU_WIDTH + 6.0, TPU_SCOOP_R], center=True).translate([0, 0, scoop_z + TPU_SCOOP_R / 2.0])
    tpu_scoop = scoop_cyl + scoop_box
    
    # 6. Dos hendiduras para puntas de repuesto (+15% más amplias):
    slot_y_offsets = [-3.50, +3.50]
    nib_z = pen_z + 1.0
    nib_slots = None
    
    r_white = 1.45  # Ø 2.90 mm (+15% sobre 2.50 mm)
    r_cone = 1.78   # Base cónica Ø 3.56 mm (+15% sobre 3.10 mm)
    w_pin = (1.50 / 2.0) if enveloping else 1.0
    w_white = (2.00 / 2.0) if enveloping else 1.5
    w_cone = (2.60 / 2.0) if enveloping else 1.8
    w_tip = 0.30
    
    for sy in slot_y_offsets:
        # A) Vástago metálico
        p_len = 5.3
        p_cx = 76.0 + p_len / 2.0
        p_cyl = make_cylinder_x(p_len + 0.2, 0.75, segments=24).translate([p_cx, sy, nib_z])
        p_rear_dome = m3d.Manifold.sphere(0.75, circular_segments=24).translate([76.0, sy, nib_z])
        
        # B) Manguito cilíndrico
        c_len = 5.0
        c_cx = 76.0 + p_len + c_len / 2.0
        c_cyl = make_cylinder_x(c_len + 0.2, r_white, segments=32).translate([c_cx, sy, nib_z])
        
        # C) Cabeza cónica recta con escalón saliente
        cn_len = 5.5
        cn_cx = 76.0 + p_len + c_len + cn_len / 2.0
        cn_cradle = make_cone_x(cn_len + 0.2, r_cone, 0.20, segments=32).translate([cn_cx - 0.1, sy, nib_z])
        tip_dome = m3d.Manifold.sphere(0.20, circular_segments=16).translate([76.0 + p_len + c_len + cn_len, sy, nib_z])
        
        x_start = 76.0
        x_white = x_start + p_len
        x_cone = x_white + c_len
        x_tip = x_cone + cn_len
        
        pts_open = [
            [x_start, sy - w_pin],
            [x_white, sy - w_pin],
            [x_white, sy - w_white],
            [x_cone, sy - w_white],
            [x_cone, sy - w_cone],
            [x_tip, sy - w_tip],
            [x_tip + 0.3, sy],
            [x_tip, sy + w_tip],
            [x_cone, sy + w_cone],
            [x_cone, sy + w_white],
            [x_white, sy + w_white],
            [x_white, sy + w_pin],
            [x_start, sy + w_pin]
        ]
        cs_open = m3d.CrossSection([pts_open])
        nib_open_solid = m3d.Manifold.extrude(cs_open, 10.0).translate([0, 0, nib_z - 0.4])
        slot = p_cyl + p_rear_dome + c_cyl + cn_cradle + tip_dome + nib_open_solid
        nib_slots = slot if nib_slots is None else (nib_slots + slot)
        
    pen_cutouts = main_cyl + open_slot + cone + nib_cyl + nib_dome + tpu_scoop
    return body - pen_cutouts - nib_slots


def build_base():
    """
    Pieza 1: Base rígida (PLA / PETG) ensanchada con 4 imanes D6x3 mm ocultos en las esquinas.
    - Altura exterior hasta costura: BASE_BODY_H = 9.60 mm.
    - Labio vertical reforzado: LIP_H = 5.40 mm (total base = 15.00 mm).
    - Apertura semicircular en el centro (X = 0.00 mm) con R = 15.0 mm que baja hasta Z = 7.20 mm.
    - Cajera interior ensanchada: 204.40 x 31.40 mm (aloja el TPU con sus 4 patitas en T rectas).
    - 4 Cavidades internas ciegas para imanes D6x3 mm (profundidad 3.40 mm, membrana 0.40 mm).
      * Pausa en slicer a Z = 9.20 mm (capa 46 a 0.20 mm).
    - Uñeras ergonómicas en la base del semicírculo en X = 0.00 mm.
    """
    base_body = rounded_box(CASE_LEN, CASE_WIDTH, BASE_BODY_H, CORNER_R)
    
    # Labio de encaje superior (exterior 34.80 x 205.40 mm, pared maciza de 1.70 mm)
    lip = rounded_box(lip_len, lip_width, LIP_H, lip_corner_r).translate([0, 0, BASE_BODY_H])
    
    # Micro-nervio táctil de retención (Snap-Fit click)
    click_bead = rounded_box(lip_len + 0.35, lip_width + 0.35, 0.70, lip_corner_r).translate([0, 0, BASE_BODY_H + LIP_H - 1.80])
    solid_base = base_body + lip + click_bead
    
    # Cajera interior para el TPU (31.40 x 204.40 mm)
    pocket_h = (BASE_BODY_H + LIP_H) - BASE_FLOOR + 2.0
    pocket = rounded_box(POCKET_LEN, POCKET_WIDTH, pocket_h, POCKET_R).translate([0, 0, BASE_FLOOR])
    
    # Apertura semicircular para agarre del lápiz (X = 0.00 mm)
    scoop_z_base = BASE_FLOOR + 5.60 + SCOOP_R  # 22.20 mm
    cyl_base = m3d.Manifold.cylinder(CASE_WIDTH + 10.0, SCOOP_R, SCOOP_R, circular_segments=48, center=True).rotate([90, 0, 0]).translate([0, 0, scoop_z_base])
    box_base = m3d.Manifold.cube([SCOOP_R * 2, CASE_WIDTH + 10.0, 20.0], center=True).translate([0, 0, scoop_z_base + 10.0])
    base_scoop = cyl_base + box_base
    
    # Uñeras ergonómicas en la base del semicírculo (X = 0.00 mm)
    scoop_z_bottom = BASE_FLOOR + 5.60  # 7.20 mm
    slot_fb = rounded_slot_3d(PRY_NOTCH_LEN, PRY_NOTCH_H * 2, PRY_NOTCH_DEPTH, r=PRY_NOTCH_R).translate([0, -CASE_WIDTH / 2.0, scoop_z_bottom])
    slot_bb = rounded_slot_3d(PRY_NOTCH_LEN, PRY_NOTCH_H * 2, PRY_NOTCH_DEPTH, r=PRY_NOTCH_R).translate([0,  CASE_WIDTH / 2.0, scoop_z_bottom])

    # 4 cavidades para imanes D6x3 mm ocultos en las 4 esquinas:
    mag_r = MAG_HOLE_D / 2.0
    base_mag_pockets = None
    for mx in [-MAG_X_OFFSET, +MAG_X_OFFSET]:
        for my in [-MAG_Y_OFFSET, +MAG_Y_OFFSET]:
            p_cyl = m3d.Manifold.cylinder(MAG_HOLE_H, mag_r, mag_r, circular_segments=32, center=False).translate([mx, my, BASE_BODY_H - MAG_MEMBRANE - MAG_HOLE_H])
            base_mag_pockets = p_cyl if base_mag_pockets is None else (base_mag_pockets + p_cyl)
            
    return solid_base - pocket - base_mag_pockets - base_scoop - slot_fb - slot_bb


def build_lid(with_phone_number=True):
    """
    Pieza 2: Tapa exterior rígida (PLA / PETG) con 4 imanes D6x3 mm ocultos en las esquinas.
    - Altura exterior neta: LID_BODY_H = 9.00 mm (total cerrado 18.60 mm).
    - Cerradura semicircular en paredes exteriores en X = 0.00 mm.
    - CIERRE REAL POR COMPRESIÓN EN TPU: 0.40 mm de apriete elástico.
    - Canal de desahogo de seguridad para el lápiz y botones con >2.5 mm de aire libre.
    - 4 Cavidades internas ciegas para imanes D6x3 mm (profundidad 3.40 mm, membrana 0.40 mm):
      * Pausa en slicer a Z = 8.60 mm (capa 43 a 0.20 mm).
    - Teléfono grabado en bajorrelieve (7.50 mm de altura).
    """
    lid_body = rounded_box(CASE_LEN, CASE_WIDTH, LID_BODY_H, CORNER_R)
    
    # Cerradura semicircular exterior que encaja en la apertura de la base (X = 0.00 mm):
    scoop_z_base = BASE_FLOOR + 5.60 + SCOOP_R
    tab_r = SCOOP_R - SCOOP_CLEARANCE
    cyl_tab = m3d.Manifold.cylinder(CASE_WIDTH + 2.0, tab_r, tab_r, circular_segments=48, center=True).rotate([90, 0, 0]).translate([0, 0, scoop_z_base + SCOOP_CLEARANCE])
    h_tab = BASE_BODY_H - (scoop_z_base - SCOOP_R)
    block_tab = m3d.Manifold.cube([SCOOP_R * 2, CASE_WIDTH, h_tab], center=False).translate([-SCOOP_R, -CASE_WIDTH / 2.0, scoop_z_base - SCOOP_R + SCOOP_CLEARANCE])
    tab_solid = block_tab ^ cyl_tab
    tab_lid_coords = tab_solid.translate([0, 0, -BASE_BODY_H])
    
    # Mantener la cerradura únicamente en las paredes exteriores largas:
    wall_inner_y = (CASE_WIDTH / 2.0 - LID_WALL) - 0.10
    inner_mask = m3d.Manifold.cube([50.0, wall_inner_y * 2, 10.0], center=True).translate([0, 0, 0])
    tab_walls_lid = tab_lid_coords - inner_mask
    lid_solid = lid_body + tab_walls_lid
    
    # 1. Zanja perimetral para alojar el labio rígido de la base:
    lip_pocket_len = lip_len + 2 * SLIDE_TOL
    lip_pocket_width = lip_width + 2 * SLIDE_TOL
    lip_pocket_depth = LIP_H + 0.30
    trench_outer = rounded_box(lip_pocket_len, lip_pocket_width, lip_pocket_depth + 1.0, 0.80).translate([0, 0, -0.5])
    trench_inner = rounded_box(POCKET_LEN - 0.40, POCKET_WIDTH - 0.40, lip_pocket_depth + 2.0, 0.80).translate([0, 0, -1.0])
    lip_trench = trench_outer - trench_inner
    
    # 2. Superficie interna de compresión sobre los hombros del TPU:
    tpu_protrusion = (BASE_FLOOR + TPU_H) - BASE_BODY_H   # 1.6 + 12.5 - 9.6 = 4.50 mm
    tpu_comp_depth = tpu_protrusion - COMPRESSION         # 4.50 - 0.40 = 4.10 mm
    tpu_comp_pocket = rounded_box(POCKET_LEN + 0.40, POCKET_WIDTH + 0.40, tpu_comp_depth + 0.50, 0.80).translate([0, 0, -0.5])
    
    # 3. Canal de desahogo de seguridad para el lápiz y botones:
    pen_relief_depth = 6.00
    pen_relief = rounded_box(TPU_LEN + 1.0, 10.60, pen_relief_depth + 0.50, 1.20).translate([0, 0, -0.5])
    
    # Detente interior para el clic de retención:
    click_groove = rounded_box(lip_pocket_len + 0.35, lip_pocket_width + 0.35, 0.90, 0.80).translate([0, 0, LIP_H - 1.80])
    internal_cuts = lip_trench + tpu_comp_pocket + pen_relief + click_groove
    
    # 4 cavidades para imanes D6x3 mm ocultos en las 4 esquinas:
    mag_r = MAG_HOLE_D / 2.0
    lid_mag_pockets = None
    for mx in [-MAG_X_OFFSET, +MAG_X_OFFSET]:
        for my in [-MAG_Y_OFFSET, +MAG_Y_OFFSET]:
            p_cyl = m3d.Manifold.cylinder(MAG_HOLE_H, mag_r, mag_r, circular_segments=32, center=False).translate([mx, my, MAG_MEMBRANE])
            lid_mag_pockets = p_cyl if lid_mag_pockets is None else (lid_mag_pockets + p_cyl)
            
    # Uñeras ergonómicas en la base de la pestaña semicircular (X = 0.00 mm):
    scoop_z_bottom = BASE_FLOOR + 5.60
    slot_fl = rounded_slot_3d(PRY_NOTCH_LEN, PRY_NOTCH_H * 2, PRY_NOTCH_DEPTH, r=PRY_NOTCH_R).translate([0, -CASE_WIDTH / 2.0, scoop_z_bottom - BASE_BODY_H])
    slot_bl = rounded_slot_3d(PRY_NOTCH_LEN, PRY_NOTCH_H * 2, PRY_NOTCH_DEPTH, r=PRY_NOTCH_R).translate([0,  CASE_WIDTH / 2.0, scoop_z_bottom - BASE_BODY_H])

    lid = lid_solid - internal_cuts - lid_mag_pockets - slot_fl - slot_bl
    
    if with_phone_number:
        cs, text_width = text_to_cross_section(PHONE_NUMBER, target_height=PHONE_TEXT_HEIGHT)
        text_3d = m3d.Manifold.extrude(cs, ENGRAVE_DEPTH + 0.20).translate([0, 0, LID_BODY_H - ENGRAVE_DEPTH])
        lid = lid - text_3d
        
    return lid


# ---------------------------------------------------------------------------
# 5. EXPORTADOR A STL CON VERIFICACIÓN DE MALLA
# ---------------------------------------------------------------------------
def export_stl(manifold, filepath):
    mesh = manifold.to_mesh()
    tm = trimesh.Trimesh(vertices=mesh.vert_properties[:, :3], faces=mesh.tri_verts)
    if not tm.is_watertight:
        trimesh.repair.fix_normals(tm)
        trimesh.repair.fix_inversion(tm)
    tm.export(filepath)
    dim = np.round(tm.bounds[1] - tm.bounds[0], 2)
    print(f" -> Guardado: {os.path.basename(filepath)}")
    print(f"    Dimensiones (X, Y, Z): {dim[0]} x {dim[1]} x {dim[2]} mm")
    print(f"    Estanqueidad (Watertight): {tm.is_watertight}")
    print(f"    Volumen: {tm.volume:.1f} mm³ | Triángulos: {len(tm.faces):,}")
    return tm


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    modelos_dir = os.path.join(base_dir, "Modelos-3D")
    if not os.path.exists(modelos_dir):
        modelos_dir = os.path.join(base_dir, "01_Modelos_3D")
    target_dir = modelos_dir if os.path.exists(modelos_dir) else base_dir

    print("=================================================================")
    print(" GENERANDO ESTUCHE V8.1 DEFINITIVO PARA XIAOMI SMART PEN 2")
    print(f" Dimensiones: {CASE_LEN:.1f} x {CASE_WIDTH:.1f} x {TOTAL_CLOSED_H:.1f} mm (Blindado con Suspensión)")
    print(f" TPU: {TPU_LEN:.1f} x {TOTAL_TPU_WIDTH:.1f} x {TPU_H:.1f} mm | Suelo: {TPU_FLOOR:.1f} mm | Uñero R={TPU_SCOOP_R:.1f} mm")
    print(f" Patitas T rectas (nervio {LEG_STEM_W:.1f} mm, apoyo {LEG_PAD_THICK:.1f} mm) | Embocadura 163.0 mm")
    print("=================================================================")
    
    tpu_insert = build_tpu_insert(enveloping=True)
    tpu_insert_rigid = build_tpu_insert(enveloping=False)
    base = build_base()
    lid_with_phone = build_lid(with_phone_number=True)
    lid_smooth = build_lid(with_phone_number=False)
    
    lid_print = lid_with_phone.rotate([180, 0, 0]).translate([0, 0, LID_BODY_H])
    lid_smooth_print = lid_smooth.rotate([180, 0, 0]).translate([0, 0, LID_BODY_H])
    
    lid_closed = lid_with_phone.translate([0, 0, BASE_BODY_H])
    assembled = base + lid_closed
    
    tpu_plate = tpu_insert.translate([0, -32.0, 0])
    base_plate = base.translate([0, 0, 0])
    lid_plate = lid_print.translate([0, 35.0, 0])
    plate_all = tpu_plate + base_plate + lid_plate
    
    files_to_export = [
        ("estuche_xiaomi_pen2_base_PLA.stl", base),
        ("estuche_xiaomi_pen2_tapa_PLA.stl", lid_print),
        ("estuche_xiaomi_pen2_tapa_lisa_PLA.stl", lid_smooth_print),
        ("estuche_xiaomi_pen2_inserto_TPU.stl", tpu_insert),
        ("estuche_xiaomi_pen2_inserto_rigido_sin_envolturas.stl", tpu_insert_rigid),
        ("estuche_xiaomi_pen2_ensamblado_cerrado.stl", assembled),
        ("estuche_xiaomi_pen2_placa_impresion.stl", plate_all)
    ]
    
    for filename, manifold_obj in files_to_export:
        filepath = os.path.join(target_dir, filename)
        export_stl(manifold_obj, filepath)
            
    print("\n=================================================================")
    print(" ¡TODOS LOS ARCHIVOS STL V8 DEFINITIVOS FUERON GENERADOS CON ÉXITO!")
    print("=================================================================")

if __name__ == "__main__":
    main()
