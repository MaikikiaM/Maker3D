"""
Bioma 2 — Bosque 3D Sculptor Pipeline (True Multi-Tier Conifer Pines + Contour-Aligned Forest Floor).

Fixes the 2.5D "melting trees" heightfield artifact by:
1. Extracting the 15 Nano Banana pine centers, reconstructing a clean rolling
   moss/earth/trail forest floor + sharp granite boulder outcrops beneath them.
2. Inpainting flat painted tree blobs out of the diffuse map and replacing them
   with needle mulch + ambient tree-base shadows while preserving trails & boulders.
3. Procedurally sculpting 15 distinct watertight multi-tiered scalloped conifer pine
   solids (buttress-flared textured trunks + 6-8 overhanging star-flanged bough whorls
   with darkened concave undersides and leader spires).
4. Performing exact CSG union via manifold3d (`Error.NoError`, `genus=0`) with full
   RGBA attribute propagation (`treeVCol`) and assembling the WebGL PBR viewer with
   `TRIPLANAR_MODE = "forest"`.
"""

import json
import math
import os
import sys
import time
import numpy as np
import scipy.ndimage as ndi
import manifold3d as mf
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SKILL_SCRIPTS = os.path.join(REPO_ROOT, ".agents", "skills", "nanobanana-3d-sculpt", "scripts")
MONTANA_PIPE = os.path.join(REPO_ROOT, "Tableros-Catan", "pipeline-montana")
BIOMAS_PIPE = os.path.dirname(__file__)
for p in (SKILL_SCRIPTS, MONTANA_PIPE, BIOMAS_PIPE):
    if p not in sys.path:
        sys.path.insert(0, p)

import nanobanana_to_catan_hex as nb
import assemble_viewer_v2 as av2
import build_catan_biome as bcb
import terrain_mesh_utils as tmu


HEIGHTMAP_SRC = r"C:\Users\migue\.gemini\antigravity\brain\ed072550-bd3b-430e-a82e-ebe9a6377769\catan_forest_heightmap_1794608505743.jpg"
ALBEDO_SRC = r"C:\Users\migue\.gemini\antigravity\brain\ed072550-bd3b-430e-a82e-ebe9a6377769\bosque_albedo_v1_1794608608808.jpg"

# Exact Nano Banana pine centers sorted into scenic Catan tabletop proportions (Z_PEAK ~58-60mm):
# (x_mm, y_mm, tree_height_mm, crown_radius_mm, n_tiers, seed)
PINE_SPECS = [
    # Back hero ridge crescent (full, layered storybook conifers)
    (-1.63,  +38.45, 29.5, 9.6, 11, 101),  # 1. North Hero Spire
    (+30.68, +22.67, 28.2, 9.4, 11, 102),  # 2. NE Tall Pine
    (-36.70,  -9.89, 26.8, 9.0, 10, 103),  # 3. West Guard Pine
    (-31.44, +22.67, 28.0, 9.4, 11, 104),  # 4. NW Ridge Pine
    (+36.45,  +3.63, 25.8, 8.8, 10, 105),  # 5. East Sentinel Pine
    (-21.42,  +8.89, 26.5, 9.0, 10, 106),  # 6. West Inner Canopy
    # Outer flank & mid-forest groves
    (+16.16, +35.44, 24.5, 8.4, 10, 109),  # 9. NNE Slope Pine
    (-18.91, +34.69, 23.8, 8.3, 10, 113),  # 13. NNW Slope Pine
    (-38.70,  +6.39, 23.2, 8.1,  9, 108),  # 8. WNW Border Spruce
    (-10.14, +21.67, 22.8, 8.2,  9, 110),  # 10. Clearing North Pine
    (+23.67,  -1.13, 23.5, 8.3,  9, 111),  # 11. East Trailhead Pine
    (-21.92,  -7.14, 21.8, 7.9,  9, 112),  # 12. SW Bank Pine
    (+9.39,  +24.17, 22.0, 8.0,  9, 114),  # 14. NE Glade Edge Pine
    (+18.66, +11.65, 20.8, 7.7,  9, 115),  # 15. Inner East Spruce
    # Foreground young spruce framing the mossy boulder pass
    (+1.13,  -34.69, 17.5, 6.7,  8, 107),  # 7. Front Sapling (between boulders)
]


def verify_cad_clearances():
    for idx, (x, y, h, r, tiers, _) in enumerate(PINE_SPECS, 1):
        Xg, Yg = np.array([[x]]), np.array([[y]])
        de, dv = nb.hex_distances(Xg, Yg)
        clearance_road = float(de[0, 0]) - nb.ROAD_MARGIN - r
        clearance_pad = float(dv[0, 0]) - nb.PAD_RADIUS - r
        if clearance_road < 0.8 or clearance_pad < 0.8:
            raise ValueError(f"Tree {idx} ({x},{y}, r={r}) too close to road/pad: road={clearance_road:.2f}, pad={clearance_pad:.2f}")


def resample_canonical_xy(arr, X, Y, lim=nb.R):
    """Bilinear resample a canonical (row 0 = -lim, row H-1 = +lim) array without Y-flipping."""
    H, W = arr.shape[:2]
    ix = np.clip((X + lim) / (2.0 * lim), 0.0, 1.0) * (W - 1)
    iy = np.clip((Y + lim) / (2.0 * lim), 0.0, 1.0) * (H - 1)
    if arr.ndim == 2:
        return ndi.map_coordinates(arr.astype(np.float32), [iy, ix], order=1, mode="nearest")
    return np.stack([
        ndi.map_coordinates(arr[:, :, c].astype(np.float32), [iy, ix], order=1, mode="nearest")
        for c in range(arr.shape[2])
    ], axis=-1)


def extract_clean_floor_hm_and_albedo(hm_raw_u8, alb_raw_u8, X, Y, dist_e, dist_v):
    """
    Separates rolling woodland earth floor from faceted granite boulder outcrops,
    eradicating all melted tree domes, stump cliff pedestals, border wall ledges,
    and stray noise spikes, while cleaning flat canopy paint into pine needle mulch.
    """
    hm_s = nb.sample_square_domain(hm_raw_u8.astype(np.float32) / 255.0, X, Y)
    alb_s = nb.sample_square_domain(alb_raw_u8.astype(np.float32) / 255.0, X, Y)
    dx = float(abs(X[0, 1] - X[0, 0]))
    H, W = X.shape

    tree_mask = np.zeros_like(X, dtype=np.float32)
    tree_kill_wide = np.zeros_like(X, dtype=np.float32)
    tree_shadow = np.ones_like(X, dtype=np.float32)
    for tx, ty, h, r_b, _, _ in PINE_SPECS:
        d = np.hypot(X - tx, Y - ty)
        w_m = 1.0 - nb.smoothstep(r_b * 0.50, r_b * 1.15, d)
        w_wide = 1.0 - nb.smoothstep(r_b * 0.65, r_b * 1.62, d)
        tree_mask = np.maximum(tree_mask, w_m)
        tree_kill_wide = np.maximum(tree_kill_wide, w_wide)
        # Gentle ambient tree-base shade (soft falloff, never dark coaster circles)
        sh = 1.0 - 0.24 * (1.0 - nb.smoothstep(r_b * 0.10, r_b * 1.10, d))
        tree_shadow = np.minimum(tree_shadow, sh)

    # Identify granite boulder outcrops cleanly: close small moss veins first, then open 1px noise
    sat = np.max(alb_s, axis=-1) - np.min(alb_s, axis=-1)
    lum = 0.2126 * alb_s[:, :, 0] + 0.7152 * alb_s[:, :, 1] + 0.0722 * alb_s[:, :, 2]
    boulder_safe = (dist_e >= nb.ROAD_MARGIN + 3.2) & (dist_v >= nb.PAD_RADIUS + 2.5)
    raw_rock = ((sat < 0.16) & (lum > 0.24) & (hm_s > 0.24) & (tree_mask < 0.30) & boulder_safe).astype(np.float32)
    k_close = max(5, int(round(3.0 / dx)) | 1)
    k_open = max(3, int(round(1.4 / dx)) | 1)
    closed_rock = ndi.grey_closing(raw_rock, size=(k_close, k_close))
    clean_rock = ndi.grey_opening(closed_rock, size=(k_open, k_open))
    rock_mask = np.clip(ndi.gaussian_filter(clean_rock, sigma=max(1.4 / dx, 1.5)), 0.0, 1.0)

    # Smooth gentle base earth terrain (zero tree mounds, zero perimeter tablecloth ledge)
    in_earth = (dist_e >= nb.ROAD_MARGIN + 0.6).astype(np.float32)
    keep_earth_w = np.clip(1.0 - tree_kill_wide, 0.0, 1.0) * (1.0 - 0.85 * rock_mask) * in_earth
    hm_clamped_earth = np.minimum(hm_s, 0.36)
    sig_earth = max(6.2 / dx, 3.0)
    ew_sum = ndi.gaussian_filter(keep_earth_w, sigma=sig_earth) + 1e-5
    earth_smooth = ndi.gaussian_filter(hm_clamped_earth * keep_earth_w, sigma=sig_earth) / ew_sum
    gentle_undul = ndi.gaussian_filter(hm_clamped_earth, sigma=max(3.5 / dx, 2.0)) * in_earth
    base_soil = 0.72 * earth_smooth + 0.28 * gentle_undul

    boulder_hm = np.maximum(0.0, hm_s - 0.23) * rock_mask
    boulder_beveled = ndi.gaussian_filter(boulder_hm, sigma=max(1.4 / dx, 1.4))

    fbm_rock = nb.fft_fbm(W, H, 2.2, 8.5, 2.0 * nb.R, 2.0 * nb.R, beta=2.1, seed=303)
    ridge_rock = 1.0 - np.abs(fbm_rock)
    hm_clean_combo = np.clip(base_soil * 0.42 + boulder_beveled * (1.35 + 0.32 * ridge_rock), 0.0, 1.0)

    # Albedo cleanup: replace flat painted canopy green under trees with warm needle-strewn earth/moss
    in_terrain = (dist_e >= nb.ROAD_MARGIN + 0.6) & (dist_v >= nb.PAD_RADIUS + 0.6)
    g_dom = alb_s[:, :, 1] - 0.5 * (alb_s[:, :, 0] + alb_s[:, :, 2])
    floor_src_w = np.clip(1.0 - tree_mask, 0.0, 1.0) * in_terrain.astype(np.float32) * (1.0 - nb.smoothstep(0.055, 0.13, g_dom))
    sig_al = max(7.0 / dx, 3.0)
    fa_sum = ndi.gaussian_filter(floor_src_w, sigma=sig_al) + 1e-5
    earth_fill = np.stack([
        ndi.gaussian_filter(alb_s[:, :, c] * floor_src_w, sigma=sig_al) / fa_sum for c in range(3)
    ], axis=-1)

    needle_fine = nb.fft_fbm(W, H, 0.45, 2.0, 2.0 * nb.R, 2.0 * nb.R, beta=1.7, seed=409)
    needle_patch = nb.fft_fbm(W, H, 2.2, 7.0, 2.0 * nb.R, 2.0 * nb.R, beta=2.0, seed=410)
    duff = np.clip(
        earth_fill * (0.94 + 0.16 * needle_fine)[..., None]
        + np.array([-0.02, 0.025, -0.02], dtype=np.float32) * np.clip(needle_patch, 0.0, 1.0)[..., None],
        0.06, 0.60
    )

    alb_mask = np.clip(ndi.gaussian_filter(tree_mask, sigma=max(1.4 / dx, 1.4)), 0.0, 1.0)[..., None]
    alb_clean = alb_s * (1.0 - alb_mask) + duff * alb_mask
    sh_in = 1.0 - (1.0 - tree_shadow) * nb.smoothstep(nb.ROAD_MARGIN + 0.5, nb.ROAD_MARGIN + 3.0, dist_e)
    alb_clean = np.clip(alb_clean * sh_in[..., None], 0.0, 1.0)

    return hm_clean_combo, np.clip(alb_clean * 255.0, 0, 255).astype(np.uint8)


def build_floor_relief_z(hm_floor, X, Y, dist_e, dist_v, relief_mm=9.2, apron_mm=13.5):
    """Map clean forest floor (gentle ~1-3mm slope near road) & beveled granite boulders (~8-9mm)."""
    base_n = np.clip(hm_floor, 0.0, 1.0)
    H, W = X.shape
    micro = 0.09 * nb.fft_fbm(W, H, 0.9, 3.2, 2.0 * nb.R, 2.0 * nb.R, beta=1.9, seed=19)
    # Gentle edge clearance curve (0->apron_mm smoothly over 13.5mm so road bank never forms a wall!)
    d_clear = np.minimum(dist_e - nb.ROAD_MARGIN, dist_v - nb.PAD_RADIUS)
    taper = nb.smoothstep(0.45, apron_mm, d_clear)
    rmax = max(float(np.percentile(base_n, 99.8)), 1e-4)
    rel = (relief_mm * np.clip(base_n / rmax, 0.0, 1.08) + micro) * taper
    return (nb.Z_BASE + np.maximum(0.0, rel)).astype(np.float32)


def build_sculpted_conifer_manifold(tx, ty, z_ground, height, r_base, n_tiers=10, seed=101):
    """
    Creates ONE guaranteed watertight (`Error.NoError`, `genus=0`) storybook conifer solid:
    - Sinuous tapered pine trunk anchored 2.6mm into ground + 5-flute root buttress flare
    - Exposed textured brown bark shaft cleanly visible under bottom skirts
    - 9-11 tightly shingled outward-flaring star-lobed bough whorls with crisp horizontal
      shadow band recesses and nuanced botanical evergreen needle palette.
    """
    rng = np.random.RandomState(seed)
    parts = []

    # 1. Curved Trunk + Root Buttress Flare
    n_th_tr = 44
    th_tr = np.linspace(0.0, 2.0 * math.pi, n_th_tr, endpoint=False)
    lean_x = float(rng.uniform(-0.85, 0.85))
    lean_y = float(rng.uniform(-0.85, 0.85))
    z_bot = float(z_ground) - 2.6
    z_top = float(z_ground) + float(height) * 0.92
    n_z = 22
    zs = np.linspace(z_bot, z_top, n_z)

    r_root_base = r_base * 0.17
    r_tip_top = max(0.30, r_base * 0.025)
    ph_root = float(rng.uniform(0.0, 2.0 * math.pi))

    verts_tr = [[tx, ty, z_bot]]
    cols_tr = [[0.22, 0.14, 0.08, 1.0]]
    for z in zs:
        s = (z - z_bot) / (z_top - z_bot)
        cx = tx + lean_x * (s ** 1.35)
        cy = ty + lean_y * (s ** 1.35)
        h_above = max(0.0, z - z_ground)
        flare = 1.0 + 0.78 * np.exp(-((max(0.0, z - (z_ground - 0.5))) / 3.0) ** 2) * (0.62 + 0.38 * np.cos(5 * th_tr + ph_root))
        bark_furrow = 1.0 + 0.07 * np.cos(14 * th_tr + 2.5 * s + ph_root)
        rr = ((1.0 - s) * r_root_base + s * r_tip_top) * flare * bark_furrow
        x = cx + rr * np.cos(th_tr)
        y = cy + rr * np.sin(th_tr)
        verts_tr.extend(np.column_stack([x, y, np.full(n_th_tr, z)]).tolist())
        furrow_tone = 0.78 + 0.25 * np.cos(14 * th_tr + 2.5 * s + ph_root) + 0.10 * min(1.0, h_above / 5.0)
        rc = np.clip(0.34 * furrow_tone, 0.09, 0.62)
        gc = np.clip(0.21 * furrow_tone, 0.05, 0.42)
        bc = np.clip(0.12 * furrow_tone, 0.03, 0.26)
        cols_tr.extend(np.column_stack([rc, gc, bc, np.ones(n_th_tr)]).tolist())
    verts_tr.append([tx + lean_x, ty + lean_y, z_top])
    cols_tr.append([0.28, 0.18, 0.10, 1.0])

    V_tr = np.array(verts_tr, dtype=np.float32)
    C_tr = np.array(cols_tr, dtype=np.float32)
    tris_tr = []
    for j in range(n_th_tr):
        jn = (j + 1) % n_th_tr
        tris_tr.append([0, 1 + jn, 1 + j])
    for k in range(n_z - 1):
        b0 = 1 + k * n_th_tr
        b1 = b0 + n_th_tr
        for j in range(n_th_tr):
            jn = (j + 1) % n_th_tr
            tris_tr.append([b0 + j, b0 + jn, b1 + jn])
            tris_tr.append([b0 + j, b1 + jn, b1 + j])
    tp = len(V_tr) - 1
    tb = 1 + (n_z - 1) * n_th_tr
    for j in range(n_th_tr):
        jn = (j + 1) % n_th_tr
        tris_tr.append([tp, tb + j, tb + jn])
    parts.append(mf.Manifold(mf.Mesh(vert_properties=np.hstack([V_tr, C_tr]).astype(np.float32), tri_verts=np.array(tris_tr, dtype=np.uint32))))

    # Subtle per-tree botanical tone shift (mix warmer spruce & deeper alpine blue-pine)
    warm_shift = float(rng.uniform(-0.025, 0.030))

    # 2. Tightly Shingled Flaring Conifer Bough Whorls
    n_th_b = 84
    th_b = np.linspace(0.0, 2.0 * math.pi, n_th_b, endpoint=False)
    trunk_clear = np.clip(height * 0.115, 2.1, 3.3)
    canopy_base_z = float(z_ground) + trunk_clear
    canopy_h = float(height) - trunk_clear

    for t_idx in range(n_tiers):
        u_t = t_idx / max(n_tiers - 1, 1)
        is_crown = (t_idx == n_tiers - 1)

        # Classic wide-base pine silhouette tapering toward crisp peak
        tier_r = r_base * (1.0 - 0.84 * (u_t ** 0.94))
        tier_r = max(tier_r, 1.45)
        z_skirt = canopy_base_z + canopy_h * (0.0 + 0.84 * u_t)
        h_tier = canopy_h * (0.185 - 0.045 * u_t)
        if is_crown:
            h_tier = canopy_h * 0.185
        # Shallow crisp bough shelf (only ~18-24% outer tip hook instead of folded vertical drapes!)
        droop = min(h_tier * 0.36, 0.16 * tier_r)
        z_hub = z_skirt + droop

        n_boughs = max(6, int(round(12.0 - 5.0 * u_t)))
        phase = t_idx * (math.pi / max(n_boughs, 1)) + float(rng.uniform(-0.22, 0.22))

        asym = (
            1.0
            + 0.10 * np.cos(2 * th_b + rng.uniform(0, 6.28))
            + 0.07 * np.sin(3 * th_b + rng.uniform(0, 6.28))
            + 0.045 * np.cos(5 * th_b + rng.uniform(0, 6.28))
        )
        # Star-pronged needle sprays with crisp jagged twig serrations
        lobe = np.abs(np.cos(0.5 * n_boughs * (th_b - phase))) ** 1.45
        twig = 0.095 * np.cos(2.0 * n_boughs * (th_b - phase) + 0.3)
        sub_twig = 0.04 * np.sin(4.0 * n_boughs * (th_b - phase) + 0.8)
        R_prof = tier_r * np.maximum(0.48, 0.56 + 0.44 * lobe + twig + sub_twig) * asym

        d_asym = 1.0 + 0.16 * np.sin(3 * th_b + rng.uniform(0, 6.28))
        tip_droop = droop * (0.55 + 0.55 * lobe) * d_asym

        s_c = np.clip((z_hub - z_bot) / (z_top - z_bot), 0.0, 1.0)
        cx = tx + lean_x * (s_c ** 1.35)
        cy = ty + lean_y * (s_c ** 1.35)

        if is_crown:
            prof = [
                (0.08, -0.04 * tip_droop, 0.16),
                (0.56, -0.35 * tip_droop, 0.34),
                (1.00, -0.72 * tip_droop + 0.12, 0.78),
                (0.92, -0.42 * tip_droop + 0.40 * h_tier, 0.94),
                (0.56,  0.32 * h_tier + 0.36 * h_tier, 0.88),
                (0.24,  0.80 * h_tier, 0.84),
            ]
            top_pole_z = float(z_hub + 0.98 * h_tier)
        else:
            # Wide flaring shingle profile: shallow dark recessed underside -> hooked outer tip -> gently arched upper bough
            prof = [
                (0.08, -0.03 * tip_droop, 0.10),   # dark recessed trunk shadow collar
                (0.45, -0.18 * tip_droop, 0.16),   # shadowed underside shelf
                (0.86, -0.68 * tip_droop, 0.26),   # outer underside lip
                (1.00, -1.00 * tip_droop + 0.10, 0.72),  # crisp downturned tip bottom
                (0.98, -0.82 * tip_droop + 0.52, 0.95),  # crisp sunlit tip upper edge
                (0.74, -0.22 * tip_droop + 0.50 * h_tier, 0.78),  # upper bough outer shingle
                (0.42,  0.35 * h_tier + 0.52 * h_tier, 0.50),     # upper inner slope
                (0.15,  1.05 * h_tier, 0.30),                     # tucked collar beneath upper whorl
            ]
            top_pole_z = float(z_hub + 1.12 * h_tier)

        verts_m = [[cx, cy, float(z_hub - 0.08 * np.mean(tip_droop))]]
        cols_m = [[0.035, 0.085, 0.035, 1.0]]
        for rf, z_off_arr, bright in prof:
            rr = np.maximum(0.22, rf * R_prof)
            zz = np.broadcast_to(z_hub + z_off_arr, (n_th_b,)).astype(np.float64).copy()
            if bright > 0.50:
                zz = zz + 0.38 * lobe * (1.0 - abs(rf - 0.65))
            x = cx + rr * np.cos(th_b)
            y = cy + rr * np.sin(th_b)
            verts_m.extend(np.column_stack([x, y, zz]).tolist())

            # Rich deep evergreen pine palette matching concept sheet under HDR lighting
            sun = bright * (0.76 + 0.26 * u_t) * (0.80 + 0.24 * lobe)
            r_c = np.clip(0.045 + 0.185 * (sun ** 1.38) + warm_shift, 0.025, 0.30)
            g_c = np.clip(0.095 + 0.285 * (sun ** 1.18) + 0.4 * warm_shift, 0.05, 0.42)
            b_c = np.clip(0.038 + 0.105 * (sun ** 1.25) - 0.3 * warm_shift, 0.025, 0.18)
            cols_m.extend(np.column_stack([r_c, g_c, b_c, np.ones(n_th_b)]).tolist())

        verts_m.append([cx, cy, top_pole_z])
        cols_m.append([0.18, 0.32, 0.13, 1.0] if is_crown else [0.09, 0.19, 0.08, 1.0])

        Vm = np.array(verts_m, dtype=np.float32)
        Cm = np.array(cols_m, dtype=np.float32)
        n_r = len(prof)
        mtris = []
        for j in range(n_th_b):
            jn = (j + 1) % n_th_b
            mtris.append([0, 1 + jn, 1 + j])
        for k in range(n_r - 1):
            b0 = 1 + k * n_th_b
            b1 = b0 + n_th_b
            for j in range(n_th_b):
                jn = (j + 1) % n_th_b
                mtris.append([b0 + j, b0 + jn, b1 + jn])
                mtris.append([b0 + j, b1 + jn, b1 + j])
        tp_m = len(Vm) - 1
        tb_m = 1 + (n_r - 1) * n_th_b
        for j in range(n_th_b):
            jn = (j + 1) % n_th_b
            mtris.append([tp_m, tb_m + j, tb_m + jn])
        parts.append(mf.Manifold(mf.Mesh(vert_properties=np.hstack([Vm, Cm]).astype(np.float32), tri_verts=np.array(mtris, dtype=np.uint32))))

    return mf.Manifold.batch_boolean(parts, mf.OpType.Add)


def render_mesh_6_views_audit(verts, faces, vcols, diff_rgb_uv, out_path, title, z_peak):
    """
    Renders true 3D multi-angle orthographic shaded point-splat/triangle-centroid
    views directly from the final Boolean-unioned geometry so the 6-view sheet
    faithfully shows real 3D trunks, overhanging boughs, boulders, and clearing!
    """
    # Compute face centroids, normals, and RGB color per face for fast crisp software rendering
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    fc = (v0 + v1 + v2) * (1.0 / 3.0)
    nrm = np.cross(v1 - v0, v2 - v0)
    nlen = np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    fn = nrm / nlen

    # Sample color per triangle: tree uses vertex vcols, ground uses UV diff_rgb_uv
    fc_vcol = (vcols[faces[:, 0]] + vcols[faces[:, 1]] + vcols[faces[:, 2]]) * (1.0 / 3.0)
    is_tree = fc_vcol[:, 3] > 0.5
    H_uv, W_uv = diff_rgb_uv.shape[:2]
    # Note: assemble_viewer_v2 uses vUv = (x+R)/(2R), v = 1 - (y+R)/(2R) over flipud texture;
    # Here diff_rgb_uv has row 0 = bottom (-R), row H-1 = top (+R) when unflipped, or check orientation:
    ux = np.clip((fc[:, 0] + nb.R) / (2.0 * nb.R) * (W_uv - 1), 0, W_uv - 1).astype(np.int32)
    vy = np.clip((nb.R - fc[:, 1]) / (2.0 * nb.R) * (H_uv - 1), 0, H_uv - 1).astype(np.int32)
    uv_rgb = diff_rgb_uv[vy, ux].astype(np.float32) / 255.0
    side_dark = np.where((fc[:, 2] < 24.9) | (np.abs(fn[:, 2]) < 0.15) & (~is_tree), 0.14, 1.0)
    fcol = np.where(is_tree[:, None], fc_vcol[:, :3], uv_rgb * np.where(fc[:, 2] < 24.8, 0.20, 1.0)[:, None])
    # Slightly darken vertical hex skirt walls cleanly
    is_skirt = (~is_tree) & (fc[:, 2] < 24.85) & (np.abs(fn[:, 2]) < 0.25)
    fcol[is_skirt] = np.array([0.11, 0.14, 0.18], dtype=np.float32)

    light_dir = np.array([0.52, -0.38, 0.76], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)
    diff_shade = np.clip(np.dot(fn, light_dir), 0.0, 1.0) * 0.78 + 0.28 + 0.12 * np.clip(fn[:, 2], 0.0, 1.0)
    lit_col = np.clip(fcol * diff_shade[:, None], 0.0, 1.0)

    # Sample densely along tall skirt wall triangles + terrain centroids for clean software preview
    wall_mask = is_skirt
    b_samples = [fc, (0.7*v0 + 0.15*v1 + 0.15*v2), (0.15*v0 + 0.7*v1 + 0.15*v2), (0.15*v0 + 0.15*v1 + 0.7*v2)]
    c_samples = [lit_col, lit_col, lit_col, lit_col]
    n_samples = [fn, fn, fn, fn]
    if np.any(wall_mask):
        wv0, wv1, wv2 = v0[wall_mask], v1[wall_mask], v2[wall_mask]
        wc, wn = lit_col[wall_mask], fn[wall_mask]
        for a_u in np.linspace(0.08, 0.92, 8):
            for a_v in (0.15, 0.50, 0.85):
                u = a_u; v = (1.0 - a_u) * a_v; w = 1.0 - u - v
                b_samples.append(u * wv0 + v * wv1 + w * wv2)
                c_samples.append(wc)
                n_samples.append(wn)
    pts_all = np.vstack(b_samples)
    col_all = np.vstack(c_samples)
    nrm_all = np.vstack(n_samples)

    pw, ph = 580, 460
    sheet = Image.new("RGB", (pw * 3, ph * 2 + 54), (229, 232, 236))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 19)
    except Exception:
        font = ImageFont.load_default()

    cx_c, cy_c, cz_c = 0.0, 0.0, 18.0
    elev = math.radians(33.0)
    for k, deg in enumerate([0, 60, 120, 180, 240, 300]):
        az = math.radians(deg)
        c_az, s_az = math.cos(az), math.sin(az)
        c_el, s_el = math.cos(elev), math.sin(elev)

        px = pts_all[:, 0] - cx_c
        py = pts_all[:, 1] - cy_c
        pz = pts_all[:, 2] - cz_c
        rx = px * c_az + py * s_az
        ry = -px * s_az + py * c_az
        cam_u = rx
        cam_v = ry * s_el + pz * c_el
        depth = -ry * c_el + pz * s_el

        # Backface cull along camera look vector
        look_w = np.array([s_az * c_el, -c_az * c_el, s_el], dtype=np.float32)
        vis = np.dot(nrm_all, look_w) > -0.15
        uu = cam_u[vis]; vv = cam_v[vis]; dd = depth[vis]; cc = col_all[vis]

        scale = 3.55
        ix = np.round(pw * 0.5 + uu * scale).astype(np.int32)
        iy = np.round(ph * 0.58 - vv * scale).astype(np.int32)
        inb = (ix >= 1) & (ix < pw - 1) & (iy >= 1) & (iy < ph - 1)
        ix, iy, dd, cc = ix[inb], iy[inb], dd[inb], cc[inb]

        ord_d = np.argsort(dd)  # back-to-front painter's sort
        ix, iy, cc = ix[ord_d], iy[ord_d], (cc[ord_d] * 255.0).astype(np.uint8)
        tile = np.full((ph, pw, 3), [229, 232, 236], dtype=np.uint8)
        for dx_p in (0, 1, -1):
            for dy_p in (0, 1, -1):
                tile[iy + dy_p, ix + dx_p] = cc

        col_idx, row_idx = k % 3, k // 3
        sheet.paste(Image.fromarray(tile), (col_idx * pw, row_idx * ph))
        draw.text((col_idx * pw + 18, row_idx * ph + 14), f"Vista {deg} deg (3D Pines)", fill=(35, 42, 50), font=font)

    draw.text((18, ph * 2 + 12), f"{title} — 15 Pinos 3D Reales Escalonados (Z_PEAK={z_peak:.1f}mm)", fill=(22, 28, 36), font=font)
    sheet.save(out_path)


def main():
    t_start = time.time()
    verify_cad_clearances()

    out_dir = os.path.join(REPO_ROOT, "Tableros-Catan", "pipeline-biomas", "bosque")
    os.makedirs(out_dir, exist_ok=True)
    viewer_html = os.path.join(REPO_ROOT, "Visores-HTML", "catan-bioma-bosque.html")

    hm_raw = np.asarray(Image.open(HEIGHTMAP_SRC).convert("L"))
    alb_raw = np.asarray(Image.open(ALBEDO_SRC).convert("RGB"))

    # 1. Compute tree-free clean height & albedo cleanly at 640x640 then sample 2048x2048 PBR
    grid_n = 640
    xs_g = np.linspace(-nb.R, nb.R, grid_n)
    Xg, Yg = np.meshgrid(xs_g, xs_g)
    dist_g, dist_vg = nb.hex_distances(Xg, Yg)
    hm_floor_g, alb_clean_g = extract_clean_floor_hm_and_albedo(hm_raw, alb_raw, Xg, Yg, dist_g, dist_vg)

    tex_size = 2048
    xs_t = np.linspace(-nb.R, nb.R, tex_size)
    Xt, Yt = np.meshgrid(xs_t, xs_t)
    dist_t, dist_vt = nb.hex_distances(Xt, Yt)
    dxt = float(xs_t[1] - xs_t[0])

    hm_floor_t = resample_canonical_xy(hm_floor_g, Xt, Yt, lim=nb.R)
    alb_clean_u8 = np.clip(resample_canonical_xy(alb_clean_g, Xt, Yt, lim=nb.R), 0, 255).astype(np.uint8)
    Zt = build_floor_relief_z(hm_floor_t, Xt, Yt, dist_t, dist_vt, relief_mm=9.2, apron_mm=13.5)
    diff_rgb, _ = bcb.refine_biome_albedo(alb_clean_u8, Zt, dist_t, dist_vt, dxt, dxt, snow_filter=False)
    norm_u8, rough_u8 = bcb.compute_enhanced_pbr_maps(Zt, diff_rgb, dxt, dxt, micro_norm_strength=0.34)

    nb.save_uv_aligned_png(diff_rgb, os.path.join(out_dir, "catan_mountain_diffuse_2k.png"))
    nb.save_uv_aligned_png(norm_u8, os.path.join(out_dir, "catan_mountain_normal_2k.png"))
    nb.save_uv_aligned_png(rough_u8, os.path.join(out_dir, "catan_mountain_roughness_2k.png"))

    clay_rgb = nb.render_shaded_clay_rgb(Zt, dist_t, dxt, dxt)
    nb.save_uv_aligned_png(clay_rgb, os.path.join(out_dir, "bosque_clay_overhead.png"))

    # 2. High-res Contour-Aligned Hex Forest Floor Solid (with Beveled Granite Boulders)
    nx = 440
    ny = int(round(nx * (nb.R / nb.APOTHEM)))
    xs_m = np.linspace(-nb.APOTHEM, nb.APOTHEM, nx)
    ys_m = np.linspace(-nb.R, nb.R, ny)
    Xm, Ym = np.meshgrid(xs_m, ys_m)
    de_m, dv_m = nb.hex_distances(Xm, Ym)
    hm_floor_m = resample_canonical_xy(hm_floor_g, Xm, Ym, lim=nb.R)
    Zm = build_floor_relief_z(hm_floor_m, Xm, Ym, de_m, dv_m, relief_mm=9.2, apron_mm=13.5)

    Xw, Yw = tmu.warp_grid_to_slopes(Xm, Ym, Zm, de_m, dv_m, iters=3, strength=0.35)
    de_w, dv_w = nb.hex_distances(Xw, Yw)
    hm_floor_w = resample_canonical_xy(hm_floor_g, Xw, Yw, lim=nb.R)
    Zw = build_floor_relief_z(hm_floor_w, Xw, Yw, de_w, dv_w, relief_mm=9.2, apron_mm=13.5)
    Zw = tmu.cap_extreme_slopes(Zw, de_w, min_step_dx=float(xs_m[1] - xs_m[0]), max_grad=1.05, passes=3)

    g_verts, g_faces, n_flips = tmu.build_contour_aligned_hex_solid(
        Xw, Yw, Zw, de_w, n_bnd=168, n_mag=32, include_magnet=True, flip_passes=4
    )
    g_wt = nb.verify_watertight(g_verts, g_faces)
    if not g_wt["watertight"]:
        raise RuntimeError(f"Forest floor not watertight: {g_wt}")
    print(f"[Floor OK] V={g_wt['V']} F={g_wt['F']} flips={n_flips} chi={g_wt['chi']}")

    # Attach 4-col RGBA property (alpha=0.0 => use 2K UV + triplanar ground shader)
    g_vcol = np.tile(np.array([[0.22, 0.27, 0.13, 0.0]], dtype=np.float32), (len(g_verts), 1))
    mf_ground = mf.Manifold(
        mf.Mesh(vert_properties=np.hstack([g_verts, g_vcol]).astype(np.float32), tri_verts=g_faces.astype(np.uint32))
    )

    # 3. Sculpt all 15 3D Multi-Tiered Conifer Pine Manifolds Anchored on Local Z_floor
    tree_mfs = []
    max_z = float(np.max(Zw))
    for i, (tx, ty, th, tr, tiers, seed) in enumerate(PINE_SPECS, 1):
        z_local = float(resample_canonical_xy(Zt, np.array([[tx]]), np.array([[ty]]), lim=nb.R)[0, 0])
        m_tree = build_sculpted_conifer_manifold(tx, ty, z_local, th, tr, n_tiers=tiers, seed=seed)
        if m_tree.status() != mf.Error.NoError or m_tree.genus() != 0:
            raise RuntimeError(f"Tree {i} invalid manifold: status={m_tree.status()}, genus={m_tree.genus()}")
        tree_mfs.append(m_tree)
        max_z = max(max_z, z_local + th)

    print(f"[Trees OK] Sculpted {len(tree_mfs)} pines (max expected Z={max_z:.2f}mm). Running batch CSG boolean...")
    t_bool = time.time()
    res = mf.Manifold.batch_boolean([mf_ground] + tree_mfs, mf.OpType.Add)
    if res.status() != mf.Error.NoError:
        raise RuntimeError(f"CSG Boolean failed: status={res.status()}, genus={res.genus()}")
    res = res.simplify(0.015)
    out_m = res.to_mesh()
    vp = np.asarray(out_m.vert_properties, dtype=np.float32)
    verts = np.ascontiguousarray(vp[:, :3], dtype=np.float32)
    vcols = np.ascontiguousarray(vp[:, 3:7], dtype=np.float32)
    faces = np.ascontiguousarray(out_m.tri_verts, dtype=np.int32)

    z_peak = float(np.max(verts[:, 2]))
    vol_cm3 = float(res.volume() / 1000.0)
    print(
        f"[CSG OK] dt={time.time()-t_bool:.2f}s | status={res.status()} genus={res.genus()} "
        f"V={len(verts)} F={len(faces)} vol={vol_cm3:.2f}cm3 Z_PEAK={z_peak:.2f}mm"
    )

    # 4. Save Binary STL + Compressed WebGL Payload (including vcols!)
    stl_path = os.path.join(out_dir, "catan_bosque.stl")
    nb.write_binary_stl(verts, faces, stl_path, label="Catan bosque 3D Pines")
    import trimesh
    tm_stl = trimesh.load(stl_path, force="mesh")
    print(f"[STL Slicer Check] watertight={tm_stl.is_watertight} euler={tm_stl.euler_number} vol={tm_stl.volume/1000.0:.2f}cm3")
    np.savez_compressed(os.path.join(out_dir, "viewer_mesh_v2.npz"), verts=verts, faces=faces, vcols=vcols)

    # 5. Render Honest 3D 6-View Audit directly from CSG Solid
    audit_path = os.path.join(out_dir, "bosque_6views_audit.png")
    render_mesh_6_views_audit(verts, faces, vcols, np.flipud(diff_rgb), audit_path, "Bioma 2 - Bosque", z_peak)

    metrics = {
        "master": {
            "V": int(len(verts)),
            "E": int(len(faces) * 3 // 2),
            "F": int(len(faces)),
            "chi": 2 - 2 * int(res.genus()),
            "genus": int(res.genus()),
            "watertight": True,
            "volume_cm3": vol_cm3,
        },
        "coverage": {"max_snow_steep": 0.0, "trees_3d": len(tree_mfs)},
        "viewer": {"V": int(len(verts)), "F": int(len(faces))},
    }
    with open(os.path.join(out_dir, "master_metrics_v2.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 6. Assemble WebGL Viewer with TRIPLANAR_MODE = "forest"
    av2.OUT_DIR = out_dir
    av2.SCRATCH_DIR = out_dir
    av2.TARGET_PATHS = [viewer_html]
    av2.TRIPLANAR_MODE = "forest"
    av2.build_viewer()

    title_str = "🌲 Bioma 2 — Bosque de Pinos 3D Catan"
    relief_actual = z_peak - nb.Z_BASE
    with open(viewer_html, "r", encoding="utf-8") as hf:
        html = hf.read()
    html = html.replace("🏔️ Cordillera Montaña Catan 3D · V3 PBR", f"{title_str} · V3 PBR")
    html = html.replace("Cordillera Montaña Catan 3D (V3)", title_str)
    html = html.replace("110.85×128×83", f"110.85×128×{z_peak:.0f}")
    html = html.replace(">83.0 mm<", f">{z_peak:.1f} mm<")
    html = html.replace('max="83.5" step="0.5" value="83.5"', f'max="{z_peak + 0.5:.1f}" step="0.5" value="{z_peak + 0.5:.1f}"')
    html = html.replace("Z=83mm", f"Z={z_peak:.0f}mm")
    html = html.replace("<span>Nieve en ≥44°</span><strong>0.000000 (0%)</strong>", f"<span>Relieve Máx</span><strong>+{relief_actual:.1f} mm (15 pinos 3D)</strong>")
    with open(viewer_html, "w", encoding="utf-8") as hf:
        hf.write(html)

    print(f"[BOSQUE COMPLETE] total={time.time()-t_start:.2f}s -> {viewer_html}")


if __name__ == "__main__":
    main()
