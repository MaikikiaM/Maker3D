"""
Unified Pure NumPy/SciPy Catan Biome Pipeline (Nano Banana 6-View 360° -> Watertight CAD + WebGL PBR).
Reuses nanobanana_to_catan_hex.py and assemble_viewer_v2.py with zero HTML duplication.
"""
import argparse
import json
import math
import os
import sys
import numpy as np
import scipy.ndimage as ndi
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SKILL_SCRIPTS = os.path.join(REPO_ROOT, ".agents", "skills", "nanobanana-3d-sculpt", "scripts")
MONTANA_PIPE = os.path.join(REPO_ROOT, "Tableros-Catan", "pipeline-montana")
BIOMAS_PIPE = os.path.dirname(__file__)
sys.path.insert(0, SKILL_SCRIPTS)
sys.path.insert(0, MONTANA_PIPE)
sys.path.insert(0, BIOMAS_PIPE)

import nanobanana_to_catan_hex as nb
import assemble_viewer_v2 as av2
import prototype_v3_cordillera as pv3
import terrain_mesh_utils as tmu


def refine_heightmap(hm_u8, gamma=0.80, blur_px=1.0, ridge_clean=0.0):
    """Smooth JPEG step quantization, optionally bridge noisy micro-crevasses, and shape slopes."""
    norm = hm_u8.astype(np.float32) / 255.0
    if ridge_clean > 0.0:
        # Bridge thin dark crevasse noise in diffusion heightmaps so peaks become broad alpine facets
        k = 9 if ridge_clean >= 0.70 else 5
        closed = ndi.grey_closing(norm, size=k)
        norm = (1.0 - ridge_clean) * norm + ridge_clean * ndi.gaussian_filter(closed, sigma=2.2)
    if blur_px > 0.0:
        norm = ndi.gaussian_filter(norm, sigma=blur_px)
    opened = np.power(np.clip(norm, 0.0, 1.0), gamma)
    # Keep float32 precision in [0..255] so gentle slopes never exhibit 8-bit stair-stepping rings
    return np.clip(opened * 255.0, 0.0, 255.0).astype(np.float32)


def refine_biome_albedo(diffuse_rgb, Z, dist_edge, dist_vert, dx, dy, snow_filter=False, snowline_mm=15.0, keep_dark_rock=False, photoreal=False):
    """Enforce cavity readability, zero perimeter-skirt creep, clean slate collar, and optional physical snow."""
    rgb = diffuse_rgb.astype(np.float32) / 255.0
    gy, gx = np.gradient(ndi.gaussian_filter(Z, 0.35 / dx), dy, dx)
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
    relief = Z - nb.Z_BASE

    cmax = np.max(rgb, axis=-1)
    cmin = np.min(rgb, axis=-1)
    lum = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    max_snow_steep = 0.0

    if snow_filter:
        sat = cmax - cmin
        snow_prob = nb.smoothstep(0.46, 0.72, lum) * (1.0 - nb.smoothstep(0.07, 0.20, sat))
        snow_allowed = (1.0 - nb.smoothstep(38.0, 44.0, slope_deg)) * nb.smoothstep(snowline_mm - 3.0, snowline_mm + 4.0, relief)
        forbidden = np.clip(ndi.gaussian_filter(snow_prob * (1.0 - snow_allowed), sigma=2.2) * 1.15, 0.0, 1.0)
        if np.any(forbidden > 0.01):
            in_terrain = ((dist_edge > nb.ROAD_MARGIN + 1.0) & (dist_vert > nb.PAD_RADIUS + 1.0)).astype(np.float32)
            rock_weight = in_terrain * np.maximum(0.0, 1.0 - snow_prob * 1.4)
            w_sum = ndi.gaussian_filter(rock_weight, sigma=24.0) + 1e-4
            rock_fill = np.stack([ndi.gaussian_filter(rgb[:, :, c] * rock_weight, sigma=24.0) / w_sum for c in range(3)], axis=-1)
            H, W = Z.shape
            g1 = nb.fft_fbm(W, H, 1.4, 5.5, 2.0 * nb.R, 2.0 * nb.R, beta=2.0, seed=71)
            g2 = nb.fft_fbm(W, H, 0.45, 1.4, 2.0 * nb.R, 2.0 * nb.R, beta=1.8, seed=72)
            cracks = 0.65 * (1.0 - np.abs(g1)) + 0.35 * g2
            slope_shade = 0.88 + 0.18 * (1.0 - nb.smoothstep(20.0, 70.0, slope_deg))
            clean_rock = np.clip(rock_fill * (0.94 + 0.16 * cracks)[..., None] * slope_shade[..., None], 0.18, 0.68)
            rgb = rgb * (1.0 - forbidden[..., None]) + clean_rock * forbidden[..., None]

        cmax2 = np.max(rgb, axis=-1)
        lum2 = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
        rem_snow = (lum2 > 0.68) & ((cmax2 - np.min(rgb, axis=-1)) < 0.12) & (slope_deg >= 44.0)
        max_snow_steep = float(np.mean(rem_snow)) * 100.0
    else:
        lum2 = lum

    # Subtle geometric cavity AO + shadow recovery inside terrain (zero edge darkening)
    z_smooth = ndi.gaussian_filter(Z, sigma=max(1.4 / dx, 2.0))
    cavity = np.clip((Z - z_smooth) * (0.028 if photoreal else 0.055), -0.05 if photoreal else -0.14, 0.06 if photoreal else 0.05)
    terrain_core = nb.smoothstep(nb.ROAD_MARGIN + 0.25, nb.ROAD_MARGIN + 1.5, dist_edge) * nb.smoothstep(nb.PAD_RADIUS + 0.25, nb.PAD_RADIUS + 1.5, dist_vert)
    rgb = np.clip(rgb * (1.0 + cavity * terrain_core)[..., None], 0.0, 1.0)
    dark_t = (terrain_core > 0.5) & (lum2 < 0.22)
    rgb = np.clip(rgb * np.where(dark_t, 1.15 if photoreal else 1.12, 1.0)[..., None], 0.0, 1.0)

    # Remove accidental grey slate road-border bleed climbing up outer sculpted banks (ONLY on non-basalt biomes!)
    if not keep_dark_rock:
        sat_pre = np.max(rgb, axis=-1) - np.min(rgb, axis=-1)
        near_rim = (dist_edge < nb.ROAD_MARGIN + 5.8) | (dist_vert < nb.PAD_RADIUS + 5.8)
        in_sculpt = (dist_edge > nb.ROAD_MARGIN + 0.15) & (dist_vert > nb.PAD_RADIUS + 0.15) & ((relief > 0.65) | (slope_deg > 16.0))
        bad_slate = in_sculpt & near_rim & (sat_pre < 0.105) & (lum2 < 0.44)
        if np.any(bad_slate):
            good_src = ((dist_edge > nb.ROAD_MARGIN + 2.2) & (dist_vert > nb.PAD_RADIUS + 2.2) & (sat_pre >= 0.11) & (relief > 0.65)).astype(np.float32)
            gw = ndi.gaussian_filter(good_src, sigma=max(4.5 / dx, 4.0)) + 1e-5
            warm_fill = np.stack([ndi.gaussian_filter(rgb[:, :, c] * good_src, sigma=max(4.5 / dx, 4.0)) / gw for c in range(3)], axis=-1)
            slate_m = np.clip(ndi.gaussian_filter(bad_slate.astype(np.float32), sigma=max(0.8 / dx, 1.5)) * 1.25, 0.0, 1.0)[..., None]
            rgb = rgb * (1.0 - slate_m) + warm_fill * slate_m

    # Strictly clamp architectural dark slate collar across the entire flat hex bezel & settlement pads (never climb raised terrain!)
    on_border = np.maximum(
        1.0 - nb.smoothstep(nb.ROAD_MARGIN + 0.40, nb.ROAD_MARGIN + 5.50, dist_edge),
        1.0 - nb.smoothstep(nb.PAD_RADIUS + 0.40, nb.PAD_RADIUS + 4.50, dist_vert),
    )
    flat_mask = 1.0 - nb.smoothstep(0.12, 0.68, relief)
    hard_rim = np.maximum(
        1.0 - nb.smoothstep(nb.ROAD_MARGIN - 0.15, nb.ROAD_MARGIN + 0.35, dist_edge),
        1.0 - nb.smoothstep(nb.PAD_RADIUS - 0.15, nb.PAD_RADIUS + 0.40, dist_vert),
    ) * (1.0 - nb.smoothstep(0.15, 0.85, relief))
    collar = np.clip(np.maximum(on_border * flat_mask, hard_rim), 0.0, 1.0)[..., None]
    slate = np.array([0.21, 0.23, 0.26], dtype=np.float32)
    rgb = rgb * (1.0 - collar) + slate * collar
    return np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8), max_snow_steep


def sculpt_albedo_microrelief(Z, diff_rgb, dist_edge, dist_vert, dx, max_bump_mm=0.95, crown_smooth_mm=0.0, water_mask=None):
    """Carve physical 3D stone domes, mineral cracks, crystalline gold outcrops & wave ribs from high-res albedo."""
    rgb_f = diff_rgb.astype(np.float32) / 255.0
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b

    is_water = np.clip((g - r - 0.025) * 4.5, 0.0, 1.0) * np.clip((b - r - 0.02) * 4.0, 0.0, 1.0)
    if water_mask is not None:
        is_water = np.maximum(is_water, np.clip(ndi.gaussian_filter(water_mask.astype(np.float32), sigma=1.5), 0.0, 1.0))
    is_foam = is_water * nb.smoothstep(0.50, 0.85, lum)
    is_gold = nb.smoothstep(0.18, 0.42, (r + g) * 0.5 - b) * nb.smoothstep(0.45, 0.78, lum)

    # Local-contrast normalized luminance so dark volcanic basalt cracks/pores carve at full physical depth matching sandstone
    s_norm = max(8.0, 2.2 / dx)
    l_local = np.maximum(ndi.gaussian_filter(lum, s_norm), 0.14)
    lum_eq = np.clip((lum / l_local) * 0.55, 0.0, 1.35)
    lum_sculpt = 0.45 * lum + 0.55 * lum_eq

    s_fine = max(0.85, 0.055 / dx if crown_smooth_mm >= 0.13 else 0.10 / dx)
    s_meso = max(3.5, 0.65 / dx)
    s_broad = max(8.0, 1.65 / dx)
    l_fine = ndi.gaussian_filter(lum_sculpt, s_fine)
    l_meso = ndi.gaussian_filter(lum_sculpt, s_meso)
    l_broad = ndi.gaussian_filter(lum_sculpt, s_broad)

    # Raw-lum scales for bright gold/cobbles + normalized cracks for dark basalt fractures/vesicles
    l_fine_raw = ndi.gaussian_filter(lum, s_fine)
    l_meso_raw = ndi.gaussian_filter(lum, s_meso)
    l_broad_raw = ndi.gaussian_filter(lum, s_broad)

    stone_domes = np.clip((l_fine - l_meso) * 2.75 + (l_meso - l_broad) * 1.15, -0.65, 0.85)
    crevice_groove = np.clip((l_meso - l_fine) - 0.04, 0.0, 0.55) * 1.80
    gold_facets = is_gold * nb.smoothstep(0.38, 0.82, lum) * (0.70 + 0.75 * np.clip(l_fine_raw - l_meso_raw + 0.25, 0.0, 0.75))

    if crown_smooth_mm >= 0.13:
        # Polish peak profiles: calm choppy micro-warts along steep canyon ridge lips, round stone crowns
        # into convex river-cobble domes (zero 80° shark teeth) while keeping rock crevices razor-deep.
        gy_m, gx_m = np.gradient(ndi.gaussian_filter(Z, max(2.0, 0.35 / dx)), dx, dx)
        macro_slope = np.hypot(gx_m, gy_m)
        crest_calm = 1.0 - 0.50 * nb.smoothstep(0.45, 1.05, macro_slope)
        wall_z_calm = 1.0 - 0.82 * nb.smoothstep(0.68, 1.32, macro_slope)

        # Round positive cobble/stone crowns into water-worn pillow domes while keeping cracks sharp
        l_dome = ndi.gaussian_filter(lum_sculpt, max(1.4, 0.14 / dx))
        dome_raw = np.clip((l_dome - l_meso) * 2.35 + (l_meso - l_broad) * 1.05, 0.0, 0.90)
        pos_d = np.power(dome_raw * crest_calm * wall_z_calm + 1e-6, 0.68)
        domed_pos = 0.62 * np.tanh(pos_d / 0.52)
        neg_d = np.minimum(0.0, stone_domes) * wall_z_calm
        rock_pos = max_bump_mm * (0.64 * domed_pos + 0.32 * gold_facets * (0.65 + 0.35 * crest_calm) * wall_z_calm)
        max_pos = 0.56 * max_bump_mm
        rock_pos = max_pos * np.tanh(rock_pos / max_pos)
        rock_pos = ndi.gaussian_filter(rock_pos, max(1.1, 0.13 / dx))
        rock_neg = max_bump_mm * (0.95 * neg_d - 0.84 * crevice_groove * wall_z_calm)
        rock_relief = rock_pos + rock_neg
    else:
        if crown_smooth_mm > 0.0:
            stone_domes = ndi.gaussian_filter(stone_domes, max(1.0, crown_smooth_mm / dx))
            stone_domes = np.sign(stone_domes) * np.power(np.abs(stone_domes) / 0.80 + 1e-6, 0.82) * 0.80
        rock_relief = max_bump_mm * (
            1.05 * stone_domes
            - 0.82 * crevice_groove
            + 0.65 * gold_facets
        )
    if water_mask is not None:
        # Sculpt BOTH submerged river stones/boulders AND zero-mean downstream waterfall foam streaks inside water
        # (never blank flat water, and zero upward transverse blocking walls!)
        sub_hp = np.clip((l_fine_raw - l_meso_raw) * 3.4 + (l_meso_raw - l_broad_raw) * 0.9, -0.18, 0.65)
        submerged_stones = np.maximum(0.0, sub_hp) * (1.0 - is_foam) * max_bump_mm * 0.62
        # High-frequency zero-mean cascade foam ribs elongated vertically along stream flow (no transverse block)
        foam_hp = l_fine_raw - ndi.gaussian_filter(lum, (max(2.2, 0.55 / dx), max(1.6, 0.30 / dx)))
        foam_ribs = np.clip(foam_hp * 2.6, -0.38, 0.48) * is_foam * max_bump_mm * 0.46
        water_relief = submerged_stones + foam_ribs
    else:
        water_relief = max_bump_mm * 0.34 * is_foam

    bump = (1.0 - is_water) * rock_relief + is_water * water_relief
    if crown_smooth_mm > 0.0:
        sig_post = min(0.042, crown_smooth_mm * 0.35) if crown_smooth_mm >= 0.13 else (crown_smooth_mm * 0.75)
        bump = ndi.gaussian_filter(bump, max(0.85, sig_post / dx))
    core = nb.smoothstep(nb.ROAD_MARGIN + 0.65, nb.ROAD_MARGIN + 2.8, dist_edge) * nb.smoothstep(nb.PAD_RADIUS + 0.65, nb.PAD_RADIUS + 2.8, dist_vert)
    return (Z + bump * core).astype(np.float32)


def deink_mineral_albedo(diff_rgb, dist_edge, dist_vert, dx):
    """Soften flat 2D comic ink outlines into natural mineral crevices & add fine geological stone grain."""
    rgb_f = diff_rgb.astype(np.float32) / 255.0
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    H, W = lum.shape

    is_water = np.clip((g - r - 0.025) * 4.5, 0.0, 1.0) * np.clip((b - r - 0.02) * 4.0, 0.0, 1.0)
    is_gold = nb.smoothstep(0.22, 0.46, (r + g) * 0.5 - b) * nb.smoothstep(0.48, 0.80, lum)
    core = nb.smoothstep(nb.ROAD_MARGIN + 0.65, nb.ROAD_MARGIN + 2.8, dist_edge) * nb.smoothstep(nb.PAD_RADIUS + 0.65, nb.PAD_RADIUS + 2.8, dist_vert)

    # Local bright neighborhood (dilated sandstone color around ink lines)
    s_ctx = max(4.0, 0.75 / dx)
    ctx_rgb = np.stack([ndi.gaussian_filter(rgb_f[..., c], s_ctx) for c in range(3)], axis=-1)
    ctx_lum = ndi.gaussian_filter(lum, s_ctx)

    # Detect sharp black/dark-umber comic outlines relative to bright neighbors
    ink_mask = np.clip((ctx_lum - lum - 0.10) * 3.8, 0.0, 1.0) * (1.0 - nb.smoothstep(0.10, 0.34, lum))
    ink_blend = (ink_mask * (1.0 - is_water) * core * 0.58)[..., None]
    # Replace dead black ink with rich warm deep mineral umber (0.38x surrounding rock) so real 3D shadows dominate
    deinked = rgb_f * (1.0 - ink_blend) + (ctx_rgb * 0.42) * ink_blend

    # Tactile porous geological stone micro-grain on sandstone/basalt (absent on calm water & mirror gold)
    grain1 = nb.fft_fbm(W, H, 0.16, 0.60, 2.0 * nb.R, 2.0 * nb.R, beta=1.9, seed=711)
    grain2 = nb.fft_fbm(W, H, 0.05, 0.18, 2.0 * nb.R, 2.0 * nb.R, beta=1.8, seed=712)
    rock_mask = (1.0 - is_water) * (1.0 - 0.65 * is_gold) * core
    deinked = np.clip(deinked * (1.0 + rock_mask[..., None] * (0.085 * grain1 + 0.055 * grain2)[..., None]), 0.0, 1.0)
    return np.clip(deinked * 255.0, 0, 255).astype(np.uint8)


def compute_enhanced_pbr_maps(Z, diff_rgb, dx, dy, micro_norm_strength=0.32, tangent_residual=False, photoreal=False, water_mask=None):
    """Generate tangent-space normal map (either macro+micro or pure high-freq residual) and material roughness."""
    gy, gx = np.gradient(Z, dy, dx)
    rgb_f = diff_rgb.astype(np.float32) / 255.0
    lum = 0.2126 * rgb_f[:, :, 0] + 0.7152 * rgb_f[:, :, 1] + 0.0722 * rgb_f[:, :, 2]

    if tangent_residual:
        # Pure tangent-space micro-grain/crevice detail centered at (128,128,255); avoids double-tilting Z normals
        hp_sigma = max(1.2, 0.09 / dx) if photoreal else max(1.8, 0.14 / dx)
        hp_gain = 1.85 if photoreal else 1.4
        hp_lum = lum - ndi.gaussian_filter(lum, sigma=hp_sigma)
        if water_mask is not None:
            hp_lum = hp_lum * (1.0 - 0.58 * np.clip(water_mask, 0.0, 1.0))
        z_res = Z - ndi.gaussian_filter(Z, sigma=max(1.6, 0.18 / dx))
        lgy, lgx = np.gradient(hp_lum * (max(micro_norm_strength, 0.1) * hp_gain), dy, dx)
        zgy, zgx = np.gradient(z_res * (0.14 if photoreal else 0.12), dy, dx)
        gx_tot = np.clip(lgx + zgx, -1.15 if photoreal else -1.05, 1.15 if photoreal else 1.05)
        gy_tot = np.clip(lgy + zgy, -1.15 if photoreal else -1.05, 1.15 if photoreal else 1.05)
    elif micro_norm_strength > 0.0:
        hp = lum - ndi.gaussian_filter(lum, sigma=2.2)
        mgy, mgx = np.gradient(hp * (micro_norm_strength * 3.5), dy, dx)
        gx_tot = gx + mgx
        gy_tot = gy + mgy
    else:
        gx_tot, gy_tot = gx, gy

    inv_len = 1.0 / np.sqrt(gx_tot * gx_tot + gy_tot * gy_tot + 1.0)
    n_x = (-gx_tot * inv_len * 0.5 + 0.5) * 255.0
    n_y = (-gy_tot * inv_len * 0.5 + 0.5) * 255.0
    n_z = (inv_len * 0.5 + 0.5) * 255.0
    normal_u8 = np.clip(np.stack([n_x, n_y, n_z], axis=-1), 0, 255).astype(np.uint8)

    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
    sat = np.max(rgb_f, axis=-1) - np.min(rgb_f, axis=-1)
    if photoreal:
        # Natural sunlit mineral response: crisp specular highlights on quartzite/wet pebbles (~0.40..0.68), zero chalkiness
        rough = 0.62 + 0.10 * nb.smoothstep(24.0, 58.0, slope_deg) + 0.04 * nb.smoothstep(0.15, 0.45, sat) - 0.22 * nb.smoothstep(0.62, 0.90, lum)
        r_u8 = np.clip(rough * 255.0, 65, 225).astype(np.uint8)
    else:
        # Matte foliage/soil (~0.84), rough sheer rock (~0.90), crisp sunlit limestone/snow/mineral crests (~0.56..0.68)
        rough = 0.80 + 0.11 * nb.smoothstep(24.0, 58.0, slope_deg) + 0.06 * nb.smoothstep(0.15, 0.45, sat) - 0.24 * nb.smoothstep(0.68, 0.94, lum)
        r_u8 = np.clip(rough * 255.0, 85, 245).astype(np.uint8)
    rough_u8 = np.stack([r_u8, r_u8, r_u8], axis=-1)
    return normal_u8, rough_u8


def synthesize_alpine_v3_albedo(X, Y, Z, dist_edge, dist_vert, dx, dy, relief_mm):
    """Warm alpine limestone/granite, emerald lower foothills, ochre scree gully, and >=44 deg snow-free caps matching V3 concept."""
    H, W = Z.shape
    rel = np.maximum(0.0, Z - nb.Z_BASE)
    gy, gx = np.gradient(ndi.gaussian_filter(Z, 0.40 / dx), dy, dx)
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
    z_smooth = ndi.gaussian_filter(Z, sigma=max(2.2 / dx, 3.0))
    curv = Z - z_smooth

    f_macro = nb.fft_fbm(W, H, 5.0, 18.0, 2.0 * nb.R, 2.0 * nb.R, beta=2.1, seed=301)
    f_rock = nb.fft_fbm(W, H, 1.2, 4.5, 2.0 * nb.R, 2.0 * nb.R, beta=2.0, seed=302)
    f_fine = nb.fft_fbm(W, H, 0.35, 1.3, 2.0 * nb.R, 2.0 * nb.R, beta=1.8, seed=303)
    cracks = 0.60 * (1.0 - np.abs(f_rock)) + 0.40 * f_fine

    # Warm sunlit sandstone/granite base (#8d7d68 -> #695b49 in fractures)
    col_hi = np.array([0.59, 0.52, 0.43], dtype=np.float32)
    col_lo = np.array([0.33, 0.29, 0.24], dtype=np.float32)
    rock_t = np.clip(0.48 + 0.24 * f_macro + 0.22 * cracks - 0.16 * nb.smoothstep(42.0, 72.0, slope_deg), 0.0, 1.0)[..., None]
    rgb = col_lo * (1.0 - rock_t) + col_hi * rock_t

    # Warm alpine valley moss along lower foothills & gentle collars (slope < 38 deg, low rel)
    in_area = nb.smoothstep(nb.ROAD_MARGIN + 0.2, nb.ROAD_MARGIN + 2.2, dist_edge) * nb.smoothstep(nb.PAD_RADIUS + 0.2, nb.PAD_RADIUS + 2.2, dist_vert)
    meadow_mask = (
        (1.0 - nb.smoothstep(4.5, relief_mm * 0.40, rel))
        * (1.0 - nb.smoothstep(26.0, 42.0, slope_deg))
        * np.clip(0.75 + 0.30 * f_macro, 0.2, 1.0)
        * in_area
    )[..., None]
    moss_c = (np.array([0.33, 0.44, 0.20], dtype=np.float32) * (0.88 + 0.24 * f_fine[..., None]))
    rgb = rgb * (1.0 - meadow_mask * 0.85) + moss_c * (meadow_mask * 0.85)

    # Warm copper/ochre scree in concave central valley floors / gullies
    gully_mask = (
        nb.smoothstep(-0.3, -1.8, curv)
        * nb.smoothstep(2.0, 8.0, rel)
        * (1.0 - nb.smoothstep(12.0, 26.0, rel))
        * (1.0 - nb.smoothstep(28.0, 46.0, slope_deg))
        * in_area
    )[..., None]
    scree_c = np.array([0.62, 0.43, 0.26], dtype=np.float32) * (0.90 + 0.20 * f_fine[..., None])
    rgb = rgb * (1.0 - gully_mask * 0.72) + scree_c * (gully_mask * 0.72)

    # Crisp clean alpine snow on ridges & cirque benches STRICTLY < 44 deg slope
    snow_mask = (
        nb.smoothstep(relief_mm * 0.22, relief_mm * 0.44, rel + 2.4 * f_macro)
        * (1.0 - nb.smoothstep(36.0, 43.4, slope_deg))
        * in_area
    )[..., None]
    snow_c = np.array([0.95, 0.97, 0.99], dtype=np.float32)
    rgb = rgb * (1.0 - snow_mask) + snow_c * snow_mask
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def render_6_equidistant_views(hm_arr, diffuse_flip_u8, relief_mm, out_path, title, apron_mm=9.5):
    """Render gapless 3x2 studio turnaround every 60 deg (0, 60, 120, 180, 240, 300 deg)."""
    N = 900
    X, Y, Z, de, dv, dx, dy = nb.synthesize_dem_from_heightmap(
        hm_arr, N, N, relief_mm=relief_mm, micro_detail_mm=0.0, x_lim=nb.R, y_lim=nb.R
    )
    Z = apply_smooth_hex_apron(X, Y, Z, de, dv, relief_mm, apron_mm=apron_mm)
    diff = np.flipud(diffuse_flip_u8)
    col = np.clip((X + nb.R) / (2.0 * nb.R) * (diff.shape[1] - 1), 0, diff.shape[1] - 1).astype(np.int32)
    row = np.clip((Y + nb.R) / (2.0 * nb.R) * (diff.shape[0] - 1), 0, diff.shape[0] - 1).astype(np.int32)
    rgb = diff[row, col].astype(np.float32) / 255.0

    gy, gx = np.gradient(Z, dy, dx)
    nz = 1.0 / np.sqrt(1.0 + gx * gx + gy * gy)
    nx_n, ny_n = -gx * nz, -gy * nz
    ndotl = np.clip(nx_n * (-0.42) + ny_n * 0.50 + nz * 0.75, 0.0, 1.0)
    shaded = np.clip(rgb * (0.48 + 0.64 * ndotl)[..., None], 0.0, 1.0)

    # Extra intermediate cliff-skirt fill on steep drops so zero vertical splat holes appear
    z_min_nb = ndi.minimum_filter(Z, size=3)
    in_hex = de >= 0.0
    xs, ys, zs, cs = X[in_hex], Y[in_hex], Z[in_hex], shaded[in_hex]
    dz = zs - z_min_nb[in_hex]
    steep = dz > 0.7
    pts_list = [np.column_stack([xs, ys, zs])]
    cols_list = [cs]
    for f in (0.33, 0.66):
        pts_list.append(np.column_stack([xs[steep], ys[steep], zs[steep] - f * dz[steep]]))
        cols_list.append(cs[steep] * 0.92)

    ang = np.radians(30.0 + 60.0 * np.arange(6))
    cx, cyh = nb.R * np.cos(ang), nb.R * np.sin(ang)
    for e in range(6):
        t = np.linspace(0, 1, 420)
        wx, wy = cx[e] + t * (cx[(e + 1) % 6] - cx[e]), cyh[e] + t * (cyh[(e + 1) % 6] - cyh[e])
        enx, eny = (cyh[(e + 1) % 6] - cyh[e]), -(cx[(e + 1) % 6] - cx[e])
        nl = math.hypot(enx, eny)
        shade_w = max(0.25, 0.38 + 0.25 * ((enx / nl) * (-0.42) + (eny / nl) * 0.50))
        for wz in np.linspace(0, nb.Z_BASE, 130):
            pts_list.append(np.column_stack([wx, wy, np.full_like(wx, wz)]))
            cols_list.append(np.tile([0.20 * shade_w * 1.65, 0.22 * shade_w * 1.65, 0.25 * shade_w * 1.65], (len(wx), 1)))

    P_all = np.vstack(pts_list)
    C_all = np.vstack(cols_list)

    def render_cam(yaw_deg, pitch_deg=34.0, W=520, H=430):
        th, ph = math.radians(yaw_deg), math.radians(pitch_deg)
        cy, sy = math.cos(th), math.sin(th)
        cp, sp = math.cos(ph), math.sin(ph)
        xr = P_all[:, 0] * cy - P_all[:, 1] * sy
        yr = P_all[:, 0] * sy + P_all[:, 1] * cy
        zr = P_all[:, 2] - (18.0 + 0.25 * relief_mm)
        yp = yr * cp - zr * sp
        zp = yr * sp + zr * cp
        sc = 3.25
        px = np.round(W / 2 + xr * sc).astype(np.int32)
        py = np.round(H * 0.55 - zp * sc).astype(np.int32)
        valid = (px >= 1) & (px < W - 1) & (py >= 1) & (py < H - 1)
        px, py, yp, C = px[valid], py[valid], yp[valid], C_all[valid]
        order = np.argsort(-yp)
        img = np.full((H, W, 3), [0.90, 0.91, 0.93], dtype=np.float32)
        for dxo in (-1, 0, 1):
            for dyo in (-1, 0, 1):
                img[py[order] + dyo, px[order] + dxo] = C[order]
        pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        draw = ImageDraw.Draw(pil)
        try:
            fnt = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 18)
        except Exception:
            fnt = ImageFont.load_default()
        draw.text((16, 12), f"Vista {int(yaw_deg)} deg", fill=(35, 42, 54), font=fnt)
        return np.asarray(pil)

    angles = [0, 60, 120, 180, 240, 300]
    row1 = np.hstack([render_cam(a) for a in angles[:3]])
    row2 = np.hstack([render_cam(a) for a in angles[3:]])
    body = np.vstack([row1, row2])
    footer = np.full((38, body.shape[1], 3), (230, 233, 238), dtype=np.uint8)
    grid = Image.fromarray(np.vstack([body, footer]))
    draw = ImageDraw.Draw(grid)
    try:
        fnt_b = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 20)
    except Exception:
        fnt_b = ImageFont.load_default()
    draw.text((18, body.shape[0] + 7), f"{title} — 6 Vistas Equidistantes 60 deg (Z_PEAK={nb.Z_BASE+relief_mm:.1f}mm)", fill=(30, 41, 59), font=fnt_b)
    grid.save(out_path)
    return out_path


def apply_smooth_hex_apron(X, Y, Z, dist_edge, dist_vert, relief_mm, apron_mm=10.0, water_mask=None):
    """Feather steep AI edge drops gradually over apron_mm starting slightly inside the flat road margin."""
    d_clear = np.minimum(dist_edge - nb.ROAD_MARGIN, dist_vert - nb.PAD_RADIUS)
    taper = nb.smoothstep(0.35, apron_mm, d_clear)
    if water_mask is not None:
        w_blur = np.clip(ndi.gaussian_filter(water_mask.astype(np.float32), sigma=3.0), 0.0, 1.0)
        taper_water = nb.smoothstep(0.35, min(3.6, apron_mm), d_clear)
        taper = taper * (1.0 - w_blur) + taper_water * w_blur
    rel = np.maximum(0.0, Z - nb.Z_BASE) * taper
    rmax = float(np.max(rel))
    if rmax > 1e-5:
        rel *= (relief_mm / rmax)
    return (nb.Z_BASE + rel).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--slug", required=True, help="Biome slug (e.g. bosque, colina, pasto, campo, desierto, oro, oceano, montana)")
    p.add_argument("--title", required=True, help="Viewer display title")
    p.add_argument("--mode", choices=["clay", "build"], required=True)
    p.add_argument("--heightmap", required=True)
    p.add_argument("--albedo", default=None)
    p.add_argument("--relief-mm", type=float, required=True)
    p.add_argument("--slope-gamma", type=float, default=0.80)
    p.add_argument("--blur-px", type=float, default=1.2)
    p.add_argument("--ridge-clean", type=float, default=0.0, help="Morphological ridge weld factor (0..1) to turn thin dark crevasses into broad rock facets.")
    p.add_argument("--apron-mm", type=float, default=9.5)
    p.add_argument("--micro-detail", type=float, default=0.14)
    p.add_argument("--micro-norm", type=float, default=0.32, help="PBR normal map luminance high-pass infusion strength.")
    p.add_argument("--nx", type=int, default=420)
    p.add_argument("--stl-nx", type=int, default=0, help="Optional separate ultra-HD grid resolution for local printable STL Master (<=10M tris), while --nx builds the <50MB WebGL viewer.")
    p.add_argument("--stl-subdivide-passes", type=int, default=None)
    p.add_argument("--stl-edge-len-max", type=float, default=None)
    p.add_argument("--stl-curv-thresh", type=float, default=None)
    p.add_argument("--tex-size", type=int, default=2048)
    p.add_argument("--contour-align", action="store_true", help="Concentrate grid toward steep walls & flip Delaunay diagonals along height contours.")
    p.add_argument("--slope-cap-grad", type=float, default=0.0, help="If >0, gently cap near-vertical cliffs to max dZ/dX gradient (e.g. 1.45 -> ~55 deg scarp).")
    p.add_argument("--warp-strength", type=float, default=0.42, help="Grid slope attraction strength when --contour-align is active.")
    p.add_argument("--triplanar-mode", choices=["none", "clay", "forest", "rock", "alpine", "alpine_crisp"], default="none")
    p.add_argument("--preserve-bezel", action="store_true", help="Keep custom bezel/pad ring art from input albedo while cleaning flat bezel height.")
    p.add_argument("--snow-smooth-sculpt", action="store_true", help="Dampen micro-relief carving over pure bright snow drifts so cirques stay smooth.")
    p.add_argument("--init-cam", choices=["iso", "hero", "south", "south30"], default="iso", help="Initial viewer camera preset ('hero' matches 0-deg Nano Banana front shot; 'south' = lower 34-deg 3/4 south view; 'south30' = 30-deg azimuth perpendicular to flat edge).")
    p.add_argument("--snowline-mm", type=float, default=15.0, help="Min relief (mm) above which physical snow is allowed when --snow-filter is active (default 15.0; lower for snow in saddles/cirques).")
    p.add_argument("--alpine-wall-tint", default="0.24,0.21,0.18", help="Rock tint (linear r,g,b) used by the alpine triplanar shader to replace streaked snow on vertical walls (dark slate: 0.115,0.12,0.13).")
    p.add_argument("--keep-dark-rock", action="store_true", help="Preserve dark basalt/slate rock cliffs near the perimeter instead of painting warm fill.")
    p.add_argument("--sculpt-from-albedo", type=float, default=0.0, help="If >0, carve physical 3D rock fractures, cobble/pebble domes, gold clusters & wave ribs (max mm) directly into Z(x,y).")
    p.add_argument("--sculpt-crown-mm", type=float, default=0.0, help="Sub-pixel spire-crown rounding radius (mm) on sculpted pebbles/ridges.")
    p.add_argument("--cliff-sculpt", type=float, default=0.0, help="Normal-direction (nx,ny,nz) 3D cliff strata/block/crack sculpting (mm) on steep rock walls.")
    p.add_argument("--cliff-sculpt-map", default=None, help="Optional float32 .npy (-1..1) texture-relief field for normal-direction 3D cliff wall & feature carving.")
    p.add_argument("--cliff-albedo-weight", type=float, default=0.0, help="Amplitude (mm) of --cliff-sculpt-map lateral normal displacement.")
    p.add_argument("--diff-quality", type=int, default=0, help="Override diffuse image quality (e.g. 92..96) for WebGL HTML embedding.")
    p.add_argument("--norm-quality", type=int, default=0, help="Override normal/roughness image quality (e.g. 85..92) for WebGL HTML embedding.")
    p.add_argument("--tangent-norm", action="store_true", help="Emit pure residual tangent-space normalMap + analytic per-vertex tangents in Three.js (kills dFdx/dFdy wall facets).")
    p.add_argument("--photoreal", action="store_true", help="Preserve photorealistic Nano Banana 4K contrast/vibrancy (skip comic deink, boost specular & exposure).")
    p.add_argument("--water-mask", default=None, help="Optional river water-channel mask PNG to prevent whitewater foam spikes & apron back-slope.")
    p.add_argument("--subdivide-passes", type=int, default=0, help="Conformal Red-Green 3D adaptive triangle refinement passes on steep walls & sculpted peaks.")
    p.add_argument("--edge-len-max", type=float, default=0.175, help="Max 3D edge length threshold (mm) for adaptive subdivision.")
    p.add_argument("--curv-thresh", type=float, default=0.0, help="Face-to-vertex normal curvature threshold (deg) to also subdivide curved ridge crests & pebbles.")
    p.add_argument("--relax-passes", type=int, default=0, help="Area-safe XY vertex relaxation passes after subdivision, re-projected onto bicubic DEM.")
    p.add_argument("--taubin-passes", type=int, default=0, help="Zero-shrinkage Taubin smoothing passes on interior terrain Z.")
    p.add_argument("--morton-sort", action="store_true", help="Morton Z-order spatial reordering (~25% tighter gzip HTML compression).")
    p.add_argument("--snow-filter", action="store_true")
    p.add_argument("--alpine-v3-palette", action="store_true", help="Warm limestone granite, green valley moss, copper scree & <44 deg snow palette.")
    p.add_argument("--auto-mineral-albedo", action="store_true", help="Fallback: paint Nano Banana DEM with V3 hyperreal mineral-runoff PBR shader.")
    p.add_argument("--viewer-html", default=None, help="Output HTML path relative to repo root")
    args = p.parse_args()

    out_dir = os.path.join("Tableros-Catan", "pipeline-biomas", args.slug)
    os.makedirs(out_dir, exist_ok=True)
    viewer_html = args.viewer_html or os.path.join("Visores-HTML", f"catan-bioma-{args.slug}.html")

    if args.heightmap.lower().endswith(".npy"):
        hm_npy = np.clip(np.load(args.heightmap).astype(np.float32), 0.0, 255.0)
        if abs(args.slope_gamma - 1.0) < 1e-4 and args.blur_px <= 0.0 and not args.ridge_clean:
            hm_arr = hm_npy
        else:
            hm_arr = refine_heightmap(hm_npy, gamma=args.slope_gamma, blur_px=args.blur_px, ridge_clean=args.ridge_clean)
    else:
        hm_img = Image.open(args.heightmap)
        hm_raw = np.asarray(hm_img)
        if hm_raw.dtype == np.uint16:
            hm_f = (hm_raw.astype(np.float32) / 65535.0) * 255.0
            hm_arr = hm_f if (abs(args.slope_gamma - 1.0) < 1e-4 and args.blur_px <= 0.0 and not args.ridge_clean) else refine_heightmap(hm_f, gamma=args.slope_gamma, blur_px=args.blur_px, ridge_clean=args.ridge_clean)
        else:
            hm_arr = refine_heightmap(np.asarray(hm_img.convert("L")), gamma=args.slope_gamma, blur_px=args.blur_px, ridge_clean=args.ridge_clean)

    Xt, Yt, Zt, dist_t, dist_vt, dxt, dyt = nb.synthesize_dem_from_heightmap(
        hm_arr, args.tex_size, args.tex_size, relief_mm=args.relief_mm, micro_detail_mm=args.micro_detail, x_lim=nb.R, y_lim=nb.R
    )
    wmask_t = None
    if args.water_mask:
        wm_arr = np.asarray(Image.open(args.water_mask).convert("L"), dtype=np.float32) / 255.0
        wmask_t = np.clip(nb.sample_square_domain(wm_arr, Xt, Yt), 0.0, 1.0)
    Zt = apply_smooth_hex_apron(Xt, Yt, Zt, dist_t, dist_vt, args.relief_mm, apron_mm=args.apron_mm, water_mask=wmask_t)
    diff_raw = None
    if args.albedo:
        alb_arr = np.asarray(Image.open(args.albedo).convert("RGB"))
        diff_raw = np.clip(nb.sample_square_domain(alb_arr, Xt, Yt), 0, 255).astype(np.uint8)
        if args.sculpt_from_albedo > 0.0:
            Zt_pre = Zt.copy() if args.snow_smooth_sculpt else None
            Zt = sculpt_albedo_microrelief(Zt, diff_raw, dist_t, dist_vt, dxt, max_bump_mm=args.sculpt_from_albedo, crown_smooth_mm=args.sculpt_crown_mm, water_mask=wmask_t)
            if args.snow_smooth_sculpt and Zt_pre is not None:
                lum_s = (0.2126 * diff_raw[..., 0] + 0.7152 * diff_raw[..., 1] + 0.0722 * diff_raw[..., 2]) / 255.0
                sat_s = (np.max(diff_raw, axis=-1) - np.min(diff_raw, axis=-1)) / 255.0
                sn_s = nb.smoothstep(0.54, 0.76, lum_s) * (1.0 - nb.smoothstep(0.10, 0.20, sat_s))
                sn_filled = ndi.binary_fill_holes(sn_s > 0.35).astype(np.float32)
                sn_base = np.maximum(sn_s, sn_filled * 0.90)
                sn_dil = ndi.maximum_filter(sn_base, size=max(3, int(round(0.95 / dxt))))
                sn_halo = np.clip(ndi.gaussian_filter(sn_dil, sigma=max(2.0, 0.65 / dxt)) * 1.15, 0.0, 1.0)
                Zt = Zt_pre + (Zt - Zt_pre) * (1.0 - 0.95 * sn_halo).astype(np.float32)
            if not args.photoreal:
                diff_raw = deink_mineral_albedo(diff_raw, dist_t, dist_vt, dxt)
    if args.slope_cap_grad > 0.0:
        n_cap_t = max(16 if args.sculpt_crown_mm >= 0.13 else 6, int(round((3.8 if args.sculpt_crown_mm >= 0.13 else 1.5) / dxt)))
        Zt = tmu.cap_extreme_slopes(Zt, dist_t, min_step_dx=dxt, max_grad=args.slope_cap_grad, passes=n_cap_t)
    clay_rgb = nb.render_shaded_clay_rgb(Zt, dist_t, dxt, dyt)
    clay_path = os.path.join(out_dir, f"{args.slug}_clay_overhead.png")
    nb.save_uv_aligned_png(clay_rgb, clay_path)

    if args.mode == "clay":
        print(f"[Clay Ready] {args.slug} (Z_PEAK={nb.Z_BASE + args.relief_mm:.1f}mm) -> {clay_path}")
        return

    if not args.albedo and not args.auto_mineral_albedo and not args.alpine_v3_palette:
        p.error("--albedo, --alpine-v3-palette, or --auto-mineral-albedo is required in --mode build")

    if args.alpine_v3_palette:
        d_u8 = synthesize_alpine_v3_albedo(Xt, Yt, Zt, dist_t, dist_vt, dxt, dyt, args.relief_mm)
        diff_rgb, max_snow = refine_biome_albedo(d_u8, Zt, dist_t, dist_vt, dxt, dyt, snow_filter=True, snowline_mm=args.relief_mm * 0.30)
        norm_u8, rough_u8 = compute_enhanced_pbr_maps(Zt, diff_rgb, dxt, dyt, micro_norm_strength=args.micro_norm, tangent_residual=args.tangent_norm)
    elif args.auto_mineral_albedo:
        rmf_m = pv3.ridged_multifractal_xy(Xt, Yt, seed=401)
        rmf_f = pv3.ridged_multifractal_xy(Xt, Yt, seed=402)
        broad = pv3.fft_fbm_canonical(Xt, Yt, 7.0, 24.0, beta=2.15, seed=404)
        z_scaled = nb.Z_BASE + (Zt - nb.Z_BASE) * (58.0 / max(args.relief_mm, 1.0))
        d_f32, _, _, slope_s, _, _, shader_stats = pv3.synthesize_hyperreal_textures(
            Xt, Yt, z_scaled, dist_t, dist_vt, dxt, dyt, rmf_m, rmf_f, broad
        )
        rel = np.maximum(0.0, Zt - nb.Z_BASE)
        in_area = nb.smoothstep(nb.ROAD_MARGIN + 0.4, nb.ROAD_MARGIN + 3.2, dist_t) * nb.smoothstep(nb.PAD_RADIUS + 0.4, nb.PAD_RADIUS + 3.2, dist_vt)
        meadow_tint = np.clip((1.0 - nb.smoothstep(3.0, args.relief_mm * 0.28, rel)) * (1.0 - nb.smoothstep(26.0, 44.0, slope_s)) * in_area, 0.0, 0.55)[..., None]
        warm_boost = np.array([1.08, 1.04, 0.96], dtype=np.float32)
        moss_col = np.array([0.28, 0.37, 0.17], dtype=np.float32)
        d_mix = np.clip(d_f32 * warm_boost, 0.0, 1.0)
        d_mix = np.clip(d_mix * (1.0 - meadow_tint * 0.48) + moss_col * (meadow_tint * 0.48), 0.0, 1.0)
        d_u8 = np.clip(d_mix * 255.0, 0, 255).astype(np.uint8)
        diff_rgb, max_snow_phys = refine_biome_albedo(d_u8, Zt, dist_t, dist_vt, dxt, dyt, snow_filter=True, snowline_mm=args.relief_mm * 0.38)
        max_snow = max(float(shader_stats.get("max_snow_steep", 0.0)), max_snow_phys)
        norm_u8, rough_u8 = compute_enhanced_pbr_maps(Zt, diff_rgb, dxt, dyt, micro_norm_strength=args.micro_norm, tangent_residual=args.tangent_norm)
    else:
        if diff_raw is None:
            alb_arr = np.asarray(Image.open(args.albedo).convert("RGB"))
            diff_raw = np.clip(nb.sample_square_domain(alb_arr, Xt, Yt), 0, 255).astype(np.uint8)
        diff_rgb, max_snow = refine_biome_albedo(diff_raw, Zt, dist_t, dist_vt, dxt, dyt, snow_filter=args.snow_filter, snowline_mm=args.snowline_mm, keep_dark_rock=args.keep_dark_rock, photoreal=args.photoreal)
        if args.preserve_bezel:
            rel_bz = np.maximum(0.0, Zt - nb.Z_BASE)
            bz_on = np.maximum(
                1.0 - nb.smoothstep(nb.ROAD_MARGIN + 0.45, nb.ROAD_MARGIN + 1.05, dist_t),
                1.0 - nb.smoothstep(nb.PAD_RADIUS + 0.35, nb.PAD_RADIUS + 0.95, dist_vt),
            ) * (1.0 - nb.smoothstep(0.14, 0.62, rel_bz))
            bz_m = np.clip(bz_on, 0.0, 1.0)[..., None]
            diff_rgb = np.clip(diff_rgb.astype(np.float32) * (1.0 - bz_m) + diff_raw.astype(np.float32) * bz_m, 0.0, 255.0).astype(np.uint8)
        norm_u8, rough_u8 = compute_enhanced_pbr_maps(Zt, diff_rgb, dxt, dyt, micro_norm_strength=args.micro_norm, tangent_residual=args.tangent_norm, photoreal=args.photoreal, water_mask=wmask_t)

    nb.save_uv_aligned_png(diff_rgb, os.path.join(out_dir, "catan_mountain_diffuse_2k.png"))
    nb.save_uv_aligned_png(norm_u8, os.path.join(out_dir, "catan_mountain_normal_2k.png"))
    nb.save_uv_aligned_png(rough_u8, os.path.join(out_dir, "catan_mountain_roughness_2k.png"))

    cliff_field_arr = None
    if args.cliff_sculpt_map:
        cliff_field_arr = np.flipud(np.load(args.cliff_sculpt_map).astype(np.float32))

    def build_lod_mesh(nx_val, subdiv_p, edge_max, curv_th, label="Mesh"):
        ny_val = int(round(nx_val * (nb.R / nb.APOTHEM)))
        Xm, Ym, Zm, dist_m, dist_vm, dxm, _ = nb.synthesize_dem_from_heightmap(
            hm_arr, nx_val, ny_val, relief_mm=args.relief_mm, micro_detail_mm=args.micro_detail, x_lim=nb.APOTHEM, y_lim=nb.R
        )
        wmask_m = np.clip(nb.sample_square_domain(wm_arr, Xm, Ym), 0.0, 1.0) if args.water_mask else None
        Zm = apply_smooth_hex_apron(Xm, Ym, Zm, dist_m, dist_vm, args.relief_mm, apron_mm=args.apron_mm, water_mask=wmask_m)
        if args.slope_cap_grad > 0.0:
            n_cap_m = max(4, int(round(1.5 / dxm)))
            Zm = tmu.cap_extreme_slopes(Zm, dist_m, min_step_dx=dxm, max_grad=args.slope_cap_grad, passes=n_cap_m)
        if args.contour_align:
            Xw, Yw = tmu.warp_grid_to_slopes(Xm, Ym, Zm, dist_m, dist_vm, iters=4, strength=args.warp_strength)
            dist_w, _ = nb.hex_distances(Xw, Yw)
            Zw = tmu.resample_canonical_xy(Zt, Xw, Yw, lim=nb.R)
            v_out, f_out, n_flips = tmu.build_contour_aligned_hex_solid(
                Xw, Yw, Zw, dist_w, n_bnd=168, n_mag=32, include_magnet=True, flip_passes=6,
                subdivide_passes=subdiv_p, edge_len_3d_max=edge_max,
                curv_deg_thresh=curv_th, relax_passes=args.relax_passes,
                taubin_passes=args.taubin_passes, Z_dem=Zt, morton_sort=args.morton_sort,
                cliff_sculpt=args.cliff_sculpt, water_mask_arr=(wm_arr if args.water_mask else None),
                cliff_albedo_field=cliff_field_arr, cliff_albedo_weight=args.cliff_albedo_weight,
            )
            print(f"[{label}] nx={nx_val} ny={ny_val} flips={n_flips} subdiv={subdiv_p} taubin={args.taubin_passes} V={len(v_out)} F={len(f_out)}")
        else:
            v_out, f_out = nb.build_solid_hex_mesh(Xm, Ym, Zm, dist_m, include_magnet=True)
        st = nb.verify_watertight(v_out, f_out)
        if not st["watertight"]:
            raise RuntimeError(f"{args.slug} {label} failed watertight check: {st}")
        return v_out, f_out, st

    verts, faces, viewer_stats = build_lod_mesh(
        args.nx, args.subdivide_passes, args.edge_len_max, args.curv_thresh,
        label="Contour-Aligned WebGL" if args.stl_nx > 0 else "Contour-Aligned Mesh",
    )
    if args.stl_nx > 0:
        stl_verts, stl_faces, stats = build_lod_mesh(
            args.stl_nx,
            args.subdivide_passes if args.stl_subdivide_passes is None else args.stl_subdivide_passes,
            args.edge_len_max if args.stl_edge_len_max is None else args.stl_edge_len_max,
            args.curv_thresh if args.stl_curv_thresh is None else args.stl_curv_thresh,
            label="Ultra-HD STL Master",
        )
        stl_path = os.path.join(out_dir, f"catan_{args.slug}.stl")
        nb.write_binary_stl(stl_verts, stl_faces, stl_path, label=f"Catan {args.slug} Master")
        del stl_verts, stl_faces
    else:
        stats = viewer_stats
        stl_path = os.path.join(out_dir, f"catan_{args.slug}.stl")
        nb.write_binary_stl(verts, faces, stl_path, label=f"Catan {args.slug}")
    np.savez_compressed(os.path.join(out_dir, "viewer_mesh_v2.npz"), verts=verts, faces=faces)

    audit_path = os.path.join(out_dir, f"{args.slug}_6views_audit.png")
    render_6_equidistant_views(hm_arr, np.flipud(diff_rgb), args.relief_mm, audit_path, args.title, apron_mm=args.apron_mm)

    z_peak = nb.Z_BASE + args.relief_mm
    metrics = {
        "master": {"V": stats["V"], "E": stats["E"], "F": stats["F"], "chi": stats["chi"], "watertight": True, "volume_cm3": stats["vol_cm3"]},
        "coverage": {"max_snow_steep": round(max_snow, 6)},
        "viewer": {"V": viewer_stats["V"], "F": viewer_stats["F"]},
    }
    with open(os.path.join(out_dir, "master_metrics_v2.json"), "w", encoding="utf-8") as mf:
        json.dump(metrics, mf, indent=2)

    av2.OUT_DIR = out_dir
    av2.SCRATCH_DIR = out_dir
    av2.TARGET_PATHS = [viewer_html]
    av2.TRIPLANAR_MODE = args.triplanar_mode
    av2.USE_SMOOTH_TANGENTS = args.tangent_norm
    av2.PHOTOREAL_GOLD = args.photoreal
    av2.ALPINE_WALL_TINT = tuple(float(v) for v in args.alpine_wall_tint.split(","))
    if args.init_cam == "south":
        av2.INIT_CAM_POS = (1.4, -373.6, 277.5)
        av2.INIT_CAM_TARGET = (1.4, -13.0, 34.2)
        av2.INIT_FOV = 18.5
        av2.INIT_SUN_POS = (150.0, -78.0, 152.0)
    elif args.init_cam == "south30":
        # az=30° (perpendicular a la arista plana inferior, exactamente alineado a FRONT 3/4 VIEW 0°)
        av2.INIT_CAM_POS = (176.0, -304.8, 245.0)
        av2.INIT_CAM_TARGET = (0.0, 0.0, 32.0)
        av2.INIT_FOV = 21.0
        av2.INIT_SUN_POS = (150.0, -78.0, 152.0)
    elif args.init_cam == "hero":
        av2.INIT_CAM_POS = (2.4, -342.8, 317.9)
        av2.INIT_CAM_TARGET = (1.4, -13.0, 34.2)
        av2.INIT_FOV = 18.5
        av2.INIT_SUN_POS = (150.0, -78.0, 152.0)
    else:
        av2.INIT_CAM_POS = (168.0, -172.0, 130.0)
        av2.INIT_CAM_TARGET = (6.0, 0.0, 34.0)
        av2.INIT_FOV = 36.0
        av2.INIT_SUN_POS = (112.0, -96.0, 138.0)
    if args.photoreal and args.tex_size >= 4096:
        av2.DIFF_QUALITY = args.diff_quality if args.diff_quality > 0 else 91
        av2.NORM_QUALITY = args.norm_quality if args.norm_quality > 0 else 85
    elif args.diff_quality > 0 or args.norm_quality > 0:
        if args.diff_quality > 0:
            av2.DIFF_QUALITY = args.diff_quality
        if args.norm_quality > 0:
            av2.NORM_QUALITY = args.norm_quality
    if args.tangent_norm:
        av2.SHADOW_BIAS = -0.00025
        av2.SHADOW_NORMAL_BIAS = 0.08
    av2.build_viewer()

    with open(viewer_html, "r", encoding="utf-8") as hf:
        html = hf.read()
    html = html.replace("🏔️ Cordillera Montaña Catan 3D · V3 PBR", f"{args.title} · V3 PBR")
    html = html.replace("Cordillera Montaña Catan 3D (V3)", args.title)
    html = html.replace("110.85×128×83", f"110.85×128×{z_peak:.0f}")
    html = html.replace(">83.0 mm<", f">{z_peak:.1f} mm<")
    html = html.replace('max="83.5" step="0.5" value="83.5"', f'max="{z_peak + 0.5:.1f}" step="0.5" value="{z_peak + 0.5:.1f}"')
    html = html.replace("Z=83mm", f"Z={z_peak:.0f}mm")
    if not args.snow_filter and not args.alpine_v3_palette:
        html = html.replace("<span>Nieve en ≥44°</span><strong>0.000000 (0%)</strong>", f"<span>Relieve Máx</span><strong>+{args.relief_mm:.1f} mm</strong>")
    else:
        html = html.replace("<span>Nieve en ≥44°</span><strong>0.000000 (0%)</strong>", f"<span>Nieve en ≥44°</span><strong>{max_snow:.6f} ({max_snow:.2f}%)</strong>")
    with open(viewer_html, "w", encoding="utf-8") as hf:
        hf.write(html)

    print(f"[BIOME OK] {args.slug}: V={stats['V']} F={stats['F']} chi={stats['chi']} vol={stats['vol_cm3']:.2f}cm3 Z={z_peak:.1f}mm audit={audit_path} viewer={viewer_html}")


if __name__ == "__main__":
    main()
