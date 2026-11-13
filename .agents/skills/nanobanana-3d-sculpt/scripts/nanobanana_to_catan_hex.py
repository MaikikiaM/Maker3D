"""
Pure NumPy + SciPy + PIL bridge projecting Nano Banana orthographic heightmaps
and geometry-locked painterly albedo textures into watertight (chi=2) Catan hex
tiles with exact CAD tolerances (R=64mm, 3.50mm road margin, Ø20mm pads, N52 pocket).
Zero trimesh dependency.
"""
import argparse
import math
import os
import struct
import tempfile
import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import Delaunay
from PIL import Image

R = 64.0
APOTHEM = R * math.cos(math.radians(30.0))  # 55.42562584 mm
Z_BASE = 25.0
ROAD_MARGIN = 3.50
PAD_RADIUS = 10.00
MAG_RADIUS = 5.10
MAG_DEPTH = 5.20


def smoothstep(e0, e1, v):
    t = np.clip((v - e0) / np.maximum(e1 - e0, 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def hex_distances(X, Y):
    """Signed interior edge distance and nearest-corner Euclidean distance."""
    normals_hex = [[math.cos(k * math.pi / 3.0), math.sin(k * math.pi / 3.0)] for k in range(6)]
    dist_edge = np.full_like(X, 999.0, dtype=np.float32)
    for nx_h, ny_h in normals_hex:
        dist_edge = np.minimum(dist_edge, APOTHEM - (X * nx_h + Y * ny_h))

    v_angles = [(30.0 + 60.0 * k) * math.pi / 180.0 for k in range(6)]
    dist_vert = np.full_like(X, 999.0, dtype=np.float32)
    for a in v_angles:
        vx, vy = R * math.cos(a), R * math.sin(a)
        dist_vert = np.minimum(dist_vert, np.hypot(X - vx, Y - vy))
    return dist_edge, dist_vert


def fft_fbm(nx, ny, min_wl_mm, max_wl_mm, width_mm, height_mm, beta=2.1, seed=42):
    rng = np.random.RandomState(seed)
    kx = np.fft.rfftfreq(nx, d=width_mm / nx)
    ky = np.fft.fftfreq(ny, d=height_mm / ny)
    KX, KY = np.meshgrid(kx, ky)
    freq = np.hypot(KX, KY)
    band = (freq >= 1.0 / max_wl_mm) & (freq <= 1.0 / min_wl_mm)
    amp = np.zeros_like(freq, dtype=np.float32)
    amp[band] = (freq[band] + 1e-6) ** (-beta / 2.0)
    amp *= smoothstep(0.8 / max_wl_mm, 1.4 / max_wl_mm, freq) * (1.0 - smoothstep(0.65 / min_wl_mm, 1.0 / min_wl_mm, freq))
    spec = amp * np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=amp.shape))
    out = np.fft.irfft2(spec, s=(ny, nx)).astype(np.float32)
    std = float(np.std(out))
    return (out - float(np.mean(out))) / (std if std > 1e-8 else 1.0)


def sample_square_domain(img_arr, X, Y):
    """Map an upright PNG array (row 0 = Y=+R) onto canonical (Y=-R..+R, X=-R..+R) grids."""
    # Flip vertically once so array row 0 matches Y=-R (bottom), matching Y-up UV mapping.
    arr = np.flipud(img_arr).astype(np.float32)
    H, W = arr.shape[:2]
    ix = np.clip((X + R) / (2.0 * R), 0.0, 1.0) * (W - 1)
    iy = np.clip((Y + R) / (2.0 * R), 0.0, 1.0) * (H - 1)
    if arr.ndim == 2:
        return ndi.map_coordinates(arr, [iy, ix], order=1, mode="nearest")
    chans = [ndi.map_coordinates(arr[:, :, c], [iy, ix], order=1, mode="nearest") for c in range(arr.shape[2])]
    return np.stack(chans, axis=-1)


def save_uv_aligned_png(arr_u8, path):
    """Flip Y so PNG row 0 maps to v=1 (+R) and row H-1 maps to v=0 (-R)."""
    Image.fromarray(np.flipud(arr_u8), mode="RGB").save(path, optimize=False, compress_level=6)


def synthesize_dem_from_heightmap(hm_arr, nx, ny, relief_mm=58.0, micro_detail_mm=0.9, x_lim=APOTHEM, y_lim=R):
    xs = np.linspace(-x_lim, x_lim, nx, dtype=np.float32)
    ys = np.linspace(-y_lim, y_lim, ny, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    dx, dy = float(xs[1] - xs[0]), float(ys[1] - ys[0])
    dist_edge, dist_vert = hex_distances(X, Y)

    raw = sample_square_domain(hm_arr, X, Y) / 255.0
    # Smooth 8-bit stair-stepping while keeping crisp ridge crests
    raw = ndi.gaussian_filter(raw, sigma=max(0.30 / dx, 0.5))

    collar = smoothstep(ROAD_MARGIN, ROAD_MARGIN + 3.2, dist_edge) * smoothstep(PAD_RADIUS, PAD_RADIUS + 3.2, dist_vert)
    shaped = np.clip(raw, 0.0, 1.0) * collar
    max_s = float(np.max(shaped))
    if max_s > 1e-5:
        shaped /= max_s

    micro = 0.0
    if micro_detail_mm > 0.0:
        f1 = 1.0 - np.abs(fft_fbm(nx, ny, 2.2, 7.5, 2.0 * x_lim, 2.0 * y_lim, seed=101))
        f2 = 1.0 - np.abs(fft_fbm(nx, ny, 0.9, 2.5, 2.0 * x_lim, 2.0 * y_lim, seed=202))
        micro = micro_detail_mm * (0.65 * (f1 - 0.35) + 0.35 * (f2 - 0.35)) * smoothstep(0.06, 0.30, shaped) * collar

    relief = np.maximum(0.0, shaped * relief_mm + micro)
    r_max = float(np.max(relief))
    if r_max > 1e-5:
        relief *= (relief_mm / r_max)
    Z = (Z_BASE + relief).astype(np.float32)
    return X, Y, Z, dist_edge, dist_vert, dx, dy


def render_shaded_clay_rgb(Z, dist_edge, dx, dy):
    """Shaded neutral clay overhead preview matching UV bounds [-R, R]x[-R, R] for AI paint-over."""
    gy, gx = np.gradient(Z, dy, dx)
    nz = 1.0 / np.sqrt(1.0 + gx * gx + gy * gy)
    nx_n, ny_n = -gx * nz, -gy * nz
    # Soft studio directional + ambient dome
    lx, ly, lz = -0.45, 0.55, 0.70
    l_len = math.sqrt(lx * lx + ly * ly + lz * lz)
    ndotl = np.clip((nx_n * lx + ny_n * ly + nz * lz) / l_len, 0.0, 1.0)
    elev_t = np.clip((Z - Z_BASE) / np.maximum(float(np.max(Z) - Z_BASE), 1.0), 0.0, 1.0)
    shade = 0.38 + 0.48 * ndotl + 0.14 * elev_t
    # Darken outer hex apron slightly so boundary collar stays obvious during paint-over
    in_inner = smoothstep(ROAD_MARGIN - 0.2, ROAD_MARGIN + 0.8, dist_edge)
    shade = shade * (0.55 + 0.45 * in_inner)
    val = np.clip(shade * 255.0, 0, 255).astype(np.uint8)
    return np.stack([val, val, val], axis=-1)


def compute_pbr_maps(Z, diffuse_rgb, dx, dy):
    """Derive tangent-space normal map and scalar roughness map from Z and painted albedo."""
    gy, gx = np.gradient(Z, dy, dx)
    inv_len = 1.0 / np.sqrt(gx * gx + gy * gy + 1.0)
    # Tangent normal encoding (+X right, +Y up, +Z out)
    n_x = (-gx * inv_len * 0.5 + 0.5) * 255.0
    n_y = (-gy * inv_len * 0.5 + 0.5) * 255.0
    n_z = (inv_len * 0.5 + 0.5) * 255.0
    normal_u8 = np.clip(np.stack([n_x, n_y, n_z], axis=-1), 0, 255).astype(np.uint8)

    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
    lum = (0.2126 * diffuse_rgb[:, :, 0] + 0.7152 * diffuse_rgb[:, :, 1] + 0.0722 * diffuse_rgb[:, :, 2]) / 255.0
    # Steep fractured rock stays rough (~0.85), high sunlit/snow crests slightly crisper (~0.52)
    rough = 0.82 + 0.10 * smoothstep(25.0, 60.0, slope_deg) - 0.25 * smoothstep(0.70, 0.96, lum)
    r_u8 = np.clip(rough * 255.0, 0, 255).astype(np.uint8)
    rough_u8 = np.stack([r_u8, r_u8, r_u8], axis=-1)
    return normal_u8, rough_u8


def build_solid_hex_mesh(X, Y, Z, dist_edge, n_bnd=144, n_mag=32, include_magnet=True):
    """Build closed watertight chi=2 2-manifold hex mesh (Z=0..Z_PEAK) via SciPy Delaunay."""
    mask_int = dist_edge >= 1.25
    pts_int = np.column_stack([X[mask_int], Y[mask_int], Z[mask_int]]).astype(np.float64)

    v_angles = [(30.0 + 60.0 * k) * math.pi / 180.0 for k in range(6)]
    corners = [(R * math.cos(a), R * math.sin(a)) for a in v_angles]
    pts_per_edge = n_bnd // 6
    bnd_xy = np.array([
        [corners[e][0] + (s / float(pts_per_edge)) * (corners[(e + 1) % 6][0] - corners[e][0]),
         corners[e][1] + (s / float(pts_per_edge)) * (corners[(e + 1) % 6][1] - corners[e][1])]
        for e in range(6) for s in range(pts_per_edge)
    ], dtype=np.float64)

    pts_top_bnd = np.column_stack([bnd_xy, np.full(n_bnd, Z_BASE, dtype=np.float64)])
    all_top_verts = np.vstack([pts_int, pts_top_bnd])
    top_bnd_start = len(pts_int)
    bot_bnd_start = len(all_top_verts)

    tri_top = Delaunay(all_top_verts[:, :2]).simplices.astype(np.int32)
    p0, p1, p2 = all_top_verts[tri_top[:, 0]], all_top_verts[tri_top[:, 1]], all_top_verts[tri_top[:, 2]]
    flip = ((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])) < 0
    tri_top[flip, 1], tri_top[flip, 2] = tri_top[flip, 2], tri_top[flip, 1]

    pts_bot_bnd = np.column_stack([bnd_xy, np.zeros(n_bnd, dtype=np.float64)])
    idx_e = np.arange(n_bnd, dtype=np.int32)
    idx_en = (idx_e + 1) % n_bnd
    t0, t1 = top_bnd_start + idx_e, top_bnd_start + idx_en
    b0, b1 = bot_bnd_start + idx_e, bot_bnd_start + idx_en
    tris_skirt = np.vstack([np.column_stack([t0, b0, b1]), np.column_stack([t0, b1, t1])]).astype(np.int32)

    if not include_magnet:
        tri_bot_local = Delaunay(bnd_xy).simplices.astype(np.int32)
        tris_bot = bot_bnd_start + tri_bot_local
        vb0, vb1, vb2 = bnd_xy[tri_bot_local[:, 0]], bnd_xy[tri_bot_local[:, 1]], bnd_xy[tri_bot_local[:, 2]]
        flip_b = ((vb1[:, 0] - vb0[:, 0]) * (vb2[:, 1] - vb0[:, 1]) - (vb1[:, 1] - vb0[:, 1]) * (vb2[:, 0] - vb0[:, 0])) > 0
        tris_bot[flip_b, 1], tris_bot[flip_b, 2] = tris_bot[flip_b, 2], tris_bot[flip_b, 1]
        verts = np.vstack([all_top_verts, pts_bot_bnd]).astype(np.float32)
        faces = np.vstack([tri_top, tris_bot, tris_skirt]).astype(np.int32)
        return verts, faces

    mag_ang = np.linspace(0.0, 2.0 * math.pi, n_mag, endpoint=False)
    mag_xy = np.column_stack([MAG_RADIUS * np.cos(mag_ang), MAG_RADIUS * np.sin(mag_ang)])
    mag_bot_start = bot_bnd_start + n_bnd
    mag_ceil_start = mag_bot_start + n_mag
    mag_ctr_idx = mag_ceil_start + n_mag

    pts_mag_bot = np.column_stack([mag_xy, np.zeros(n_mag, dtype=np.float64)])
    pts_mag_ceil = np.column_stack([mag_xy, np.full(n_mag, MAG_DEPTH, dtype=np.float64)])
    pt_mag_ctr = np.array([[0.0, 0.0, MAG_DEPTH]], dtype=np.float64)

    ann_pts = np.vstack([bnd_xy, mag_xy])
    tri_ann = Delaunay(ann_pts).simplices
    tri_ann = tri_ann[np.hypot(ann_pts[tri_ann].mean(axis=1)[:, 0], ann_pts[tri_ann].mean(axis=1)[:, 1]) > MAG_RADIUS * 1.001]
    l2g = np.concatenate([np.arange(bot_bnd_start, bot_bnd_start + n_bnd, dtype=np.int32),
                          np.arange(mag_bot_start, mag_bot_start + n_mag, dtype=np.int32)])
    tris_bot = l2g[tri_ann]
    vb0, vb1, vb2 = ann_pts[tri_ann[:, 0]], ann_pts[tri_ann[:, 1]], ann_pts[tri_ann[:, 2]]
    flip_b = ((vb1[:, 0] - vb0[:, 0]) * (vb2[:, 1] - vb0[:, 1]) - (vb1[:, 1] - vb0[:, 1]) * (vb2[:, 0] - vb0[:, 0])) > 0
    tris_bot[flip_b, 1], tris_bot[flip_b, 2] = tris_bot[flip_b, 2], tris_bot[flip_b, 1]

    jm = np.arange(n_mag, dtype=np.int32)
    jmn = (jm + 1) % n_mag
    mb0, mb1 = mag_bot_start + jm, mag_bot_start + jmn
    mc0, mc1 = mag_ceil_start + jm, mag_ceil_start + jmn
    tris_mag_wall = np.vstack([np.column_stack([mc0, mb1, mb0]), np.column_stack([mc0, mc1, mb1])]).astype(np.int32)
    tris_mag_ceil = np.column_stack([np.full(n_mag, mag_ctr_idx, dtype=np.int32), mc1, mc0]).astype(np.int32)

    verts = np.vstack([all_top_verts, pts_bot_bnd, pts_mag_bot, pts_mag_ceil, pt_mag_ctr]).astype(np.float32)
    faces = np.vstack([tri_top, tris_bot, tris_skirt, tris_mag_wall, tris_mag_ceil]).astype(np.int32)
    return verts, faces


def verify_watertight(verts, faces):
    V, F = len(verts), len(faces)
    u = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]]).astype(np.int64)
    v = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]]).astype(np.int64)
    fwd, rev = np.sort(u * V + v), np.sort(v * V + u)
    unique_fwd = bool(np.all(fwd[1:] != fwd[:-1]))
    paired = bool(np.array_equal(fwd, rev))
    E = len(fwd) // 2
    chi = int(V - E + F)
    v0, v1, v2 = verts[faces[:, 0]].astype(np.float64), verts[faces[:, 1]].astype(np.float64), verts[faces[:, 2]].astype(np.float64)
    cross = np.cross(v1 - v0, v2 - v0)
    min_area = float(0.5 * np.min(np.linalg.norm(cross, axis=1)))
    vol_cm3 = float(np.sum(np.einsum("ij,ij->i", v0, cross)) / 6000.0)
    return {"V": V, "E": E, "F": F, "chi": chi, "watertight": unique_fwd and paired and chi == 2 and min_area > 0.0,
            "min_area": min_area, "vol_cm3": vol_cm3}


def write_binary_stl(verts, faces, path, label="NanoBanana Catan Hex"):
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    nrm = np.cross(v1 - v0, v2 - v0)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    dtype = np.dtype([("normal", "<3f4"), ("v0", "<3f4"), ("v1", "<3f4"), ("v2", "<3f4"), ("attr", "<u2")])
    buf = np.empty(len(faces), dtype=dtype)
    buf["normal"], buf["v0"], buf["v1"], buf["v2"], buf["attr"] = nrm, v0, v1, v2, 0
    with open(path, "wb") as f:
        f.write(label.encode("ascii")[:80].ljust(80, b"\x00"))
        f.write(struct.pack("<I", len(faces)))
        buf.tofile(f)


def write_obj_mtl(verts, faces, obj_path, tex_filename):
    mtl_path = os.path.splitext(obj_path)[0] + ".mtl"
    mtl_name = os.path.basename(mtl_path)
    with open(mtl_path, "w", encoding="utf-8") as mf:
        mf.write(f"newmtl tile_mat\nKa 0.05 0.05 0.05\nKd 1.0 1.0 1.0\nKs 0.08 0.08 0.08\nNs 24.0\nmap_Kd {tex_filename}\n")
    u = np.clip((verts[:, 0] + R) / (2.0 * R), 0.0, 1.0)
    v = np.clip((verts[:, 1] + R) / (2.0 * R), 0.0, 1.0)
    with open(obj_path, "w", encoding="utf-8") as of:
        of.write(f"mtllib {mtl_name}\nusemtl tile_mat\n")
        for x, y, z in verts:
            of.write(f"v {x:.4f} {y:.4f} {z:.4f}\n")
        for uu, vv in zip(u, v):
            of.write(f"vt {uu:.6f} {vv:.6f}\n")
        for a, b, c in faces + 1:
            of.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")


def run_pipeline(hm_arr, albedo_arr, out_dir, prefix, relief_mm, nx, tex_size, micro_detail_mm, include_magnet):
    os.makedirs(out_dir, exist_ok=True)
    # UV textures ALWAYS cover exact canonical [-R, R] x [-R, R] square!
    Xt, Yt, Zt, dist_t, _, dxt, dyt = synthesize_dem_from_heightmap(
        hm_arr, tex_size, tex_size, relief_mm=relief_mm, micro_detail_mm=micro_detail_mm, x_lim=R, y_lim=R
    )
    clay_rgb = render_shaded_clay_rgb(Zt, dist_t, dxt, dyt)
    clay_path = os.path.join(out_dir, f"{prefix}_clay_overhead.png")
    save_uv_aligned_png(clay_rgb, clay_path)

    diffuse_rgb = clay_rgb if albedo_arr is None else np.clip(sample_square_domain(albedo_arr, Xt, Yt), 0, 255).astype(np.uint8)
    normal_u8, rough_u8 = compute_pbr_maps(Zt, diffuse_rgb, dxt, dyt)

    diff_name = f"{prefix}_diffuse.png"
    save_uv_aligned_png(diffuse_rgb, os.path.join(out_dir, diff_name))
    save_uv_aligned_png(normal_u8, os.path.join(out_dir, f"{prefix}_normal.png"))
    save_uv_aligned_png(rough_u8, os.path.join(out_dir, f"{prefix}_roughness.png"))

    Xm, Ym, Zm, dist_m, _, _, _ = synthesize_dem_from_heightmap(
        hm_arr, nx, int(round(nx * (R / APOTHEM))), relief_mm=relief_mm, micro_detail_mm=micro_detail_mm, x_lim=APOTHEM, y_lim=R
    )
    verts, faces = build_solid_hex_mesh(Xm, Ym, Zm, dist_m, include_magnet=include_magnet)
    stats = verify_watertight(verts, faces)

    stl_path = os.path.join(out_dir, f"{prefix}.stl")
    obj_path = os.path.join(out_dir, f"{prefix}.obj")
    write_binary_stl(verts, faces, stl_path, label=prefix)
    write_obj_mtl(verts, faces, obj_path, diff_name)
    print(f"[OK] {prefix}: V={stats['V']} E={stats['E']} F={stats['F']} chi={stats['chi']} watertight={stats['watertight']} vol={stats['vol_cm3']:.2f}cm3 -> {stl_path}")
    if not stats["watertight"]:
        raise RuntimeError(f"Non-watertight mesh generated: {stats}")
    return stats


def main():
    p = argparse.ArgumentParser(description="Project Nano Banana heightmap + albedo onto watertight Catan hex CAD tile.")
    p.add_argument("--heightmap", help="Path to strictly top-down grayscale heightmap image.")
    p.add_argument("--albedo", default=None, help="Optional path to UV-aligned top-down diffuse albedo PNG/JPG.")
    p.add_argument("--out-dir", default=".", help="Output folder for STL, OBJ/MTL, and PBR PNG textures.")
    p.add_argument("--prefix", default="catan_nanobanana_tile", help="Output filename prefix.")
    p.add_argument("--relief-mm", type=float, default=58.0, help="Target peak height above Z=25mm plinth (default 58.0).")
    p.add_argument("--nx", type=int, default=260, help="Horizontal mesh grid samples across apothem span.")
    p.add_argument("--tex-size", type=int, default=1024, help="UV texture resolution (default 1024).")
    p.add_argument("--micro-detail", type=float, default=0.8, help="Physical millimeter micro-relief amplitude (0 to disable).")
    p.add_argument("--no-magnet", action="store_true", help="Disable bottom Ø10.2x5.2mm N52 magnet pocket.")
    p.add_argument("--self-test", action="store_true", help="Run a synthetic heightmap/albedo verification in a temp directory.")
    args = p.parse_args()

    if args.self_test:
        with tempfile.TemporaryDirectory() as tmp:
            yy, xx = np.mgrid[0:512, 0:512]
            r = np.hypot((xx - 256) / 200.0, (yy - 256) / 200.0)
            hm = np.clip(np.exp(-2.6 * r * r) * 255.0, 0, 255).astype(np.uint8)
            alb = np.stack([np.clip(80 + 0.65 * hm, 0, 255), np.clip(95 + 0.45 * hm, 0, 255), np.clip(70 + 0.55 * hm, 0, 255)], axis=-1).astype(np.uint8)
            run_pipeline(hm, alb, tmp, "self_test_tile", 42.0, nx=96, tex_size=256, micro_detail_mm=0.5, include_magnet=True)
        return

    if not args.heightmap:
        p.error("--heightmap is required unless --self-test is set.")
    hm_arr = np.asarray(Image.open(args.heightmap).convert("L"))
    albedo_arr = np.asarray(Image.open(args.albedo).convert("RGB")) if args.albedo else None
    run_pipeline(hm_arr, albedo_arr, args.out_dir, args.prefix, args.relief_mm, args.nx, args.tex_size, args.micro_detail, not args.no_magnet)


if __name__ == "__main__":
    main()
