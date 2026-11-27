"""Independent CAD invariant check and 2D-to-3D Clay parity crop audit for Montana v4."""
import json
import os
import struct
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
MONTANA_DIR = os.path.join(ROOT, "montana")
WORK_DIR = os.path.join(MONTANA_DIR, "work_v4")

R = 64.0
APOTHEM = 55.42562584
Z_BASE = 25.0
ROAD_MARGIN = 3.50
PAD_RADIUS = 10.00


def hex_dist_pts(x, y):
    ax, ay = np.abs(x), np.abs(y)
    return np.minimum(APOTHEM - ax, (2.0 * APOTHEM - ax - np.sqrt(3.0) * ay) * 0.5)


def vert_dist_pts(x, y):
    angles = np.radians([30, 90, 150, 210, 270, 330])
    vx = R * np.cos(angles)[None, :]
    vy = R * np.sin(angles)[None, :]
    return np.min(np.hypot(x[:, None] - vx, y[:, None] - vy), axis=1)


def audit_stl_invariants(stl_path):
    """Audit binary STL bounds, road/pad flatness, magnet pocket, and degenerate triangle count."""
    with open(stl_path, "rb") as f:
        header = f.read(80)
        n_tris = struct.unpack("<I", f.read(4))[0]

    dt = np.dtype([
        ("n", "<f4", (3,)),
        ("v0", "<f4", (3,)),
        ("v1", "<f4", (3,)),
        ("v2", "<f4", (3,)),
        ("attr", "<u2"),
    ])
    tris = np.memmap(stl_path, dtype=dt, mode="r", offset=84, shape=(n_tris,))

    degen = 0
    z_min, z_max = 1e9, -1e9
    x_min, x_max = 1e9, -1e9
    y_min, y_max = 1e9, -1e9
    max_road_err = 0.0
    max_pad_err = 0.0
    mag_pocket_hits = 0

    chunk = 500_000
    for i0 in range(0, n_tris, chunk):
        c = tris[i0 : i0 + chunk]
        v0, v1, v2 = c["v0"], c["v1"], c["v2"]
        cr = np.cross(v1 - v0, v2 - v0)
        area2 = np.linalg.norm(cr, axis=1)
        degen += int(np.sum(area2 <= 1e-10))

        pts = np.concatenate([v0, v1, v2], axis=0)
        z_min = min(z_min, float(pts[:, 2].min()))
        z_max = max(z_max, float(pts[:, 2].max()))
        x_min = min(x_min, float(pts[:, 0].min()))
        x_max = max(x_max, float(pts[:, 0].max()))
        y_min = min(y_min, float(pts[:, 1].min()))
        y_max = max(y_max, float(pts[:, 1].max()))

        top_m = pts[:, 2] >= (Z_BASE - 0.5)
        if np.any(top_m):
            pt = pts[top_m]
            dh = hex_dist_pts(pt[:, 0], pt[:, 1])
            dv = vert_dist_pts(pt[:, 0], pt[:, 1])
            in_road = (dh >= -0.05) & (dh <= ROAD_MARGIN - 0.15)
            if np.any(in_road):
                max_road_err = max(max_road_err, float(np.max(np.abs(pt[in_road, 2] - Z_BASE))))
            in_pad = dv <= (PAD_RADIUS - 0.15)
            if np.any(in_pad):
                max_pad_err = max(max_pad_err, float(np.max(np.abs(pt[in_pad, 2] - Z_BASE))))

        bot_m = (pts[:, 2] >= 4.5) & (pts[:, 2] <= 5.4) & (np.hypot(pts[:, 0], pts[:, 1]) <= 5.25)
        mag_pocket_hits += int(np.sum(bot_m))

    metrics_path = os.path.join(MONTANA_DIR, "master_metrics_v2.json")
    metrics = json.load(open(metrics_path, encoding="utf-8")) if os.path.exists(metrics_path) else {}
    res = {
        "stl_mb": round(os.path.getsize(stl_path) / (1024 * 1024), 2),
        "n_tris": int(n_tris),
        "degen": int(degen),
        "bounds_x": [round(x_min, 3), round(x_max, 3)],
        "bounds_y": [round(y_min, 3), round(y_max, 3)],
        "bounds_z": [round(z_min, 3), round(z_max, 3)],
        "max_road_err_mm": round(max_road_err, 5),
        "max_pad_err_mm": round(max_pad_err, 5),
        "mag_pocket_verts": int(mag_pocket_hits),
        "master_metrics": metrics,
    }
    print("[STL Audit]", json.dumps(res, indent=2))
    assert n_tris <= 10_000_000, f"STL tris {n_tris} exceeds 10M limit!"
    assert degen == 0, f"Found {degen} degenerate triangles!"
    assert max_road_err < 0.05, f"Road lane not flat: {max_road_err}"
    assert max_pad_err < 0.05, f"Pad not flat: {max_pad_err}"
    assert mag_pocket_hits > 0, "Bottom magnet pocket missing!"
    if "master" in metrics:
        assert metrics["master"]["watertight"] is True
        assert metrics["master"]["chi"] == 2
    # Check that high vertical rock walls on the 360-deg north face stay crisp rock (zero north snow bleed)
    alb = np.flipud(np.asarray(Image.open(os.path.join(MONTANA_DIR, "catan_mountain_diffuse_2k.png")).convert("RGB")) / 255.0)
    z_grid = np.flipud(np.load(os.path.join(WORK_DIR, "montana_v4_hm_2k.npy")).astype(np.float32)) / 255.0 * 45.2 + Z_BASE
    from scipy import ndimage as ndi
    ah, aw = z_grid.shape
    a_sm = np.asarray(Image.fromarray((alb * 255).astype(np.uint8)).resize((aw, ah), Image.BILINEAR)) / 255.0
    dx = 2.0 * R / (aw - 1)
    gy, gx = np.gradient(ndi.gaussian_filter(z_grid, 0.35 / dx), dx, dx)
    sl = np.degrees(np.arctan(np.hypot(gx, gy)))
    lum = 0.2126 * a_sm[..., 0] + 0.7152 * a_sm[..., 1] + 0.0722 * a_sm[..., 2]
    sat = np.ptp(a_sm, axis=-1)
    y_world = np.linspace(-R, R, ah)[:, None]
    north_steep_snow = float(np.mean((y_world > 12.0) & (sl >= 46.0) & (lum > 0.72) & (sat < 0.11))) * 100.0
    print(f"[North Wall Snow Audit] north>=46deg snow = {north_steep_snow:.4f}%")
    assert north_steep_snow < 0.5, f"North cliff snow bleed too high: {north_steep_snow:.4f}%"
    return res


def render_mesh_crop_clay(verts, faces, x0, x1, y0, y1, out_sz=420):
    """Rasterize true 3D untextured clay triangles in [x0, x1] x [y0, y1] orthographic view."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cx = (v0[:, 0] + v1[:, 0] + v2[:, 0]) / 3.0
    cy = (v0[:, 1] + v1[:, 1] + v2[:, 1]) / 3.0
    cz = (v0[:, 2] + v1[:, 2] + v2[:, 2]) / 3.0
    keep = (cx >= x0 - 1.5) & (cx <= x1 + 1.5) & (cy >= y0 - 1.5) & (cy <= y1 + 1.5) & (cz >= Z_BASE - 0.1)
    idx = np.where(keep)[0]
    p0, p1, p2 = v0[idx], v1[idx], v2[idx]
    nrm = np.cross(p1 - p0, p2 - p0)
    nlen = np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    nrm = nrm / nlen
    pos_z = nrm[:, 2] > 0.02
    p0, p1, p2, nrm = p0[pos_z], p1[pos_z], p2[pos_z], nrm[pos_z]

    ld = np.array([0.66, -0.35, 0.66], dtype=np.float32)
    ld /= np.linalg.norm(ld)
    shade = np.clip(0.32 + 0.68 * np.maximum(0.0, np.sum(nrm * ld[None, :], axis=1)), 0.12, 1.0)
    # Crisp raster via barycentric scanline over small bounding boxes
    img = np.full((out_sz, out_sz), 0.78, dtype=np.float32)
    zbuf = np.full((out_sz, out_sz), -1e9, dtype=np.float32)
    sx = (out_sz - 1) / (x1 - x0)
    sy = (out_sz - 1) / (y1 - y0)

    def to_px(pt):
        px = (pt[:, 0] - x0) * sx
        py = (y1 - pt[:, 1]) * sy  # north=+y -> top row=0
        return px, py

    xA, yA = to_px(p0)
    xB, yB = to_px(p1)
    xC, yC = to_px(p2)
    denom = (yB - yC) * (xA - xC) + (xC - xB) * (yA - yC)
    valid = np.abs(denom) > 1e-5
    xA, yA, xB, yB, xC, yC, p0, p1, p2, shade, denom = (
        arr[valid] for arr in (xA, yA, xB, yB, xC, yC, p0, p1, p2, shade, denom)
    )
    ux0 = np.clip(np.floor(np.minimum(np.minimum(xA, xB), xC)).astype(np.int32), 0, out_sz - 1)
    ux1 = np.clip(np.ceil(np.maximum(np.maximum(xA, xB), xC)).astype(np.int32), 0, out_sz - 1)
    uy0 = np.clip(np.floor(np.minimum(np.minimum(yA, yB), yC)).astype(np.int32), 0, out_sz - 1)
    uy1 = np.clip(np.ceil(np.maximum(np.maximum(yA, yB), yC)).astype(np.int32), 0, out_sz - 1)
    for k in range(len(denom)):
        ax0, ax1, ay0, ay1 = int(ux0[k]), int(ux1[k]), int(uy0[k]), int(uy1[k])
        d = float(denom[k])
        bx, by, cx_, cy_, ax_, ay_ = float(xB[k]), float(yB[k]), float(xC[k]), float(yC[k]), float(xA[k]), float(yA[k])
        z0, z1, z2, sh = float(p0[k, 2]), float(p1[k, 2]), float(p2[k, 2]), float(shade[k])
        for py in range(ay0, ay1 + 1):
            for px in range(ax0, ax1 + 1):
                w1 = ((by - cy_) * (px - cx_) + (cx_ - bx) * (py - cy_)) / d
                w2 = ((cy_ - ay_) * (px - cx_) + (ax_ - cx_) * (py - cy_)) / d
                w3 = 1.0 - w1 - w2
                if w1 >= -0.002 and w2 >= -0.002 and w3 >= -0.002:
                    zz = w1 * z0 + w2 * z1 + w3 * z2
                    if zz > zbuf[py, px]:
                        zbuf[py, px] = zz
                        img[py, px] = sh
    return np.repeat(np.clip(img * 255.0, 0, 255).astype(np.uint8)[..., None], 3, axis=-1)


def render_parity_crops():
    """Generate side-by-side 2D Albedo vs 3D Mesh Clay zoomed crops (SKILL.md Rule #7)."""
    z = np.load(os.path.join(MONTANA_DIR, "viewer_mesh_v2.npz"))
    verts, faces = z["verts"], z["faces"]
    alb = np.asarray(Image.open(os.path.join(MONTANA_DIR, "catan_mountain_diffuse_2k.png")).convert("RGB"))
    ah, aw = alb.shape[:2]

    quads = [
        ("Front Canyon & Copper Scree", -20.0, 22.0, -48.0, -6.0),
        ("Central Peak Granite Clefts", -18.0, 24.0, -8.0, 34.0),
        ("Left Spur Snow Bowl & Arete", -48.0, -6.0, -24.0, 18.0),
        ("North 180 Cirques & Buttresses", -28.0, 28.0, 4.0, 48.0),
    ]
    rows = []
    try:
        fnt = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 18)
    except Exception:
        fnt = ImageFont.load_default()

    for title, x0, x1, y0, y1 in quads:
        # Map world [-R, +R] -> pixel [0, aw] (row 0 = +R = north!)
        u0 = int(np.clip((x0 + R) / (2.0 * R) * aw, 0, aw - 1))
        u1 = int(np.clip((x1 + R) / (2.0 * R) * aw, 0, aw - 1))
        v0 = int(np.clip((R - y1) / (2.0 * R) * ah, 0, ah - 1))
        v1 = int(np.clip((R - y0) / (2.0 * R) * ah, 0, ah - 1))
        crop_2d = Image.fromarray(alb[v0:v1, u0:u1]).resize((420, 420), Image.LANCZOS)
        crop_3d = Image.fromarray(render_mesh_crop_clay(verts, faces, x0, x1, y0, y1, out_sz=420))
        d1 = ImageDraw.Draw(crop_2d); d1.text((10, 8), f"2D Albedo: {title}", fill=(255, 240, 180), font=fnt)
        d2 = ImageDraw.Draw(crop_3d); d2.text((10, 8), f"3D Clay Mesh: {title}", fill=(20, 30, 45), font=fnt)
        rows.append(np.hstack([np.asarray(crop_2d), np.asarray(crop_3d)]))

    panel = np.vstack([
        np.hstack([rows[0], rows[1]]),
        np.hstack([rows[2], rows[3]]),
    ])
    os.makedirs(WORK_DIR, exist_ok=True)
    out_p = os.path.join(WORK_DIR, "v4_2d_vs_3d_parity_crops.png")
    Image.fromarray(panel).save(out_p)
    print(f"[Parity OK] Saved {out_p} ({panel.shape[1]}x{panel.shape[0]})")


if __name__ == "__main__":
    audit_stl_invariants(os.path.join(MONTANA_DIR, "catan_montana.stl"))
    render_parity_crops()
