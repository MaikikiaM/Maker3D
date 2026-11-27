"""
Auditor de Montaña DEF (reusa los rasterizadores de montana_audit_parity.py).
  * Invariantes Catan sobre el STL Master (binario, memmap): degenerados, carril/pads planos, bolsillo imán, <=10M tris.
  * Nieve REAL sobre la malla final: fracción de caras con pendiente >= 44° cuyo color de albedo es nieve.
  * Distribución de pendientes del terreno (sin muros de ~85°).
  * Recortes comparativos 2D Albedo vs Arcilla 3D (regla #7 del SKILL).
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import montana_audit_parity as ap  # noqa: E402

R, Z_BASE = ap.R, ap.Z_BASE


def stl_audit(stl_path, metrics_path):
    import struct
    with open(stl_path, "rb") as f:
        f.read(80)
        n_tris = struct.unpack("<I", f.read(4))[0]
    dt = np.dtype([("n", "<f4", (3,)), ("v0", "<f4", (3,)), ("v1", "<f4", (3,)), ("v2", "<f4", (3,)), ("attr", "<u2")])
    tris = np.memmap(stl_path, dtype=dt, mode="r", offset=84, shape=(n_tris,))
    degen = 0
    zmin, zmax = 1e9, -1e9
    road_err = pad_err = 0.0
    mag = 0
    for i0 in range(0, n_tris, 500_000):
        c = tris[i0:i0 + 500_000]
        v0, v1, v2 = c["v0"], c["v1"], c["v2"]
        degen += int(np.sum(np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1) <= 1e-10))
        pts = np.concatenate([v0, v1, v2], axis=0)
        zmin = min(zmin, float(pts[:, 2].min())); zmax = max(zmax, float(pts[:, 2].max()))
        top = pts[:, 2] >= (Z_BASE - 0.5)
        if np.any(top):
            pt = pts[top]
            dh = ap.hex_dist_pts(pt[:, 0], pt[:, 1])
            dv = ap.vert_dist_pts(pt[:, 0], pt[:, 1])
            m_r = (dh >= -0.05) & (dh <= ap.ROAD_MARGIN - 0.15)
            if np.any(m_r):
                road_err = max(road_err, float(np.max(np.abs(pt[m_r, 2] - Z_BASE))))
            m_p = dv <= (ap.PAD_RADIUS - 0.15)
            if np.any(m_p):
                pad_err = max(pad_err, float(np.max(np.abs(pt[m_p, 2] - Z_BASE))))
        mag += int(np.sum((pts[:, 2] >= 4.5) & (pts[:, 2] <= 5.4) & (np.hypot(pts[:, 0], pts[:, 1]) <= 5.25)))
    metrics = json.load(open(metrics_path, encoding="utf-8")) if os.path.exists(metrics_path) else {}
    res = {
        "stl_mb": round(os.path.getsize(stl_path) / 1048576, 2), "n_tris": int(n_tris), "degen": degen,
        "z": [round(zmin, 3), round(zmax, 3)], "road_err_mm": round(road_err, 5), "pad_err_mm": round(pad_err, 5),
        "mag_pocket_verts": mag, "master": metrics.get("master"), "viewer": metrics.get("viewer"),
    }
    print("[STL Audit]", json.dumps(res))
    assert n_tris <= 10_000_000 and degen == 0 and road_err < 0.05 and pad_err < 0.05 and mag > 0
    if res["master"]:
        assert res["master"]["watertight"] is True and res["master"]["chi"] == 2
    return res


def mesh_audits(mesh_dir):
    z = np.load(os.path.join(mesh_dir, "viewer_mesh_v2.npz"))
    v, f = z["verts"].astype(np.float64), z["faces"]
    alb = np.asarray(Image.open(os.path.join(mesh_dir, "catan_mountain_diffuse_2k.png")).convert("RGB"), dtype=np.float32) / 255.0
    ah, aw = alb.shape[:2]
    p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cen = (p0 + p1 + p2) / 3.0
    nrm = np.cross(p1 - p0, p2 - p0)
    area = np.linalg.norm(nrm, axis=1) + 1e-12
    nz = np.abs(nrm[:, 2]) / area
    slope = np.degrees(np.arccos(np.clip(nz, 0, 1)))
    terr = (cen[:, 2] > Z_BASE + 0.3) & (ap.hex_dist_pts(cen[:, 0], cen[:, 1]) > ap.ROAD_MARGIN + 0.5)
    px = np.clip(np.round((cen[:, 0] + R) / (2 * R) * (aw - 1)), 0, aw - 1).astype(int)
    py = np.clip(np.round((R - cen[:, 1]) / (2 * R) * (ah - 1)), 0, ah - 1).astype(int)
    col = alb[py, px]
    lum = 0.2126 * col[:, 0] + 0.7152 * col[:, 1] + 0.0722 * col[:, 2]
    sat = col.max(1) - col.min(1)
    snow = (lum > 0.68) & (sat < 0.12)
    w = area * terr
    snow_steep = float(np.sum(w * (snow & (slope >= 44.0))) / max(np.sum(w), 1e-9) * 100.0)
    snow_total = float(np.sum(w * snow) / max(np.sum(w), 1e-9) * 100.0)
    # pendiente macro (suavizada por celdas de 1 mm) para estadísticas de "muros"
    pct = [float(np.sum(w * (slope >= t)) / max(np.sum(w), 1e-9) * 100.0) for t in (60.0, 70.0, 80.0)]
    print(f"[Mesh Audit] snow_total={snow_total:.2f}% | snow on >=44deg faces={snow_steep:.4f}% | faces>=60/70/80deg = {pct[0]:.1f}/{pct[1]:.1f}/{pct[2]:.1f}% | Ztop={cen[:,2].max():.2f}")
    return {"snow_total": snow_total, "snow_steep": snow_steep, "steep_pct": pct}, (v, f, alb)


def parity_crops(mesh_dir, out_path, bundle):
    v, f, alb = bundle
    ah, aw = alb.shape[:2]
    quads = [("Cañon Sur + cascajo", -22, 22, -50, -6), ("Aguja central", -18, 24, -8, 34),
             ("Flanco izq. neveros", -48, -6, -24, 18), ("Circo Norte", -26, 26, 6, 50)]
    try:
        fnt = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 18)
    except Exception:
        fnt = ImageFont.load_default()
    cells = []
    for t, x0, x1, y0, y1 in quads:
        u0 = int(np.clip((x0 + R) / (2 * R) * aw, 0, aw - 1)); u1 = int(np.clip((x1 + R) / (2 * R) * aw, 0, aw - 1))
        v0 = int(np.clip((R - y1) / (2 * R) * ah, 0, ah - 1)); v1 = int(np.clip((R - y0) / (2 * R) * ah, 0, ah - 1))
        c2 = Image.fromarray((alb[v0:v1, u0:u1] * 255).astype(np.uint8)).resize((420, 420), Image.LANCZOS)
        c3 = Image.fromarray(ap.render_mesh_crop_clay(v, f, x0, x1, y0, y1, out_sz=420))
        ImageDraw.Draw(c2).text((8, 6), "2D " + t, fill=(255, 240, 120), font=fnt)
        ImageDraw.Draw(c3).text((8, 6), "3D " + t, fill=(20, 30, 45), font=fnt)
        cells.append(np.hstack([np.asarray(c2), np.asarray(c3)]))
    Image.fromarray(np.vstack([np.hstack(cells[:2]), np.hstack(cells[2:])])).save(out_path)
    print("[Parity] saved", out_path)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--slug", default="montana_def")
    a.add_argument("--skip-stl", action="store_true")
    a = a.parse_args()
    d = os.path.join(HERE, a.slug)
    if not a.skip_stl:
        stl_audit(os.path.join(d, f"catan_{a.slug}.stl"), os.path.join(d, "master_metrics_v2.json"))
    res, bundle = mesh_audits(d)
    os.makedirs(os.path.join(d, "work"), exist_ok=True)
    parity_crops(d, os.path.join(d, "work", "def_2d_vs_3d_parity_crops.png"), bundle)
