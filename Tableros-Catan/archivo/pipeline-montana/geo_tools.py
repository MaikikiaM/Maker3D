"""Herramientas de analisis geometrico y render ortografico por splatting (z-buffer).
Coordenadas del casco: X lateral (+X = lado izquierdo del casco visto de frente? se verifica),
Y: frente negativo (-97) -> nuca positivo (+223), Z: arriba (1616 -> 1948).
"""
import os
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = r"c:\Users\migue\Proyectos IA varios\Miscelanea"
CASCO = os.path.join(BASE_DIR, r"Diseños 3D\Casco")
GREEB = os.path.join(CASCO, "Greeblies")
SCR = os.path.join(BASE_DIR, "scratch")
CACHE = os.path.join(SCR, "helmet_samples.npz")


def load_helmet():
    return trimesh.load(os.path.join(CASCO, "Phase 2 Main Body.stl"))


def load_greeblies():
    out = {}
    for key, fn in [("mouth", "Mouth Center Detail.stl"), ("ear_l", "Ear Piece Left.stl"),
                    ("ear_r", "Ear Piece Right.stl"), ("aer_l", "Aerator Left - Mesh.stl"),
                    ("aer_r", "Aerator Right - Mesh.stl")]:
        out[key] = trimesh.load(os.path.join(GREEB, fn))
    return out


def sample_mesh(mesh, n, seed=0):
    pts, fidx = trimesh.sample.sample_surface(mesh, n, seed=seed)
    nrm = mesh.face_normals[fidx]
    return pts.astype(np.float32), nrm.astype(np.float32), fidx.astype(np.int32)


def get_helmet_samples(n=6_000_000):
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        return d["p"], d["n"], d["f"]
    m = load_helmet()
    p, nn, f = sample_mesh(m, n)
    np.savez(CACHE, p=p, n=nn, f=f)
    return p, nn, f


# Vistas: (vector hacia el que mira la camara, eje 'derecha' en pantalla, eje 'arriba' en pantalla)
VIEWS = {
    "front": (np.array([0, 1, 0.]), np.array([1, 0, 0.]), np.array([0, 0, 1.])),   # camara en -Y mirando +Y
    "back": (np.array([0, -1, 0.]), np.array([-1, 0, 0.]), np.array([0, 0, 1.])),
    "left": (np.array([-1, 0, 0.]), np.array([0, 1, 0.]), np.array([0, 0, 1.])),   # camara en +X mirando -X; frente a la izquierda
    "right": (np.array([1, 0, 0.]), np.array([0, -1, 0.]), np.array([0, 0, 1.])),
    "top": (np.array([0, 0, -1.]), np.array([1, 0, 0.]), np.array([0, -1, 0.])),  # frente arriba
}


def view_basis(name_or_dir):
    if isinstance(name_or_dir, str):
        return VIEWS[name_or_dir]
    d = np.asarray(name_or_dir, float)
    d /= np.linalg.norm(d)
    up = np.array([0, 0, 1.])
    r = np.cross(d, up)
    if np.linalg.norm(r) < 1e-6:
        r = np.array([1, 0, 0.])
    r /= np.linalg.norm(r)
    u = np.cross(r, d)
    return d, r, u


def splat(layers, view, mmpp=0.35, bounds=None, light=None, bg=(255, 255, 255), dilate=1):
    """layers: lista de (pts, nrms, colors(N,3 uint8)). Devuelve imagen PIL y metadatos de proyeccion."""
    d, r, u = view_basis(view)
    allp = np.concatenate([L[0] for L in layers])
    alln = np.concatenate([L[1] for L in layers])
    allc = np.concatenate([L[2] for L in layers]).astype(np.float32)
    sx = allp @ r
    sy = allp @ u
    dep = allp @ d
    if bounds is None:
        x0, x1, y0, y1 = sx.min() - 5, sx.max() + 5, sy.min() - 5, sy.max() + 5
    else:
        x0, x1, y0, y1 = bounds
    W = int((x1 - x0) / mmpp)
    H = int((y1 - y0) / mmpp)
    px = ((sx - x0) / mmpp).astype(np.int64)
    py = ((y1 - sy) / mmpp).astype(np.int64)
    ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    # back-face culling suave: solo puntos que miran a camara
    facing = (alln @ d) < 0.05
    ok &= facing
    px, py, dep, nn, cc = px[ok], py[ok], dep[ok], alln[ok], allc[ok]
    key = py * W + px
    order = np.lexsort((dep, key))
    key_s = key[order]
    first = np.ones(len(key_s), bool)
    first[1:] = key_s[1:] != key_s[:-1]
    sel = order[first]
    if light is None:
        light = -d + 0.35 * u + 0.25 * r
    light = light / np.linalg.norm(light)
    lam = np.clip(nn[sel] @ light, 0, 1)
    shade = 0.35 + 0.65 * lam
    col = np.clip(cc[sel] * shade[:, None], 0, 255)
    img = np.zeros((H * W, 3), np.float32)
    img[:] = bg
    mask = np.zeros(H * W, bool)
    depth = np.full(H * W, np.inf, np.float32)
    img[key[sel]] = col
    mask[key[sel]] = True
    depth[key[sel]] = dep[sel]
    img = img.reshape(H, W, 3)
    mask = mask.reshape(H, W)
    # relleno de huecos por dilatacion simple
    for _ in range(dilate):
        holes = ~mask
        acc = np.zeros_like(img)
        cnt = np.zeros((H, W), np.float32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                sm = np.roll(np.roll(mask, dy, 0), dx, 1)
                si = np.roll(np.roll(img, dy, 0), dx, 1)
                acc += si * sm[..., None]
                cnt += sm
        fill = holes & (cnt >= 5)
        img[fill] = acc[fill] / cnt[fill][:, None]
        mask = mask | fill
    meta = dict(x0=x0, x1=x1, y0=y0, y1=y1, mmpp=mmpp, W=W, H=H, r=r, u=u, d=d)
    return Image.fromarray(img.astype(np.uint8)), meta


def draw_grid(img, meta, step=10, label_every=20, color=(0, 140, 255)):
    im = img.copy()
    dr = ImageDraw.Draw(im, "RGBA")
    x0, x1, y0, y1, mmpp = meta["x0"], meta["x1"], meta["y0"], meta["y1"], meta["mmpp"]
    gx = np.arange(np.ceil(x0 / step) * step, x1, step)
    gy = np.arange(np.ceil(y0 / step) * step, y1, step)
    for v in gx:
        X = (v - x0) / mmpp
        strong = abs(v % label_every) < 1e-6
        dr.line([(X, 0), (X, meta["H"])], fill=color + ((110,) if strong else (45,)), width=1)
        if strong:
            dr.text((X + 2, 2), f"{v:.0f}", fill=(0, 60, 200, 255))
    for v in gy:
        Y = (y1 - v) / mmpp
        strong = abs(v % label_every) < 1e-6
        dr.line([(0, Y), (meta["W"], Y)], fill=color + ((110,) if strong else (45,)), width=1)
        if strong:
            dr.text((2, Y + 2), f"{v:.0f}", fill=(0, 60, 200, 255))
    return im


def to_px(meta, P):
    P = np.atleast_2d(P)
    sx = P @ meta["r"]
    sy = P @ meta["u"]
    return np.column_stack([(sx - meta["x0"]) / meta["mmpp"], (meta["y1"] - sy) / meta["mmpp"]])
