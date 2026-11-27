import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from geo_tools import *

FE_CACHE = os.path.join(SCR, "helmet_feature_edges.npz")


def feature_edges(mesh, ang_deg=22.0):
    if os.path.exists(FE_CACHE):
        d = np.load(FE_CACHE)
        return d["a"], d["b"], d["ang"]
    ang = mesh.face_adjacency_angles
    sel = ang > np.radians(ang_deg)
    e = mesh.face_adjacency_edges[sel]
    V = mesh.vertices
    a, b = V[e[:, 0]], V[e[:, 1]]
    np.savez(FE_CACHE, a=a, b=b, ang=ang[sel])
    return a, b, ang[sel]


def depth_buffer(p, n, view, meta):
    d, r, u = meta["d"], meta["r"], meta["u"]
    sx, sy, dep = p @ r, p @ u, p @ d
    W, H, mmpp = meta["W"], meta["H"], meta["mmpp"]
    px = ((sx - meta["x0"]) / mmpp).astype(np.int64)
    py = ((meta["y1"] - sy) / mmpp).astype(np.int64)
    ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    zb = np.full(H * W, np.inf, np.float32)
    np.minimum.at(zb, (py[ok] * W + px[ok]), dep[ok])
    zb = zb.reshape(H, W)
    # erosion: minimo local 3x3 para robustez
    from scipy.ndimage import minimum_filter
    return minimum_filter(zb, size=5)


def draw_features(img, meta, zb, a, b, ang, tol=2.5, steps=6):
    dr = ImageDraw.Draw(img, "RGBA")
    d, r, u = meta["d"], meta["r"], meta["u"]
    W, H, mmpp = meta["W"], meta["H"], meta["mmpp"]
    t = np.linspace(0, 1, steps)
    for i0 in range(0, len(a), 200000):
        A, B, G = a[i0:i0 + 200000], b[i0:i0 + 200000], ang[i0:i0 + 200000]
        mid = 0.5 * (A + B)
        sx, sy, dep = mid @ r, mid @ u, mid @ d
        px = ((sx - meta["x0"]) / mmpp).astype(np.int64)
        py = ((meta["y1"] - sy) / mmpp).astype(np.int64)
        ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        vis = np.zeros(len(A), bool)
        vis[ok] = dep[ok] <= zb[py[ok], px[ok]] + tol
        PA, PB = to_px(meta, A[vis]), to_px(meta, B[vis])
        GG = G[vis]
        for (x0, y0), (x1, y1), g in zip(PA, PB, GG):
            c = (200, 0, 0, 255) if g > np.radians(60) else (230, 90, 0, 200)
            dr.line([(x0, y0), (x1, y1)], fill=c, width=1)
    return img


if __name__ == "__main__":
    mesh = load_helmet()
    a, b, ang = feature_edges(mesh)
    print("feature edges:", len(a))
    p, n, f = get_helmet_samples()
    g = load_greeblies()
    layers = [(p, n, np.full((len(p), 3), 238, np.uint8))]
    gcol = {"mouth": (60, 60, 60), "ear_l": (170, 220, 170), "ear_r": (170, 220, 170), "aer_l": (150, 150, 220), "aer_r": (150, 150, 220)}
    for k, m in g.items():
        gp, gn, _ = sample_mesh(m, 300_000)
        layers.append((gp, gn, np.tile(np.array(gcol[k], np.uint8), (len(gp), 1))))
    out = os.path.join(SCR, "geo")
    views = {"front": "front", "left": "left", "back": "back", "top": "top"}
    for name, v in views.items():
        img, meta = splat(layers, v, mmpp=0.3)
        zb = depth_buffer(p, n, v, meta)
        img = draw_features(img, meta, zb, a, b, ang)
        img = draw_grid(img, meta, step=10, label_every=20)
        img.save(os.path.join(out, f"feat_{name}.png"))
        print(name)
