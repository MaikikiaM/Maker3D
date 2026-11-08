import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from geo_tools import *

p, n, f = get_helmet_samples()
g = load_greeblies()
layers = [(p, n, np.full((len(p), 3), 235, np.uint8))]
gcol = {"mouth": (40, 40, 40), "ear_l": (150, 200, 150), "ear_r": (150, 200, 150), "aer_l": (120, 120, 200), "aer_r": (120, 120, 200)}
for k, m in g.items():
    gp, gn, _ = sample_mesh(m, 400_000)
    layers.append((gp, gn, np.tile(np.array(gcol[k], np.uint8), (len(gp), 1))))

out = os.path.join(SCR, "geo")
os.makedirs(out, exist_ok=True)
for v in ["front", "left", "back", "top"]:
    img, meta = splat(layers, v, mmpp=0.3)
    img = draw_grid(img, meta, step=10, label_every=20)
    img.save(os.path.join(out, f"bare_{v}.png"))
    print(v, img.size)
