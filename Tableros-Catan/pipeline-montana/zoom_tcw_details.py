import os
import numpy as np
from PIL import Image, ImageEnhance

ART = r"C:\Users\migue\.gemini\antigravity\brain\708952eb-13ab-449a-b9c1-cf788f04feb2"
OUT = r"c:\Users\migue\Proyectos IA varios\Miscelanea\scratch\geo"

# 1. Official TCW Headshot bright/contrast zoom on cheek hollow + jaw tube
im1 = Image.open(os.path.join(ART, "thorn_official_headshot.png")).convert("RGB")
w1, h1 = im1.size
# Cheek hollow + jaw tube + teeth region
crop1 = im1.crop((int(w1 * 0.08), int(h1 * 0.35), int(w1 * 0.72), int(h1 * 0.80)))
# Brighten shadows so dark crimson vs white/grey in shadow is 100% clear
arr = np.asarray(crop1, dtype=np.float32) / 255.0
arr_gamma = np.clip(arr ** 0.55, 0, 1) * 255.0
im1_bright = Image.fromarray(arr_gamma.astype(np.uint8))
im1_bright = ImageEnhance.Contrast(im1_bright).enhance(1.35)
im1_bright.resize((crop1.width * 2, crop1.height * 2), Image.Resampling.NEAREST).save(os.path.join(OUT, "zoom_tcw_shadow_lifted.png"))

# 2. Vector diagram zoom on cheek hollow + jaw tube + whiskers
im2 = Image.open(os.path.join(ART, r".user_uploaded\media_1792280941574_1d9305e8.png")).convert("RGB")
w2, h2 = im2.size
crop2 = im2.crop((int(w2 * 0.05), int(h2 * 0.42), int(w2 * 0.85), int(h2 * 0.98)))
crop2.resize((crop2.width * 2, crop2.height * 2), Image.Resampling.NEAREST).save(os.path.join(OUT, "zoom_vec_cheek_tube.png"))

# 3. Also check the 3 images from step 310 to see all references the user sent!
for idx, fn in enumerate(["media_1792245174317_edc90dfd.png", "media_1792245188728_6271f387.png", "media_1792245235973_de5ec8f4.png"]):
    p = os.path.join(ART, ".user_uploaded", fn)
    if os.path.exists(p):
        im = Image.open(p)
        print(f"Step 310 ref {idx}: {fn} size={im.size}")

print("Done:", w1, h1, "|", w2, h2)
