"""Genera laminas comparativas lado a lado con el MISMO angulo de camara que las referencias canonicas."""
import os
import sys
from PIL import Image, ImageDraw

BASE_DIR = r"c:\Users\migue\Proyectos IA varios\Miscelanea"
SCR = os.path.join(BASE_DIR, "scratch")
GEO = os.path.join(SCR, "geo")
ART = r"C:\Users\migue\.gemini\antigravity\brain\708952eb-13ab-449a-b9c1-cf788f04feb2"


def fit_box(im, w, h, bg=(245, 246, 248)):
    im = im.convert("RGB")
    s = min(w / im.width, h / im.height)
    nw, nh = max(1, int(im.width * s)), max(1, int(im.height * s))
    rs = im.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (w, h), bg)
    canvas.paste(rs, ((w - nw) // 2, (h - nh) // 2))
    return canvas


def build_sheet(tag="v2"):
    pairs = [
        ("Canon TCW S6 (CGI)", os.path.join(ART, "thorn_official_headshot.png"),
         f"Prototipo {tag} - Mismo angulo 3/4", os.path.join(SCR, f"proto_{tag}_tcw_ref_side.png")),
        ("Zoom Canon TCW (Sombra levantada)", os.path.join(GEO, "zoom_tcw_shadow_lifted.png"),
         f"Prototipo {tag} - Macro Mismo Angulo", os.path.join(SCR, f"proto_{tag}_macro_side_jaw.png")),
        ("Diagrama Vectorial TCW", os.path.join(GEO, "vec_full_zoom.png"),
         f"Prototipo {tag} - Macro Ala Derecha", os.path.join(SCR, f"proto_{tag}_macro_wing.png")),
        ("Casco Fisico Trasera", os.path.join(GEO, "crop_rear_detail.png"),
         f"Prototipo {tag} - Trasera", os.path.join(SCR, f"proto_{tag}_back.png")),
    ]
    cw, ch = 620, 600
    sheet = Image.new("RGB", (cw * 2 + 30, (ch + 36) * len(pairs) + 20), (22, 24, 30))
    dr = ImageDraw.Draw(sheet)
    for idx, (t1, p1, t2, p2) in enumerate(pairs):
        y0 = 10 + idx * (ch + 36)
        dr.text((20, y0 + 8), t1, fill=(240, 242, 246))
        dr.text((cw + 30, y0 + 8), t2, fill=(255, 110, 110))
        if os.path.exists(p1):
            sheet.paste(fit_box(Image.open(p1), cw, ch), (10, y0 + 28))
        if os.path.exists(p2):
            sheet.paste(fit_box(Image.open(p2), cw, ch), (cw + 20, y0 + 28))
    out = os.path.join(SCR, f"compare_{tag}.png")
    sheet.save(out)
    print("Saved comparison sheet:", out)


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "v2"
    build_sheet(tag)
