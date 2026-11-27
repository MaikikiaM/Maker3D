from PIL import Image
import os

O = r"c:\Users\migue\Proyectos IA varios\Miscelanea\scratch\geo"
im1 = Image.open(r"C:\Users\migue\.gemini\antigravity\brain\708952eb-13ab-449a-b9c1-cf788f04feb2\thorn_official_headshot.png")
im2 = Image.open(r"C:\Users\migue\.gemini\antigravity\brain\708952eb-13ab-449a-b9c1-cf788f04feb2\.user_uploaded\media_1792280941574_1d9305e8.png")
print("headshot:", im1.size, "vector:", im2.size)

# Crops of official TCW screenshot
w, h = im1.size
im1.crop((int(w*0.20), int(h*0.02), int(w*0.82), int(h*0.30))).save(os.path.join(O, "tcw_dome_wing.png"))
im1.crop((int(w*0.25), int(h*0.22), int(w*0.85), int(h*0.62))).save(os.path.join(O, "tcw_eyes_cheeks.png"))
im1.crop((int(w*0.05), int(h*0.42), int(w*0.85), int(h*0.82))).save(os.path.join(O, "tcw_jaw_bolt_whiskers.png"))

# Crops of clean vector diagram
w2, h2 = im2.size
im2.crop((int(w2*0.12), int(h2*0.06), int(w2*0.95), int(h2*0.95))).save(os.path.join(O, "vec_full_zoom.png"))
