import os
from PIL import Image
F = r"c:\Users\migue\Proyectos IA varios\Miscelanea\Diseños 3D\Casco\Fotos"
O = r"c:\Users\migue\Proyectos IA varios\Miscelanea\scratch\geo"
crops = [
    ("WhatsApp Image 2026-10-17 at 17.40.56 (3).jpeg", (100, 850, 1000, 1500), "crop_side_lower.png"),
    ("WhatsApp Image 2026-10-17 at 17.40.56 (3).jpeg", (150, 0, 1100, 650), "crop_side_upper.png"),
    ("WhatsApp Image 2026-10-17 at 17.40.56.jpeg", (150, 300, 1050, 1000), "crop_front_mid.png"),
    ("WhatsApp Image 2026-10-17 at 17.40.56.jpeg", (150, 950, 1100, 1500), "crop_front_low.png"),
    ("WhatsApp Image 2026-10-17 at 17.40.56 (2).jpeg", (150, 500, 1100, 1300), "crop_34_face.png"),
]
for fn, box, out in crops:
    im = Image.open(os.path.join(F, fn))
    im.crop(box).save(os.path.join(O, out))
    print(out, im.size)
