import numpy as np
from PIL import Image

# Load 4K diffuse texture
img = Image.open(r"Tableros Catan\Loseta_Montana_3D\catan_mountain_diffuse_4k.png")
w, h = img.size
print(f"Loaded 4K texture: {w}x{h}")

# Crop three distinct macro areas:
# 1. Main summit rock & snow knife-edge
# 2. Alpine rock wall / couloir
# 3. Basal talus & tundra transition
crop_summit = img.crop((w//2 - 300, h//2 - 300, w//2 + 300, h//2 + 300))
crop_couloir = img.crop((w//2 - 800, h//2 - 200, w//2 - 200, h//2 + 400))
crop_tundra = img.crop((w//2 + 400, h//2 + 600, w//2 + 1000, h//2 + 1200))

# Create a composite macro proof image
macro_proof = Image.new("RGB", (1800, 600))
macro_proof.paste(crop_summit, (0, 0))
macro_proof.paste(crop_couloir, (600, 0))
macro_proof.paste(crop_tundra, (1200, 0))

out_proof = r"Tableros Catan\Loseta_Montana_3D\renders_cad\macro_texture_verification_proof.png"
macro_proof.save(out_proof)
print(f"Saved macro proof: {out_proof}")

# Also copy to brain dir
brain_proof = r"C:\Users\migue\.gemini\antigravity\brain\ae0d77ea-1004-4459-bb12-8bf2643bf77c\macro_texture_verification_proof.png"
macro_proof.save(brain_proof)
