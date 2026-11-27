import os
import base64

print("=== BUILDING INTERACTIVE 360-DEGREE TURNTABLE SCRUBBER HTML ===")

frames_dir = r"scratch\turntable_frames"
out_dir = r"Tableros Catan\Loseta_Montana_3D\renders_cad"
brain_dir = r"C:\Users\migue\.gemini\antigravity\brain\ae0d77ea-1004-4459-bb12-8bf2643bf77c"

n_frames = 36
b64_frames = []

for i in range(n_frames):
    p = os.path.join(frames_dir, f"frame_{i:02d}.png")
    with open(p, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('ascii')
        b64_frames.append(f"data:image/png;base64,{b64}")

html_scrubber = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>Scrubber 360° Loseta Montaña Catan</title>
  <style>
    body {{ background: #090d16; color: #e2e8f0; font-family: sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; margin: 0; }}
    #turntable-container {{ position: relative; width: 680px; height: 580px; display: flex; align-items: center; justify-content: center; background: #0b1120; border-radius: 12px; border: 1px solid #1e293b; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }}
    #frame-img {{ max-width: 100%; max-height: 100%; object-fit: contain; }}
    .controls {{ margin-top: 18px; width: 680px; display: flex; flex-direction: column; gap: 8px; }}
    .slider {{ width: 100%; accent-color: #38bdf8; cursor: pointer; }}
    .info {{ display: flex; justify-content: space-between; font-size: 0.85rem; color: #94a3b8; font-weight: bold; }}
  </style>
</head>
<body>
  <div id="turntable-container">
    <img id="frame-img" src="{b64_frames[0]}">
  </div>
  <div class="controls">
    <div class="info">
      <span>Ángulo de Giro: <span id="angle-val" style="color:#38bdf8;">0°</span></span>
      <span>Fotograma: <span id="frame-val" style="color:#34d399;">1 / 36</span></span>
    </div>
    <input type="range" class="slider" min="0" max="35" value="0" oninput="updateFrame(this.value)">
    <div style="text-align: center; font-size: 0.75rem; color: #64748b; margin-top: 4px;">
      Arrastra el deslizador horizontalmente para girar 360° alrededor de la cumbre
    </div>
  </div>
  <script>
    const frames = {b64_frames};
    function updateFrame(idx) {{
      document.getElementById('frame-img').src = frames[idx];
      document.getElementById('angle-val').textContent = (idx * 10) + '°';
      document.getElementById('frame-val').textContent = (parseInt(idx) + 1) + ' / 36';
    }}
  </script>
</body>
</html>
"""

out_html = os.path.join(out_dir, "catan_mountain_turntable_scrubber.html")
with open(out_html, 'w', encoding='utf-8') as f:
    f.write(html_scrubber)

import shutil
shutil.copy(out_html, os.path.join(brain_dir, "catan_mountain_turntable_scrubber.html"))
print(f"Saved: {out_html} and copied to brain ({os.path.getsize(out_html)/1024/1024:.2f} MB)")
