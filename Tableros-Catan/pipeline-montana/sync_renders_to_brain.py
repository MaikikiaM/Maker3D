import os
import shutil

brain_dir = r"C:\Users\migue\.gemini\antigravity\brain\ae0d77ea-1004-4459-bb12-8bf2643bf77c"
cad_dir = r"c:\Users\migue\Proyectos IA varios\Miscelanea\Tableros Catan\Loseta_Montana_3D\renders_cad"

renders = [
    "render_mountain_isometric.png",
    "render_mountain_top_ortho.png",
    "render_mountain_macro_zoom_proof.png",
    "render_mountain_6_player_perspectives.png",
    "render_solid_cross_section.png",
    "macro_texture_verification_proof.png"
]

print("Syncing renders from renders_cad to brain directory...")
for r in renders:
    src = os.path.join(cad_dir, r)
    dst = os.path.join(brain_dir, r)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"Synced: {r} ({os.path.getsize(dst)/1024:.1f} KB)")
    else:
        print(f"Not found: {r}")

print("Sync completed.")
