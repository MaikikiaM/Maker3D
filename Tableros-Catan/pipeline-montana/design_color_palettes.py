import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import shutil

print("=== DESIGNING 45 TONES OF GREY + 15 TONES OF GREEN ===")

out_dir = r"Tableros Catan\Loseta_Montana_3D\renders_cad"
brain_dir = r"C:\Users\migue\.gemini\antigravity\brain\ae0d77ea-1004-4459-bb12-8bf2643bf77c"
os.makedirs(out_dir, exist_ok=True)

# -------------------------------------------------------------
# 1. 45 DISTINCT TONES OF GREY (From White 1.0 to Black 0.04)
# -------------------------------------------------------------
# We span from pure arctic white (tone 1) down to deep basalt black (tone 45).
# To provide authentic rock realism, each tone has subtle geological tinting:
# - High summit snow (cool slight blue tint)
# - Quartzite & granodiorite (warm slight neutral/cream tint)
# - Gneiss & weathered slate (subtle charcoal slate tint)
# - Basalt & crevice occlusion (neutral deep black)

greys_45 = []
for i in range(45):
    # Normalized luminance factor from 1.0 down to 0.04
    t = i / 44.0  # 0 at pure white, 1 at deep black
    val = 0.985 * (1.0 - t)**1.12 + 0.035
    
    # Subtle geological color temperature modulation
    if i < 8:
        # High snow: cool slight sky tint
        r = val * 0.99
        g = val * 0.995
        b = val * 1.00
    elif i < 20:
        # Quartzite / pale granite: neutral warm
        r = val * 1.01
        g = val * 1.00
        b = val * 0.98
    elif i < 35:
        # Classic granite / gneiss: neutral stone
        r = val * 0.99
        g = val * 1.00
        b = val * 1.01
    else:
        # Deep slate / crevice shadow: dark neutral
        r = val * 0.98
        g = val * 0.99
        b = val * 1.00
    greys_45.append(np.clip([r, g, b], 0.0, 1.0))

greys_45 = np.array(greys_45, dtype=np.float32)

# -------------------------------------------------------------
# 2. 15 DISTINCT TONES OF GREEN (Sub-Alpine Mosses to Lichens)
# -------------------------------------------------------------
# 15 distinct botanical and rupestral tones:
# 1: Pale Chartreuse Lichen (Xanthoria candelaria)
# 2: Sulphur Alpine Lichen (Rhizocarpon geographicum)
# 3: Sage Rock Lichen (Stereocaulon alpinum)
# 4: Golden Olive Heath (Loiseleuria procumbens)
# 5: Sunlit High Tundra Turf (Kobresia myosuroides)
# 6: Spring Scree Bryophyte (Racomitrium lanuginosum)
# 7: Alpine Meadow Fescue (Festuca varia)
# 8: Peat Moss (Sphagnum fuscum)
# 9: Sub-Alpine Crowberry Green (Empetrum nigrum)
# 10: Dwarf Willow Turf (Salix herbacea)
# 11: Wet Gutter Liverwort (Marchantia polymorpha)
# 12: Damp Slate Crevice Moss (Grimmia alpestris)
# 13: Weathered Moraine Lichen (Lecanora rupicola)
# 14: Dark Peated Humus (Carex curvula)
# 15: Deep Basalt Shadow Moss (Andreaea rupestris)

greens_15_hex = [
    "#c6d67e",  # 1. Pale Chartreuse Lichen
    "#b4c96b",  # 2. Sulphur Alpine Lichen
    "#9fb85d",  # 3. Sage Rock Lichen
    "#8ca84f",  # 4. Golden Olive Heath
    "#769841",  # 5. Sunlit High Tundra Turf
    "#638936",  # 6. Spring Scree Bryophyte
    "#527b2d",  # 7. Alpine Meadow Fescue
    "#436c25",  # 8. Peat Moss
    "#375e20",  # 9. Sub-Alpine Crowberry
    "#2e521c",  # 10. Dwarf Willow Turf
    "#264618",  # 11. Wet Gutter Liverwort
    "#203c15",  # 12. Damp Slate Crevice Moss
    "#344422",  # 13. Weathered Moraine Lichen
    "#27381b",  # 14. Dark Peated Humus
    "#192b12",  # 15. Deep Basalt Shadow Moss
]

greens_15 = []
for h in greens_15_hex:
    r = int(h[1:3], 16) / 255.0
    g = int(h[3:5], 16) / 255.0
    b = int(h[5:7], 16) / 255.0
    greens_15.append([r, g, b])
greens_15 = np.array(greens_15, dtype=np.float32)

print(f"Constructed 45 Greys: shape {greys_45.shape}, luminance from {np.mean(greys_45[0]):.3f} to {np.mean(greys_45[-1]):.3f}")
print(f"Constructed 15 Greens: shape {greens_15.shape}, green channel from {greens_15[0,1]:.3f} to {greens_15[-1,1]:.3f}")

# Save palette arrays to scratch
np.save('scratch/greys_45.npy', greys_45)
np.save('scratch/greens_15.npy', greens_15)

# Render verification swatch poster
fig = plt.figure(figsize=(14, 8), facecolor='#090d16')

# Title
plt.suptitle("Carta Cromática de Ingeniería: 45 Tonos de Gris + 15 Tonos de Verde",
             color='#38bdf8', fontsize=16, fontweight='bold', y=0.96)

# Subplot 1: 45 Tonos de Gris
ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=2)
ax1.set_facecolor('#0f172a')
for i in range(45):
    rect = patches.Rectangle((i, 0), 1, 1, facecolor=greys_45[i], edgecolor='none')
    ax1.add_patch(rect)
    # Numbering
    txt_col = 'black' if np.mean(greys_45[i]) > 0.5 else 'white'
    ax1.text(i + 0.5, 0.5, f"{i+1}", color=txt_col, fontsize=7, ha='center', va='center', fontweight='bold')

ax1.set_xlim(0, 45)
ax1.set_ylim(0, 1)
ax1.set_xticks(np.arange(0.5, 45.5, 5))
ax1.set_xticklabels([f"Tono {i+1}" for i in range(0, 45, 5)], color='#94a3b8', fontsize=8.5)
ax1.set_yticks([])
ax1.set_title("45 Tonos de Grises Geológicos (Cumbre, Nieve, Glaciar, Cuarcita, Granito, Gneis, Pizarra y Basalto)",
              color='#e2e8f0', fontsize=11, fontweight='bold', pad=8)

# Subplot 2: 15 Tonos de Verde
ax2 = plt.subplot2grid((5, 1), (3, 0), rowspan=2)
ax2.set_facecolor('#0f172a')
for j in range(15):
    rect = patches.Rectangle((j, 0), 1, 1, facecolor=greens_15[j], edgecolor='none')
    ax2.add_patch(rect)
    txt_col = 'black' if np.mean(greens_15[j]) > 0.45 else 'white'
    ax2.text(j + 0.5, 0.5, f"V{j+1}", color=txt_col, fontsize=9, ha='center', va='center', fontweight='bold')

ax2.set_xlim(0, 15)
ax2.set_ylim(0, 1)
ax2.set_xticks(np.arange(0.5, 15.5, 1))
ax2.set_xticklabels([f"V{j+1}" for j in range(15)], color='#94a3b8', fontsize=8.5)
ax2.set_yticks([])
ax2.set_title("15 Tonos de Verde Botánico Sub-Alpino (Líquenes Rupestres, Musgo de Turbera, Brezo y Humus)",
              color='#34d399', fontsize=11, fontweight='bold', pad=8)

plt.tight_layout(rect=[0, 0.03, 1, 0.93])
poster_path = os.path.join(out_dir, "render_palette_60_tones_audit.png")
plt.savefig(poster_path, dpi=200, bbox_inches='tight', facecolor='#090d16')
plt.close()

shutil.copy2(poster_path, os.path.join(brain_dir, "render_palette_60_tones_audit.png"))
print(f"Saved: {poster_path} and synced to brain")
