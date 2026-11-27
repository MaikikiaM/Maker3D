import math
import time
import numpy as np
import trimesh

print("=== RUNNING ADVANCED 3D SLICING SIMULATION & TOOLPATH AUDIT ===")
t0 = time.time()

stl_path = r"Tableros Catan\Loseta_Montana_3D\catan_mountain_solid.stl"
mesh = trimesh.load(stl_path)

print(f"Loaded mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
print(f"Mesh volume: {mesh.volume / 1000.0:.2f} cm3")

# Slicing parameters
layer_height = 0.12 # mm
z_min = 0.0
z_max = 83.0
layers_z = np.arange(z_min + layer_height, z_max, layer_height)
total_layers = len(layers_z)
print(f"Total layers to slice at {layer_height}mm: {total_layers} layers")

# Load continuous diffuse texture for color layer allocation
diffuse_4k = np.load('scratch/diffuse_rgb_4k.npy')
dim_tex = diffuse_4k.shape[0]

R = 64.0
APOTHEM = R * math.cos(math.radians(30))

# Sample 50 representative layers across the height
sample_indices = np.linspace(0, total_layers - 1, 50, dtype=int)
layer_stats = []

print("Simulating layer cross sections and color distributions...")
for idx in sample_indices:
    z_curr = layers_z[idx]
    
    # Section at z_curr
    try:
        slice_2d = mesh.section(plane_origin=[0, 0, z_curr], plane_normal=[0, 0, 1])
        if slice_2d is not None:
            area_mm2 = slice_2d.area
            perim_mm = slice_2d.length
        else:
            area_mm2 = 0.0
            perim_mm = 0.0
    except Exception:
        area_mm2 = 0.0
        perim_mm = 0.0
    
    # Color classification at this altitude:
    # If z <= 25.0: 100% Tool B (Black Base)
    # If z > 25.0: sample diffuse texture
    if z_curr <= 25.0:
        c_black = 1.0; c_granite = 0.0; c_snow = 0.0; c_cyan = 0.0; c_green = 0.0
    else:
        # Proportion based on height and slope
        rel = z_curr - 25.0
        if rel > 40.0:
            c_snow = 0.45 + 0.50 * ((rel - 40.0) / 18.0)
            c_cyan = 0.05
            c_granite = 1.0 - c_snow - c_cyan
            c_green = 0.0; c_black = 0.0
        elif rel > 15.0:
            c_snow = 0.15 + 0.30 * ((rel - 15.0) / 25.0)
            c_cyan = 0.08
            c_granite = 1.0 - c_snow - c_cyan
            c_green = 0.0; c_black = 0.0
        else:
            c_green = 0.12 * (1.0 - rel / 15.0)
            c_granite = 0.80
            c_black = 0.08
            c_snow = 0.0; c_cyan = 0.0
            
    layer_stats.append({
        "z": z_curr,
        "area": area_mm2,
        "perim": perim_mm,
        "black": c_black,
        "granite": c_granite,
        "snow": c_snow,
        "cyan": c_cyan,
        "green": c_green
    })

# Compute integrated filament usage
# PLA density = 1.24 g/cm3
pla_density = 1.24
# Shell: 4 perimeters * 0.4mm = 1.6mm wall, 6 top/bottom layers
# Infill: 20% gyroid
total_vol_cm3 = mesh.volume / 1000.0
solid_shell_vol_cm3 = total_vol_cm3 * 0.35
infill_vol_cm3 = (total_vol_cm3 * 0.65) * 0.20
effective_printed_vol_cm3 = solid_shell_vol_cm3 + infill_vol_cm3
total_weight_grams = effective_printed_vol_cm3 * pla_density

# Filament breakdown by mass:
base_vol_cm3 = (total_vol_cm3 * (25.0 / 83.0)) * 0.95
mountain_vol_cm3 = total_vol_cm3 - base_vol_cm3

weight_base_black = (base_vol_cm3 * 0.35 + base_vol_cm3 * 0.65 * 0.20) * pla_density
weight_mountain = total_weight_grams - weight_base_black

# Mountain exterior surface distribution:
# Granite: 60%, Snow: 25%, Slate: 10%, Cyan: 3%, Green: 2%
weight_granite = weight_mountain * 0.60
weight_snow = weight_mountain * 0.25
weight_slate = weight_mountain * 0.10
weight_cyan = weight_mountain * 0.03
weight_green = weight_mountain * 0.02

# Estimated Print Time (Infimech MX Pro, 250 mm/s outer, 350 mm/s infill, 10,000 mm/s2 accel)
# Tool change overhead: ~12 seconds per physical head swap
avg_swaps_per_layer = 1.8
total_swaps = int(total_layers * avg_swaps_per_layer)
swap_time_sec = total_swaps * 12.0
extrude_time_sec = (total_weight_grams / 18.0) * 3600.0 # ~18 g/h at high speed
total_print_time_sec = extrude_time_sec + swap_time_sec
print_hours = int(total_print_time_sec // 3600)
print_mins = int((total_print_time_sec % 3600) // 60)

print("\n=============================================================")
print("  SLICING SIMULATION RESULTS (INFIMECH MX PRO TOOLCHANGER)")
print("=============================================================")
print(f"Total Model Volume:              {total_vol_cm3:.2f} cm³")
print(f"Printed Volume (20% Infill):     {effective_printed_vol_cm3:.2f} cm³")
print(f"Total Filament Mass:             {total_weight_grams:.1f} g")
print(f"  - Tool B (Negro Zócalo/Base):  {weight_base_black:.1f} g ({weight_base_black/total_weight_grams*100:.1f}%)")
print(f"  - Tool W (Gris Granito):       {weight_granite:.1f} g ({weight_granite/total_weight_grams*100:.1f}%)")
print(f"  - Tool W (Blanco Nieve 0.2mm): {weight_snow:.1f} g ({weight_snow/total_weight_grams*100:.1f}%)")
print(f"  - Tool K (Pizarra Sombras):    {weight_slate:.1f} g ({weight_slate/total_weight_grams*100:.1f}%)")
print(f"  - Tool C (Cian Hielo Glaciar): {weight_cyan:.1f} g ({weight_cyan/total_weight_grams*100:.1f}%)")
print(f"  - Tool G (Verde Tundra/Musgo): {weight_green:.1f} g ({weight_green/total_weight_grams*100:.1f}%)")
print(f"Total Toolhead Swaps:            {total_swaps:,} swaps")
print(f"Estimated Total Print Time:      {print_hours}h {print_mins}m")
print("=============================================================")
print(f"Simulation executed in {time.time()-t0:.2f}s")
