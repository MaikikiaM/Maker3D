import os
import json

print("=== GENERATING PROFESSIONAL SLICER PROFILES FOR BAMBU / ORCA / PRUSA ===")

out_dir = r"Tableros Catan\Loseta_Montana_3D"

# 1. BambuStudio / OrcaSlicer Process Profile JSON
bambu_profile = {
    "name": "0.16mm Optimal Catan Mountain 5.11M",
    "from": "system",
    "inherits": "0.16mm Optimal @BBL X1C",
    "is_custom_defined": "1",
    "version": "1.8.0.0",
    "layer_height": "0.16",
    "first_layer_height": "0.20",
    "initial_layer_print_height": "0.20",
    "wall_loops": "4",
    "top_shell_layers": "5",
    "bottom_shell_layers": "4",
    "sparse_infill_density": "15%",
    "sparse_infill_pattern": "gyroid",
    "enable_support": "0",
    "support_type": "none",
    "reduce_infill_retraction": "1",
    "wipe_on_loops": "1",
    "seam_position": "back",
    "outer_wall_speed": "120",
    "inner_wall_speed": "200",
    "sparse_infill_speed": "250",
    "top_surface_speed": "100",
    "travel_speed": "500",
    "initial_layer_speed": "35",
    "nozzle_temperature": ["215"],
    "bed_temperature": ["60"],
    "cooling_fan_speed": ["100"],
    "notes": "Custom high-precision alpine profile for Catan 3D Mountain with 5.11M faces. Zero supports needed."
}

bambu_path = os.path.join(out_dir, "bambu_orcaslicer_mountain_profile.json")
with open(bambu_path, 'w', encoding='utf-8') as f:
    json.dump(bambu_profile, f, indent=2)
print(f"Saved: {bambu_path}")

# 2. PrusaSlicer MMU3 / MK4 Profile INI
prusa_ini = """# PrusaSlicer Print Profile for Catan 3D Mountain Monolith
[print:0.16mm Mountain 5.11M @MK4]
layer_height = 0.16
first_layer_height = 0.20
perimeters = 4
top_solid_layers = 5
bottom_solid_layers = 4
fill_density = 15%
fill_pattern = gyroid
support_material = 0
infill_speed = 180
perimeter_speed = 120
external_perimeter_speed = 80
travel_speed = 300
first_layer_speed = 30
retract_length = 0.8
retract_lift = 0.2
wipe = 1
extrusion_width = 0.45
first_layer_extrusion_width = 0.50
perimeter_extrusion_width = 0.45
infill_extrusion_width = 0.45
solid_infill_extrusion_width = 0.45
top_infill_extrusion_width = 0.40
temperature = 215
bed_temperature = 60
fan_below_layer_time = 100
"""

prusa_path = os.path.join(out_dir, "prusaslicer_mmu3_mountain_profile.ini")
with open(prusa_path, 'w', encoding='utf-8') as f:
    f.write(prusa_ini)
print(f"Saved: {prusa_path}")

print("Slicer profiles generated successfully!")
