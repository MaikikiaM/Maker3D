import os
import math
import time
import numpy as np

print("=== GENERATING PRODUCTION SAMPLE G-CODE FOR THE FIRST 10 LAYERS ===")
t0 = time.time()

out_dir = r"Tableros Catan\Loseta_Montana_3D"
gcode_path = os.path.join(out_dir, "catan_mountain_base_sample.gcode")

R = 64.0
APOTHEM = R * math.cos(math.radians(30))
layer_h = 0.20
nozzle_w = 0.42
filament_area = math.pi * ((1.75 / 2.0) ** 2)

v_angles = (30 + 60 * np.arange(7)) * np.pi / 180.0
hex_vx = R * np.cos(v_angles)
hex_vy = R * np.sin(v_angles)

gcode_lines = [
    "; Catan 3D Mountain 5.11M Production Sample G-Code",
    "; Printer: Bambu Lab X1C / Generic High-Flow FDM",
    "; Filament: PLA (1.75mm)",
    "; Layer Height: 0.20mm | Nozzle: 0.40mm",
    "M140 S60 ; Set Bed Temp",
    "M104 S215 ; Set Nozzle Temp",
    "M190 S60 ; Wait Bed Temp",
    "M109 S215 ; Wait Nozzle Temp",
    "G28 ; Home All Axes",
    "G29 ; Auto Bed Leveling",
    "G90 ; Absolute Positioning",
    "M83 ; Relative Extrusion Mode",
    "G1 Z2.0 F3000 ; Lift Nozzle",
    "G1 X-55.0 Y-60.0 F9000 ; Move to Start",
    "G1 Z0.20 F1200 ; First Layer Height"
]

e_total = 0.0

# 10 Layers (Z = 0.20 to Z = 2.00 mm)
for lay_idx in range(1, 11):
    z = lay_idx * layer_h
    gcode_lines.append(f"\n; LAYER {lay_idx} at Z = {z:.2f} mm")
    gcode_lines.append(f"G1 Z{z:.2f} F1200")
    
    # Outer wall
    start_x = hex_vx[0]; start_y = hex_vy[0]
    gcode_lines.append(f"G1 X{start_x:.3f} Y{start_y:.3f} F9000")
    gcode_lines.append(f"G1 E0.8 F2400 ; Unretract")
    
    for k in range(1, 7):
        p_x = hex_vx[k]; p_y = hex_vy[k]
        dist = math.hypot(p_x - hex_vx[k-1], p_y - hex_vy[k-1])
        vol_extruded = dist * nozzle_w * layer_h
        e_len = vol_extruded / filament_area
        e_total += e_len
        gcode_lines.append(f"G1 X{p_x:.3f} Y{p_y:.3f} E{e_len:.4f} F7200")
        
    gcode_lines.append("G1 E-0.8 F2400 ; Retract")
    
    # Magnet cavity circular wall (if z <= 5.2 mm)
    if z <= 5.2:
        gcode_lines.append(f"; Magnet Cavity Perimeter (Radius 5.10 mm)")
        gcode_lines.append(f"G1 X5.100 Y0.000 F9000")
        gcode_lines.append("G1 E0.8 F2400 ; Unretract")
        n_arc_pts = 36
        for a_step in range(1, n_arc_pts + 1):
            ang_c = a_step * (2 * math.pi / n_arc_pts)
            cx = 5.1 * math.cos(ang_c)
            cy = 5.1 * math.sin(ang_c)
            arc_len = 5.1 * (2 * math.pi / n_arc_pts)
            e_arc = (arc_len * nozzle_w * layer_h) / filament_area
            e_total += e_arc
            gcode_lines.append(f"G1 X{cx:.3f} Y{cy:.3f} E{e_arc:.4f} F4800")
        gcode_lines.append("G1 E-0.8 F2400 ; Retract")

gcode_lines.extend([
    "\n; End of Sample Extrusion",
    "G1 Z10.0 F3000 ; Lift Nozzle",
    "G1 X-50 Y50 F9000 ; Present Part",
    "M104 S0 ; Turn off Nozzle",
    "M140 S0 ; Turn off Bed",
    "M84 ; Disable Steppers"
])

with open(gcode_path, 'w', encoding='utf-8') as f_g:
    f_g.write("\n".join(gcode_lines) + "\n")

print(f"Generated Sample G-Code: {gcode_path} ({os.path.getsize(gcode_path)/1024:.1f} KB, {len(gcode_lines)} lines)")
print(f"Total Filament Extruded: {e_total:.2f} mm")
print(f"G-code generation finished in {time.time()-t0:.2f}s.")
