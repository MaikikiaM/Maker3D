import os
import math
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt

print("=== COMPUTING TRAPEZOIDAL VELOCITY PROFILE & TOOLPATH KINEMATICS ===")
t0 = time.time()

out_dir = r"Tableros Catan\Loseta_Montana_3D\renders_cad"
brain_dir = r"C:\Users\migue\.gemini\antigravity\brain\ae0d77ea-1004-4459-bb12-8bf2643bf77c"

# Bambu X1-Carbon / CoreXY Kinematics Parameters
v_max_outer = 150.0 # mm/s
v_max_inner = 250.0 # mm/s
v_max_infill = 300.0 # mm/s
v_max_travel = 500.0 # mm/s
accel = 10000.0 # mm/s^2 (1G acceleration)
jerk = 12.0 # mm/s (junction deviation)

# Simulate toolpath segment lengths across Layer 125 (Z = 25.0 mm):
# Outer perimeter: 6 segments of 64.00 mm = 384.0 mm
# 3 Inner perimeters: 6 * 63.6 + 6 * 63.2 + 6 * 62.8 = 1137.6 mm
# Solid rectilinear infill: 180 passes, average length 58.0 mm = 10,440.0 mm
# Travel moves: 180 moves, average length 15.0 mm = 2,700.0 mm

segments = [
    {"type": "Perímetro Exterior", "length": 64.00, "v_target": v_max_outer, "count": 6},
    {"type": "Paredes Interiores", "length": 63.20, "v_target": v_max_inner, "count": 18},
    {"type": "Relleno Sólido", "length": 58.00, "v_target": v_max_infill, "count": 180},
    {"type": "Desplazamientos G0", "length": 15.00, "v_target": v_max_travel, "count": 180}
]

# Trapezoidal profile calculation for a segment of length L:
# Accel distance: d_acc = (v_target^2 - v_start^2) / (2 * accel)
def calc_trapezoid_time(L, v_target, v_start=12.0, v_end=12.0, a=accel):
    d_acc = (v_target**2 - v_start**2) / (2.0 * a)
    d_dec = (v_target**2 - v_end**2) / (2.0 * a)
    
    if d_acc + d_dec <= L:
        # Reaches cruise speed
        d_cruise = L - (d_acc + d_dec)
        t_acc = (v_target - v_start) / a
        t_dec = (v_target - v_end) / a
        t_cruise = d_cruise / v_target
        return t_acc + t_cruise + t_dec, v_target
    else:
        # Triangular profile (does not reach v_target)
        v_peak = math.sqrt((2.0 * a * L + v_start**2 + v_end**2) / 2.0)
        t_acc = (v_peak - v_start) / a
        t_dec = (v_peak - v_end) / a
        return t_acc + t_dec, v_peak

total_layer_time = 0.0
print(f"{'Segment Type':25s} | {'Len (mm)':8s} | {'Target v':8s} | {'Peak v':8s} | {'Time/seg':8s} | {'Total Time':10s}")
print("-" * 75)

for s in segments:
    t_seg, v_actual = calc_trapezoid_time(s["length"], s["v_target"])
    tot_s = t_seg * s["count"]
    total_layer_time += tot_s
    print(f"{s['type']:25s} | {s['length']:8.2f} | {s['v_target']:8.0f} | {v_actual:8.1f} | {t_seg*1000:7.1f} ms | {tot_s:8.2f} s")

# Add retraction / Z-hop overhead (0.8mm retract at 40 mm/s + Z-hop 0.2mm)
retract_time = 180 * (0.8 / 40.0 * 2.0 + 0.015) # seconds
total_layer_time += retract_time
print(f"Retractions & Z-hops:     {retract_time:.2f} s")
print(f"Total Layer 125 Print Time: {total_layer_time:.2f} s ({total_layer_time/60.0:.2f} min)")

# Velocity vs Time plot for a single perimeter move vs infill move
dt = 0.0005 # 0.5 ms
t_p, _ = calc_trapezoid_time(64.0, v_max_outer)
time_axis_p = np.arange(0, t_p, dt)
v_p = []
d_acc_p = (v_max_outer**2 - jerk**2) / (2 * accel)
t_acc_p = (v_max_outer - jerk) / accel
t_dec_start_p = t_p - t_acc_p

for t_i in time_axis_p:
    if t_i < t_acc_p:
        v_p.append(jerk + accel * t_i)
    elif t_i < t_dec_start_p:
        v_p.append(v_max_outer)
    else:
        v_p.append(max(jerk, v_max_outer - accel * (t_i - t_dec_start_p)))
v_p = np.array(v_p)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor='#090d16')
ax1.set_facecolor('#0f172a')
ax2.set_facecolor('#0f172a')

# 1. Velocity Profile of Outer Perimeter Extrusion
ax1.plot(time_axis_p * 1000.0, v_p, color='#38bdf8', lw=2.5, label='Velocidad de Boquilla v(t) [mm/s]')
ax1.fill_between(time_axis_p * 1000.0, 0, v_p, color='#0284c7', alpha=0.25)
ax1.axhline(v_max_outer, color='#34d399', linestyle='--', label=f'Velocidad de Crucero ({v_max_outer:.0f} mm/s)')
ax1.axhline(jerk, color='#f59e0b', linestyle=':', label=f'Velocidad en Vértice / Jerk ({jerk:.0f} mm/s)')

ax1.set_xlabel("Tiempo Transcurrido (Milisegundos ms)", color='#e2e8f0', fontsize=11, fontweight='bold')
ax1.set_ylabel("Velocidad de Avance (mm/s)", color='#e2e8f0', fontsize=11, fontweight='bold')
ax1.set_title("Perfil Trapezoidal de Velocidad (Arista de 64.00 mm a 1G Aceleración)", color='#38bdf8', fontsize=12, fontweight='bold')
ax1.tick_params(colors='#94a3b8')
ax1.grid(True, color='#334155', linestyle=':', alpha=0.6)
ax1.legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='#e2e8f0', loc='center right')

# 2. Time Breakdown Pie Chart
labels_pie = ["Relleno Sólido", "Retracciones & Z-Hop", "Paredes Interiores", "Desplazamientos G0", "Perímetro Exterior"]
times_pie = [180 * calc_trapezoid_time(58.0, v_max_infill)[0], retract_time, 18 * calc_trapezoid_time(63.2, v_max_inner)[0], 180 * calc_trapezoid_time(15.0, v_max_travel)[0], 6 * t_p]
colors_pie = ['#38bdf8', '#f43f5e', '#34d399', '#f59e0b', '#a855f7']

wedges, texts, autotexts = ax2.pie(times_pie, labels=labels_pie, autopct='%1.1f%%', colors=colors_pie,
                                   textprops=dict(color='#e2e8f0', fontsize=9.5),
                                   wedgeprops=dict(edgecolor='#0f172a', linewidth=2))
for at in autotexts:
    at.set_color('#090d16')
    at.set_fontweight('bold')
ax2.set_title("Desglose del Tiempo de Impresión de la Capa 125 (Z = 25.0 mm)", color='#38bdf8', fontsize=12, fontweight='bold')

plt.suptitle("Cinemática de Boquilla y Simulación de Firmware (Klipper / Bambu Micro-Step)",
             color='#e2e8f0', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()

out_kinematics = os.path.join(out_dir, "render_toolpath_kinematics.png")
plt.savefig(out_kinematics, dpi=200, bbox_inches='tight', facecolor='#090d16')
plt.close()

shutil.copy(out_kinematics, os.path.join(brain_dir, "render_toolpath_kinematics.png"))
print(f"Saved: {out_kinematics} and copied to brain")
print(f"Kinematics simulation finished in {time.time()-t0:.2f}s.")
