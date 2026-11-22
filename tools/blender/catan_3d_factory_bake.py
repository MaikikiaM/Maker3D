"""
Módulo 2 de la Fábrica Modular 3D Catán (Blender 5.3 Look-Dev PBR 3D + Smart UV + Cycles Bake):
  - Importa el sólido CSG estanco (tile_master.stl) generado por catan_3d_factory_csg.py.
  - Separa con precisión milimétrica el material del zócalo B-Rep (Plinth_Gunmetal) de la
    superficie escultórica (Biome_PBR_3D) sin costuras ni bordes borrosos.
  - Evalúa en Cycles 3D puro (Object Coords + True Normal + ShaderNodeBevel + AmbientOcclusion
    + Voronoi/Noise 3D + splat_mask.png):
      * Granito frío fracturado con dry-brush en aristas vivas y AO oscuro en fisuras.
      * Nieve alpina en hoyas cóncavas (< 38.7° de pendiente, 0.0000% en acantilados >= 44°).
      * Cañón terracota mineral y pradera oliva-musgo orgánica.
  - Despliega Smart UV 3D (cero estiramiento vertical en acantilados) y bakea Albedo,
    Normal y Roughness en Cycles para exportar tile_viewer.glb + renders de auditoría.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import addon_utils
import bmesh
import bpy
import mathutils
import numpy as np

R = 64.0000
APOTHEM = R * math.cos(math.radians(30.0))  # 55.425626 mm
Z_BASE = 25.0000
ROAD_MARGIN = 3.50
PAD_RADIUS = 10.00


def _hex_distances_pt(x: float, y: float) -> Tuple[float, float]:
    de = 999.0
    for k in range(6):
        ang = k * math.pi / 3.0
        de = min(de, APOTHEM - (x * math.cos(ang) + y * math.sin(ang)))
    dv = 999.0
    for k in range(6):
        ang = math.radians(30.0 + 60.0 * k)
        dv = min(dv, math.hypot(x - R * math.cos(ang), y - R * math.sin(ang)))
    return de, dv


def _new_material(name: str) -> Tuple[bpy.types.Material, bpy.types.NodeTree, bpy.types.Node, bpy.types.Node]:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat, nt, bsdf, out


def _assign_plinth_vs_terrain_slots(obj: bpy.types.Object) -> None:
    """Asigna slot 0 (Plinth_Gunmetal) a la base/carril/pads y slot 1 (Biome_Terrain) al relieve."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    for f in bm.faces:
        c = f.calc_center_median()
        if c.z < Z_BASE - 0.03:
            f.material_index = 0
        elif c.z <= Z_BASE + 0.08:
            de, dv = _hex_distances_pt(c.x, c.y)
            if de < ROAD_MARGIN + 0.12 or dv < PAD_RADIUS + 0.12:
                f.material_index = 0
            else:
                f.material_index = 1
        else:
            f.material_index = 1
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


def build_cycles_montana_materials(splat_path: Path) -> Tuple[bpy.types.Material, bpy.types.Material]:
    """Construye los materiales PBR 3D en Cycles (Plinth_Gunmetal + Biome_Montana_3D)."""

    # --- 1. Zócalo Gunmetal Pizarra Cepillado ---
    m_pl, nt, bsdf, _ = _new_material("Plinth_Gunmetal")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    n_p = nt.nodes.new("ShaderNodeTexNoise")
    n_p.inputs["Scale"].default_value = 0.70
    n_p.inputs["Detail"].default_value = 6.0
    nt.links.new(tc.outputs["Object"], n_p.inputs["Vector"])
    ao_p = nt.nodes.new("ShaderNodeAmbientOcclusion")
    ao_p.samples = 8
    ao_p.inputs["Distance"].default_value = 2.8
    r_p = nt.nodes.new("ShaderNodeValToRGB")
    r_p.color_ramp.elements[0].position = 0.25
    r_p.color_ramp.elements[0].color = (0.042, 0.046, 0.054, 1.0)
    r_p.color_ramp.elements[1].position = 0.85
    r_p.color_ramp.elements[1].color = (0.075, 0.082, 0.094, 1.0)
    nt.links.new(n_p.outputs["Fac"], r_p.inputs["Fac"])
    mx_p = nt.nodes.new("ShaderNodeMix")
    mx_p.data_type = "RGBA"
    mx_p.blend_type = "MULTIPLY"
    mx_p.inputs["Factor"].default_value = 0.60
    nt.links.new(r_p.outputs["Color"], mx_p.inputs["A"])
    nt.links.new(ao_p.outputs["Color"], mx_p.inputs["B"])
    nt.links.new(mx_p.outputs["Result"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.48

    # --- 2. Terreno + Granito + Nieve + Terracota 3D ---
    m_bm, nt, bsdf, _ = _new_material("Biome_Montana_3D")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    geom = nt.nodes.new("ShaderNodeNewGeometry")

    # Mapeo XY [-R..R] -> [0..1] para leer splat_mask.png
    map_xy = nt.nodes.new("ShaderNodeMapping")
    map_xy.inputs["Location"].default_value = (0.5, 0.5, 0.0)
    map_xy.inputs["Scale"].default_value = (1.0 / (2.0 * R), 1.0 / (2.0 * R), 1.0)
    nt.links.new(tc.outputs["Object"], map_xy.inputs["Vector"])

    t_splat = nt.nodes.new("ShaderNodeTexImage")
    im_s = bpy.data.images.load(str(splat_path), check_existing=True)
    im_s.colorspace_settings.name = "Non-Color"
    t_splat.image = im_s
    t_splat.extension = "EXTEND"
    nt.links.new(map_xy.outputs["Vector"], t_splat.inputs["Vector"])
    sep_sp = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(t_splat.outputs["Color"], sep_sp.inputs["Color"])
    # R = terracotta canyon, G = snow cirque mask, B = scree region

    sep_xyz = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep_xyz.inputs["Vector"])
    sep_nrm = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geom.outputs["True Normal"], sep_nrm.inputs["Vector"])

    # Ruidos 3D reutilizables
    n_macro = nt.nodes.new("ShaderNodeTexNoise")
    n_macro.inputs["Scale"].default_value = 0.14
    n_macro.inputs["Detail"].default_value = 8.0
    n_macro.inputs["Roughness"].default_value = 0.62
    nt.links.new(tc.outputs["Object"], n_macro.inputs["Vector"])

    n_med = nt.nodes.new("ShaderNodeTexNoise")
    n_med.inputs["Scale"].default_value = 0.42
    n_med.inputs["Detail"].default_value = 10.0
    n_med.inputs["Roughness"].default_value = 0.68
    nt.links.new(tc.outputs["Object"], n_med.inputs["Vector"])

    n_fine = nt.nodes.new("ShaderNodeTexNoise")
    n_fine.inputs["Scale"].default_value = 1.65
    n_fine.inputs["Detail"].default_value = 8.0
    n_fine.inputs["Roughness"].default_value = 0.65
    nt.links.new(tc.outputs["Object"], n_fine.inputs["Vector"])

    vor_rk = nt.nodes.new("ShaderNodeTexVoronoi")
    vor_rk.feature = "DISTANCE_TO_EDGE"
    vor_rk.inputs["Scale"].default_value = 0.24
    # Warp anisótropo Z*0.28 + ruido orgánico para vetas de granito naturales (cero malla tortuga)
    map_strata = nt.nodes.new("ShaderNodeMapping")
    map_strata.inputs["Scale"].default_value = (1.0, 1.0, 0.28)
    nt.links.new(tc.outputs["Object"], map_strata.inputs["Vector"])

    w_sub = nt.nodes.new("ShaderNodeVectorMath")
    w_sub.operation = "SUBTRACT"
    w_sub.inputs[1].default_value = (0.5, 0.5, 0.5)
    nt.links.new(n_med.outputs["Color"], w_sub.inputs[0])

    w_sc = nt.nodes.new("ShaderNodeVectorMath")
    w_sc.operation = "SCALE"
    w_sc.inputs["Scale"].default_value = 8.5
    nt.links.new(w_sub.outputs["Vector"], w_sc.inputs[0])

    w_add = nt.nodes.new("ShaderNodeVectorMath")
    w_add.operation = "ADD"
    nt.links.new(map_strata.outputs["Vector"], w_add.inputs[0])
    nt.links.new(w_sc.outputs["Vector"], w_add.inputs[1])
    nt.links.new(w_add.outputs["Vector"], vor_rk.inputs["Vector"])

    ao = nt.nodes.new("ShaderNodeAmbientOcclusion")
    ao.samples = 8
    ao.inputs["Distance"].default_value = 5.8

    bevel = nt.nodes.new("ShaderNodeBevel")
    bevel.samples = 4
    bevel.inputs["Radius"].default_value = 0.16
    vdist = nt.nodes.new("ShaderNodeVectorMath")
    vdist.operation = "DISTANCE"
    nt.links.new(bevel.outputs["Normal"], vdist.inputs[0])
    nt.links.new(geom.outputs["True Normal"], vdist.inputs[1])

    edge_ramp = nt.nodes.new("ShaderNodeValToRGB")
    edge_ramp.color_ramp.elements[0].position = 0.18
    edge_ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    edge_ramp.color_ramp.elements[1].position = 0.78
    edge_ramp.color_ramp.elements[1].color = (0.50, 0.50, 0.50, 1.0)
    nt.links.new(vdist.outputs["Value"], edge_ramp.inputs["Fac"])

    # --- A. Shader Granito Fracturado Frío (Azulado-Pizarra Profundo + Grano Mineral) ---
    rk_col_r = nt.nodes.new("ShaderNodeValToRGB")
    rk_col_r.color_ramp.elements[0].position = 0.24
    rk_col_r.color_ramp.elements[0].color = (0.055, 0.068, 0.088, 1.0)
    rk_col_r.color_ramp.elements[1].position = 0.78
    rk_col_r.color_ramp.elements[1].color = (0.265, 0.295, 0.335, 1.0)
    nt.links.new(n_med.outputs["Fac"], rk_col_r.inputs["Fac"])

    # Macro-variación cálida-fría en zonas amplias de roca
    rk_macro_t = nt.nodes.new("ShaderNodeValToRGB")
    rk_macro_t.color_ramp.elements[0].position = 0.25
    rk_macro_t.color_ramp.elements[0].color = (0.86, 0.90, 0.96, 1.0)
    rk_macro_t.color_ramp.elements[1].position = 0.78
    rk_macro_t.color_ramp.elements[1].color = (1.06, 1.02, 0.96, 1.0)
    nt.links.new(n_macro.outputs["Fac"], rk_macro_t.inputs["Fac"])

    mx_rk_mc = nt.nodes.new("ShaderNodeMix")
    mx_rk_mc.data_type = "RGBA"
    mx_rk_mc.blend_type = "MULTIPLY"
    mx_rk_mc.inputs["Factor"].default_value = 0.55
    nt.links.new(rk_col_r.outputs["Color"], mx_rk_mc.inputs["A"])
    nt.links.new(rk_macro_t.outputs["Color"], mx_rk_mc.inputs["B"])

    # Micro-moteado mineral granítico (n_fine)
    rk_grain = nt.nodes.new("ShaderNodeValToRGB")
    rk_grain.color_ramp.elements[0].position = 0.30
    rk_grain.color_ramp.elements[0].color = (0.76, 0.79, 0.82, 1.0)
    rk_grain.color_ramp.elements[1].position = 0.75
    rk_grain.color_ramp.elements[1].color = (1.14, 1.16, 1.18, 1.0)
    nt.links.new(n_fine.outputs["Fac"], rk_grain.inputs["Fac"])

    mx_rk_gr = nt.nodes.new("ShaderNodeMix")
    mx_rk_gr.data_type = "RGBA"
    mx_rk_gr.blend_type = "MULTIPLY"
    mx_rk_gr.inputs["Factor"].default_value = 0.65
    nt.links.new(mx_rk_mc.outputs["Result"], mx_rk_gr.inputs["A"])
    nt.links.new(rk_grain.outputs["Color"], mx_rk_gr.inputs["B"])

    crk_r = nt.nodes.new("ShaderNodeValToRGB")
    crk_r.color_ramp.elements[0].position = 0.008
    crk_r.color_ramp.elements[0].color = (0.34, 0.37, 0.41, 1.0)
    crk_r.color_ramp.elements[1].position = 0.095
    crk_r.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    nt.links.new(vor_rk.outputs["Distance"], crk_r.inputs["Fac"])

    mx_rk_cr = nt.nodes.new("ShaderNodeMix")
    mx_rk_cr.data_type = "RGBA"
    mx_rk_cr.blend_type = "MULTIPLY"
    mx_rk_cr.inputs["Factor"].default_value = 0.44
    nt.links.new(mx_rk_gr.outputs["Result"], mx_rk_cr.inputs["A"])
    nt.links.new(crk_r.outputs["Color"], mx_rk_cr.inputs["B"])

    # Dry-brush sutil y controlado en crestas afiladas
    mx_rk_ed = nt.nodes.new("ShaderNodeMix")
    mx_rk_ed.data_type = "RGBA"
    mx_rk_ed.inputs["B"].default_value = (0.43, 0.47, 0.52, 1.0)
    nt.links.new(edge_ramp.outputs["Color"], mx_rk_ed.inputs["Factor"])
    nt.links.new(mx_rk_cr.outputs["Result"], mx_rk_ed.inputs["A"])

    # --- B. Shader Pradera Oliva-Musgo + Glera Izquierda ---
    moss_r = nt.nodes.new("ShaderNodeValToRGB")
    moss_r.color_ramp.elements[0].position = 0.26
    moss_r.color_ramp.elements[0].color = (0.056, 0.082, 0.024, 1.0)
    moss_r.color_ramp.elements[1].position = 0.78
    moss_r.color_ramp.elements[1].color = (0.185, 0.222, 0.058, 1.0)
    nt.links.new(n_macro.outputs["Fac"], moss_r.inputs["Fac"])

    # Suelo pedregoso gris-pardo en la glera izquierda (splat.B * n_fine)
    scree_col = nt.nodes.new("ShaderNodeValToRGB")
    scree_col.color_ramp.elements[0].position = 0.28
    scree_col.color_ramp.elements[0].color = (0.12, 0.13, 0.13, 1.0)
    scree_col.color_ramp.elements[1].position = 0.74
    scree_col.color_ramp.elements[1].color = (0.29, 0.31, 0.31, 1.0)
    nt.links.new(n_fine.outputs["Fac"], scree_col.inputs["Fac"])

    scree_w = nt.nodes.new("ShaderNodeMath")
    scree_w.operation = "MULTIPLY"
    scree_w.use_clamp = True
    nt.links.new(sep_sp.outputs["Blue"], scree_w.inputs[0])
    nt.links.new(n_med.outputs["Fac"], scree_w.inputs[1])

    mx_moss_sc = nt.nodes.new("ShaderNodeMix")
    mx_moss_sc.data_type = "RGBA"
    nt.links.new(scree_w.outputs["Value"], mx_moss_sc.inputs["Factor"])
    nt.links.new(moss_r.outputs["Color"], mx_moss_sc.inputs["A"])
    nt.links.new(scree_col.outputs["Color"], mx_moss_sc.inputs["B"])

    # --- C. Shader Senda Rocosa / Ocre Terroso (natural y apagado, mezclado con grava alpina) ---
    terra_r = nt.nodes.new("ShaderNodeValToRGB")
    terra_r.color_ramp.elements[0].position = 0.24
    terra_r.color_ramp.elements[0].color = (0.18, 0.105, 0.060, 1.0)
    terra_r.color_ramp.elements[1].position = 0.78
    terra_r.color_ramp.elements[1].color = (0.37, 0.205, 0.115, 1.0)
    nt.links.new(n_fine.outputs["Fac"], terra_r.inputs["Fac"])

    # --- D. Máscaras Semánticas 3D (Pendiente + Altura + Splat) ---
    z_rock = nt.nodes.new("ShaderNodeMapRange")
    z_rock.inputs["From Min"].default_value = Z_BASE + 2.0
    z_rock.inputs["From Max"].default_value = Z_BASE + 5.2
    z_rock.inputs["To Min"].default_value = 0.0
    z_rock.inputs["To Max"].default_value = 1.0
    nt.links.new(sep_xyz.outputs["Z"], z_rock.inputs["Value"])

    n_z_jit = nt.nodes.new("ShaderNodeMapRange")
    n_z_jit.inputs["From Min"].default_value = 0.0
    n_z_jit.inputs["From Max"].default_value = 1.0
    n_z_jit.inputs["To Min"].default_value = -0.25
    n_z_jit.inputs["To Max"].default_value = 0.25
    nt.links.new(n_med.outputs["Fac"], n_z_jit.inputs["Value"])

    steep_m = nt.nodes.new("ShaderNodeMapRange")
    steep_m.inputs["From Min"].default_value = 0.80  # ~37 deg -> empieza roca
    steep_m.inputs["From Max"].default_value = 0.66  # ~48 deg -> 100% roca
    steep_m.inputs["To Min"].default_value = 0.0
    steep_m.inputs["To Max"].default_value = 1.0
    nt.links.new(sep_nrm.outputs["Z"], steep_m.inputs["Value"])

    m_rk_z = nt.nodes.new("ShaderNodeMath")
    m_rk_z.operation = "ADD"
    m_rk_z.use_clamp = True
    nt.links.new(z_rock.outputs["Result"], m_rk_z.inputs[0])
    nt.links.new(n_z_jit.outputs["Result"], m_rk_z.inputs[1])

    m_rk_fac = nt.nodes.new("ShaderNodeMath")
    m_rk_fac.operation = "MAXIMUM"
    m_rk_fac.use_clamp = True
    nt.links.new(m_rk_z.outputs["Value"], m_rk_fac.inputs[0])
    nt.links.new(steep_m.outputs["Result"], m_rk_fac.inputs[1])

    mx_moss_rk = nt.nodes.new("ShaderNodeMix")
    mx_moss_rk.data_type = "RGBA"
    nt.links.new(m_rk_fac.outputs["Value"], mx_moss_rk.inputs["Factor"])
    nt.links.new(mx_moss_sc.outputs["Result"], mx_moss_rk.inputs["A"])
    nt.links.new(mx_rk_ed.outputs["Result"], mx_moss_rk.inputs["B"])

    # 2) Aplicar Terracota quebrada por grava en el suelo del cañón (splat.R * gentle_slope * rock_break)
    gentle_path = nt.nodes.new("ShaderNodeMapRange")
    gentle_path.inputs["From Min"].default_value = 0.68
    gentle_path.inputs["From Max"].default_value = 0.86
    gentle_path.inputs["To Min"].default_value = 0.0
    gentle_path.inputs["To Max"].default_value = 1.0
    nt.links.new(sep_nrm.outputs["Z"], gentle_path.inputs["Value"])

    path_break = nt.nodes.new("ShaderNodeMapRange")
    path_break.inputs["From Min"].default_value = 0.28
    path_break.inputs["From Max"].default_value = 0.72
    path_break.inputs["To Min"].default_value = 0.12
    path_break.inputs["To Max"].default_value = 0.74
    nt.links.new(n_med.outputs["Fac"], path_break.inputs["Value"])

    path_w0 = nt.nodes.new("ShaderNodeMath")
    path_w0.operation = "MULTIPLY"
    path_w0.use_clamp = True
    nt.links.new(sep_sp.outputs["Red"], path_w0.inputs[0])
    nt.links.new(gentle_path.outputs["Result"], path_w0.inputs[1])

    path_w = nt.nodes.new("ShaderNodeMath")
    path_w.operation = "MULTIPLY"
    path_w.use_clamp = True
    nt.links.new(path_w0.outputs["Value"], path_w.inputs[0])
    nt.links.new(path_break.outputs["Result"], path_w.inputs[1])

    mx_terra = nt.nodes.new("ShaderNodeMix")
    mx_terra.data_type = "RGBA"
    nt.links.new(path_w.outputs["Value"], mx_terra.inputs["Factor"])
    nt.links.new(mx_moss_rk.outputs["Result"], mx_terra.inputs["A"])
    nt.links.new(terra_r.outputs["Color"], mx_terra.inputs["B"])

    # 3) Nieve Alpina: 0.0000% en pendientes >= 44 deg (nz <= 0.72), solo en circos y collados designados (cero puntos sueltos en agujas)
    sn_slope = nt.nodes.new("ShaderNodeMapRange")
    sn_slope.inputs["From Min"].default_value = 0.72  # ~44.0 deg -> 0% nieve estricto
    sn_slope.inputs["From Max"].default_value = 0.83  # ~33.9 deg -> 100% nieve
    sn_slope.inputs["To Min"].default_value = 0.0
    sn_slope.inputs["To Max"].default_value = 1.0
    nt.links.new(sep_nrm.outputs["Z"], sn_slope.inputs["Value"])

    sn_elev = nt.nodes.new("ShaderNodeMapRange")
    sn_elev.inputs["From Min"].default_value = Z_BASE + 1.6
    sn_elev.inputs["From Max"].default_value = Z_BASE + 4.4
    sn_elev.inputs["To Min"].default_value = 0.0
    sn_elev.inputs["To Max"].default_value = 1.0
    nt.links.new(sep_xyz.outputs["Z"], sn_elev.inputs["Value"])

    sn_cirque_w = nt.nodes.new("ShaderNodeMath")
    sn_cirque_w.operation = "MULTIPLY"
    sn_cirque_w.use_clamp = True
    nt.links.new(sep_sp.outputs["Green"], sn_cirque_w.inputs[0])
    nt.links.new(sn_elev.outputs["Result"], sn_cirque_w.inputs[1])

    sn_m2 = nt.nodes.new("ShaderNodeMath")
    sn_m2.operation = "MULTIPLY"
    sn_m2.use_clamp = True
    nt.links.new(sn_cirque_w.outputs["Value"], sn_m2.inputs[0])
    nt.links.new(sn_slope.outputs["Result"], sn_m2.inputs[1])

    # Borde orgánico de nieve roto con ruido 3D
    sn_thresh = nt.nodes.new("ShaderNodeMapRange")
    sn_thresh.inputs["From Min"].default_value = 0.16
    sn_thresh.inputs["From Max"].default_value = 0.34
    sn_thresh.inputs["To Min"].default_value = 0.0
    sn_thresh.inputs["To Max"].default_value = 1.0

    sn_jit = nt.nodes.new("ShaderNodeMath")
    sn_jit.operation = "MULTIPLY_ADD"
    sn_jit.inputs[1].default_value = 0.16
    nt.links.new(n_med.outputs["Fac"], sn_jit.inputs[0])
    nt.links.new(sn_m2.outputs["Value"], sn_jit.inputs[2])
    sn_jit2 = nt.nodes.new("ShaderNodeMath")
    sn_jit2.operation = "SUBTRACT"
    sn_jit2.inputs[1].default_value = 0.08
    nt.links.new(sn_jit.outputs["Value"], sn_jit2.inputs[0])
    nt.links.new(sn_jit2.outputs["Value"], sn_thresh.inputs["Value"])

    # Garantizar estrictamente 0% nieve donde Normal.Z <= 0.72 (~44 deg)
    sn_hard_gate = nt.nodes.new("ShaderNodeMapRange")
    sn_hard_gate.inputs["From Min"].default_value = 0.72
    sn_hard_gate.inputs["From Max"].default_value = 0.75
    sn_hard_gate.inputs["To Min"].default_value = 0.0
    sn_hard_gate.inputs["To Max"].default_value = 1.0
    nt.links.new(sep_nrm.outputs["Z"], sn_hard_gate.inputs["Value"])

    snow_fac = nt.nodes.new("ShaderNodeMath")
    snow_fac.operation = "MULTIPLY"
    snow_fac.use_clamp = True
    nt.links.new(sn_thresh.outputs["Result"], snow_fac.inputs[0])
    nt.links.new(sn_hard_gate.outputs["Result"], snow_fac.inputs[1])

    snow_col = (0.90, 0.93, 0.96, 1.0)
    mx_snow = nt.nodes.new("ShaderNodeMix")
    mx_snow.data_type = "RGBA"
    mx_snow.inputs["B"].default_value = snow_col
    nt.links.new(snow_fac.outputs["Value"], mx_snow.inputs["Factor"])
    nt.links.new(mx_terra.outputs["Result"], mx_snow.inputs["A"])

    # Aplicar Ambient Occlusion global en grietas/uniones de rocas y talud (más suave sobre nieve)
    ao_ramp = nt.nodes.new("ShaderNodeValToRGB")
    ao_ramp.color_ramp.elements[0].position = 0.16
    ao_ramp.color_ramp.elements[0].color = (0.22, 0.25, 0.30, 1.0)
    ao_ramp.color_ramp.elements[1].position = 0.85
    ao_ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    nt.links.new(ao.outputs["AO"], ao_ramp.inputs["Fac"])

    ao_fac_inv = nt.nodes.new("ShaderNodeMath")
    ao_fac_inv.operation = "MULTIPLY_ADD"
    ao_fac_inv.inputs[1].default_value = -0.45
    ao_fac_inv.inputs[2].default_value = 0.78
    nt.links.new(snow_fac.outputs["Value"], ao_fac_inv.inputs[0])

    mx_ao = nt.nodes.new("ShaderNodeMix")
    mx_ao.data_type = "RGBA"
    mx_ao.blend_type = "MULTIPLY"
    nt.links.new(ao_fac_inv.outputs["Value"], mx_ao.inputs["Factor"])
    nt.links.new(mx_snow.outputs["Result"], mx_ao.inputs["A"])
    nt.links.new(ao_ramp.outputs["Color"], mx_ao.inputs["B"])
    nt.links.new(mx_ao.outputs["Result"], bsdf.inputs["Base Color"])

    # Bump 3D (micro-grano granito + musgo, atenuado en nieve)
    bump = nt.nodes.new("ShaderNodeBump")
    bump_s = nt.nodes.new("ShaderNodeMath")
    bump_s.operation = "MULTIPLY_ADD"
    bump_s.inputs[1].default_value = -0.26
    bump_s.inputs[2].default_value = 0.34
    nt.links.new(snow_fac.outputs["Value"], bump_s.inputs[0])
    nt.links.new(bump_s.outputs["Value"], bump.inputs["Strength"])
    bump.inputs["Distance"].default_value = 0.25
    h_mix = nt.nodes.new("ShaderNodeMath")
    h_mix.operation = "MULTIPLY_ADD"
    h_mix.inputs[1].default_value = 0.65
    nt.links.new(n_fine.outputs["Fac"], h_mix.inputs[0])
    nt.links.new(vor_rk.outputs["Distance"], h_mix.inputs[2])
    nt.links.new(h_mix.outputs["Value"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    # Roughness: roca/pradera mate (~0.85), nieve ligeramente más satinada (~0.46)
    mx_rough = nt.nodes.new("ShaderNodeMix")
    mx_rough.data_type = "FLOAT"
    mx_rough.inputs["A"].default_value = 0.85
    mx_rough.inputs["B"].default_value = 0.46
    nt.links.new(snow_fac.outputs["Value"], mx_rough.inputs["Factor"])
    nt.links.new(mx_rough.outputs["Result"], bsdf.inputs["Roughness"])

    return m_pl, m_bm


def bake_and_export(
    scene: bpy.types.Scene,
    obj: bpy.types.Object,
    out_dir: Path,
    bake_res: int = 2048,
    samples: int = 8,
) -> Tuple[Path, Path, Path]:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    print(f"  [UV] Smart UV Project 3D ({len(obj.data.polygons)} caras)...")
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(66.0), island_margin=0.0025)
    bpy.ops.object.mode_set(mode="OBJECT")

    scene.cycles.samples = samples
    albedo_p = out_dir / "baked_albedo.png"
    normal_p = out_dir / "baked_normal.png"
    rough_p = out_dir / "baked_roughness.png"

    def run_bake(bake_type: str, img_name: str, out_path: Path, is_color: bool, res: int) -> None:
        im = bpy.data.images.new(img_name, width=res, height=res, alpha=False)
        if not is_color:
            im.colorspace_settings.name = "Non-Color"
        img_nodes = []
        for mat in obj.data.materials:
            if not mat or not mat.node_tree:
                continue
            tn = mat.node_tree.nodes.new("ShaderNodeTexImage")
            tn.image = im
            mat.node_tree.nodes.active = tn
            img_nodes.append((mat.node_tree, tn))

        if bake_type == "DIFFUSE":
            scene.render.bake.use_pass_direct = False
            scene.render.bake.use_pass_indirect = False
            scene.render.bake.use_pass_color = True
        scene.render.bake.margin = 6
        bpy.ops.object.bake(type=bake_type)
        im.filepath_raw = str(out_path)
        im.file_format = "PNG"
        im.save()
        for nt, tn in img_nodes:
            nt.nodes.remove(tn)
        bpy.data.images.remove(im)

    print(f"  [Bake] Cycles DIFFUSE ({bake_res}x{bake_res}, samples={samples})...")
    run_bake("DIFFUSE", "BakeAlb", albedo_p, True, bake_res)
    norm_res = min(2048, bake_res)
    print(f"  [Bake] Cycles NORMAL ({norm_res}x{norm_res})...")
    run_bake("NORMAL", "BakeNrm", normal_p, False, norm_res)
    print(f"  [Bake] Cycles ROUGHNESS ({norm_res}x{norm_res})...")
    run_bake("ROUGHNESS", "BakeRgh", rough_p, False, norm_res)

    # Reemplazar por material PBR limpio cableado a los 3 mapas bakeados
    mat, nt, bsdf, _ = _new_material("Catan_Baked_PBR")
    tc = nt.nodes.new("ShaderNodeTexCoord")

    t_a = nt.nodes.new("ShaderNodeTexImage")
    im_a = bpy.data.images.load(str(albedo_p), check_existing=False); im_a.pack()
    t_a.image = im_a
    nt.links.new(tc.outputs["UV"], t_a.inputs["Vector"])
    nt.links.new(t_a.outputs["Color"], bsdf.inputs["Base Color"])

    t_n = nt.nodes.new("ShaderNodeTexImage")
    im_n = bpy.data.images.load(str(normal_p), check_existing=False); im_n.colorspace_settings.name = "Non-Color"; im_n.pack()
    t_n.image = im_n
    nt.links.new(tc.outputs["UV"], t_n.inputs["Vector"])
    nmap = nt.nodes.new("ShaderNodeNormalMap")
    nt.links.new(t_n.outputs["Color"], nmap.inputs["Color"])
    nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])

    t_r = nt.nodes.new("ShaderNodeTexImage")
    im_r = bpy.data.images.load(str(rough_p), check_existing=False); im_r.colorspace_settings.name = "Non-Color"; im_r.pack()
    t_r.image = im_r
    nt.links.new(tc.outputs["UV"], t_r.inputs["Vector"])
    nt.links.new(t_r.outputs["Color"], bsdf.inputs["Roughness"])

    obj.data.materials.clear()
    obj.data.materials.append(mat)
    return albedo_p, normal_p, rough_p


def render_audits(scene: bpy.types.Scene, out_dir: Path, samples: int = 16) -> None:
    world = bpy.data.worlds.new("StudioWorld")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.78, 0.80, 0.82, 1.0)
        bg.inputs["Strength"].default_value = 0.72

    for name, loc, energy, col, ang in [
        ("KeySun", (95.0, -70.0, 120.0), 3.0, (1.0, 0.98, 0.95), math.radians(10.0)),
        ("FillSun", (-85.0, -35.0, 80.0), 1.25, (0.88, 0.94, 1.0), math.radians(24.0)),
        ("RimSun", (-15.0, 95.0, 70.0), 1.35, (0.98, 0.99, 1.0), math.radians(16.0)),
    ]:
        ld = bpy.data.lights.new(name=name, type="SUN")
        ld.energy, ld.color, ld.angle = energy, col, ang
        lo = bpy.data.objects.new(name, ld)
        lo.location = loc
        lo.rotation_euler = (mathutils.Vector((0.0, 0.0, 28.0)) - mathutils.Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        scene.collection.objects.link(lo)

    scene.cycles.samples = samples
    cam_d = bpy.data.cameras.new("AuditCam")
    cam_o = bpy.data.objects.new("AuditCam", cam_d)
    scene.collection.objects.link(cam_o)
    scene.camera = cam_o

    mat_clay = bpy.data.materials.new("Clay_Audit")
    mat_clay.use_nodes = True
    bsdf_c = mat_clay.node_tree.nodes.get("Principled BSDF")
    if bsdf_c:
        bsdf_c.inputs["Base Color"].default_value = (0.60, 0.59, 0.57, 1.0)
        bsdf_c.inputs["Roughness"].default_value = 0.55

    def render_grid(views, out_png: Path, cols: int, rows: int, res_px: int, overrides=None):
        scene.render.resolution_x = res_px
        scene.render.resolution_y = res_px
        tiles = []
        with tempfile.TemporaryDirectory() as td:
            tp = Path(td)
            for idx, (cx, cy, cz, tgt) in enumerate(views):
                bpy.context.view_layer.material_override = overrides[idx] if overrides else None
                cam_o.data.type = "PERSP"
                cam_o.data.lens = 52.0
                cam_o.location = (cx, cy, cz)
                cam_o.rotation_euler = (mathutils.Vector(tgt) - mathutils.Vector((cx, cy, cz))).to_track_quat("-Z", "Y").to_euler()
                vp = tp / f"v_{idx}.png"
                scene.render.filepath = str(vp)
                bpy.ops.render.render(write_still=True)
                im = bpy.data.images.load(str(vp), check_existing=False)
                tiles.append(np.array(im.pixels[:], dtype=np.float32).reshape((res_px, res_px, 4)))
                bpy.data.images.remove(im)
        bpy.context.view_layer.material_override = None
        row_strips = [np.concatenate(tiles[r * cols:(r + 1) * cols], axis=1) for r in range(rows - 1, -1, -1)]
        canvas = np.concatenate(row_strips, axis=0)
        out_im = bpy.data.images.new(out_png.stem, width=cols * res_px, height=rows * res_px, alpha=True)
        out_im.pixels.foreach_set(canvas.ravel())
        out_im.filepath_raw = str(out_png)
        out_im.file_format = "PNG"
        out_im.save()
        bpy.data.images.remove(out_im)

    front_az = math.radians(-60.0)
    views_fb = [(175.0 * math.cos(az), 175.0 * math.sin(az), 102.0, (0.0, 0.0, 30.0)) for az in (front_az, front_az + math.pi, front_az, front_az + math.pi)]
    render_grid(views_fb, out_dir / "audit_front_back_pbr_clay.png", 2, 2, 560, overrides=[None, None, mat_clay, mat_clay])

    views_6 = [(190.0 * math.cos(front_az + math.radians(a)), 190.0 * math.sin(front_az + math.radians(a)), 112.0, (0.0, 0.0, 30.0)) for a in (0, 60, 120, 180, 240, 300)]
    render_grid(views_6, out_dir / "audit_6views_pbr.png", 3, 2, 460, overrides=None)
    render_grid(views_6, out_dir / "audit_6views_clay.png", 3, 2, 460, overrides=[mat_clay] * 6)


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(description="Fábrica Modular 3D Catán - Módulo 2 Look-Dev + Bake Cycles")
    p.add_argument("--stl", required=True)
    p.add_argument("--splat", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--bake-res", type=int, default=2048)
    p.add_argument("--bake-samples", type=int, default=8)
    p.add_argument("--render-samples", type=int, default=16)
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    try:
        addon_utils.enable("cycles")
    except Exception:
        pass
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    if hasattr(scene.cycles, "use_denoising"):
        try:
            scene.cycles.use_denoising = True
        except Exception:
            pass
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"

    bpy.ops.wm.stl_import(filepath=str(Path(args.stl).resolve()))
    obj = bpy.context.selected_objects[0]
    obj.name = "Catan_Factory_Master"

    m_pl, m_bm = build_cycles_montana_materials(Path(args.splat).resolve())
    obj.data.materials.clear()
    obj.data.materials.append(m_pl)
    obj.data.materials.append(m_bm)
    _assign_plinth_vs_terrain_slots(obj)

    bake_and_export(scene, obj, out_dir, bake_res=args.bake_res, samples=args.bake_samples)

    glb_path = out_dir / "tile_viewer.glb"
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.export_scene.gltf(filepath=str(glb_path), export_format="GLB", use_selection=True, export_apply=True)

    render_audits(scene, out_dir, samples=args.render_samples)
    print(f"[Bake Done] GLB={glb_path}")


if __name__ == "__main__":
    main()
