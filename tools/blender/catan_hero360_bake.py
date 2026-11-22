"""
Motor Modular Maker3D — Proyeccion Multi-Camara Cycles sobre Smart UV 3D + 3D Normal Displace
Convierte cualquier geometria Hero 360° (B-Rep + relieve estanco) + 6 vistas Nano Banana calibradas
en textura PBR 4K sin estiramiento vertical, micro-relieve fisico en 360°, STL maestro, GLB y renders de auditoria.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import bmesh
import bpy
import mathutils
from mathutils.bvhtree import BVHTree
import numpy as np


def _load_img_topdown_np(path: str) -> np.ndarray:
    """Carga PNG/JPG como array float32 (H, W, 3) top-down [0..1] usando bpy puro (sin PIL)."""
    im = bpy.data.images.load(str(path), check_existing=False)
    im.colorspace_settings.name = "Non-Color"
    w, h = im.size[0], im.size[1]
    ch = im.channels
    buf = np.empty(w * h * ch, dtype=np.float32)
    im.pixels.foreach_get(buf)
    bpy.data.images.remove(im)
    arr = buf.reshape((h, w, ch))[..., :3]
    return np.flipud(arr).copy()


def _save_np_rgb_png(arr_topdown: np.ndarray, path: Path) -> None:
    """Guarda array float32 (H, W, 3) top-down [0..1] a PNG usando bpy puro."""
    h, w = arr_topdown.shape[:2]
    rgba = np.ones((h, w, 4), dtype=np.float32)
    rgba[..., :3] = np.clip(arr_topdown[..., :3], 0.0, 1.0)
    rgba_bot = np.flipud(rgba)
    im = bpy.data.images.new("__out_png", width=w, height=h, alpha=True)
    im.pixels.foreach_set(rgba_bot.ravel())
    im.filepath_raw = str(path)
    im.file_format = "PNG"
    im.save()
    bpy.data.images.remove(im)

R = 64.0
APOTHEM = R * math.sqrt(3.0) * 0.5
Z_BASE = 25.0
ROAD_MARGIN = 3.50
PAD_RADIUS = 10.00


def _hex_distances_pt(x: float, y: float) -> Tuple[float, float]:
    ax, ay = abs(x), abs(y)
    de = APOTHEM - max(ax, 0.5 * ax + (math.sqrt(3.0) * 0.5) * ay)
    dv = 999.0
    for k in range(6):
        a = math.radians(30.0 + 60.0 * k)
        d = math.hypot(x - R * math.cos(a), y - R * math.sin(a))
        if d < dv:
            dv = d
    return de, dv


def _hex_distances_arr(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ax, ay = np.abs(x), np.abs(y)
    de = APOTHEM - np.maximum(ax, 0.5 * ax + (math.sqrt(3.0) * 0.5) * ay)
    dv = np.full_like(x, 999.0, dtype=np.float64)
    for k in range(6):
        a = math.radians(30.0 + 60.0 * k)
        dv = np.minimum(dv, np.hypot(x - R * math.cos(a), y - R * math.sin(a)))
    return de, dv


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / max(e1 - e0, 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _new_material(name: str):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat, nt, bsdf, out


def assign_plinth_vs_terrain_slots(obj: bpy.types.Object) -> np.ndarray:
    """Asigna slot 0 (Plinth_Gunmetal) a base/carril/pads y slot 1 (Hero360_Terrain) al relieve."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    is_ter_face = np.zeros(len(bm.faces), dtype=bool)
    for fi, f in enumerate(bm.faces):
        c = f.calc_center_median()
        if c.z < Z_BASE - 0.03:
            f.material_index = 0
        elif c.z <= Z_BASE + 0.14:
            de, dv = _hex_distances_pt(c.x, c.y)
            if de < ROAD_MARGIN + 0.25 or dv < PAD_RADIUS + 0.25:
                f.material_index = 0
            else:
                f.material_index = 1
                is_ter_face[fi] = True
        else:
            f.material_index = 1
            is_ter_face[fi] = True
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return is_ter_face


def subdivide_steep_terrain_edges(
    obj: bpy.types.Object,
    max_edge_mm: float = 0.29,
    max_passes: int = 3,
    target_max_faces: int = 2350000,
) -> None:
    """Subdivide adaptativamente aristas largas del relieve (> max_edge_mm) para eliminar facetas en paredes verticales."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    def _face_is_terrain(f) -> bool:
        c = f.calc_center_median()
        if c.z > Z_BASE + 0.08:
            return True
        if c.z < Z_BASE - 0.03:
            return False
        de, dv = _hex_distances_pt(c.x, c.y)
        return not (de < ROAD_MARGIN + 0.25 or dv < PAD_RADIUS + 0.25)

    for p_i in range(max_passes):
        cur_faces = len(bm.faces)
        if cur_faces >= target_max_faces:
            break
        ter_flags = [_face_is_terrain(f) for f in bm.faces]
        cand = []
        for e in bm.edges:
            if len(e.link_faces) != 2:
                continue
            if not (ter_flags[e.link_faces[0].index] and ter_flags[e.link_faces[1].index]):
                continue
            v0, v1 = e.verts[0].co, e.verts[1].co
            if v0.z <= Z_BASE + 0.04 or v1.z <= Z_BASE + 0.04:
                continue
            length = (v1 - v0).length
            if length > max_edge_mm:
                cand.append((length, e))
        if not cand:
            break
        budget = int((target_max_faces - cur_faces) / 1.85)
        if budget <= 0:
            break
        if len(cand) > budget:
            cand.sort(key=lambda t: t[0], reverse=True)
            cand = cand[:budget]
        edges_split = [t[1] for t in cand]
        bmesh.ops.subdivide_edges(
            bm,
            edges=edges_split,
            cuts=1,
            use_grid_fill=True,
            use_single_edge=True,
            quad_corner_type="INNER_VERT",
        )
        ngons = [f for f in bm.faces if len(f.verts) > 3]
        if ngons:
            bmesh.ops.triangulate(bm, faces=ngons, quad_method="SHORT_EDGE", ngon_method="BEAUTY")
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        print(f"  [Subdiv {p_i + 1}] split={len(edges_split)} aristas > {max_edge_mm:.2f} mm -> {len(bm.faces)} caras")

    bm.normal_update()
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


def build_plinth_material() -> bpy.types.Material:
    """Material gunmetal pizarra cepillado con anillos sutiles en carril y pads."""
    m_pl, nt, bsdf, _ = _new_material("Plinth_Gunmetal")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    n_p = nt.nodes.new("ShaderNodeTexNoise")
    n_p.inputs["Scale"].default_value = 0.72
    n_p.inputs["Detail"].default_value = 6.0
    nt.links.new(tc.outputs["Object"], n_p.inputs["Vector"])
    ao_p = nt.nodes.new("ShaderNodeAmbientOcclusion")
    ao_p.samples = 8
    ao_p.inputs["Distance"].default_value = 3.0
    r_p = nt.nodes.new("ShaderNodeValToRGB")
    r_p.color_ramp.elements[0].position = 0.25
    r_p.color_ramp.elements[0].color = (0.044, 0.048, 0.056, 1.0)
    r_p.color_ramp.elements[1].position = 0.85
    r_p.color_ramp.elements[1].color = (0.078, 0.084, 0.096, 1.0)
    nt.links.new(n_p.outputs["Fac"], r_p.inputs["Fac"])
    mx_p = nt.nodes.new("ShaderNodeMix")
    mx_p.data_type = "RGBA"
    mx_p.blend_type = "MULTIPLY"
    mx_p.inputs["Factor"].default_value = 0.55
    nt.links.new(r_p.outputs["Color"], mx_p.inputs["A"])
    nt.links.new(ao_p.outputs["Color"], mx_p.inputs["B"])
    nt.links.new(mx_p.outputs["Result"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.46
    return m_pl


def setup_camera_projections_and_shader(
    obj: bpy.types.Object,
    cams: Dict[str, dict],
) -> bpy.types.Material:
    """
    Calcula 6 capas UV proyectadas exactas (UV_000..UV_300) + visibilidad BVH en Blender C++
    con peso Winner-Takes-Most ndotv^4 x mascara erosionada, y construye el shader de mezcla Cycles.
    """
    cam_keys = ["0", "60", "120", "180", "240", "300"]
    me = obj.data

    # 1. Extraer vertices y normales del objeto en numpy
    n_verts = len(me.vertices)
    coords = np.empty(n_verts * 3, dtype=np.float64)
    normals = np.empty(n_verts * 3, dtype=np.float64)
    me.vertices.foreach_get("co", coords)
    me.vertices.foreach_get("normal", normals)
    V = coords.reshape(n_verts, 3)
    N = normals.reshape(n_verts, 3)

    de_v, dv_v = _hex_distances_arr(V[:, 0], V[:, 1])
    is_ter_v = (V[:, 2] >= Z_BASE - 0.02) & (de_v >= ROAD_MARGIN - 0.1) & (dv_v >= PAD_RADIUS - 0.1)

    # Evitar proyectar el bisel gris del zocalo sobre el faldon exterior de hierba (tirar UV hacia la pradera limpia)
    r_xy = np.maximum(np.hypot(V[:, 0], V[:, 1]), 1e-6)
    rim_pull = np.where(
        is_ter_v & (V[:, 2] < Z_BASE + 4.2),
        np.clip(np.maximum(ROAD_MARGIN + 3.20 - de_v, PAD_RADIUS + 3.00 - dv_v), 0.0, 3.30),
        0.0,
    )
    V_proj = V.copy()
    V_proj[:, 0] -= (V[:, 0] / r_xy) * rim_pull
    V_proj[:, 1] -= (V[:, 1] / r_xy) * rim_pull
    V_proj[:, 2] += 0.45 * (rim_pull > 0.0)

    # BVHTree para test de oclusion exacto por rayo en C++
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    bvh = BVHTree.FromBMesh(bm)

    # Matriz de adyacencia rapida para propagar pesos a micro-grietas ocultas
    edges_i0 = np.array([e.verts[0].index for e in bm.edges], dtype=np.int32)
    edges_i1 = np.array([e.verts[1].index for e in bm.edges], dtype=np.int32)
    bm.free()

    W_cams = np.zeros((n_verts, len(cam_keys)), dtype=np.float32)
    UVs_cams = np.zeros((len(cam_keys), n_verts, 2), dtype=np.float32)
    M_cams = np.zeros((len(cam_keys), n_verts), dtype=np.float32)

    print("  [Proj] Calculando proyeccion UV + BVH raycast 360° para 6 camaras...")
    for ci, k in enumerate(cam_keys):
        cd = cams[k]
        cam_pos = np.array(cd["cam_xyz"], dtype=np.float64)
        fwd = np.array(cd["fwd"], dtype=np.float64)
        right_r = np.array(cd["right_r"], dtype=np.float64)
        up_r = np.array(cd["up_r"], dtype=np.float64)
        W_img, H_img = float(cd["W"]), float(cd["H"])
        f_px = (H_img * 0.5) / math.tan(math.radians(cd["fov_deg"] * 0.5))

        rel = V_proj - cam_pos[None, :]
        xc = rel @ right_r
        yc = rel @ up_r
        zc = np.maximum(rel @ fwd, 1e-4)

        u_px = W_img * 0.5 + f_px * (xc / zc)
        v_pil = H_img * 0.5 - f_px * (yc / zc)
        u_bl = np.clip(u_px / W_img, 0.0, 1.0)
        v_bl = np.clip(1.0 - (v_pil / H_img), 0.0, 1.0)
        UVs_cams[ci, :, 0] = u_bl.astype(np.float32)
        UVs_cams[ci, :, 1] = v_bl.astype(np.float32)

        # Mascara erosionada del terreno en esta vista (cero bordes de cielo/zocalo)
        msk_arr = _load_img_topdown_np(cd["mask"])[..., 0]
        mh, mw = msk_arr.shape
        iu = np.clip(np.round(u_px), 0, mw - 1).astype(np.int32)
        iv = np.clip(np.round(v_pil), 0, mh - 1).astype(np.int32)
        m_val = msk_arr[iv, iu]
        M_cams[ci] = m_val

        dir_to_cam = cam_pos[None, :] - V
        dist_to_cam = np.linalg.norm(dir_to_cam, axis=1)
        dir_n = dir_to_cam / np.maximum(dist_to_cam[:, None], 1e-6)
        ndotv = np.sum(N * dir_n, axis=1)

        in_bounds = (u_px > 2.0) & (u_px < W_img - 3.0) & (v_pil > 2.0) & (v_pil < H_img - 3.0)
        cand_idx = np.where(is_ter_v & in_bounds & (m_val > 0.03) & (ndotv > 0.08))[0]

        vis = np.zeros(n_verts, dtype=bool)
        vec_M = mathutils.Vector
        for vi_idx in cand_idx:
            p = V[vi_idx] + N[vi_idx] * 0.14
            d = dir_n[vi_idx]
            max_d = float(dist_to_cam[vi_idx]) - 0.5
            hit_loc, _, _, hit_d = bvh.ray_cast(vec_M(p), vec_M(d), max_d)
            if hit_loc is None:
                vis[vi_idx] = True

        # Peso Winner-Takes-Most fuerte para preservar nitidez fotografica por faceta
        w_k = np.where(vis, (np.maximum(ndotv, 0.0) ** 4.0) * m_val, 0.0).astype(np.float32)
        W_cams[:, ci] = w_k
        print(f"    cam {int(k):03d}°: visible={(vis).sum()} verts, mean_w={(w_k[vis].mean() if np.any(vis) else 0.0):.3f}")

    # Propagar pesos de vertices vecinos hacia micro-hendiduras ocultas y faldon exterior (18 pasadas)
    w_sum = np.sum(W_cams, axis=1)
    for _ in range(18):
        missing = is_ter_v & (w_sum < 1e-4)
        if not np.any(missing):
            break
        acc = np.zeros_like(W_cams)
        np.add.at(acc, edges_i0, W_cams[edges_i1])
        np.add.at(acc, edges_i1, W_cams[edges_i0])
        acc_s = np.sum(acc, axis=1)
        fill = missing & (acc_s > 1e-5)
        W_cams[fill] = acc[fill]
        w_sum = np.sum(W_cams, axis=1)

    # Ultimo respaldo limpio por mejor ndotv respetando m_val > 0.02 donde aun falte algun vertice interno
    still_miss = is_ter_v & (w_sum < 1e-5)
    if np.any(still_miss):
        best_fb = np.full(still_miss.sum(), 0, dtype=np.int32)
        best_score = np.full(still_miss.sum(), -1e9, dtype=np.float32)
        m_indices = np.where(still_miss)[0]
        for ci, k in enumerate(cam_keys):
            cd = cams[k]
            cam_pos = np.array(cd["cam_xyz"], dtype=np.float64)
            dvec = cam_pos[None, :] - V[m_indices]
            dvec /= np.maximum(np.linalg.norm(dvec, axis=1, keepdims=True), 1e-6)
            sc = np.sum(N[m_indices] * dvec, axis=1).astype(np.float32)
            # Solo donde cae dentro de imagen Y de la mascara erosionada
            u_b, v_b = UVs_cams[ci, m_indices, 0], UVs_cams[ci, m_indices, 1]
            ok = (u_b > 0.02) & (u_b < 0.98) & (v_b > 0.02) & (v_b < 0.98) & (M_cams[ci, m_indices] > 0.02)
            better = ok & (sc > best_score)
            best_score[better] = sc[better]
            best_fb[better] = ci
        for j, vi_idx in enumerate(m_indices):
            W_cams[vi_idx, best_fb[j]] = 1.0
        w_sum = np.sum(W_cams, axis=1)

    # Normalizar pesos para que sumen exactamente 1.0 y suavizar ligeramente fronteras de oclusion (2 pasadas)
    W_cams = W_cams / np.maximum(w_sum[:, None], 1e-6)
    deg_w = np.zeros(n_verts, dtype=np.float32)
    np.add.at(deg_w, edges_i0, 1.0)
    np.add.at(deg_w, edges_i1, 1.0)
    deg_w = np.maximum(deg_w, 1.0)
    for _ in range(2):
        acc = np.zeros_like(W_cams)
        np.add.at(acc, edges_i0, W_cams[edges_i1])
        np.add.at(acc, edges_i1, W_cams[edges_i0])
        nbr = acc / deg_w[:, None]
        W_cams = np.where(is_ter_v[:, None], 0.72 * W_cams + 0.28 * nbr, W_cams)
        W_cams = W_cams / np.maximum(np.sum(W_cams, axis=1, keepdims=True), 1e-6)

    # Escribir los 6 UV layers + 2 Color Attributes (CamW0=[w0,w60,w120,w180], CamW1=[w240,w300,0,1])
    n_loops = len(me.loops)
    loop_vi = np.empty(n_loops, dtype=np.int32)
    me.loops.foreach_get("vertex_index", loop_vi)

    for ci, k in enumerate(cam_keys):
        uv_name = f"UV_{int(k):03d}"
        uv_lay = me.uv_layers.new(name=uv_name)
        uv_flat = UVs_cams[ci, loop_vi, :].ravel()
        uv_lay.data.foreach_set("uv", uv_flat)

    ca0 = me.color_attributes.new(name="CamW0", type="FLOAT_COLOR", domain="POINT")
    ca1 = me.color_attributes.new(name="CamW1", type="FLOAT_COLOR", domain="POINT")
    rgba0 = np.column_stack([W_cams[:, 0], W_cams[:, 1], W_cams[:, 2], W_cams[:, 3]]).astype(np.float32).ravel()
    rgba1 = np.column_stack([W_cams[:, 4], W_cams[:, 5], np.zeros(n_verts), np.ones(n_verts)]).astype(np.float32).ravel()
    ca0.data.foreach_set("color", rgba0)
    ca1.data.foreach_set("color", rgba1)
    me.update()

    # Construir nodo shader Cycles que mezcla exactamente sum(Image_k(UV_k) * W_k)
    m_ter, nt, bsdf, _ = _new_material("Hero360_Terrain")
    a0 = nt.nodes.new("ShaderNodeAttribute")
    a0.attribute_name = "CamW0"
    sep0 = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(a0.outputs["Color"], sep0.inputs["Color"])

    a1 = nt.nodes.new("ShaderNodeAttribute")
    a1.attribute_name = "CamW1"
    sep1 = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(a1.outputs["Color"], sep1.inputs["Color"])

    weight_sockets = [
        sep0.outputs["Red"],
        sep0.outputs["Green"],
        sep0.outputs["Blue"],
        a0.outputs["Alpha"],
        sep1.outputs["Red"],
        sep1.outputs["Green"],
    ]

    accum_vec = None
    for ci, k in enumerate(cam_keys):
        cd = cams[k]
        uv_n = nt.nodes.new("ShaderNodeUVMap")
        uv_n.uv_map = f"UV_{int(k):03d}"

        tex = nt.nodes.new("ShaderNodeTexImage")
        im = bpy.data.images.load(cd["image"], check_existing=True)
        tex.image = im
        tex.extension = "EXTEND"
        nt.links.new(uv_n.outputs["UV"], tex.inputs["Vector"])

        vsc = nt.nodes.new("ShaderNodeVectorMath")
        vsc.operation = "SCALE"
        nt.links.new(tex.outputs["Color"], vsc.inputs[0])
        nt.links.new(weight_sockets[ci], vsc.inputs["Scale"])

        if accum_vec is None:
            accum_vec = vsc.outputs["Vector"]
        else:
            vadd = nt.nodes.new("ShaderNodeVectorMath")
            vadd.operation = "ADD"
            nt.links.new(accum_vec, vadd.inputs[0])
            nt.links.new(vsc.outputs["Vector"], vadd.inputs[1])
            accum_vec = vadd.outputs["Vector"]

    nt.links.new(accum_vec, bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.85
    return m_ter


def _taubin_smooth_terrain_only(obj: bpy.types.Object, is_ter_v: np.ndarray, passes: int = 2) -> None:
    """Suavizado Taubin sin encogimiento solo en el interior seguro del relieve (protege bordes B-Rep)."""
    me = obj.data
    n_verts = len(me.vertices)
    coords = np.empty(n_verts * 3, dtype=np.float64)
    me.vertices.foreach_get("co", coords)
    V = coords.reshape(n_verts, 3)

    de, dv = _hex_distances_arr(V[:, 0], V[:, 1])
    safe = is_ter_v & (de > ROAD_MARGIN + 0.9) & (dv > PAD_RADIUS + 0.9) & (V[:, 2] > Z_BASE + 0.20)

    e0 = np.array([e.vertices[0] for e in me.edges], dtype=np.int32)
    e1 = np.array([e.vertices[1] for e in me.edges], dtype=np.int32)
    deg = np.zeros(n_verts, dtype=np.float64)
    np.add.at(deg, e0, 1.0)
    np.add.at(deg, e1, 1.0)
    deg = np.maximum(deg, 1.0)

    for _ in range(passes):
        for lam in (0.32, -0.33):
            avg = np.zeros_like(V)
            np.add.at(avg, e0, V[e1])
            np.add.at(avg, e1, V[e0])
            avg /= deg[:, None]
            V[safe] += lam * (avg[safe] - V[safe])

    V[:, 2] = np.maximum(V[:, 2], 0.0)
    me.vertices.foreach_set("co", V.ravel())
    me.update()


def sculpt_3d_from_baked_albedo(obj: bpy.types.Object, albedo_path: Path) -> None:
    """
    Lee el albedo 4K ya horneado sobre Smart UV 3D y esculpe grietas/crestas fisicas
    en direccion de la normal 3D real (nx, ny, nz), afectando tambien a caras traseras y acantilados >=55°.
    """
    arr = _load_img_topdown_np(str(albedo_path))
    ah, aw = arr.shape[:2]

    me = obj.data
    n_verts = len(me.vertices)
    n_loops = len(me.loops)
    coords = np.empty(n_verts * 3, dtype=np.float64)
    normals = np.empty(n_verts * 3, dtype=np.float64)
    me.vertices.foreach_get("co", coords)
    me.vertices.foreach_get("normal", normals)
    V = coords.reshape(n_verts, 3)
    N = normals.reshape(n_verts, 3)

    uv_lay = me.uv_layers["UVMap"]
    uv_flat = np.empty(n_loops * 2, dtype=np.float32)
    uv_lay.data.foreach_get("uv", uv_flat)
    uv_l = uv_flat.reshape(n_loops, 2)
    loop_vi = np.empty(n_loops, dtype=np.int32)
    me.loops.foreach_get("vertex_index", loop_vi)

    # Promediar UV por vertice (primer loop valido dentro de su isla UV) e interpolacion bilineal exacta
    v_uv = np.zeros((n_verts, 2), dtype=np.float32)
    v_uv[loop_vi] = uv_l

    fu = np.clip(v_uv[:, 0], 0.0, 1.0) * (aw - 1)
    fv = np.clip(1.0 - v_uv[:, 1], 0.0, 1.0) * (ah - 1)
    u0 = np.floor(fu).astype(np.int32)
    v0 = np.floor(fv).astype(np.int32)
    u1 = np.minimum(u0 + 1, aw - 1)
    v1 = np.minimum(v0 + 1, ah - 1)
    du = (fu - u0)[:, None]
    dv_uv = (fv - v0)[:, None]
    rgb = (
        arr[v0, u0] * (1.0 - du) * (1.0 - dv_uv)
        + arr[v0, u1] * du * (1.0 - dv_uv)
        + arr[v1, u0] * (1.0 - du) * dv_uv
        + arr[v1, u1] * du * dv_uv
    )
    lum = 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]
    sat = np.max(rgb, axis=1) - np.min(rgb, axis=1)

    de, dv = _hex_distances_arr(V[:, 0], V[:, 1])
    is_ter_v = (V[:, 2] >= Z_BASE + 0.04) & (de >= ROAD_MARGIN + 0.45) & (dv >= PAD_RADIUS + 0.45)
    safe_margin = (
        is_ter_v
        & _smoothstep(ROAD_MARGIN + 0.5, ROAD_MARGIN + 2.4, de).astype(bool)
    )
    weight_border = (
        _smoothstep(ROAD_MARGIN + 0.5, ROAD_MARGIN + 2.6, de)
        * _smoothstep(PAD_RADIUS + 0.5, PAD_RADIUS + 2.4, dv)
        * _smoothstep(Z_BASE + 0.05, Z_BASE + 0.85, V[:, 2])
    )
    # No deformar nieve limpia; esculpir rocas, barrancos, piedras y musgo
    snow_mask = _smoothstep(0.68, 0.84, lum) * (1.0 - _smoothstep(0.10, 0.20, sat)) * _smoothstep(Z_BASE + 3.0, Z_BASE + 8.0, V[:, 2])

    # Contraste local normalizado sobre grafo de malla densificada (funciona en cualquier pendiente 0°..90°!)
    e0 = np.array([e.vertices[0] for e in me.edges], dtype=np.int32)
    e1 = np.array([e.vertices[1] for e in me.edges], dtype=np.int32)
    deg = np.zeros(n_verts, dtype=np.float32)
    np.add.at(deg, e0, 1.0)
    np.add.at(deg, e1, 1.0)
    deg = np.maximum(deg, 1.0)

    lum_smooth = lum.copy()
    for _ in range(52):
        acc = np.zeros(n_verts, dtype=np.float32)
        np.add.at(acc, e0, lum_smooth[e1])
        np.add.at(acc, e1, lum_smooth[e0])
        lum_smooth = np.where(is_ter_v, acc / deg, lum_smooth)

    # Normalizar rocas oscuras para tallar grietas a la misma profundidad fisica
    lum_eq = np.clip((lum / np.maximum(lum_smooth, 0.14)) * 0.52, 0.0, 1.35)
    lum_sc = 0.45 * lum + 0.55 * lum_eq

    l_sm_wide = lum_sc.copy()
    for _ in range(38):
        acc = np.zeros(n_verts, dtype=np.float32)
        np.add.at(acc, e0, l_sm_wide[e1])
        np.add.at(acc, e1, l_sm_wide[e0])
        l_sm_wide = np.where(is_ter_v, acc / deg, l_sm_wide)

    l_sm_mid = lum_sc.copy()
    for _ in range(6):
        acc = np.zeros(n_verts, dtype=np.float32)
        np.add.at(acc, e0, l_sm_mid[e1])
        np.add.at(acc, e1, l_sm_mid[e0])
        l_sm_mid = np.where(is_ter_v, acc / deg, l_sm_mid)

    crevice_macro = np.clip((l_sm_wide - l_sm_mid) - 0.006, 0.0, 0.55) * 2.80
    crevice_micro = np.clip((l_sm_mid - lum_sc) - 0.004, 0.0, 0.45) * 2.60
    ridge = np.clip((l_sm_mid - l_sm_wide) * 2.20 + np.clip(lum_sc - l_sm_mid, 0.0, 0.35) * 1.40, 0.0, 0.55)

    disp_mm = weight_border * (1.0 - 0.90 * snow_mask) * (0.58 * ridge - 0.72 * crevice_macro - 0.52 * crevice_micro)
    disp_mm = np.clip(disp_mm, -0.82, 0.44)

    V_new = V + N * disp_mm[:, None]
    # Clamping vertical minimo sobre el zocalo dentro del borde seguro
    V_new[is_ter_v, 2] = np.maximum(V_new[is_ter_v, 2], Z_BASE + 0.02)
    me.vertices.foreach_set("co", V_new.ravel())
    me.update()

    _taubin_smooth_terrain_only(obj, safe_margin, passes=1)
    print(f"  [Sculpt3D] Desplazamiento por Normal 3D completado (range=[{disp_mm.min():.3f}, {disp_mm.max():.3f}] mm)")


def bake_pbr_maps(scene: bpy.types.Scene, obj: bpy.types.Object, out_dir: Path, is_ter_face: np.ndarray, albedo_res: int = 4096, samples: int = 8) -> Tuple[Path, Path, Path]:
    """Hornea DIFFUSE 4K desde las 6 camaras, esculpe normales 3D, aplica suavizado organico y hornea NORMAL + ROUGHNESS."""
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    print(f"  [UV] Abriendo Smart UV Project 3D ({len(obj.data.polygons)} caras)...")
    uv_main = obj.data.uv_layers.new(name="UVMap")
    for ul in obj.data.uv_layers:
        ul.active = (ul.name == "UVMap")
        ul.active_render = (ul.name == "UVMap")
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(82.0), island_margin=0.001, area_weight=1.0, correct_aspect=True)
    bpy.ops.object.mode_set(mode="OBJECT")
    for ul in obj.data.uv_layers:
        ul.active = (ul.name == "UVMap")
        ul.active_render = (ul.name == "UVMap")

    albedo_p = out_dir / "baked_albedo.png"
    normal_p = out_dir / "baked_normal.png"
    rough_p = out_dir / "baked_roughness.png"

    im_a = bpy.data.images.new("Bake_Albedo", width=albedo_res, height=albedo_res)
    for m in obj.data.materials:
        n = m.node_tree.nodes.new("ShaderNodeTexImage")
        n.name = "__BAKE__"
        n.image = im_a
        m.node_tree.nodes.active = n

    scene.cycles.samples = samples
    scene.render.bake.use_pass_direct = False
    scene.render.bake.use_pass_indirect = False
    scene.render.bake.use_pass_color = True
    scene.render.bake.margin = 6
    print(f"  [Bake] Cycles DIFFUSE ({albedo_res}x{albedo_res})...")
    bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR"})
    im_a.filepath_raw = str(albedo_p)
    im_a.file_format = "PNG"
    im_a.save()

    # Esculpir fisicamente en 3D sobre la normal real leyendo el albedo recien horneado
    sculpt_3d_from_baked_albedo(obj, albedo_p)
    obj.data.polygons.foreach_set("use_smooth", is_ter_face.tolist())
    obj.data.update()

    # Conectar Bump sutil + Roughness por luminancia del albedo en Hero360_Terrain y hornear Normal + Roughness
    map_res = max(2048, min(albedo_res, 4096))
    m_ter = bpy.data.materials.get("Hero360_Terrain")
    if m_ter:
        nt = m_ter.node_tree
        bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
        uv_m = nt.nodes.new("ShaderNodeUVMap")
        uv_m.uv_map = "UVMap"
        t_alb = nt.nodes.new("ShaderNodeTexImage")
        im_disk = bpy.data.images.load(str(albedo_p), check_existing=False)
        t_alb.image = im_disk
        nt.links.new(uv_m.outputs["UV"], t_alb.inputs["Vector"])

        rgb2bw = nt.nodes.new("ShaderNodeRGBToBW")
        nt.links.new(t_alb.outputs["Color"], rgb2bw.inputs["Color"])

        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.24
        bump.inputs["Distance"].default_value = 0.16
        nt.links.new(rgb2bw.outputs["Val"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

        # Nieve satinada (~0.45), roca/pradera mate (~0.86)
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.inputs["From Min"].default_value = 0.65
        mr.inputs["From Max"].default_value = 0.85
        mr.inputs["To Min"].default_value = 0.86
        mr.inputs["To Max"].default_value = 0.45
        nt.links.new(rgb2bw.outputs["Val"], mr.inputs["Value"])
        nt.links.new(mr.outputs["Result"], bsdf.inputs["Roughness"])

    im_n = bpy.data.images.new("Bake_Normal", width=map_res, height=map_res)
    im_n.colorspace_settings.name = "Non-Color"
    for m in obj.data.materials:
        m.node_tree.nodes["__BAKE__"].image = im_n
    print(f"  [Bake] Cycles NORMAL ({map_res}x{map_res})...")
    bpy.ops.object.bake(type="NORMAL", normal_space="TANGENT")
    im_n.filepath_raw = str(normal_p)
    im_n.file_format = "PNG"
    im_n.save()

    im_r = bpy.data.images.new("Bake_Roughness", width=map_res, height=map_res)
    im_r.colorspace_settings.name = "Non-Color"
    for m in obj.data.materials:
        m.node_tree.nodes["__BAKE__"].image = im_r
    print(f"  [Bake] Cycles ROUGHNESS ({map_res}x{map_res})...")
    bpy.ops.object.bake(type="ROUGHNESS")
    im_r.filepath_raw = str(rough_p)
    im_r.file_format = "PNG"
    im_r.save()

    # Limpiar UV layers auxiliares de camara y dejar material PBR unico empacado sobre UVMap
    while len(obj.data.uv_layers) > 1:
        for ul in list(obj.data.uv_layers):
            if ul.name != "UVMap":
                obj.data.uv_layers.remove(ul)
                break
    for attr in list(obj.data.color_attributes):
        obj.data.color_attributes.remove(attr)

    mat, nt, bsdf, _ = _new_material("Catan_Hero360_PBR")
    tc = nt.nodes.new("ShaderNodeTexCoord")

    ta = nt.nodes.new("ShaderNodeTexImage")
    ia = bpy.data.images.load(str(albedo_p), check_existing=False); ia.pack()
    ta.image = ia
    nt.links.new(tc.outputs["UV"], ta.inputs["Vector"])
    nt.links.new(ta.outputs["Color"], bsdf.inputs["Base Color"])

    tn = nt.nodes.new("ShaderNodeTexImage")
    inm = bpy.data.images.load(str(normal_p), check_existing=False); inm.colorspace_settings.name = "Non-Color"; inm.pack()
    tn.image = inm
    nt.links.new(tc.outputs["UV"], tn.inputs["Vector"])
    nmap = nt.nodes.new("ShaderNodeNormalMap")
    nt.links.new(tn.outputs["Color"], nmap.inputs["Color"])
    nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])

    tr = nt.nodes.new("ShaderNodeTexImage")
    ir = bpy.data.images.load(str(rough_p), check_existing=False); ir.colorspace_settings.name = "Non-Color"; ir.pack()
    tr.image = ir
    nt.links.new(tc.outputs["UV"], tr.inputs["Vector"])
    nt.links.new(tr.outputs["Color"], bsdf.inputs["Roughness"])

    obj.data.materials.clear()
    obj.data.materials.append(mat)
    obj.data.polygons.foreach_set("use_smooth", is_ter_face.tolist())
    obj.data.update()
    return albedo_p, normal_p, rough_p


def render_6views_audit(scene: bpy.types.Scene, obj: bpy.types.Object, cams: Dict[str, dict], out_dir: Path, samples: int = 16) -> None:
    world = bpy.data.worlds.new("StudioHeroWorld")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.86, 0.87, 0.89, 1.0)
        bg.inputs["Strength"].default_value = 0.82

    for name, loc, energy, col, ang in [
        ("KeySun", (95.0, -70.0, 125.0), 2.3, (1.0, 0.98, 0.96), math.radians(14.0)),
        ("FillSun", (-85.0, -40.0, 85.0), 1.35, (0.90, 0.95, 1.0), math.radians(28.0)),
        ("RimSun", (-10.0, 95.0, 75.0), 1.25, (0.98, 0.99, 1.0), math.radians(20.0)),
    ]:
        ld = bpy.data.lights.new(name=name, type="SUN")
        ld.energy, ld.color, ld.angle = energy, col, ang
        lo = bpy.data.objects.new(name, ld)
        lo.location = loc
        lo.rotation_euler = (mathutils.Vector((0.0, 0.0, 29.0)) - mathutils.Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        scene.collection.objects.link(lo)

    scene.cycles.samples = samples
    cam_d = bpy.data.cameras.new("AuditCam")
    cam_d.clip_end = 2500.0
    cam_o = bpy.data.objects.new("AuditCam", cam_d)
    scene.collection.objects.link(cam_o)
    scene.camera = cam_o

    mat_clay = bpy.data.materials.new("Clay_Audit")
    mat_clay.use_nodes = True
    bsdf_c = mat_clay.node_tree.nodes.get("Principled BSDF")
    if bsdf_c:
        bsdf_c.inputs["Base Color"].default_value = (0.48, 0.49, 0.51, 1.0)
        bsdf_c.inputs["Roughness"].default_value = 0.46

    def _set_cam_from_dict(cd: dict) -> None:
        cam_pos = mathutils.Vector(cd["cam_xyz"])
        right = mathutils.Vector(cd["right_r"]).normalized()
        up = mathutils.Vector(cd["up_r"]).normalized()
        fwd = mathutils.Vector(cd["fwd"]).normalized()
        # Blender camera local axes: +X=right, +Y=up, -Z=fwd -> R = [right | up | -fwd]
        rot_m = mathutils.Matrix((
            (right.x, up.x, -fwd.x),
            (right.y, up.y, -fwd.y),
            (right.z, up.z, -fwd.z),
        ))
        cam_o.location = cam_pos
        cam_o.rotation_euler = rot_m.to_euler()
        cam_d.type = "PERSP"
        cam_d.sensor_fit = "VERTICAL"
        cam_d.sensor_height = 24.0
        cam_d.lens = 12.0 / math.tan(math.radians(cd["fov_deg"] * 0.5))

    def render_keys(keys: List[str], res_px: int, is_clay: bool) -> List[np.ndarray]:
        orig_mats = list(obj.data.materials)
        fill_l = bpy.data.lights.get("FillSun")
        key_l = bpy.data.lights.get("KeySun")
        orig_bg = bg.inputs["Strength"].default_value if bg else 0.82
        orig_fill = fill_l.energy if fill_l else 1.35
        orig_key = key_l.energy if key_l else 2.3
        if is_clay:
            obj.data.materials.clear()
            obj.data.materials.append(mat_clay)
            if bg:
                bg.inputs["Strength"].default_value = 0.34
            if fill_l:
                fill_l.energy = 0.72
            if key_l:
                key_l.energy = 2.65
        scene.render.resolution_x = res_px
        scene.render.resolution_y = res_px
        import tempfile
        tiles = []
        with tempfile.TemporaryDirectory() as td:
            for i, k in enumerate(keys):
                _set_cam_from_dict(cams[k])
                tp = Path(td) / f"v_{i}.png"
                scene.render.filepath = str(tp)
                bpy.ops.render.render(write_still=True)
                tiles.append(_load_img_topdown_np(str(tp)))
        if is_clay:
            obj.data.materials.clear()
            for m in orig_mats:
                obj.data.materials.append(m)
            if bg:
                bg.inputs["Strength"].default_value = orig_bg
            if fill_l:
                fill_l.energy = orig_fill
            if key_l:
                key_l.energy = orig_key
        return tiles

    cam_keys = ["0", "60", "120", "180", "240", "300"]
    pbr_tiles = render_keys(cam_keys, res_px=520, is_clay=False)
    clay_tiles = render_keys(cam_keys, res_px=520, is_clay=True)

    c_pbr = np.vstack([np.hstack(pbr_tiles[:3]), np.hstack(pbr_tiles[3:])])
    c_clay = np.vstack([np.hstack(clay_tiles[:3]), np.hstack(clay_tiles[3:])])
    _save_np_rgb_png(c_pbr, out_dir / "audit_6views_pbr.png")
    _save_np_rgb_png(c_clay, out_dir / "audit_6views_clay.png")

    fb = np.vstack([
        np.hstack([pbr_tiles[0], pbr_tiles[3]]),
        np.hstack([clay_tiles[0], clay_tiles[3]]),
    ])
    _save_np_rgb_png(fb, out_dir / "audit_front_back_pbr_clay.png")

    # Ortografico cenital
    cam_d.type = "ORTHO"
    cam_d.ortho_scale = 136.0
    cam_o.location = (0.0, 0.0, 220.0)
    cam_o.rotation_euler = (0.0, 0.0, 0.0)
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024
    scene.render.filepath = str(out_dir / "audit_ortho.png")
    bpy.ops.render.render(write_still=True)


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(description="Fábrica Modular Maker3D — Cycles Smart UV 3D Camera-Projection Bake")
    p.add_argument("--stl", required=True)
    p.add_argument("--cameras", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--bake-res", type=int, default=4096)
    p.add_argument("--bake-samples", type=int, default=8)
    p.add_argument("--render-samples", type=int, default=16)
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.cameras, "r", encoding="utf-8") as f:
        cams = json.load(f)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    import addon_utils
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
    obj.name = "Catan_Hero360"

    subdivide_steep_terrain_edges(obj, max_edge_mm=0.24, max_passes=3, target_max_faces=2200000)

    m_pl = build_plinth_material()
    m_ter = setup_camera_projections_and_shader(obj, cams)
    obj.data.materials.clear()
    obj.data.materials.append(m_pl)
    obj.data.materials.append(m_ter)
    is_ter_face = assign_plinth_vs_terrain_slots(obj)
    obj.data.polygons.foreach_set("use_smooth", is_ter_face.tolist())
    obj.data.update()

    bake_pbr_maps(scene, obj, out_dir, is_ter_face=is_ter_face, albedo_res=args.bake_res, samples=args.bake_samples)

    master_stl = out_dir / "tile_master.stl"
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.wm.stl_export(filepath=str(master_stl), export_selected_objects=True, ascii_format=False)

    glb_path = out_dir / "tile_viewer.glb"
    bpy.ops.export_scene.gltf(filepath=str(glb_path), export_format="GLB", use_selection=True, export_apply=True)

    render_6views_audit(scene, obj, cams, out_dir, samples=args.render_samples)
    print(f"[Hero360 Bake Done] STL={master_stl}, GLB={glb_path}")


if __name__ == "__main__":
    main()
