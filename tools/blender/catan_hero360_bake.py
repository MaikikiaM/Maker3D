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

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
import geology_micro_detail as gmd
import macro_bake_helpers as mbh


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
    """Asigna slot 0 (Plinth_Gunmetal) a base/carril/pads y slot 1 (Hero360_Terrain) al relieve (preserva slot 2 Talus si existe)."""
    me = obj.data
    n_polys = len(me.polygons)
    centers = np.empty(n_polys * 3, dtype=np.float64)
    me.polygons.foreach_get("center", centers)
    C = centers.reshape(n_polys, 3)

    de, dv = _hex_distances_arr(C[:, 0], C[:, 1])
    is_ter_face = (C[:, 2] >= Z_BASE - 0.03) & (
        (C[:, 2] > Z_BASE + 0.14) | ((de >= ROAD_MARGIN + 0.25) & (dv >= PAD_RADIUS + 0.25))
    )

    cur_idx = np.empty(n_polys, dtype=np.int32)
    me.polygons.foreach_get("material_index", cur_idx)
    # Conservar slot 2 (Talus_Rock) si ya fue marcado
    is_talus = (cur_idx == 2) & is_ter_face
    new_idx = np.where(is_talus, 2, np.where(is_ter_face, 1, 0)).astype(np.int32)
    me.polygons.foreach_set("material_index", new_idx.tolist())
    me.update()
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

    max_len_sq = max_edge_mm * max_edge_mm
    z_thr = Z_BASE + 0.04
    for p_i in range(max_passes):
        cur_faces = len(bm.faces)
        if cur_faces >= target_max_faces:
            break
        cand = []
        for e in bm.edges:
            if len(e.link_faces) != 2:
                continue
            v0, v1 = e.verts[0].co, e.verts[1].co
            if v0.z <= z_thr or v1.z <= z_thr:
                continue
            diff = v1 - v0
            lsq = diff.length_squared
            if lsq > max_len_sq:
                f0, f1 = e.link_faces[0], e.link_faces[1]
                if _face_is_terrain(f0) and _face_is_terrain(f1):
                    cand.append((math.sqrt(lsq), e))
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


def apply_geologic_microstrata(
    obj: bpy.types.Object,
    is_ter_face: np.ndarray,
    snow_mask_v: np.ndarray | None = None,
) -> Tuple[float, float]:
    """Aplica micro-estratificación rocosa y diaclasas Voronoi (>= 0.28 mm ancho) en pendientes >= 40°."""
    me = obj.data
    n_verts = len(me.vertices)
    coords = np.empty(n_verts * 3, dtype=np.float64)
    normals = np.empty(n_verts * 3, dtype=np.float64)
    me.vertices.foreach_get("co", coords)
    me.vertices.foreach_get("normal", normals)
    V = coords.reshape(n_verts, 3)
    N = normals.reshape(n_verts, 3)

    disp_mm = gmd.compute_geologic_microstrata_displacement(V, N, snow_mask=snow_mask_v)
    V_new = V + N * disp_mm[:, None]

    # No modificar vértices del zócalo ni invadir debajo de Z_BASE+0.02 en terreno
    de, dv = _hex_distances_arr(V[:, 0], V[:, 1])
    is_ter_v = (V[:, 2] >= Z_BASE + 0.04) & (de >= ROAD_MARGIN + 0.45) & (dv >= PAD_RADIUS + 0.45)
    V_new = np.where(is_ter_v[:, None], V_new, V)
    V_new[is_ter_v, 2] = np.maximum(V_new[is_ter_v, 2], Z_BASE + 0.02)
    me.vertices.foreach_set("co", V_new.ravel())
    me.update()

    _taubin_smooth_terrain_only(obj, is_ter_v, passes=1)
    dmin, dmax = float(disp_mm.min()), float(disp_mm.max())
    print(f"  [MicroStrata] Rango desplazamiento = [{dmin:.4f}, {dmax:.4f}] mm")
    return dmin, dmax


def build_talus_material_from_cameras(cams: Dict[str, dict], talus_manifest_path: str) -> bpy.types.Material:
    """Shader Cycles para slot 2 (Pedregal Poly Haven): 70% tono proyectado de cámaras + 30% roca triplanar CC0 + AO contacto."""
    cam_keys = ["0", "60", "120", "180", "240", "300"]
    m_tal, nt, bsdf, _ = _new_material("Hero360_Talus")
    a0 = nt.nodes.new("ShaderNodeAttribute"); a0.attribute_name = "CamW0"
    sep0 = nt.nodes.new("ShaderNodeSeparateColor"); nt.links.new(a0.outputs["Color"], sep0.inputs["Color"])
    a1 = nt.nodes.new("ShaderNodeAttribute"); a1.attribute_name = "CamW1"
    sep1 = nt.nodes.new("ShaderNodeSeparateColor"); nt.links.new(a1.outputs["Color"], sep1.inputs["Color"])
    w_socks = [sep0.outputs["Red"], sep0.outputs["Green"], sep0.outputs["Blue"], a0.outputs["Alpha"], sep1.outputs["Red"], sep1.outputs["Green"]]

    accum_vec = None
    for ci, k in enumerate(cam_keys):
        uv_n = nt.nodes.new("ShaderNodeUVMap"); uv_n.uv_map = f"UV_{int(k):03d}"
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(cams[k]["image"], check_existing=True)
        tex.extension = "EXTEND"
        nt.links.new(uv_n.outputs["UV"], tex.inputs["Vector"])
        vsc = nt.nodes.new("ShaderNodeVectorMath"); vsc.operation = "SCALE"
        nt.links.new(tex.outputs["Color"], vsc.inputs[0]); nt.links.new(w_socks[ci], vsc.inputs["Scale"])
        if accum_vec is None:
            accum_vec = vsc.outputs["Vector"]
        else:
            va = nt.nodes.new("ShaderNodeVectorMath"); va.operation = "ADD"
            nt.links.new(accum_vec, va.inputs[0]); nt.links.new(vsc.outputs["Vector"], va.inputs[1])
            accum_vec = va.outputs["Vector"]

    manifest = json.loads(Path(talus_manifest_path).read_text(encoding="utf-8"))
    repo_root = _SCRIPT_DIR.parent.parent
    rock_texs = [str((repo_root / p).resolve()) if not Path(p).is_absolute() else p for p in manifest.get("unique_rock_textures", [])]
    rock_texs = [p for p in rock_texs if Path(p).exists()]
    # Preferir textura de granito/roca terrestre frente a regolito lunar si está disponible
    pref = [p for p in rock_texs if "boulder" in Path(p).name.lower() or "rock_07" in Path(p).name.lower()]
    if pref:
        rock_texs = pref
    if rock_texs:
        tc_box = nt.nodes.new("ShaderNodeTexCoord")
        mapping = nt.nodes.new("ShaderNodeMapping")
        mapping.inputs["Scale"].default_value = (0.28, 0.28, 0.28)
        nt.links.new(tc_box.outputs["Object"], mapping.inputs["Vector"])
        tex_r = nt.nodes.new("ShaderNodeTexImage")
        tex_r.image = bpy.data.images.load(rock_texs[0], check_existing=True)
        tex_r.projection = "BOX"
        tex_r.projection_blend = 0.25
        nt.links.new(mapping.outputs["Vector"], tex_r.inputs["Vector"])

        mix_col = nt.nodes.new("ShaderNodeMix"); mix_col.data_type = "RGBA"
        mix_col.inputs["Factor"].default_value = 0.72
        nt.links.new(accum_vec, mix_col.inputs["A"])
        nt.links.new(tex_r.outputs["Color"], mix_col.inputs["B"])
        base_col_sock = mix_col.outputs["Result"]
    else:
        base_col_sock = accum_vec

    # Oscurecimiento de contacto en la base de las rocas (AO local)
    ao_c = nt.nodes.new("ShaderNodeAmbientOcclusion")
    ao_c.samples = 6; ao_c.only_local = True; ao_c.inputs["Distance"].default_value = 1.6
    ao_map = nt.nodes.new("ShaderNodeMapRange")
    ao_map.inputs["From Min"].default_value = 0.0; ao_map.inputs["From Max"].default_value = 1.0
    ao_map.inputs["To Min"].default_value = 0.50; ao_map.inputs["To Max"].default_value = 1.0
    nt.links.new(ao_c.outputs["AO"], ao_map.inputs["Value"])
    mul_ao = nt.nodes.new("ShaderNodeVectorMath"); mul_ao.operation = "SCALE"
    nt.links.new(base_col_sock, mul_ao.inputs[0]); nt.links.new(ao_map.outputs["Result"], mul_ao.inputs["Scale"])
    nt.links.new(mul_ao.outputs["Vector"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.88
    return m_tal


def build_plinth_texture(out_dir: Path, res: int = 1024) -> Path:
    """Genera textura CAD 1K gunmetal oscura (#1F242B) con aros concentricos en los 6 pads y biseles."""
    lin_x = np.linspace(-R, R, res, dtype=np.float64)
    lin_y = np.linspace(R, -R, res, dtype=np.float64)
    X, Y = np.meshgrid(lin_x, lin_y)
    de, dv = _hex_distances_arr(X, Y)

    rng = np.random.default_rng(888)
    grain = rng.normal(0.0, 0.014, (res, res))
    # Mezcla cepillada horizontal sutil
    brushed = 0.55 * grain + 0.45 * np.roll(grain, 1, axis=1)
    base = np.array([0.148, 0.156, 0.172], dtype=np.float64)[None, None, :] * (1.0 + brushed[..., None])

    rim_hi = np.exp(-0.5 * ((de - 0.90) / 0.30) ** 2) * 0.044
    curb_sh = np.exp(-0.5 * ((de - 3.55) / 0.38) ** 2) * 0.056
    pad_groove = np.exp(-0.5 * ((dv - 8.20) / 0.30) ** 2) * 0.044
    pad_hi = np.exp(-0.5 * ((dv - 9.00) / 0.26) ** 2) * 0.028
    col = np.clip(base + (rim_hi - curb_sh - pad_groove + pad_hi)[..., None], 0.05, 0.28).astype(np.float32)

    plinth_p = out_dir / "plinth_albedo_1k.png"
    _save_np_rgb_png(col, plinth_p)
    return plinth_p


def build_plinth_material(out_dir: Path) -> Tuple[bpy.types.Material, Path]:
    """Material gunmetal pizarra oscuro con aros de poblado y biseles CAD dedicados."""
    plinth_p = build_plinth_texture(out_dir, res=1024)
    m_pl, nt, bsdf, _ = _new_material("Plinth_Gunmetal")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    t_pl = nt.nodes.new("ShaderNodeTexImage")
    im_pl = bpy.data.images.load(str(plinth_p), check_existing=False)
    t_pl.image = im_pl
    nt.links.new(tc.outputs["UV"], t_pl.inputs["Vector"])
    nt.links.new(t_pl.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.44
    bsdf.inputs["Metallic"].default_value = 0.18
    return m_pl, plinth_p


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

    # Micro-ajuste minimo de borde (<= 0.55 mm) sin estiramiento radial visible sobre pradera
    r_xy = np.maximum(np.hypot(V[:, 0], V[:, 1]), 1e-6)
    rim_pull = np.where(
        is_ter_v & (V[:, 2] < Z_BASE + 1.25),
        np.clip(np.maximum(ROAD_MARGIN + 0.55 - de_v, PAD_RADIUS + 0.55 - dv_v), 0.0, 0.65),
        0.0,
    )
    V_proj = V.copy()
    V_proj[:, 0] -= (V[:, 0] / r_xy) * rim_pull
    V_proj[:, 1] -= (V[:, 1] / r_xy) * rim_pull
    V_proj[:, 2] += 0.20 * (rim_pull > 0.0)

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

        # Peso Winner-Takes-Most intenso para preservar nitidez fotografica por faceta
        w_k = np.where(vis, (np.maximum(ndotv, 0.0) ** 5.5) * m_val, 0.0).astype(np.float32)
        W_cams[:, ci] = w_k
        print(f"    cam {int(k):03d}°: visible={(vis).sum()} verts, mean_w={(w_k[vis].mean() if np.any(vis) else 0.0):.3f}")

    # Propagar pesos de vertices vecinos hacia micro-hendiduras ocultas y faldon exterior (22 pasadas)
    w_sum = np.sum(W_cams, axis=1)
    for _ in range(22):
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
            u_b, v_b = UVs_cams[ci, m_indices, 0], UVs_cams[ci, m_indices, 1]
            ok = (u_b > 0.02) & (u_b < 0.98) & (v_b > 0.02) & (v_b < 0.98) & (M_cams[ci, m_indices] > 0.02)
            better = ok & (sc > best_score)
            best_score[better] = sc[better]
            best_fb[better] = ci
        for j, vi_idx in enumerate(m_indices):
            W_cams[vi_idx, best_fb[j]] = 1.0
        w_sum = np.sum(W_cams, axis=1)

    # Normalizar pesos y difuminar 1 pasada SOLO en costuras entre camaras (nunca sobre caras nitidas!)
    W_cams = W_cams / np.maximum(w_sum[:, None], 1e-6)
    deg_w = np.zeros(n_verts, dtype=np.float32)
    np.add.at(deg_w, edges_i0, 1.0)
    np.add.at(deg_w, edges_i1, 1.0)
    deg_w = np.maximum(deg_w, 1.0)
    seam_v = is_ter_v & (np.max(W_cams, axis=1) < 0.72)
    acc = np.zeros_like(W_cams)
    np.add.at(acc, edges_i0, W_cams[edges_i1])
    np.add.at(acc, edges_i1, W_cams[edges_i0])
    nbr = acc / deg_w[:, None]
    W_cams = np.where(seam_v[:, None], 0.76 * W_cams + 0.24 * nbr, W_cams)
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

    rim_guard = (
        (1.0 - _smoothstep(ROAD_MARGIN + 0.55, ROAD_MARGIN + 1.65, de_v) * _smoothstep(PAD_RADIUS + 0.45, PAD_RADIUS + 1.50, dv_v))
        * (1.0 - _smoothstep(Z_BASE + 1.4, Z_BASE + 2.8, V[:, 2]))
        * is_ter_v.astype(np.float32)
    )
    ca0 = me.color_attributes.new(name="CamW0", type="FLOAT_COLOR", domain="POINT")
    ca1 = me.color_attributes.new(name="CamW1", type="FLOAT_COLOR", domain="POINT")
    rgba0 = np.column_stack([W_cams[:, 0], W_cams[:, 1], W_cams[:, 2], W_cams[:, 3]]).astype(np.float32).ravel()
    rgba1 = np.column_stack([W_cams[:, 4], W_cams[:, 5], rim_guard, np.ones(n_verts)]).astype(np.float32).ravel()
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

    # Reemplazar SOLO grises frios metalicos en el borde exterior (rim_guard * low_sat) por musgo alpino natural 3D
    sep_hsv = nt.nodes.new("ShaderNodeSeparateColor")
    sep_hsv.mode = "HSV"
    nt.links.new(accum_vec, sep_hsv.inputs["Color"])
    low_sat = nt.nodes.new("ShaderNodeMapRange")
    low_sat.clamp = True
    low_sat.inputs["From Min"].default_value = 0.08
    low_sat.inputs["From Max"].default_value = 0.24
    low_sat.inputs["To Min"].default_value = 1.0
    low_sat.inputs["To Max"].default_value = 0.0
    nt.links.new(sep_hsv.outputs["Green"], low_sat.inputs["Value"])

    moss_fac = nt.nodes.new("ShaderNodeMath")
    moss_fac.operation = "MULTIPLY"
    moss_fac.use_clamp = True
    nt.links.new(sep1.outputs["Blue"], moss_fac.inputs[0])
    nt.links.new(low_sat.outputs["Result"], moss_fac.inputs[1])

    tc_m = nt.nodes.new("ShaderNodeTexCoord")
    n_moss = nt.nodes.new("ShaderNodeTexNoise")
    n_moss.inputs["Scale"].default_value = 2.6
    n_moss.inputs["Detail"].default_value = 7.0
    nt.links.new(tc_m.outputs["Object"], n_moss.inputs["Vector"])
    r_moss = nt.nodes.new("ShaderNodeValToRGB")
    r_moss.color_ramp.elements[0].position = 0.25
    r_moss.color_ramp.elements[0].color = (0.032, 0.056, 0.012, 1.0)
    r_moss.color_ramp.elements[1].position = 0.80
    r_moss.color_ramp.elements[1].color = (0.095, 0.135, 0.030, 1.0)
    nt.links.new(n_moss.outputs["Fac"], r_moss.inputs["Fac"])

    mix_rim = nt.nodes.new("ShaderNodeMix")
    mix_rim.data_type = "RGBA"
    nt.links.new(moss_fac.outputs["Value"], mix_rim.inputs["Factor"])
    nt.links.new(accum_vec, mix_rim.inputs["A"])
    nt.links.new(r_moss.outputs["Color"], mix_rim.inputs["B"])

    nt.links.new(mix_rim.outputs["Result"], bsdf.inputs["Base Color"])
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

    # Promediar UV por vertice usando SOLO loops del relieve (slot 1) para aislar de UVs del zocalo
    poly_mat = np.empty(len(me.polygons), dtype=np.int32)
    me.polygons.foreach_get("material_index", poly_mat)
    loop_tot = np.empty(len(me.polygons), dtype=np.int32)
    me.polygons.foreach_get("loop_total", loop_tot)
    is_ter_loop = np.repeat(poly_mat == 1, loop_tot)
    is_slot2_v = np.zeros(n_verts, dtype=bool)
    is_slot2_v[loop_vi[np.repeat(poly_mat == 2, loop_tot)]] = True
    v_uv = np.zeros((n_verts, 2), dtype=np.float32)
    v_uv[loop_vi[is_ter_loop]] = uv_l[is_ter_loop]

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
    is_ter_v = (V[:, 2] >= Z_BASE + 0.04) & (de >= ROAD_MARGIN + 0.45) & (dv >= PAD_RADIUS + 0.45) & (~is_slot2_v)
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


def _compute_uv_coverage_pct(me: bpy.types.Mesh, face_mask: np.ndarray) -> float:
    """Calcula exactamente qué % del cuadrado UV [0,1]^2 cubren las caras seleccionadas."""
    uv_lay = me.uv_layers["UVMap"]
    n_loops = len(me.loops)
    n_polys = len(me.polygons)
    uv_arr = np.empty(n_loops * 2, dtype=np.float64)
    uv_lay.data.foreach_get("uv", uv_arr)
    uv_l = uv_arr.reshape(n_loops, 2)
    loop_start = np.empty(n_polys, dtype=np.int32)
    loop_tot = np.empty(n_polys, dtype=np.int32)
    me.polygons.foreach_get("loop_start", loop_start)
    me.polygons.foreach_get("loop_total", loop_tot)
    sel = face_mask & (loop_tot == 3)
    if not np.any(sel):
        return 0.0
    s = loop_start[sel]
    p0, p1, p2 = uv_l[s], uv_l[s + 1], uv_l[s + 2]
    area = 0.5 * np.abs((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1]))
    return float(np.sum(area) * 100.0)


def _transfer_macro_uv_from_base(obj: bpy.types.Object, base_stl_path: Path, poly_mat: np.ndarray) -> float:
    """
    Despliega UVMap limpio sobre la geometría macro tile_base.stl (sin micro-fracturas ni piedras,
    ~72% cobertura sin fragmentos diminutos) y transfiere UVs por triángulo más cercano sin cruzar costuras.
    """
    me = obj.data
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.wm.stl_import(filepath=str(base_stl_path.resolve()))
    src_obj = bpy.context.selected_objects[0]
    src_obj.name = "_UV_Base_Ref"
    src_me = src_obj.data
    bpy.context.view_layer.objects.active = src_obj

    n_sp = len(src_me.polygons)
    s_ctrs = np.empty(n_sp * 3, dtype=np.float64)
    src_me.polygons.foreach_get("center", s_ctrs)
    Cs = s_ctrs.reshape(n_sp, 3)
    de_s, dv_s = _hex_distances_arr(Cs[:, 0], Cs[:, 1])
    src_ter = (Cs[:, 2] >= Z_BASE - 0.03) & ((Cs[:, 2] > Z_BASE + 0.14) | ((de_s >= ROAD_MARGIN + 0.25) & (dv_s >= PAD_RADIUS + 0.25)))

    src_me.uv_layers.new(name="UVMap")
    src_me.polygons.foreach_set("select", src_ter.tolist())
    src_me.update()
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_mode(type="FACE")
    bpy.ops.uv.smart_project(angle_limit=math.radians(89.0), island_margin=0.0003, area_weight=1.0, correct_aspect=True)
    bpy.ops.uv.pack_islands(shape_method="CONCAVE", rotate=True, margin=0.0004)
    bpy.ops.object.mode_set(mode="OBJECT")

    # Extraer geometría + UVs de src_obj en numpy
    ns_v = len(src_me.vertices)
    ns_l = len(src_me.loops)
    sv_co = np.empty(ns_v * 3, dtype=np.float64); src_me.vertices.foreach_get("co", sv_co); sV = sv_co.reshape(ns_v, 3)
    sl_vi = np.empty(ns_l, dtype=np.int32); src_me.loops.foreach_get("vertex_index", sl_vi)
    sl_uv = np.empty(ns_l * 2, dtype=np.float64); src_me.uv_layers["UVMap"].data.foreach_get("uv", sl_uv); sUV = sl_uv.reshape(ns_l, 2)
    s_ls = np.empty(n_sp, dtype=np.int32); src_me.polygons.foreach_get("loop_start", s_ls)

    s_tri_pts = np.stack([sV[sl_vi[s_ls]], sV[sl_vi[s_ls + 1]], sV[sl_vi[s_ls + 2]]], axis=1)  # (Fs, 3, 3)
    s_tri_uvs = np.stack([sUV[s_ls], sUV[s_ls + 1], sUV[s_ls + 2]], axis=1)  # (Fs, 3, 2)

    bm_s = bmesh.new()
    bm_s.from_mesh(src_me)
    bm_s.faces.ensure_lookup_table()
    bvh_s = BVHTree.FromBMesh(bm_s)
    bm_s.free()

    # Seleccionar triángulo más cercano de base_stl para cada cara Slot 1 de destino
    n_dp = len(me.polygons)
    d_ctrs = np.empty(n_dp * 3, dtype=np.float64); me.polygons.foreach_get("center", d_ctrs); Cd = d_ctrs.reshape(n_dp, 3)
    slot1_idx = np.where(poly_mat == 1)[0]
    src_match = np.zeros(len(slot1_idx), dtype=np.int32)
    vec_M = mathutils.Vector
    for ki, fi in enumerate(slot1_idx):
        _, _, idx_s, _ = bvh_s.find_nearest(vec_M(Cd[fi]))
        src_match[ki] = idx_s if idx_s is not None else 0

    # Eliminar src_obj para liberar RAM
    bpy.data.objects.remove(src_obj, do_unlink=True)
    bpy.data.meshes.remove(src_me)

    # Proyección baricéntrica vectorizada por triángulo fuente (mantiene todos los loops en la misma isla UV)
    nd_v = len(me.vertices)
    nd_l = len(me.loops)
    dv_co = np.empty(nd_v * 3, dtype=np.float64); me.vertices.foreach_get("co", dv_co); dV = dv_co.reshape(nd_v, 3)
    dl_vi = np.empty(nd_l, dtype=np.int32); me.loops.foreach_get("vertex_index", dl_vi)
    d_ls = np.empty(n_dp, dtype=np.int32); me.polygons.foreach_get("loop_start", d_ls)

    A = s_tri_pts[src_match, 0]  # (K, 3)
    B = s_tri_pts[src_match, 1]
    C = s_tri_pts[src_match, 2]
    v0 = B - A
    v1 = C - A
    d00 = np.sum(v0 * v0, axis=1)
    d01 = np.sum(v0 * v1, axis=1)
    d11 = np.sum(v1 * v1, axis=1)
    denom = np.maximum(d00 * d11 - d01 * d01, 1e-12)

    uv_dst = np.zeros((nd_l, 2), dtype=np.float32)
    s_l0 = d_ls[slot1_idx]
    uvA = s_tri_uvs[src_match, 0]
    uvB = s_tri_uvs[src_match, 1]
    uvC = s_tri_uvs[src_match, 2]
    for k_corner in range(3):
        l_ids = s_l0 + k_corner
        P = dV[dl_vi[l_ids]]
        v2 = P - A
        d20 = np.sum(v2 * v0, axis=1)
        d21 = np.sum(v2 * v1, axis=1)
        vb = (d11 * d20 - d01 * d21) / denom
        vc = (d00 * d21 - d01 * d20) / denom
        va = 1.0 - vb - vc
        w = np.clip(np.stack([va, vb, vc], axis=1), -0.15, 1.15)
        w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-6)
        uv_c = w[:, 0:1] * uvA + w[:, 1:2] * uvB + w[:, 2:3] * uvC
        uv_dst[l_ids] = np.clip(uv_c, 0.0, 1.0).astype(np.float32)

    if "UVMap" not in me.uv_layers:
        me.uv_layers.new(name="UVMap")
    for ul in me.uv_layers:
        ul.active = (ul.name == "UVMap")
        ul.active_render = (ul.name == "UVMap")
    me.uv_layers["UVMap"].data.foreach_set("uv", uv_dst.ravel())
    me.update()

    # Si hay rocas (slot 2), desplegarlas aparte y empaquetarlas en banda derecha [0.895, 0.998]
    bpy.ops.object.select_all(action="DESELECT"); obj.select_set(True); bpy.context.view_layer.objects.active = obj
    slot2_mask = poly_mat == 2
    if np.any(slot2_mask):
        me.polygons.foreach_set("select", slot2_mask.tolist()); me.update()
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_mode(type="FACE")
        bpy.ops.uv.smart_project(angle_limit=math.radians(88.0), island_margin=0.001, area_weight=1.0, correct_aspect=True)
        bpy.ops.uv.pack_islands(shape_method="CONCAVE", rotate=True, margin=0.001)
        bpy.ops.object.mode_set(mode="OBJECT")

        uv_cur = np.empty(nd_l * 2, dtype=np.float32)
        me.uv_layers["UVMap"].data.foreach_get("uv", uv_cur)
        uv_l = uv_cur.reshape(nd_l, 2)
        loop_tot = np.empty(n_dp, dtype=np.int32); me.polygons.foreach_get("loop_total", loop_tot)
        l_s1 = np.repeat(poly_mat == 1, loop_tot)
        l_s2 = np.repeat(slot2_mask, loop_tot)
        uv_l[l_s1, 0] = 0.002 + uv_l[l_s1, 0] * 0.890
        uv_l[l_s1, 1] = 0.002 + uv_l[l_s1, 1] * 0.996
        uv_l[l_s2, 0] = 0.896 + uv_l[l_s2, 0] * 0.102
        uv_l[l_s2, 1] = 0.002 + uv_l[l_s2, 1] * 0.996
        me.uv_layers["UVMap"].data.foreach_set("uv", uv_l.ravel())
        me.update()

    occ = _compute_uv_coverage_pct(me, poly_mat >= 1)
    print(f"  [Macro UV Transfer] Cobertura geométrica terreno+talus = {occ:.1f}%")
    return occ


def unwrap_terrain_uv_optimized(
    obj: bpy.types.Object,
    is_ter_face: np.ndarray,
    out_dir: Path | None = None,
    try_angles: Tuple[float, ...] = (66.0, 74.0, 82.0),
) -> float:
    """Despliega UVMap exclusivo para el relieve buscando la mayor ocupación posible (>=75% objetivo)."""
    me = obj.data
    if out_dir is not None and (out_dir / "tile_base.stl").exists() and len(me.polygons) > 3000000:
        poly_mat = np.empty(len(me.polygons), dtype=np.int32)
        me.polygons.foreach_get("material_index", poly_mat)
        return _transfer_macro_uv_from_base(obj, out_dir / "tile_base.stl", poly_mat)

    if "UVMap" not in me.uv_layers:
        me.uv_layers.new(name="UVMap")
    for ul in me.uv_layers:
        ul.active = (ul.name == "UVMap")
        ul.active_render = (ul.name == "UVMap")

    best_occ = -1.0
    best_uv = None
    sel_list = is_ter_face.tolist()
    for ang_deg in try_angles:
        bpy.ops.object.mode_set(mode="OBJECT")
        me.polygons.foreach_set("select", sel_list)
        me.update()
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_mode(type="FACE")
        bpy.ops.uv.smart_project(
            angle_limit=math.radians(ang_deg),
            island_margin=0.0005,
            area_weight=1.0,
            correct_aspect=True,
        )
        bpy.ops.object.mode_set(mode="OBJECT")
        occ = _compute_uv_coverage_pct(me, is_ter_face)
        print(f"    [UV Search] angle_limit={ang_deg:.0f}° margin=0.0005 -> cobertura teórica = {occ:.1f}%")
        if occ > best_occ:
            best_occ = occ
            best_uv = np.empty(len(me.loops) * 2, dtype=np.float32)
            me.uv_layers["UVMap"].data.foreach_get("uv", best_uv)
        if occ >= 76.0:
            break

    if best_uv is not None:
        me.uv_layers["UVMap"].data.foreach_set("uv", best_uv)
        me.update()
    return best_occ


def bake_pbr_maps(
    scene: bpy.types.Scene,
    obj: bpy.types.Object,
    out_dir: Path,
    is_ter_face: np.ndarray,
    plinth_albedo_p: Path,
    albedo_res: int = 4096,
    normal_res: int | None = None,
    rough_res: int | None = None,
    samples: int = 8,
    save_unsharp_to: Path | None = None,
    bake_pbr: bool = True,
) -> Tuple[Path, Path | None, Path | None]:
    """
    Aisla UV del zocalo vs relieve (>=75% del atlas exclusivo para el terreno), hornea DIFFUSE,
    aplica Unsharp libre de halos, esculpe normales 3D bilineales, suaviza solo caras organicas
    y hornea NORMAL/ROUGHNESS con micro-grano mineral.
    """
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    n_ter = int(np.sum(is_ter_face))
    print(f"  [UV] Smart UV Project 3D exclusivo en relieve ({n_ter}/{len(is_ter_face)} caras)...")
    try_angs = (66.0, 74.0, 82.0) if albedo_res >= 8192 else (82.0,)
    unwrap_terrain_uv_optimized(obj, is_ter_face, out_dir=out_dir, try_angles=try_angs)

    # Escribir UVs planares XY en el mismo UVMap SOLO sobre caras del zocalo (slot 0)
    me = obj.data
    n_loops = len(me.loops)
    uv_arr = np.empty(n_loops * 2, dtype=np.float32)
    me.uv_layers["UVMap"].data.foreach_get("uv", uv_arr)
    uv_l = uv_arr.reshape(n_loops, 2)
    n_verts = len(me.vertices)
    v_coords = np.empty(n_verts * 3, dtype=np.float32)
    me.vertices.foreach_get("co", v_coords)
    V_xy = v_coords.reshape(n_verts, 3)[:, :2]
    loop_vi = np.empty(n_loops, dtype=np.int32)
    me.loops.foreach_get("vertex_index", loop_vi)
    loop_tot = np.empty(len(me.polygons), dtype=np.int32)
    me.polygons.foreach_get("loop_total", loop_tot)
    is_plinth_loop = np.repeat(~is_ter_face, loop_tot)
    uv_l[is_plinth_loop] = np.clip((V_xy[loop_vi[is_plinth_loop]] + R) / (2.0 * R), 0.0, 1.0)
    me.uv_layers["UVMap"].data.foreach_set("uv", uv_l.ravel())
    me.update()

    for ul in obj.data.uv_layers:
        ul.active = (ul.name == "UVMap")
        ul.active_render = (ul.name == "UVMap")

    albedo_p = save_unsharp_to or (out_dir / "baked_albedo.png")
    normal_p = out_dir / "baked_normal.png"
    rough_p = out_dir / "baked_roughness.png"

    im_dummy = bpy.data.images.new("Bake_Dummy", width=64, height=64)
    im_a = bpy.data.images.new("Bake_Albedo", width=albedo_res, height=albedo_res)
    m_pl = bpy.data.materials.get("Plinth_Gunmetal")
    m_ter = bpy.data.materials.get("Hero360_Terrain")
    m_tal = bpy.data.materials.get("Hero360_Talus")
    for m, target_im in [(m_pl, im_dummy), (m_ter, im_a), (m_tal, im_a)]:
        if m:
            n = m.node_tree.nodes.new("ShaderNodeTexImage")
            n.name = "__BAKE__"
            n.image = target_im
            m.node_tree.nodes.active = n

    scene.cycles.samples = samples
    scene.render.bake.use_pass_direct = False
    scene.render.bake.use_pass_indirect = False
    scene.render.bake.use_pass_color = True
    scene.render.bake.margin = 6
    print(f"  [Bake] Cycles DIFFUSE ({albedo_res}x{albedo_res} exclusivo terreno)...")
    bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR"})
    im_a.filepath_raw = str(albedo_p)
    im_a.file_format = "PNG"
    im_a.save()

    # Unsharp limpio en el interior de las islas UV (cero halos oscuros en bordes)
    arr4 = _load_img_topdown_np(str(albedo_p))
    valid4 = np.max(arr4, axis=-1) > 0.015
    for _ in range(2):
        p_m = np.pad(valid4, 1, mode="constant", constant_values=False)
        valid4 = p_m[1:-1, 1:-1] & p_m[:-2, 1:-1] & p_m[2:, 1:-1] & p_m[1:-1, :-2] & p_m[1:-1, 2:]
    p_rgb = np.pad(arr4, ((1, 1), (1, 1), (0, 0)), mode="edge")
    blur4 = (
        4.0 * arr4
        + 2.0 * (p_rgb[:-2, 1:-1] + p_rgb[2:, 1:-1] + p_rgb[1:-1, :-2] + p_rgb[1:-1, 2:])
        + 1.0 * (p_rgb[:-2, :-2] + p_rgb[:-2, 2:] + p_rgb[2:, :-2] + p_rgb[2:, 2:])
    ) / 16.0
    arr4_sharp = np.where(valid4[..., None], np.clip(arr4 + 0.36 * (arr4 - blur4), 0.0, 1.0), arr4)
    _save_np_rgb_png(arr4_sharp, albedo_p)
    occ_pct = 100.0 * float((np.max(arr4_sharp, axis=-1) > 0.012).mean())
    print(f"  [UV Stats] Cobertura activa terreno en atlas {albedo_res}x{albedo_res} = {occ_pct:.1f}%")

    if not bake_pbr:
        return albedo_p, None, None

    # Esculpir fisicamente en 3D sobre la normal real leyendo el albedo nitido bilineal
    sculpt_3d_from_baked_albedo(obj, albedo_p)
    obj.data.polygons.foreach_set("use_smooth", is_ter_face.tolist())
    obj.data.update()

    # Conectar micro-grano mineral + albedo en Bump para NORMAL nítido y ROUGHNESS diferenciada
    map_n_res = normal_res if normal_res else max(2048, min(albedo_res, 4096))
    map_r_res = rough_res if rough_res else max(2048, min(albedo_res, 4096))
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

        tc_obj = nt.nodes.new("ShaderNodeTexCoord")
        noise_rock = nt.nodes.new("ShaderNodeTexNoise")
        noise_rock.inputs["Scale"].default_value = 3.8
        noise_rock.inputs["Detail"].default_value = 8.0
        noise_rock.inputs["Roughness"].default_value = 0.62
        nt.links.new(tc_obj.outputs["Object"], noise_rock.inputs["Vector"])

        mix_h = nt.nodes.new("ShaderNodeMix")
        mix_h.data_type = "FLOAT"
        mix_h.inputs["Factor"].default_value = 0.42
        nt.links.new(rgb2bw.outputs["Val"], mix_h.inputs["A"])
        nt.links.new(noise_rock.outputs["Fac"], mix_h.inputs["B"])

        bump_str = nt.nodes.new("ShaderNodeMapRange")
        bump_str.inputs["From Min"].default_value = 0.66
        bump_str.inputs["From Max"].default_value = 0.84
        bump_str.inputs["To Min"].default_value = 0.42
        bump_str.inputs["To Max"].default_value = 0.06
        nt.links.new(rgb2bw.outputs["Val"], bump_str.inputs["Value"])

        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Distance"].default_value = 0.14
        nt.links.new(bump_str.outputs["Result"], bump.inputs["Strength"])
        nt.links.new(mix_h.outputs["Result"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

        # Nieve satinada (~0.42), roca/grietas mate (~0.90)
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.inputs["From Min"].default_value = 0.65
        mr.inputs["From Max"].default_value = 0.85
        mr.inputs["To Min"].default_value = 0.90
        mr.inputs["To Max"].default_value = 0.42
        nt.links.new(rgb2bw.outputs["Val"], mr.inputs["Value"])
        nt.links.new(mr.outputs["Result"], bsdf.inputs["Roughness"])

    im_n = bpy.data.images.new("Bake_Normal", width=map_n_res, height=map_n_res)
    im_n.colorspace_settings.name = "Non-Color"
    if m_ter:
        m_ter.node_tree.nodes["__BAKE__"].image = im_n
    print(f"  [Bake] Cycles NORMAL ({map_n_res}x{map_n_res})...")
    bpy.ops.object.bake(type="NORMAL", normal_space="TANGENT")
    im_n.filepath_raw = str(normal_p)
    im_n.file_format = "PNG"
    im_n.save()

    im_r = bpy.data.images.new("Bake_Roughness", width=map_r_res, height=map_r_res)
    im_r.colorspace_settings.name = "Non-Color"
    if m_ter:
        m_ter.node_tree.nodes["__BAKE__"].image = im_r
    print(f"  [Bake] Cycles ROUGHNESS ({map_r_res}x{map_r_res})...")
    bpy.ops.object.bake(type="ROUGHNESS")
    im_r.filepath_raw = str(rough_p)
    im_r.file_format = "PNG"
    im_r.save()

    # Limpiar UV layers auxiliares de camara y asignar 2 materiales PBR dedicados (Zocalo + Terreno 4K)
    while len(obj.data.uv_layers) > 1:
        for ul in list(obj.data.uv_layers):
            if ul.name != "UVMap":
                obj.data.uv_layers.remove(ul)
                break
    for attr in list(obj.data.color_attributes):
        obj.data.color_attributes.remove(attr)

    mat_pl_final, nt_p, bsdf_p, _ = _new_material("Catan_Hero360_Plinth")
    tc_p = nt_p.nodes.new("ShaderNodeTexCoord")
    t_plf = nt_p.nodes.new("ShaderNodeTexImage")
    im_plf = bpy.data.images.load(str(plinth_albedo_p), check_existing=False); im_plf.pack()
    t_plf.image = im_plf
    nt_p.links.new(tc_p.outputs["UV"], t_plf.inputs["Vector"])
    nt_p.links.new(t_plf.outputs["Color"], bsdf_p.inputs["Base Color"])
    bsdf_p.inputs["Roughness"].default_value = 0.44
    bsdf_p.inputs["Metallic"].default_value = 0.18

    mat_ter_final, nt, bsdf, _ = _new_material("Catan_Hero360_Terrain")
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
    bsdf.inputs["Metallic"].default_value = 0.02

    obj.data.materials.clear()
    obj.data.materials.append(mat_pl_final)
    obj.data.materials.append(mat_ter_final)
    obj.data.polygons.foreach_set("material_index", is_ter_face.astype(np.int32).tolist())
    obj.data.polygons.foreach_set("use_smooth", is_ter_face.tolist())
    obj.data.update()
    return albedo_p, normal_p, rough_p


def sculpt_3d_from_highpass_bandlimited(
    obj: bpy.types.Object,
    hp_topdown: np.ndarray,
    albedo_path: Path,
    max_amp_mm: float = 0.12,
) -> None:
    """
    F6.3: Micro-desplazamiento físico desde HP filtrado a >= 0.25 mm de ancho (sigma ~4.5 px en 8K),
    amplitud <= 0.12 mm, solo en terreno (slot 1) con pendiente >= 35° y fuera de nieve/bordes.
    """
    hp_lum = (0.2126 * hp_topdown[..., 0] + 0.7152 * hp_topdown[..., 1] + 0.0722 * hp_topdown[..., 2]).astype(np.float32)
    # Filtrado separable 2D en NumPy puro (ancho >= 0.25 mm ~ 13 px en 8K)
    rad = max(2, int(round(4.5 * (hp_lum.shape[0] / 8192.0))))
    hp_filt = hp_lum.copy()
    k = 2 * rad + 1
    for _ in range(3):
        p_x = np.pad(hp_filt, ((0, 0), (rad, rad)), mode="edge")
        c_x = np.cumsum(np.pad(p_x, ((0, 0), (1, 0)), mode="constant"), axis=1, dtype=np.float64)
        hp_filt = ((c_x[:, k:] - c_x[:, :-k]) / k).astype(np.float32)
        p_y = np.pad(hp_filt, ((rad, rad), (0, 0)), mode="edge")
        c_y = np.cumsum(np.pad(p_y, ((1, 0), (0, 0)), mode="constant"), axis=0, dtype=np.float64)
        hp_filt = ((c_y[k:, :] - c_y[:-k, :]) / k).astype(np.float32)

    alb = _load_img_topdown_np(str(albedo_path))
    ah, aw = alb.shape[:2]

    me = obj.data
    n_verts = len(me.vertices)
    n_loops = len(me.loops)
    coords = np.empty(n_verts * 3, dtype=np.float64)
    normals = np.empty(n_verts * 3, dtype=np.float64)
    me.vertices.foreach_get("co", coords)
    me.vertices.foreach_get("normal", normals)
    V = coords.reshape(n_verts, 3)
    N = normals.reshape(n_verts, 3)

    poly_mat = np.empty(len(me.polygons), dtype=np.int32)
    me.polygons.foreach_get("material_index", poly_mat)
    loop_tot = np.empty(len(me.polygons), dtype=np.int32)
    me.polygons.foreach_get("loop_total", loop_tot)
    loop_vi = np.empty(n_loops, dtype=np.int32)
    me.loops.foreach_get("vertex_index", loop_vi)
    uv_flat = np.empty(n_loops * 2, dtype=np.float32)
    me.uv_layers["UVMap"].data.foreach_get("uv", uv_flat)
    uv_l = uv_flat.reshape(n_loops, 2)

    is_ter_loop = np.repeat(poly_mat == 1, loop_tot)
    is_talus_v = np.zeros(n_verts, dtype=bool)
    is_talus_v[loop_vi[np.repeat(poly_mat == 2, loop_tot)]] = True

    v_uv = np.zeros((n_verts, 2), dtype=np.float32)
    v_uv[loop_vi[is_ter_loop]] = uv_l[is_ter_loop]

    # Muestreo bilineal sobre HP filtrado y albedo
    hh, hw = hp_filt.shape
    fu = np.clip(v_uv[:, 0], 0.0, 1.0) * (hw - 1)
    fv = np.clip(1.0 - v_uv[:, 1], 0.0, 1.0) * (hh - 1)
    u0, v0 = np.floor(fu).astype(np.int32), np.floor(fv).astype(np.int32)
    u1, v1 = np.minimum(u0 + 1, hw - 1), np.minimum(v0 + 1, hh - 1)
    du, dv_u = (fu - u0), (fv - v0)
    hp_v = (
        hp_filt[v0, u0] * (1.0 - du) * (1.0 - dv_u)
        + hp_filt[v0, u1] * du * (1.0 - dv_u)
        + hp_filt[v1, u0] * (1.0 - du) * dv_u
        + hp_filt[v1, u1] * du * dv_u
    )

    fa_u = np.clip(v_uv[:, 0], 0.0, 1.0) * (aw - 1)
    fa_v = np.clip(1.0 - v_uv[:, 1], 0.0, 1.0) * (ah - 1)
    iu, iv = np.clip(np.round(fa_u), 0, aw - 1).astype(np.int32), np.clip(np.round(fa_v), 0, ah - 1).astype(np.int32)
    rgb = alb[iv, iu]
    lum = 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]
    sat = np.max(rgb, axis=1) - np.min(rgb, axis=1)

    de, dv = _hex_distances_arr(V[:, 0], V[:, 1])
    safe_b = _smoothstep(ROAD_MARGIN + 2.0, ROAD_MARGIN + 4.0, de) * _smoothstep(PAD_RADIUS + 2.0, PAD_RADIUS + 4.0, dv) * _smoothstep(Z_BASE + 0.4, Z_BASE + 1.5, V[:, 2])
    # Refuerzo completo en pendientes >= 35° y relieve suave (~45%) en cojines de musgo 3D en valles (< 35°)
    slope_w = (0.45 + 0.55 * (1.0 - _smoothstep(0.76, 0.84, N[:, 2]))) * _smoothstep(-0.20, 0.05, N[:, 2])
    snow_m = _smoothstep(0.68, 0.84, lum) * (1.0 - _smoothstep(0.10, 0.20, sat)) * _smoothstep(Z_BASE + 3.0, Z_BASE + 8.0, V[:, 2])
    m = safe_b * slope_w * (1.0 - 0.90 * snow_m) * (~is_talus_v).astype(np.float64)

    disp = np.clip(m * (hp_v / 0.14) * max_amp_mm, -max_amp_mm, max_amp_mm)
    V_new = V + N * disp[:, None]
    is_ter_v = (V[:, 2] >= Z_BASE + 0.04) & (de >= ROAD_MARGIN + 0.45) & (dv >= PAD_RADIUS + 0.45) & (~is_talus_v)
    V_new[is_ter_v, 2] = np.maximum(V_new[is_ter_v, 2], Z_BASE + 0.02)
    me.vertices.foreach_set("co", V_new.ravel())
    me.update()
    print(f"  [MicroHP Sculpt] Desplazamiento >=0.25mm aplicado (range=[{disp.min():.4f}, {disp.max():.4f}] mm)")


def _bake_normal_and_roughness_passB(
    scene: bpy.types.Scene,
    obj: bpy.types.Object,
    out_dir: Path,
    albedo_p: Path,
    plinth_albedo_p: Path,
    hp_topdown: np.ndarray,
    is_ter_face: np.ndarray,
    map_n_res: int = 8192,
    map_r_res: int = 4096,
    samples: int = 6,
) -> Tuple[Path, Path]:
    """Hornea NORMAL 8K (incluyendo relieve derivado de HP + grano mineral) y ROUGHNESS 4K, y asigna materiales finales."""
    normal_p = out_dir / "baked_normal.png"
    rough_p = out_dir / "baked_roughness.png"
    hp_height_p = out_dir / "macro_hp_height_8k.png"

    hp_l = 0.2126 * hp_topdown[..., 0] + 0.7152 * hp_topdown[..., 1] + 0.0722 * hp_topdown[..., 2]
    hp_h = np.clip(0.5 + hp_l / 0.36, 0.0, 1.0)
    _save_np_rgb_png(np.repeat(hp_h[..., None], 3, axis=-1), hp_height_p)

    m_pl, nt_p, _, _ = _new_material("Bake_Plinth_Tmp")
    im_d = bpy.data.images.new("Bake_Dummy_B", width=64, height=64)
    tp = nt_p.nodes.new("ShaderNodeTexImage"); tp.image = im_d; nt_p.nodes.active = tp

    m_ter, nt, bsdf, _ = _new_material("Bake_Terrain_Tmp")
    uv_m = nt.nodes.new("ShaderNodeUVMap"); uv_m.uv_map = "UVMap"
    t_alb = nt.nodes.new("ShaderNodeTexImage"); t_alb.image = bpy.data.images.load(str(albedo_p), check_existing=False)
    nt.links.new(uv_m.outputs["UV"], t_alb.inputs["Vector"])
    rgb2bw = nt.nodes.new("ShaderNodeRGBToBW"); nt.links.new(t_alb.outputs["Color"], rgb2bw.inputs["Color"])

    t_hp = nt.nodes.new("ShaderNodeTexImage")
    im_hph = bpy.data.images.load(str(hp_height_p), check_existing=False); im_hph.colorspace_settings.name = "Non-Color"
    t_hp.image = im_hph
    nt.links.new(uv_m.outputs["UV"], t_hp.inputs["Vector"])

    tc_obj = nt.nodes.new("ShaderNodeTexCoord")
    noise_rock = nt.nodes.new("ShaderNodeTexNoise")
    noise_rock.inputs["Scale"].default_value = 4.2
    noise_rock.inputs["Detail"].default_value = 9.0
    noise_rock.inputs["Roughness"].default_value = 0.64
    nt.links.new(tc_obj.outputs["Object"], noise_rock.inputs["Vector"])

    # Mezclar altura HP macro (70%) + micro-ruido mineral (30%)
    mix_h = nt.nodes.new("ShaderNodeMix"); mix_h.data_type = "FLOAT"
    mix_h.inputs["Factor"].default_value = 0.30
    nt.links.new(t_hp.outputs["Color"], mix_h.inputs["A"])
    nt.links.new(noise_rock.outputs["Fac"], mix_h.inputs["B"])

    bump_str = nt.nodes.new("ShaderNodeMapRange")
    bump_str.inputs["From Min"].default_value = 0.66; bump_str.inputs["From Max"].default_value = 0.84
    bump_str.inputs["To Min"].default_value = 0.48; bump_str.inputs["To Max"].default_value = 0.06
    nt.links.new(rgb2bw.outputs["Val"], bump_str.inputs["Value"])

    bump = nt.nodes.new("ShaderNodeBump"); bump.inputs["Distance"].default_value = 0.16
    nt.links.new(bump_str.outputs["Result"], bump.inputs["Strength"])
    nt.links.new(mix_h.outputs["Result"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = 0.65; mr.inputs["From Max"].default_value = 0.85
    mr.inputs["To Min"].default_value = 0.90; mr.inputs["To Max"].default_value = 0.42
    nt.links.new(rgb2bw.outputs["Val"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], bsdf.inputs["Roughness"])

    im_n = bpy.data.images.new("Bake_Normal_B", width=map_n_res, height=map_n_res); im_n.colorspace_settings.name = "Non-Color"
    im_r = bpy.data.images.new("Bake_Rough_B", width=map_r_res, height=map_r_res); im_r.colorspace_settings.name = "Non-Color"
    tb = nt.nodes.new("ShaderNodeTexImage"); tb.image = im_n; nt.nodes.active = tb

    poly_mat = np.empty(len(obj.data.polygons), dtype=np.int32)
    obj.data.polygons.foreach_get("material_index", poly_mat)
    obj.data.materials.clear()
    obj.data.materials.append(m_pl)
    obj.data.materials.append(m_ter)
    obj.data.materials.append(m_ter)
    obj.data.polygons.foreach_set("material_index", poly_mat.tolist())

    for ul in obj.data.uv_layers:
        ul.active = (ul.name == "UVMap")
        ul.active_render = (ul.name == "UVMap")

    scene.cycles.samples = samples
    scene.render.bake.margin = 6
    print(f"  [Bake PassB] Cycles NORMAL ({map_n_res}x{map_n_res})...")
    bpy.ops.object.bake(type="NORMAL", normal_space="TANGENT")
    im_n.filepath_raw = str(normal_p); im_n.file_format = "PNG"; im_n.save()

    tb.image = im_r
    print(f"  [Bake PassB] Cycles ROUGHNESS ({map_r_res}x{map_r_res})...")
    bpy.ops.object.bake(type="ROUGHNESS")
    im_r.filepath_raw = str(rough_p); im_r.file_format = "PNG"; im_r.save()

    # Limpiar UV/atributos auxiliares y dejar exactamente 2 materiales PBR limpios (Plinth + Terreno/Talus 8K)
    while len(obj.data.uv_layers) > 1:
        for ul in list(obj.data.uv_layers):
            if ul.name != "UVMap":
                obj.data.uv_layers.remove(ul); break
    for attr in list(obj.data.color_attributes):
        obj.data.color_attributes.remove(attr)

    mat_pl_f, nt_pf, bsdf_pf, _ = _new_material("Catan_Hero360_Plinth")
    tc_pf = nt_pf.nodes.new("ShaderNodeTexCoord"); t_plf = nt_pf.nodes.new("ShaderNodeTexImage")
    im_plf = bpy.data.images.load(str(plinth_albedo_p), check_existing=False); im_plf.pack()
    t_plf.image = im_plf; nt_pf.links.new(tc_pf.outputs["UV"], t_plf.inputs["Vector"]); nt_pf.links.new(t_plf.outputs["Color"], bsdf_pf.inputs["Base Color"])
    bsdf_pf.inputs["Roughness"].default_value = 0.44; bsdf_pf.inputs["Metallic"].default_value = 0.18

    mat_ter_f, ntf, bsdff, _ = _new_material("Catan_Hero360_Terrain")
    tcf = ntf.nodes.new("ShaderNodeTexCoord")
    ta = ntf.nodes.new("ShaderNodeTexImage"); ia = bpy.data.images.load(str(albedo_p), check_existing=False); ia.pack(); ta.image = ia
    ntf.links.new(tcf.outputs["UV"], ta.inputs["Vector"]); ntf.links.new(ta.outputs["Color"], bsdff.inputs["Base Color"])
    tn = ntf.nodes.new("ShaderNodeTexImage"); inm = bpy.data.images.load(str(normal_p), check_existing=False); inm.colorspace_settings.name = "Non-Color"; inm.pack(); tn.image = inm
    ntf.links.new(tcf.outputs["UV"], tn.inputs["Vector"]); nmap = ntf.nodes.new("ShaderNodeNormalMap"); ntf.links.new(tn.outputs["Color"], nmap.inputs["Color"]); ntf.links.new(nmap.outputs["Normal"], bsdff.inputs["Normal"])
    tr = ntf.nodes.new("ShaderNodeTexImage"); ir = bpy.data.images.load(str(rough_p), check_existing=False); ir.colorspace_settings.name = "Non-Color"; ir.pack(); tr.image = ir
    ntf.links.new(tcf.outputs["UV"], tr.inputs["Vector"]); ntf.links.new(tr.outputs["Color"], bsdff.inputs["Roughness"])
    bsdff.inputs["Metallic"].default_value = 0.02

    obj.data.materials.clear()
    obj.data.materials.append(mat_pl_f)
    obj.data.materials.append(mat_ter_f)
    obj.data.polygons.foreach_set("material_index", is_ter_face.astype(np.int32).tolist())
    obj.data.polygons.foreach_set("use_smooth", is_ter_face.tolist())
    obj.data.update()
    return normal_p, rough_p


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
        orig_idx = np.empty(len(obj.data.polygons), dtype=np.int32)
        obj.data.polygons.foreach_get("material_index", orig_idx)
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
            obj.data.polygons.foreach_set("material_index", orig_idx.tolist())
            obj.data.update()
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


def render_clay_f2_preview(scene: bpy.types.Scene, obj: bpy.types.Object, cams: Dict[str, dict], out_dir: Path, samples: int = 16) -> None:
    """Render rápido de 2 vistas en arcilla (1024 px, 0° detalle + 180°) para Puerta F2."""
    world = bpy.data.worlds.new("ClayPreviewWorld")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.86, 0.87, 0.89, 1.0)
        bg.inputs["Strength"].default_value = 0.34
    for name, loc, energy, col, ang in [
        ("KeySun", (95.0, -70.0, 125.0), 2.75, (1.0, 0.98, 0.96), math.radians(12.0)),
        ("FillSun", (-85.0, -40.0, 85.0), 0.72, (0.90, 0.95, 1.0), math.radians(28.0)),
        ("RimSun", (-10.0, 95.0, 75.0), 1.35, (0.98, 0.99, 1.0), math.radians(18.0)),
    ]:
        ld = bpy.data.lights.new(name=name, type="SUN")
        ld.energy, ld.color, ld.angle = energy, col, ang
        lo = bpy.data.objects.new(name, ld)
        lo.location = loc
        lo.rotation_euler = (mathutils.Vector((0.0, 0.0, 29.0)) - mathutils.Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        scene.collection.objects.link(lo)

    mat_clay = bpy.data.materials.new("Clay_F2")
    mat_clay.use_nodes = True
    bsdf_c = mat_clay.node_tree.nodes.get("Principled BSDF")
    if bsdf_c:
        bsdf_c.inputs["Base Color"].default_value = (0.50, 0.51, 0.53, 1.0)
        bsdf_c.inputs["Roughness"].default_value = 0.44
    obj.data.materials.clear()
    obj.data.materials.append(mat_clay)

    cam_d = bpy.data.cameras.new("F2ClayCam")
    cam_d.clip_end = 2500.0
    cam_o = bpy.data.objects.new("F2ClayCam", cam_d)
    scene.collection.objects.link(cam_o)
    scene.camera = cam_o
    scene.cycles.samples = samples
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024

    for tag, cd, zoom_fov in [
        ("f2_clay_000_zoom", cams["0"], 7.5),
        ("f2_clay_180", cams["180"], cams["180"]["fov_deg"]),
    ]:
        cam_pos = mathutils.Vector(cd["cam_xyz"])
        right = mathutils.Vector(cd["right_r"]).normalized()
        up = mathutils.Vector(cd["up_r"]).normalized()
        fwd = mathutils.Vector(cd["fwd"]).normalized()
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
        cam_d.lens = 12.0 / math.tan(math.radians(zoom_fov * 0.5))
        scene.render.filepath = str(out_dir / f"{tag}.png")
        bpy.ops.render.render(write_still=True)
    print(f"[ClayF2 Preview Done] {out_dir / 'f2_clay_000_zoom.png'}, {out_dir / 'f2_clay_180.png'}")


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(description="Fábrica Modular Maker3D — Cycles Smart UV 3D Camera-Projection Bake")
    p.add_argument("--stl", required=False, default="")
    p.add_argument("--cameras", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--bake-res", type=int, default=4096)
    p.add_argument("--normal-res", type=int, default=None)
    p.add_argument("--rough-res", type=int, default=None)
    p.add_argument("--bake-samples", type=int, default=8)
    p.add_argument("--render-samples", type=int, default=16)
    p.add_argument("--max-edge-mm", type=float, default=0.21)
    p.add_argument("--target-max-faces", type=int, default=2900000)
    p.add_argument("--subdiv-passes", type=int, default=3)
    p.add_argument("--micro-strata", action="store_true")
    p.add_argument("--talus-manifest", default=None)
    p.add_argument("--macro-dir", default=None)
    p.add_argument("--stage", choices=["full", "clay-preview", "export-geology", "passA", "render-macros", "passB"], default="full")
    p.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    p.add_argument("--use-v4-front", default=None)
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.cameras, "r", encoding="utf-8") as f:
        cams = json.load(f)

    if args.stage == "render-macros":
        macro_dir = Path(args.macro_dir).resolve() if args.macro_dir else (out_dir / "macro")
        mbh.render_macro_passes(out_dir / "passA_scene.blend", macro_dir, samples=args.render_samples, res=2048, device=args.device)
        return

    if args.stage == "passB":
        macro_dir = Path(args.macro_dir).resolve() if args.macro_dir else (out_dir / "macro")
        bpy.ops.wm.open_mainfile(filepath=str(out_dir / "passA_scene.blend"))
        mbh.relink_blend_images([out_dir, macro_dir, macro_dir / "renders", macro_dir / "aligned_hp", out_dir.parent / "views_2k"])
        scene = bpy.context.scene
        mbh.configure_cycles_device(scene, args.device)
        obj = bpy.data.objects.get("Catan_Hero360") or bpy.context.selected_objects[0]
        bpy.ops.object.select_all(action="DESELECT"); obj.select_set(True); bpy.context.view_layer.objects.active = obj

        # Mantener mask de terreno+talus (slot >= 1) y proteger slot 2 si existe
        poly_mat = np.empty(len(obj.data.polygons), dtype=np.int32)
        obj.data.polygons.foreach_get("material_index", poly_mat)
        is_ter_face = poly_mat >= 1

        hp_top, ao_top = mbh.project_highpass_and_ao(scene, obj, macro_dir, atlas_res=args.bake_res, samples=max(2, args.bake_samples // 2))
        passA_alb = _load_img_topdown_np(str(out_dir / "passA_albedo_8k.png"))
        # F6.1: albedo_final = passA * exp(HP) * lerp(1, AO_cavidad, 0.30)
        ao_mul = 1.0 - 0.30 * (1.0 - np.clip(ao_top, 0.0, 1.0))
        albedo_final = np.clip(passA_alb * np.exp(hp_top) * ao_mul[..., None], 0.0, 1.0)
        final_alb_p = out_dir / "baked_albedo.png"
        _save_np_rgb_png(albedo_final, final_alb_p)

        # Esculpido físico principal desde albedo + micro-desplazamiento desde HP filtrado >= 0.25 mm
        sculpt_3d_from_baked_albedo(obj, final_alb_p)
        sculpt_3d_from_highpass_bandlimited(obj, hp_top, final_alb_p, max_amp_mm=0.12)
        obj.data.polygons.foreach_set("use_smooth", is_ter_face.tolist())
        obj.data.update()

        plinth_p = out_dir / "plinth_albedo_1k.png"
        if not plinth_p.exists():
            plinth_p = build_plinth_texture(out_dir)

        map_n_res = args.normal_res or args.bake_res
        map_r_res = args.rough_res or max(2048, min(args.bake_res, 4096))
        _bake_normal_and_roughness_passB(
            scene, obj, out_dir, final_alb_p, plinth_p, hp_top, is_ter_face,
            map_n_res=map_n_res, map_r_res=map_r_res, samples=args.bake_samples,
        )

        master_stl = out_dir / "tile_master.stl"
        bpy.ops.object.select_all(action="DESELECT"); obj.select_set(True); bpy.context.view_layer.objects.active = obj
        bpy.ops.wm.stl_export(filepath=str(master_stl), export_selected_objects=True, ascii_format=False)
        glb_path = out_dir / "tile_viewer.glb"
        bpy.ops.export_scene.gltf(filepath=str(glb_path), export_format="GLB", use_selection=True, export_apply=True)
        render_6views_audit(scene, obj, cams, out_dir, samples=args.render_samples)
        print(f"[PassB Done] STL={master_stl}, GLB={glb_path}")
        return

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    mbh.configure_cycles_device(scene, args.device)
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

    if args.subdiv_passes > 0:
        subdivide_steep_terrain_edges(
            obj,
            max_edge_mm=args.max_edge_mm,
            max_passes=args.subdiv_passes,
            target_max_faces=args.target_max_faces,
        )

    is_ter_face = assign_plinth_vs_terrain_slots(obj)
    if args.talus_manifest:
        tm_data = json.loads(Path(args.talus_manifest).read_text(encoding="utf-8"))
        rf_npy = Path(tm_data["rock_faces_npy"])
        rc_npy = rf_npy.with_suffix("").with_suffix(".rock_centers.npy")
        is_rock = None
        if rf_npy.exists():
            arr_r = np.load(str(rf_npy))
            if len(arr_r) == len(obj.data.polygons):
                is_rock = arr_r
        if is_rock is None and rc_npy.exists():
            rock_ctrs = np.load(str(rc_npy))
            kd = mathutils.kdtree.KDTree(len(rock_ctrs))
            for i_rc, pt_rc in enumerate(rock_ctrs):
                kd.insert(pt_rc, i_rc)
            kd.balance()
            n_p = len(obj.data.polygons)
            ctrs_p = np.empty(n_p * 3, dtype=np.float64)
            obj.data.polygons.foreach_get("center", ctrs_p)
            C_p = ctrs_p.reshape(n_p, 3)
            is_rock = np.zeros(n_p, dtype=bool)
            cand_val = np.where(is_ter_face & (C_p[:, 2] <= Z_BASE + 19.0))[0]
            for fi_v in cand_val:
                _, _, dist_rc = kd.find(C_p[fi_v])
                if dist_rc <= 0.025:
                    is_rock[fi_v] = True
        if is_rock is not None:
            poly_idx = np.where(is_rock, 2, np.where(is_ter_face, 1, 0)).astype(np.int32)
            obj.data.polygons.foreach_set("material_index", poly_idx.tolist())
            obj.data.update()
            print(f"  [Talus Slot2] Asignadas {int(is_rock.sum())} caras de roca al slot 2")

    if args.micro_strata:
        apply_geologic_microstrata(obj, is_ter_face)
        is_ter_face = assign_plinth_vs_terrain_slots(obj)

    obj.data.polygons.foreach_set("use_smooth", is_ter_face.tolist())
    obj.data.update()

    if args.stage == "export-geology":
        geo_stl = out_dir / "tile_geology.stl"
        bpy.ops.object.select_all(action="DESELECT"); obj.select_set(True); bpy.context.view_layer.objects.active = obj
        bpy.ops.wm.stl_export(filepath=str(geo_stl), export_selected_objects=True, ascii_format=False)
        print(f"[ExportGeology Done] {geo_stl} ({len(obj.data.polygons)} caras)")
        return

    if args.stage == "clay-preview":
        render_clay_f2_preview(scene, obj, cams, out_dir, samples=args.render_samples)
        return

    # Guardar índices de material antes de regenerar slots (conserva slot 2 de rocas si existe)
    orig_poly_mat = np.empty(len(obj.data.polygons), dtype=np.int32)
    obj.data.polygons.foreach_get("material_index", orig_poly_mat)

    m_pl, plinth_albedo_p = build_plinth_material(out_dir)
    m_ter = setup_camera_projections_and_shader(obj, cams)
    obj.data.materials.clear()
    obj.data.materials.append(m_pl)
    obj.data.materials.append(m_ter)
    if args.talus_manifest and np.any(orig_poly_mat == 2):
        m_tal = build_talus_material_from_cameras(cams, args.talus_manifest)
        obj.data.materials.append(m_tal)
        obj.data.polygons.foreach_set("material_index", orig_poly_mat.tolist())
        is_ter_face = (orig_poly_mat >= 1)
    else:
        is_ter_face = assign_plinth_vs_terrain_slots(obj)
    obj.data.polygons.foreach_set("use_smooth", is_ter_face.tolist())
    obj.data.update()

    passA_alb_p = out_dir / "passA_albedo_8k.png" if args.stage == "passA" else None
    bake_pbr_maps(
        scene,
        obj,
        out_dir,
        is_ter_face=is_ter_face,
        plinth_albedo_p=plinth_albedo_p,
        albedo_res=args.bake_res,
        normal_res=args.normal_res,
        rough_res=args.rough_res,
        samples=args.bake_samples,
        save_unsharp_to=passA_alb_p,
        bake_pbr=(args.stage != "passA"),
    )

    if args.stage == "passA":
        mat_pl_pa, _, _, _ = _new_material("Catan_Hero360_Plinth_PassA")
        mat_ter_pa, nt_pa, bsdf_pa, _ = _new_material("Catan_Hero360_Terrain_PassA")
        tc_pa = nt_pa.nodes.new("ShaderNodeTexCoord")
        ta_pa = nt_pa.nodes.new("ShaderNodeTexImage")
        ia_pa = bpy.data.images.load(str(passA_alb_p), check_existing=False)
        ta_pa.image = ia_pa
        nt_pa.links.new(tc_pa.outputs["UV"], ta_pa.inputs["Vector"])
        nt_pa.links.new(ta_pa.outputs["Color"], bsdf_pa.inputs["Base Color"])
        bsdf_pa.inputs["Roughness"].default_value = 0.86
        cur_idx = np.empty(len(obj.data.polygons), dtype=np.int32)
        obj.data.polygons.foreach_get("material_index", cur_idx)
        obj.data.materials.clear()
        obj.data.materials.append(mat_pl_pa)
        obj.data.materials.append(mat_ter_pa)
        if np.any(cur_idx == 2):
            obj.data.materials.append(mat_ter_pa)
        obj.data.polygons.foreach_set("material_index", cur_idx.tolist())
        obj.data.polygons.foreach_set("use_smooth", is_ter_face.tolist())
        obj.data.update()
        blend_p = out_dir / "passA_scene.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_p))
        print(f"[PassA Done] Albedo={passA_alb_p}, Blend={blend_p}")
        return

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
