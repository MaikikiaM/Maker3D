"""
Helpers de Blender para Parches Macro 'Posición Humana' (F5 + F6) en Montaña Hero 360° v05:
- Generación de 16 cámaras macro + verificación de cobertura >= 85% por BVH.
- Renderizado de pases PBR, Arcilla y Máscara exacta (2048x2048).
- Proyección High-Pass 16-bit acumulada en EXR 8K (Σ(w·HP) / Σw) y AO de cavidad 8K.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import bmesh
import bpy
import mathutils
from mathutils.bvhtree import BVHTree
import numpy as np

R = 64.0
Z_BASE = 25.0
ROAD_MARGIN = 3.50
PAD_RADIUS = 10.00


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / max(e1 - e0, 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _camera_basis_looking_at(cam_xyz: np.ndarray, target_xyz: np.ndarray, fov_deg: float, res: int = 2048) -> Dict[str, Any]:
    fwd = target_xyz - cam_xyz
    fwd /= max(np.linalg.norm(fwd), 1e-8)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(fwd, world_up)) > 0.98:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(fwd, world_up)
    right /= max(np.linalg.norm(right), 1e-8)
    up = np.cross(right, fwd)
    up /= max(np.linalg.norm(up), 1e-8)
    return {
        "cam_xyz": cam_xyz.tolist(),
        "target_xyz": target_xyz.tolist(),
        "fwd": fwd.tolist(),
        "right_r": right.tolist(),
        "up_r": up.tolist(),
        "fov_deg": float(fov_deg),
        "W": int(res),
        "H": int(res),
    }


def build_16_macro_cameras(obj: bpy.types.Object, res: int = 2048) -> Dict[str, dict]:
    """
    Construye 16 cámaras cercanas (~40-45 mm campo -> ~48 px/mm en 2048 px):
    - Anillo A: 6 cámaras a elevación 22° (az 0..300)
    - Anillo B: 6 cámaras a elevación 48° (az 30..330)
    - 3 cámaras altas (70°) sobre pico principal y ambos circos
    - 1 cámara rasante de valle sur
    Si la cobertura BVH con ndotv >= 0.35 baja de 85%, agrega cámaras de refuerzo.
    """
    cams: List[Tuple[str, Dict[str, Any]]] = []

    def _cam_pos(az_deg: float, el_deg: float, dist: float, tgt: np.ndarray) -> np.ndarray:
        az = math.radians(az_deg)
        el = math.radians(el_deg)
        offset = np.array([
            -dist * math.cos(el) * math.sin(az),
            -dist * math.cos(el) * math.cos(az),
            dist * math.sin(el),
        ], dtype=np.float64)
        return tgt + offset

    # Anillo A: 6 cámaras bajas (22°) mirando a media ladera
    for idx, az in enumerate(range(0, 360, 60)):
        a = math.radians(az)
        tgt = np.array([-18.0 * math.sin(a), -18.0 * math.cos(a), Z_BASE + 15.0], dtype=np.float64)
        cxyz = _cam_pos(float(az), 24.0, 84.0, tgt)
        cams.append((f"A{idx}_{az:03d}", _camera_basis_looking_at(cxyz, tgt, fov_deg=35.0, res=res)))

    # Anillo B: 6 cámaras medias (48°) cubriendo crestas y contrafuertes
    for idx, az in enumerate(range(30, 390, 60)):
        a = math.radians(az)
        tgt = np.array([-15.0 * math.sin(a), -15.0 * math.cos(a), Z_BASE + 18.0], dtype=np.float64)
        cxyz = _cam_pos(float(az), 48.0, 82.0, tgt)
        cams.append((f"B{idx}_{az:03d}", _camera_basis_looking_at(cxyz, tgt, fov_deg=36.0, res=res)))

    # 3 cámaras altas (70°) sobre pico y ambos circos glaciares
    high_specs = [
        ("C0_peak", 15.0, 70.0, 80.0, np.array([0.0, 2.0, Z_BASE + 28.0]), 36.0),
        ("C1_cirqueW", 115.0, 70.0, 80.0, np.array([-20.0, 5.0, Z_BASE + 15.0]), 36.0),
        ("C2_cirqueE", 245.0, 70.0, 80.0, np.array([20.0, 5.0, Z_BASE + 15.0]), 36.0),
    ]
    for cid, az, el, dist, tgt, fov in high_specs:
        cams.append((cid, _camera_basis_looking_at(_cam_pos(az, el, dist, tgt), tgt, fov_deg=fov, res=res)))

    # 1 cámara rasante específica para fondo del valle sur y derrubio
    tgt_v = np.array([0.0, -24.0, Z_BASE + 4.0], dtype=np.float64)
    cams.append(("D0_valleyS", _camera_basis_looking_at(_cam_pos(0.0, 32.0, 78.0, tgt_v), tgt_v, fov_deg=36.0, res=res)))

    cam_dict = {cid: cdata for cid, cdata in cams}
    cov_pct = compute_macro_coverage(obj, cam_dict)
    print(f"  [MacroCams] Cobertura BVH (ndotv >= 0.35) con {len(cam_dict)} cámaras = {cov_pct:.1f}%")

    # Añadir cámaras de refuerzo si aún no alcanza el 85%
    extra_specs = [
        ("E0_ridgeN", 180.0, 58.0, 84.0, np.array([0.0, 22.0, Z_BASE + 14.0]), 38.0),
        ("E1_valleyE", 300.0, 44.0, 82.0, np.array([26.0, -14.0, Z_BASE + 6.0]), 38.0),
        ("E2_valleyW", 60.0, 44.0, 82.0, np.array([-26.0, -14.0, Z_BASE + 6.0]), 38.0),
        ("E3_valleyNE", 210.0, 44.0, 82.0, np.array([24.0, 20.0, Z_BASE + 6.0]), 38.0),
        ("E4_valleyNW", 150.0, 44.0, 82.0, np.array([-24.0, 20.0, Z_BASE + 6.0]), 38.0),
        ("E5_valleyS2", 0.0, 62.0, 84.0, np.array([0.0, -28.0, Z_BASE + 4.0]), 38.0),
    ]
    for cid, az, el, dist, tgt, fov in extra_specs:
        if cov_pct >= 85.0:
            break
        cam_dict[cid] = _camera_basis_looking_at(_cam_pos(az, el, dist, tgt), tgt, fov_deg=fov, res=res)
        cov_pct = compute_macro_coverage(obj, cam_dict)
        print(f"  [MacroCams+] +{cid} -> cobertura = {cov_pct:.1f}%")

    return cam_dict


def compute_macro_coverage(obj: bpy.types.Object, cams: Dict[str, dict], ndotv_thr: float = 0.35) -> float:
    """Calcula el % de área del terreno visible con ndotv >= ndotv_thr desde al menos una cámara macro."""
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    bvh = BVHTree.FromBMesh(bm)

    n_polys = len(me.polygons)
    centers = np.empty(n_polys * 3, dtype=np.float64)
    normals = np.empty(n_polys * 3, dtype=np.float64)
    areas = np.empty(n_polys, dtype=np.float64)
    mats = np.empty(n_polys, dtype=np.int32)
    me.polygons.foreach_get("center", centers)
    me.polygons.foreach_get("normal", normals)
    me.polygons.foreach_get("area", areas)
    me.polygons.foreach_get("material_index", mats)
    C = centers.reshape(n_polys, 3)
    N = normals.reshape(n_polys, 3)
    bm.free()

    ter_idx = np.where((mats >= 1) & (C[:, 2] >= Z_BASE + 0.10))[0]
    if len(ter_idx) == 0:
        return 0.0
    # Submuestreo uniforme ponderado o paso determinista hasta 40k caras para test BVH en < 1.5s
    step = max(1, len(ter_idx) // 35000)
    samp_idx = ter_idx[::step]
    Cs = C[samp_idx]
    Ns = N[samp_idx]
    As = areas[samp_idx]

    covered = np.zeros(len(samp_idx), dtype=bool)
    vec_M = mathutils.Vector
    for cd in cams.values():
        cam_pos = np.array(cd["cam_xyz"], dtype=np.float64)
        fwd = np.array(cd["fwd"], dtype=np.float64)
        right_r = np.array(cd["right_r"], dtype=np.float64)
        up_r = np.array(cd["up_r"], dtype=np.float64)
        W_img, H_img = float(cd["W"]), float(cd["H"])
        f_px = (H_img * 0.5) / math.tan(math.radians(cd["fov_deg"] * 0.5))

        rel = Cs - cam_pos[None, :]
        xc = rel @ right_r
        yc = rel @ up_r
        zc = rel @ fwd
        in_front = zc > 5.0
        u_px = W_img * 0.5 + f_px * (xc / np.maximum(zc, 1e-4))
        v_pil = H_img * 0.5 - f_px * (yc / np.maximum(zc, 1e-4))
        in_frame = in_front & (u_px > 16.0) & (u_px < W_img - 16.0) & (v_pil > 16.0) & (v_pil < H_img - 16.0)

        d2c = cam_pos[None, :] - Cs
        dist = np.linalg.norm(d2c, axis=1)
        d_n = d2c / np.maximum(dist[:, None], 1e-6)
        ndotv = np.sum(Ns * d_n, axis=1)

        cand = np.where((~covered) & in_frame & (ndotv >= ndotv_thr))[0]
        for ci in cand:
            p = Cs[ci] + Ns[ci] * 0.14
            hit_loc, _, _, _ = bvh.ray_cast(vec_M(p), vec_M(d_n[ci]), float(dist[ci]) - 0.5)
            if hit_loc is None:
                covered[ci] = True

    return float(np.sum(As[covered]) / max(np.sum(As), 1e-6) * 100.0)


def _setup_studio_world_and_lights(scene: bpy.types.Scene) -> Tuple[Any, Any, Any]:
    world = bpy.data.worlds.new("MacroRenderWorld")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.86, 0.87, 0.89, 1.0)
        bg.inputs["Strength"].default_value = 0.82
    fill_l = key_l = None
    for name, loc, energy, col, ang in [
        ("MacroKeySun", (95.0, -70.0, 125.0), 2.3, (1.0, 0.98, 0.96), math.radians(14.0)),
        ("MacroFillSun", (-85.0, -40.0, 85.0), 1.35, (0.90, 0.95, 1.0), math.radians(28.0)),
        ("MacroRimSun", (-10.0, 95.0, 75.0), 1.25, (0.98, 0.99, 1.0), math.radians(20.0)),
    ]:
        ld = bpy.data.lights.new(name=name, type="SUN")
        ld.energy, ld.color, ld.angle = energy, col, ang
        lo = bpy.data.objects.new(name, ld)
        lo.location = loc
        lo.rotation_euler = (mathutils.Vector((0.0, 0.0, 29.0)) - mathutils.Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        scene.collection.objects.link(lo)
        if name == "MacroKeySun":
            key_l = ld
        elif name == "MacroFillSun":
            fill_l = ld
    return bg, key_l, fill_l


def render_macro_passes(
    blend_path: Path,
    macro_dir: Path,
    samples: int = 16,
    res: int = 2048,
) -> Path:
    """Carga passA_scene.blend, construye las 16 cámaras macro y renderiza PBR, Arcilla y Máscara 2048 px."""
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"

    obj = bpy.data.objects.get("Catan_Hero360") or bpy.context.selected_objects[0]
    macro_dir.mkdir(parents=True, exist_ok=True)
    renders_dir = macro_dir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)

    cams = build_16_macro_cameras(obj, res=res)
    bg, key_l, fill_l = _setup_studio_world_and_lights(scene)

    cam_d = bpy.data.cameras.new("MacroCam")
    cam_d.clip_start = 1.0
    cam_d.clip_end = 2500.0
    cam_o = bpy.data.objects.new("MacroCam", cam_d)
    scene.collection.objects.link(cam_o)
    scene.camera = cam_o

    def _apply_cam(cd: dict) -> None:
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
        cam_d.lens = 12.0 / math.tan(math.radians(cd["fov_deg"] * 0.5))

    orig_mats = list(obj.data.materials)
    orig_idx = np.empty(len(obj.data.polygons), dtype=np.int32)
    obj.data.polygons.foreach_get("material_index", orig_idx)

    # Material Arcilla
    mat_clay = bpy.data.materials.new("MacroClay")
    mat_clay.use_nodes = True
    bsdf_c = mat_clay.node_tree.nodes.get("Principled BSDF")
    if bsdf_c:
        bsdf_c.inputs["Base Color"].default_value = (0.50, 0.51, 0.53, 1.0)
        bsdf_c.inputs["Roughness"].default_value = 0.45

    # Materiales Emisión para Máscara exacta de terreno+talus vs fondo/zócalo
    def _emit_mat(name: str, col: Tuple[float, float, float, float]):
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        nt = m.node_tree
        nt.nodes.clear()
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = col
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
        return m

    m_blk = _emit_mat("MaskZero", (0.0, 0.0, 0.0, 1.0))
    m_wht = _emit_mat("MaskOne", (1.0, 1.0, 1.0, 1.0))

    scene.render.resolution_x = res
    scene.render.resolution_y = res

    # 1. Pases PBR provisionales
    print(f"  [MacroRender] Renderizando {len(cams)} pases PBR ({res}x{res})...")
    for cid, cd in cams.items():
        _apply_cam(cd)
        p_out = renders_dir / f"patch_{cid}_pbr.png"
        scene.render.filepath = str(p_out)
        bpy.ops.render.render(write_still=True)
        cd["pbr_render"] = str(p_out)

    # 2. Pases Arcilla
    obj.data.materials.clear()
    obj.data.materials.append(mat_clay)
    if bg:
        bg.inputs["Strength"].default_value = 0.34
    if fill_l:
        fill_l.energy = 0.72
    if key_l:
        key_l.energy = 2.65
    print(f"  [MacroRender] Renderizando {len(cams)} pases Arcilla...")
    for cid, cd in cams.items():
        _apply_cam(cd)
        c_out = renders_dir / f"patch_{cid}_clay.png"
        scene.render.filepath = str(c_out)
        bpy.ops.render.render(write_still=True)
        cd["clay_render"] = str(c_out)

    # 3. Pases Máscara binaria Cycles 1 sample (zocalo=0, terreno+talus=1, fondo=0)
    scene.cycles.samples = 1
    if bg:
        bg.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
        bg.inputs["Strength"].default_value = 0.0
    obj.data.materials.clear()
    obj.data.materials.append(m_blk)
    obj.data.materials.append(m_wht)
    obj.data.materials.append(m_wht)
    obj.data.polygons.foreach_set("material_index", np.where(orig_idx >= 1, 1, 0).tolist())
    obj.data.update()
    print(f"  [MacroRender] Renderizando {len(cams)} máscaras exactas...")
    for cid, cd in cams.items():
        _apply_cam(cd)
        m_out = renders_dir / f"patch_{cid}_mask.png"
        scene.render.filepath = str(m_out)
        bpy.ops.render.render(write_still=True)
        cd["mask"] = str(m_out)

    cam_json = macro_dir / "macro_cameras.json"
    cam_json.write_text(json.dumps(cams, indent=2), encoding="utf-8")
    print(f"[MacroRender Done] Guardado {cam_json}")
    return cam_json


def _load_img_topdown_np(path: str, non_color: bool = True) -> np.ndarray:
    im = bpy.data.images.load(str(path), check_existing=False)
    im.colorspace_settings.name = "Non-Color" if non_color else "sRGB"
    w, h = im.size[0], im.size[1]
    ch = im.channels
    buf = np.empty(w * h * ch, dtype=np.float32)
    im.pixels.foreach_get(buf)
    bpy.data.images.remove(im)
    return np.flipud(buf.reshape((h, w, ch))[..., :3]).copy()


def _save_np_rgb_png_bottomup_from_topdown(arr_topdown: np.ndarray, path: Path, non_color: bool = False) -> None:
    h, w, _ = arr_topdown.shape
    flipped = np.flipud(np.clip(arr_topdown, 0.0, 1.0))
    rgba = np.ones((h, w, 4), dtype=np.float32)
    rgba[..., :3] = flipped
    im = bpy.data.images.new(path.stem, width=w, height=h, alpha=True)
    im.colorspace_settings.name = "Non-Color" if non_color else "sRGB"
    im.pixels.foreach_set(rgba.ravel())
    im.filepath_raw = str(path)
    im.file_format = "PNG"
    im.save()
    bpy.data.images.remove(im)


def project_highpass_and_ao(
    scene: bpy.types.Scene,
    obj: bpy.types.Object,
    macro_dir: Path,
    atlas_res: int = 8192,
    samples: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Proyecta todos los parches High-Pass aceptados en macro_report.json sobre UVMap 8K
    acumulando Σ(w·HP) y Σw en float32 EXR, y hornea AO de cavidad 8K.
    Devuelve (HP_atlas_topdown float32 [H,W,3], AO_topdown float32 [H,W]).
    """
    cams = json.loads((macro_dir / "macro_cameras.json").read_text(encoding="utf-8"))
    report = json.loads((macro_dir / "macro_report.json").read_text(encoding="utf-8"))
    accepted_ids = [cid for cid in sorted(cams.keys()) if report["patches"].get(cid, {}).get("accepted")]
    if not accepted_ids:
        print("  [MacroHP] WARN: 0 parches aceptados; usando HP=0.")
        zero_hp = np.zeros((atlas_res, atlas_res, 3), dtype=np.float32)
        ao = bake_cavity_ao(scene, obj, macro_dir / "ao_cavity_8k.png", res=atlas_res, samples=6)
        return zero_hp, ao

    print(f"  [MacroHP] Proyectando {len(accepted_ids)} parches High-Pass al atlas {atlas_res}x{atlas_res}...")
    me = obj.data
    n_verts = len(me.vertices)
    coords = np.empty(n_verts * 3, dtype=np.float64)
    normals = np.empty(n_verts * 3, dtype=np.float64)
    me.vertices.foreach_get("co", coords)
    me.vertices.foreach_get("normal", normals)
    V = coords.reshape(n_verts, 3)
    N = normals.reshape(n_verts, 3)

    # Identificar vértices de terreno+talus (z >= Z_BASE - 0.02 dentro del borde seguro)
    n_polys = len(me.polygons)
    poly_mat = np.empty(n_polys, dtype=np.int32)
    me.polygons.foreach_get("material_index", poly_mat)
    loop_vi = np.empty(len(me.loops), dtype=np.int32)
    me.loops.foreach_get("vertex_index", loop_vi)
    loop_tot = np.empty(n_polys, dtype=np.int32)
    me.polygons.foreach_get("loop_total", loop_tot)
    is_ter_v = np.zeros(n_verts, dtype=bool)
    is_ter_loop = np.repeat(poly_mat >= 1, loop_tot)
    is_ter_v[loop_vi[is_ter_loop]] = True

    bm = bmesh.new()
    bm.from_mesh(me)
    bvh = BVHTree.FromBMesh(bm)
    bm.free()

    hp_accum_num = np.zeros((atlas_res, atlas_res, 3), dtype=np.float32)
    hp_accum_den = np.zeros((atlas_res, atlas_res), dtype=np.float32)

    orig_mats = list(obj.data.materials)
    scene.cycles.samples = samples
    scene.render.bake.use_pass_direct = False
    scene.render.bake.use_pass_indirect = False
    scene.render.bake.margin = 4

    vec_M = mathutils.Vector
    batch_sz = 4
    for b_start in range(0, len(accepted_ids), batch_sz):
        batch = accepted_ids[b_start : b_start + batch_sz]
        uv_names = []
        W_b = np.zeros((n_verts, len(batch)), dtype=np.float32)

        for bi, cid in enumerate(batch):
            cd = cams[cid]
            cam_pos = np.array(cd["cam_xyz"], dtype=np.float64)
            fwd = np.array(cd["fwd"], dtype=np.float64)
            right_r = np.array(cd["right_r"], dtype=np.float64)
            up_r = np.array(cd["up_r"], dtype=np.float64)
            W_img, H_img = float(cd["W"]), float(cd["H"])
            f_px = (H_img * 0.5) / math.tan(math.radians(cd["fov_deg"] * 0.5))

            rel = V - cam_pos[None, :]
            xc = rel @ right_r
            yc = rel @ up_r
            zc = np.maximum(rel @ fwd, 1e-4)
            u_px = W_img * 0.5 + f_px * (xc / zc)
            v_pil = H_img * 0.5 - f_px * (yc / zc)
            u_bl = np.clip(u_px / W_img, 0.0, 1.0)
            v_bl = np.clip(1.0 - (v_pil / H_img), 0.0, 1.0)

            uvn = f"_UV_M_{bi}"
            uv_names.append(uvn)
            if uvn in me.uv_layers:
                me.uv_layers.remove(me.uv_layers[uvn])
            uv_lay = me.uv_layers.new(name=uvn)
            uv_lay.data.foreach_set("uv", np.column_stack([u_bl, v_bl])[loop_vi].astype(np.float32).ravel())

            msk_arr = _load_img_topdown_np(cd["mask"], non_color=True)[..., 0]
            mh, mw = msk_arr.shape
            iu = np.clip(np.round(u_px), 0, mw - 1).astype(np.int32)
            iv = np.clip(np.round(v_pil), 0, mh - 1).astype(np.int32)
            m_val = msk_arr[iv, iu]

            d2c = cam_pos[None, :] - V
            dist = np.linalg.norm(d2c, axis=1)
            dn = d2c / np.maximum(dist[:, None], 1e-6)
            ndotv = np.sum(N * dn, axis=1)
            in_b = (u_px > 6.0) & (u_px < W_img - 6.0) & (v_pil > 6.0) & (v_pil < H_img - 6.0)
            cand_idx = np.where(is_ter_v & in_b & (m_val > 0.15) & (ndotv > 0.18))[0]

            vis = np.zeros(n_verts, dtype=bool)
            for vi_idx in cand_idx:
                p = V[vi_idx] + N[vi_idx] * 0.14
                hit_loc, _, _, _ = bvh.ray_cast(vec_M(p), vec_M(dn[vi_idx]), float(dist[vi_idx]) - 0.5)
                if hit_loc is None:
                    vis[vi_idx] = True

            # Atenuación suave hacia los bordes del encuadre para evitar cortes
            edge_dist = np.minimum(np.minimum(u_bl, 1.0 - u_bl), np.minimum(v_bl, 1.0 - v_bl))
            edge_fade = _smoothstep(0.01, 0.08, edge_dist)
            W_b[:, bi] = np.where(vis, (np.maximum(ndotv, 0.0) ** 4.0) * m_val * edge_fade, 0.0).astype(np.float32)

        ca_name = "_MacroW"
        if ca_name in me.color_attributes:
            me.color_attributes.remove(me.color_attributes[ca_name])
        ca = me.color_attributes.new(name=ca_name, type="FLOAT_COLOR", domain="POINT")
        ca.data.foreach_set("color", W_b.ravel())
        me.update()

        for ul in me.uv_layers:
            ul.active = (ul.name == "UVMap")
            ul.active_render = (ul.name == "UVMap")

        # 1. Material de emisión para Σ(w·HP) codificado en [0..1] con offset +0.5 (se decodifica por pixel!)
        # Como EMIT recorta si negativo en PNG pero en float_buffer=True preserva negativos reales:
        m_num = bpy.data.materials.new("_BakeHPNum")
        m_num.use_nodes = True
        nt = m_num.node_tree
        nt.nodes.clear()
        out_n = nt.nodes.new("ShaderNodeOutputMaterial")
        em_n = nt.nodes.new("ShaderNodeEmission")
        nt.links.new(em_n.outputs["Emission"], out_n.inputs["Surface"])

        att = nt.nodes.new("ShaderNodeAttribute")
        att.attribute_name = ca_name
        sep = nt.nodes.new("ShaderNodeSeparateColor")
        nt.links.new(att.outputs["Color"], sep.inputs["Color"])
        w_sockets = [sep.outputs["Red"], sep.outputs["Green"], sep.outputs["Blue"], att.outputs["Alpha"]]

        acc_num = None
        acc_den = None
        loaded_ims = []
        for bi, cid in enumerate(batch):
            hp_p = Path(report["patches"][cid]["hp_file"])
            uv_n = nt.nodes.new("ShaderNodeUVMap"); uv_n.uv_map = uv_names[bi]
            tex = nt.nodes.new("ShaderNodeTexImage")
            im = bpy.data.images.load(str(hp_p), check_existing=False)
            im.colorspace_settings.name = "Non-Color"
            tex.image = im; tex.extension = "EXTEND"
            loaded_ims.append(im)
            nt.links.new(uv_n.outputs["UV"], tex.inputs["Vector"])

            # Decodificar (Color - 0.5) * 0.40
            vsub = nt.nodes.new("ShaderNodeVectorMath"); vsub.operation = "SUBTRACT"
            vsub.inputs[1].default_value = (0.5, 0.5, 0.5)
            nt.links.new(tex.outputs["Color"], vsub.inputs[0])

            vmul = nt.nodes.new("ShaderNodeVectorMath"); vmul.operation = "SCALE"
            vmul.inputs["Scale"].default_value = 0.40
            nt.links.new(vsub.outputs["Vector"], vmul.inputs[0])

            vw = nt.nodes.new("ShaderNodeVectorMath"); vw.operation = "SCALE"
            nt.links.new(vmul.outputs["Vector"], vw.inputs[0])
            nt.links.new(w_sockets[bi], vw.inputs["Scale"])

            if acc_num is None:
                acc_num = vw.outputs["Vector"]
                acc_den = w_sockets[bi]
            else:
                va = nt.nodes.new("ShaderNodeVectorMath"); va.operation = "ADD"
                nt.links.new(acc_num, va.inputs[0]); nt.links.new(vw.outputs["Vector"], va.inputs[1])
                acc_num = va.outputs["Vector"]
                da = nt.nodes.new("ShaderNodeMath"); da.operation = "ADD"
                nt.links.new(acc_den, da.inputs[0]); nt.links.new(w_sockets[bi], da.inputs[1])
                acc_den = da.outputs["Value"]

        # Sesgo +0.5 para garantizar EMIT siempre no negativo, aunque float_buffer aguanta directo
        vbias = nt.nodes.new("ShaderNodeVectorMath"); vbias.operation = "ADD"
        vbias.inputs[1].default_value = (0.5, 0.5, 0.5)
        nt.links.new(acc_num, vbias.inputs[0])
        nt.links.new(vbias.outputs["Vector"], em_n.inputs["Color"])

        im_f = bpy.data.images.new(f"Bake_HP_Num_{b_start}", width=atlas_res, height=atlas_res, float_buffer=True)
        im_f.colorspace_settings.name = "Non-Color"
        tn_b = nt.nodes.new("ShaderNodeTexImage"); tn_b.image = im_f; nt.nodes.active = tn_b

        m_blk = bpy.data.materials.new("_BakeHPZero")
        m_blk.use_nodes = True
        im_d = bpy.data.images.new("DummyHP", width=64, height=64)
        t_d = m_blk.node_tree.nodes.new("ShaderNodeTexImage"); t_d.image = im_d; m_blk.node_tree.nodes.active = t_d

        obj.data.materials.clear()
        obj.data.materials.append(m_blk)
        obj.data.materials.append(m_num)
        obj.data.materials.append(m_num)
        bpy.ops.object.bake(type="EMIT")

        px_buf = np.empty(atlas_res * atlas_res * 4, dtype=np.float32)
        im_f.pixels.foreach_get(px_buf)
        num_batch = np.flipud(px_buf.reshape((atlas_res, atlas_res, 4))[..., :3]) - 0.5
        hp_accum_num += num_batch

        # 2. Hornear Σw en la misma pasada
        nt.links.new(acc_den, em_n.inputs["Color"])
        bpy.ops.object.bake(type="EMIT")
        im_f.pixels.foreach_get(px_buf)
        den_batch = np.flipud(px_buf.reshape((atlas_res, atlas_res, 4))[..., 0])
        hp_accum_den += den_batch

        # Limpiar imágenes y materiales temporales del lote
        bpy.data.images.remove(im_f)
        bpy.data.images.remove(im_d)
        for im in loaded_ims:
            bpy.data.images.remove(im)
        bpy.data.materials.remove(m_num)
        bpy.data.materials.remove(m_blk)
        for uvn in uv_names:
            if uvn in me.uv_layers:
                me.uv_layers.remove(me.uv_layers[uvn])
        if ca_name in me.color_attributes:
            me.color_attributes.remove(me.color_attributes[ca_name])
        print(f"    [MacroHP Batch {b_start // batch_sz + 1}] OK ({len(batch)} parches)")

    obj.data.materials.clear()
    for m in orig_mats:
        obj.data.materials.append(m)
    obj.data.polygons.foreach_set("material_index", poly_mat.tolist())
    obj.data.update()

    fade_w = _smoothstep(0.05, 0.30, hp_accum_den)[..., None].astype(np.float32)
    hp_final = np.where(hp_accum_den[..., None] > 1e-4, hp_accum_num / np.maximum(hp_accum_den[..., None], 1e-4), 0.0)
    hp_final = np.clip(hp_final * fade_w, -0.18, 0.18).astype(np.float32)

    # Guardar EXR de auditoría
    exr_p = macro_dir / "macro_hp_accum_8k.exr"
    im_exr = bpy.data.images.new("MacroHP_EXR", width=atlas_res, height=atlas_res, float_buffer=True)
    im_exr.colorspace_settings.name = "Non-Color"
    rgba_exr = np.zeros((atlas_res, atlas_res, 4), dtype=np.float32)
    rgba_exr[..., :3] = np.flipud(hp_final)
    rgba_exr[..., 3] = np.flipud(hp_accum_den)
    im_exr.pixels.foreach_set(rgba_exr.ravel())
    im_exr.filepath_raw = str(exr_p)
    im_exr.file_format = "OPEN_EXR"
    im_exr.save()
    bpy.data.images.remove(im_exr)

    ao = bake_cavity_ao(scene, obj, macro_dir / "ao_cavity_8k.png", res=atlas_res, samples=6)
    return hp_final, ao


def bake_cavity_ao(scene: bpy.types.Scene, obj: bpy.types.Object, out_path: Path, res: int = 8192, samples: int = 6) -> np.ndarray:
    """Hornea mapa AO geométrico local en 8K para reforzar micro-estratos y pie de rocas."""
    me = obj.data
    orig_mats = list(me.materials)
    poly_mat = np.empty(len(me.polygons), dtype=np.int32)
    me.polygons.foreach_get("material_index", poly_mat)

    for ul in me.uv_layers:
        ul.active = (ul.name == "UVMap")
        ul.active_render = (ul.name == "UVMap")

    m_ao = bpy.data.materials.new("_BakeAO")
    m_ao.use_nodes = True
    nt = m_ao.node_tree
    nt.nodes.clear()
    ao_n = nt.nodes.new("ShaderNodeAmbientOcclusion")
    ao_n.samples = 8
    ao_n.only_local = True
    ao_n.inputs["Distance"].default_value = 2.2
    em_n = nt.nodes.new("ShaderNodeEmission")
    out_n = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(ao_n.outputs["AO"], em_n.inputs["Color"])
    nt.links.new(em_n.outputs["Emission"], out_n.inputs["Surface"])

    im_ao = bpy.data.images.new("Bake_AO_8K", width=res, height=res)
    im_ao.colorspace_settings.name = "Non-Color"
    t_b = nt.nodes.new("ShaderNodeTexImage"); t_b.image = im_ao; nt.nodes.active = t_b

    m_z = bpy.data.materials.new("_BakeAOZero"); m_z.use_nodes = True
    im_zd = bpy.data.images.new("DummyAO", width=64, height=64)
    t_zd = m_z.node_tree.nodes.new("ShaderNodeTexImage"); t_zd.image = im_zd; m_z.node_tree.nodes.active = t_zd

    me.materials.clear()
    me.materials.append(m_z)
    me.materials.append(m_ao)
    me.materials.append(m_ao)

    scene.cycles.samples = samples
    scene.render.bake.margin = 4
    print(f"  [Bake] Cycles AO Cavidad ({res}x{res})...")
    bpy.ops.object.bake(type="EMIT")

    buf = np.empty(res * res * 4, dtype=np.float32)
    im_ao.pixels.foreach_get(buf)
    ao_top = np.flipud(buf.reshape((res, res, 4))[..., 0]).copy()
    im_ao.filepath_raw = str(out_path)
    im_ao.file_format = "PNG"
    im_ao.save()

    bpy.data.images.remove(im_ao)
    bpy.data.images.remove(im_zd)
    bpy.data.materials.remove(m_ao)
    bpy.data.materials.remove(m_z)
    me.materials.clear()
    for m in orig_mats:
        me.materials.append(m)
    me.polygons.foreach_set("material_index", poly_mat.tolist())
    me.update()
    return ao_top
