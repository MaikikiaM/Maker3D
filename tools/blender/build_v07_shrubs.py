"""
Motor Híbrido v07-shrubs8k (0 créditos Colab) para Catán Montaña Hero 360°:
  1) `blender-export-gbuf`: hornea en Blender CPU (samples=1, sin denoiser) los mapas 8K de
     posición (pos_8k.npy), normal (nrm_8k.npy), visibilidad exacta 16 cámaras (vis_pack_8k.npy)
     y extrae la malla + aristas (`mesh_data.npz`).
  2) `python-project-and-sculpt`:
     - Proyección Multi-Banda Burt-Adelson ganador-dominante (Low: S^5, Detail: (S/Smax)^32)
       que elimina el promediado destructivo entre cámaras sobre el suelo del valle.
     - Relleno auténtico `Quilt 2K` de pradera perimetral exterior (`D0_valleyS`) sin bandas lisas.
     - Esculpido botánico 3D real sobre la malla de 65 µm: ~1,650 mini-arbustos en cojín con
       flósculos tipo coliflor (`+0.48..+1.28 mm`) + ~1,100 guijarros facetados de cauce
       (`+0.24..+0.58 mm`) + musgo tapizante + relieve de granito en escarpes >= 48°, 0 cráteres.
     - Collar AO de contacto analítico + `ao_cavity_8k.png` en `baked_albedo.png`.
  3) `blender-finalize`: carga `V_new.npy`, hornea NORMAL 8K + ROUGHNESS 4K, exporta
     `tile_master.stl`, `tile_viewer.glb` y renders `audit_6views_pbr/clay/ortho`.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORK_DIR = REPO_ROOT / "Tableros-Catan" / "pipeline-biomas" / "montana_hero360" / "work_v05"
DEFAULT_SCRATCH_DIR = Path(
    r"C:\Users\migue\.gemini\antigravity\brain\34342c68-2385-41c9-a165-20883e62734d\scratch\v07_gbuf"
)

R = 64.0
APOTHEM = R * math.sqrt(3.0) * 0.5
Z_BASE = 25.0
ROAD_MARGIN = 3.50
PAD_RADIUS = 10.00


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / max(e1 - e0, 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _hex_distances_arr(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ax = np.abs(x)
    ay = np.abs(y)
    de = APOTHEM - np.maximum(ax, 0.5 * ax + (math.sqrt(3.0) * 0.5) * ay)
    dv = np.full_like(x, 999.0, dtype=np.float64)
    for k in range(6):
        a = math.radians(30.0 + 60.0 * k)
        dv = np.minimum(dv, np.hypot(x - R * math.cos(a), y - R * math.sin(a)))
    return de, dv


# ---------------------------------------------------------------------------
# SUBFASE 1A: Blender CPU -> G-Buffer 8K + mesh_data.npz
# ---------------------------------------------------------------------------
def run_blender_export_gbuf(work_dir: Path, gbuf_dir: Path, res: int = 8192) -> None:
    import bpy

    t0 = time.time()
    gbuf_dir.mkdir(parents=True, exist_ok=True)
    macro_dir = work_dir / "macro"
    blend_p = work_dir / "gpu_job_scene.blend"
    if not blend_p.exists():
        blend_p = work_dir / "passA_scene.blend"

    print(f"[v07 1A] Abriendo {blend_p}...", flush=True)
    bpy.ops.wm.open_mainfile(filepath=str(blend_p))
    obj = bpy.data.objects.get("Catan_Hero360") or bpy.context.selected_objects[0]
    me = obj.data
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    for ul in list(me.uv_layers):
        if ul.name != "UVMap":
            me.uv_layers.remove(ul)
        else:
            ul.active = True
            ul.active_render = True
    for attr in list(me.color_attributes):
        me.color_attributes.remove(attr)

    n_v = len(me.vertices)
    n_p = len(me.polygons)
    n_l = len(me.loops)
    n_e = len(me.edges)

    co = np.empty(n_v * 3, dtype=np.float64)
    nm = np.empty(n_v * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    me.vertices.foreach_get("normal", nm)
    V = co.reshape(n_v, 3).astype(np.float32)
    N = nm.reshape(n_v, 3).astype(np.float32)

    pm = np.empty(n_p, dtype=np.int32)
    lt = np.empty(n_p, dtype=np.int32)
    lv = np.empty(n_l, dtype=np.int32)
    uv_flat = np.empty(n_l * 2, dtype=np.float32)
    me.polygons.foreach_get("material_index", pm)
    me.polygons.foreach_get("loop_total", lt)
    me.loops.foreach_get("vertex_index", lv)
    me.uv_layers["UVMap"].data.foreach_get("uv", uv_flat)
    uv_l = uv_flat.reshape(n_l, 2)

    ev = np.empty(n_e * 2, dtype=np.int32)
    me.edges.foreach_get("vertices", ev)
    E = ev.reshape(n_e, 2)

    is_ter_loop = np.repeat(pm >= 1, lt)
    is_ter_v = np.zeros(n_v, dtype=bool)
    is_ter_v[lv[is_ter_loop]] = True
    is_talus_v = np.zeros(n_v, dtype=bool)
    is_talus_v[lv[np.repeat(pm == 2, lt)]] = True

    v_uv = np.zeros((n_v, 2), dtype=np.float32)
    v_uv[lv[is_ter_loop]] = uv_l[is_ter_loop]

    # Guardar solo aristas donde ambos extremos pertenecen a terreno/talus para acelerar grafos
    ter_e = is_ter_v[E[:, 0]] & is_ter_v[E[:, 1]]
    e0_ter = E[ter_e, 0].astype(np.int32)
    e1_ter = E[ter_e, 1].astype(np.int32)

    np.savez(
        gbuf_dir / "mesh_data.npz",
        V=V,
        N=N,
        v_uv=v_uv,
        poly_mat=pm.astype(np.int16),
        is_ter_v=is_ter_v,
        is_talus_v=is_talus_v,
        e0_ter=e0_ter,
        e1_ter=e1_ter,
    )
    print(f"[v07 1A] Guardado mesh_data.npz ({n_v:,} verts, {len(e0_ter):,} aristas terreno)", flush=True)

    # Configurar Cycles CPU ultra-rápido para EMIT 1 sample sin denoiser
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 1
    if hasattr(sc.cycles, "use_denoising"):
        sc.cycles.use_denoising = False
    sc.render.bake.use_pass_direct = False
    sc.render.bake.use_pass_indirect = False
    sc.render.bake.margin = 6

    m_blk = bpy.data.materials.new("_Gbuf_Zero")
    m_blk.use_nodes = True
    im_d = bpy.data.images.new("_Gbuf_Dummy", width=64, height=64)
    t_d = m_blk.node_tree.nodes.new("ShaderNodeTexImage")
    t_d.image = im_d
    m_blk.node_tree.nodes.active = t_d

    m_gb = bpy.data.materials.new("_Gbuf_Emit")
    m_gb.use_nodes = True
    nt = m_gb.node_tree
    nt.nodes.clear()
    out_n = nt.nodes.new("ShaderNodeOutputMaterial")
    em_n = nt.nodes.new("ShaderNodeEmission")
    geo_n = nt.nodes.new("ShaderNodeNewGeometry")
    nt.links.new(em_n.outputs["Emission"], out_n.inputs["Surface"])

    im_b = bpy.data.images.new("_Gbuf_Atlas", width=res, height=res, float_buffer=True)
    im_b.colorspace_settings.name = "Non-Color"
    tb = nt.nodes.new("ShaderNodeTexImage")
    tb.image = im_b
    nt.nodes.active = tb

    obj.data.materials.clear()
    obj.data.materials.append(m_blk)
    obj.data.materials.append(m_gb)
    obj.data.materials.append(m_gb)
    obj.data.polygons.foreach_set("material_index", pm.tolist())
    obj.data.update()

    px = np.empty(res * res * 4, dtype=np.float32)

    # 1. Posición mundo top-down
    nt.links.new(geo_n.outputs["Position"], em_n.inputs["Color"])
    bpy.ops.object.bake(type="EMIT")
    im_b.pixels.foreach_get(px)
    pos_8k = np.flipud(px.reshape((res, res, 4))[..., :3]).copy()
    np.save(gbuf_dir / "pos_8k.npy", pos_8k)
    del pos_8k
    print(f"[v07 1A] pos_8k.npy listo ({time.time() - t0:.1f}s)", flush=True)

    # 2. Normal mundo top-down
    nt.links.new(geo_n.outputs["Normal"], em_n.inputs["Color"])
    bpy.ops.object.bake(type="EMIT")
    im_b.pixels.foreach_get(px)
    nrm_8k = np.flipud(px.reshape((res, res, 4))[..., :3]).copy()
    np.save(gbuf_dir / "nrm_8k.npy", nrm_8k)
    del nrm_8k
    print(f"[v07 1A] nrm_8k.npy listo ({time.time() - t0:.1f}s)", flush=True)

    # 3. Visibilidad exacta de las 16 cámaras en 6 pases EMIT RGB (3 cámaras por pase)
    cams = json.loads((macro_dir / "macro_cameras.json").read_text(encoding="utf-8"))
    rep = json.loads((macro_dir / "macro_report.json").read_text(encoding="utf-8"))
    acc_ids = [cid for cid in sorted(cams.keys()) if rep["patches"].get(cid, {}).get("accepted")]
    vis_npz = np.load(macro_dir / "macro_vis_cache.npz")

    att_n = nt.nodes.new("ShaderNodeAttribute")
    att_n.attribute_name = "_VisPack"
    nt.links.new(att_n.outputs["Color"], em_n.inputs["Color"])

    vis_pack = np.zeros((res, res, len(acc_ids)), dtype=np.uint8)
    rgba_v = np.ones((n_v, 4), dtype=np.float32)
    ca = me.color_attributes.new(name="_VisPack", type="FLOAT_COLOR", domain="POINT")

    for b0 in range(0, len(acc_ids), 3):
        batch = acc_ids[b0 : b0 + 3]
        rgba_v[:, :3] = 0.0
        for ch, cid in enumerate(batch):
            key = f"vis_{cid}"
            if key in vis_npz:
                bits = np.unpackbits(vis_npz[key])[:n_v].astype(np.float32)
                rgba_v[:, ch] = bits

        ca.data.foreach_set("color", rgba_v.ravel())
        me.update()

        bpy.ops.object.bake(type="EMIT")
        im_b.pixels.foreach_get(px)
        blk = np.flipud(px.reshape((res, res, 4))[..., :len(batch)])
        vis_pack[..., b0 : b0 + len(batch)] = np.clip(np.round(blk * 255.0), 0, 255).astype(np.uint8)
        print(f"[v07 1A] vis_pack lote {b0 // 3 + 1}/{(len(acc_ids) + 2) // 3} OK ({time.time() - t0:.1f}s)", flush=True)

    np.save(gbuf_dir / "vis_pack_8k.npy", vis_pack)
    (gbuf_dir / "gbuf_meta.json").write_text(
        json.dumps({"res": res, "accepted_ids": acc_ids}, indent=2), encoding="utf-8"
    )
    print(f"[v07 1A] G-Buffer completo en {time.time() - t0:.1f}s -> {gbuf_dir}", flush=True)


# ---------------------------------------------------------------------------
# SUBFASE 1B: Proyección Multi-Banda Nítida + Quilt + Esculpido 3D Botánico
# ---------------------------------------------------------------------------
def _value_noise_3d_fast(x: np.ndarray, y: np.ndarray, z: np.ndarray, seed: int = 17) -> np.ndarray:
    ix = np.floor(x).astype(np.int64)
    iy = np.floor(y).astype(np.int64)
    iz = np.floor(z).astype(np.int64)
    fx = x - ix
    fy = y - iy
    fz = z - iz
    ux = fx * fx * (3.0 - 2.0 * fx)
    uy = fy * fy * (3.0 - 2.0 * fy)
    uz = fz * fz * (3.0 - 2.0 * fz)

    def _h(ox: int, oy: int, oz: int) -> np.ndarray:
        h = ((ix + ox) * 73856093) ^ ((iy + oy) * 19349663) ^ ((iz + oz) * 83492791) ^ (seed * 2654435761)
        h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
        return (h.astype(np.float64) / 2147483647.5) - 1.0

    c000, c100 = _h(0, 0, 0), _h(1, 0, 0)
    c010, c110 = _h(0, 1, 0), _h(1, 1, 0)
    c001, c101 = _h(0, 0, 1), _h(1, 0, 1)
    c011, c111 = _h(0, 1, 1), _h(1, 1, 1)
    x00 = c000 + ux * (c100 - c000)
    x10 = c010 + ux * (c110 - c010)
    x01 = c001 + ux * (c101 - c001)
    x11 = c011 + ux * (c111 - c011)
    y0 = x00 + uy * (x10 - x00)
    y1 = x01 + uy * (x11 - x01)
    return y0 + uz * (y1 - y0)


def _poisson_select_2d(
    pts_xy: np.ndarray,
    radii: np.ndarray,
    scores: np.ndarray,
    min_spacing_factor: float = 0.68,
    max_count: int = 1900,
    seed: int = 2027,
) -> np.ndarray:
    """Selección Poisson-Disk veloz con hash espacial 2D respetando jerarquía de señal con ligero jitter natural."""
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(0.86, 1.14, size=len(scores))
    order = np.argsort(-(scores * jitter))
    cell = float(np.max(radii)) * min_spacing_factor * 1.1 + 1e-4
    grid: Dict[Tuple[int, int], List[int]] = {}
    chosen_idx: List[int] = []
    chosen_xy: List[Tuple[float, float]] = []
    chosen_r: List[float] = []

    for ci in order:
        if len(chosen_idx) >= max_count:
            break
        x = float(pts_xy[ci, 0])
        y = float(pts_xy[ci, 1])
        ri = float(radii[ci])
        gx, gy = int(math.floor(x / cell)), int(math.floor(y / cell))
        ok = True
        for ox in (-2, -1, 0, 1, 2):
            for oy in (-2, -1, 0, 1, 2):
                bucket = grid.get((gx + ox, gy + oy))
                if not bucket:
                    continue
                for j in bucket:
                    cx, cy = chosen_xy[j]
                    cr = chosen_r[j]
                    if math.hypot(x - cx, y - cy) < min_spacing_factor * (ri + cr):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            k = len(chosen_idx)
            chosen_idx.append(int(ci))
            chosen_xy.append((x, y))
            chosen_r.append(ri)
            grid.setdefault((gx, gy), []).append(k)
    return np.asarray(chosen_idx, dtype=np.int32)


def run_python_project_and_sculpt(work_dir: Path, gbuf_dir: Path) -> None:
    import cv2
    from scipy.spatial import cKDTree

    sys.path.insert(0, str(REPO_ROOT / "tools" / "blender"))
    import geology_micro_detail as gmd

    t0 = time.time()
    macro_dir = work_dir / "macro"
    meta = json.loads((gbuf_dir / "gbuf_meta.json").read_text(encoding="utf-8"))
    res = int(meta["res"])
    acc_ids: List[str] = meta["accepted_ids"]
    cams = json.loads((macro_dir / "macro_cameras.json").read_text(encoding="utf-8"))
    rep = json.loads((macro_dir / "macro_report.json").read_text(encoding="utf-8"))

    print(f"[v07 1B] Cargando parches RGB 2K ({len(acc_ids)} cámaras) y descomponiendo Low + Detail...", flush=True)
    patches_low: List[np.ndarray] = []
    patches_det: List[np.ndarray] = []
    patches_msk: List[np.ndarray] = []
    cam_params = []

    for cid in acc_ids:
        raw_p = Path(rep["patches"][cid].get("aligned_rgb_file") or "")
        if not raw_p.exists():
            raw_p = macro_dir / "aligned_rgb" / f"patch_{cid}_aligned.png"
        im_bgr = cv2.imread(str(raw_p), cv2.IMREAD_COLOR)
        if im_bgr is None:
            raise FileNotFoundError(f"No se pudo leer {raw_p}")
        rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        low = cv2.GaussianBlur(rgb, (0, 0), 22.0)
        det = rgb - low

        msk_p = Path(cams[cid]["mask"])
        if not msk_p.exists():
            msk_p = macro_dir / "renders" / msk_p.name
        msk = (cv2.imread(str(msk_p), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0)
        if msk.shape[:2] != rgb.shape[:2]:
            msk = cv2.resize(msk, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

        patches_low.append(low)
        patches_det.append(det)
        patches_msk.append(msk)

        cd = cams[cid]
        cp = np.array(cd["cam_xyz"], dtype=np.float32)
        fw = np.array(cd["fwd"], dtype=np.float32)
        rr = np.array(cd["right_r"], dtype=np.float32)
        ur = np.array(cd["up_r"], dtype=np.float32)
        W_i, H_i = float(cd["W"]), float(cd["H"])
        f_px = float((H_i * 0.5) / math.tan(math.radians(cd["fov_deg"] * 0.5)))
        cam_az = math.atan2(float(cp[1]), float(cp[0]))
        cam_params.append((cid, cp, fw, rr, ur, W_i, H_i, f_px, cam_az))

    # Preparar Quilt 2K periódico sin costuras desde zona pura de musgo/grava de D0_valleyS (evitando el carril inferior)
    d0_idx = acc_ids.index("D0_valleyS")
    d0_rgb = patches_low[d0_idx] + patches_det[d0_idx]
    d0_msk = patches_msk[d0_idx]
    y_q0, y_q1, x_q0, x_q1 = 1320, 1832, 680, 1192
    # Verificar que toda la región elegida pertenece al terreno útil
    if float(d0_msk[y_q0:y_q1, x_q0:x_q1].mean()) < 0.98:
        y_q0, y_q1, x_q0, x_q1 = 1280, 1792, 700, 1212
    q_patch = np.clip(d0_rgb[y_q0:y_q1, x_q0:x_q1], 0.0, 1.0)
    # Espejado 2x2 C0-continuo (1024x1024) sin costura alguna
    q_top = np.concatenate([q_patch, q_patch[:, ::-1]], axis=1)
    q_tile = np.concatenate([q_top, q_top[::-1, :]], axis=0).astype(np.float32)
    q_low = cv2.GaussianBlur(q_tile, (0, 0), 22.0)
    q_det = q_tile - q_low
    qt_h, qt_w = q_det.shape[:2]

    # Cargar G-buffer y passA por memory-map
    pos_mm = np.load(gbuf_dir / "pos_8k.npy", mmap_mode="r")
    nrm_mm = np.load(gbuf_dir / "nrm_8k.npy", mmap_mode="r")
    vis_mm = np.load(gbuf_dir / "vis_pack_8k.npy", mmap_mode="r")

    passA_bgr = cv2.imread(str(work_dir / "passA_albedo_8k.png"), cv2.IMREAD_COLOR)
    passA_rgb = cv2.cvtColor(passA_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    del passA_bgr
    ao_cav = cv2.imread(str(macro_dir / "ao_cavity_8k.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

    albedo_out = np.zeros((res, res, 3), dtype=np.float32)
    hero_s_ids = {"A0_000", "B0_030", "B5_330", "D0_valleyS"}
    strip_h = 512
    n_cams = len(acc_ids)

    print("[v07 1B] Proyectando atlas 8K por tiras (Burt-Adelson Winner-Dominant + Quilt perimetral)...", flush=True)
    for y0 in range(0, res, strip_h):
        y1 = min(res, y0 + strip_h)
        h_s = y1 - y0
        P = np.asarray(pos_mm[y0:y1], dtype=np.float32)
        N = np.asarray(nrm_mm[y0:y1], dtype=np.float32)
        V8 = np.asarray(vis_mm[y0:y1], dtype=np.float32) / 255.0
        pa = passA_rgb[y0:y1]

        valid = P[..., 2] > 1.0
        steep = 1.0 - _smoothstep(0.45, 0.78, np.abs(N[..., 2]))
        flat = 1.0 - steep
        p_az = np.arctan2(P[..., 1], P[..., 0])

        S_stack = np.zeros((h_s, res, n_cams), dtype=np.float32)
        Low_stack = np.zeros((h_s, res, n_cams, 3), dtype=np.float32)
        Det_stack = np.zeros((h_s, res, n_cams, 3), dtype=np.float32)

        for ci, (cid, cp, fw, rr, ur, W_i, H_i, f_px, cam_az) in enumerate(cam_params):
            rel = P - cp[None, None, :]
            xc = np.einsum("hwk,k->hw", rel, rr)
            yc = np.einsum("hwk,k->hw", rel, ur)
            zc_raw = np.einsum("hwk,k->hw", rel, fw)
            zc = np.maximum(zc_raw, 1e-3)

            u_px = np.float32(W_i * 0.5) + np.float32(f_px) * (xc / zc)
            v_pil = np.float32(H_i * 0.5) - np.float32(f_px) * (yc / zc)
            u_n = u_px / np.float32(W_i)
            v_n = v_pil / np.float32(H_i)
            edge_d = np.minimum(np.minimum(u_n, 1.0 - u_n), np.minimum(v_n, 1.0 - v_n))
            in_fr = valid & (zc_raw > 1.0) & (u_px >= 0.0) & (u_px < W_i - 1.0) & (v_pil >= 0.0) & (v_pil < H_i - 1.0)

            map_x = np.where(in_fr, u_px, -10.0).astype(np.float32)
            map_y = np.where(in_fr, v_pil, -10.0).astype(np.float32)

            m_s = cv2.remap(patches_msk[ci], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            low_s = cv2.remap(patches_low[ci], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            det_s = cv2.remap(patches_det[ci], map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            Low_stack[:, :, ci, :] = low_s
            Det_stack[:, :, ci, :] = det_s

            d2c = -rel
            dist = np.linalg.norm(d2c, axis=-1)
            ndotv = np.sum(N * (d2c / np.maximum(dist[..., None], 1e-5)), axis=-1)
            vis_f = V8[..., ci] * m_s * _smoothstep(0.008, 0.065, edge_d) * in_fr.astype(np.float32)

            # Puntuación geométrica por rol para impedir promediado entre tomas de suelo
            hb = 1.35 if cid in hero_s_ids else 1.0
            score_cliff = (np.maximum(ndotv, 0.0) ** 3.0) * hb
            az_match = 0.5 + 0.5 * np.cos(p_az - cam_az)

            if cid == "D0_valleyS":
                in_south = (P[..., 1] < -5.0) & (np.abs(P[..., 0]) < 27.0) & (P[..., 2] < 38.5)
                score_meadow = np.maximum(ndotv, 0.15) * np.where(in_south, 1.65, 0.55)
            elif cid == "C0_peak":
                score_meadow = np.maximum(ndotv, 0.15) * _smoothstep(33.0, 43.0, P[..., 2]) * 0.65
            elif cid in ("C1_cirqueW", "C2_cirqueE"):
                score_meadow = np.maximum(ndotv, 0.15) * _smoothstep(31.0, 38.0, P[..., 2]) * 0.90
            else:
                ring_b = 1.25 if cid.startswith("B") else 1.0
                score_meadow = (0.35 + 0.65 * (az_match ** 3.0)) * np.maximum(ndotv, 0.15) * ring_b

            S_stack[..., ci] = vis_f * (steep * score_cliff + flat * score_meadow)

        S_max = np.max(S_stack, axis=-1)
        # Banda Low suave (S^5)
        W_low = S_stack ** 5.0
        W_low_sum = np.maximum(np.sum(W_low, axis=-1, keepdims=True), 1e-8)
        proj_low = np.sum(Low_stack * W_low[..., None], axis=2) / W_low_sum

        # Banda Detail ganador-dominante ((S / Smax)^32 -> >95% de texels de valle de 1 sola cámara)
        S_rel = S_stack / np.maximum(S_max[..., None], 1e-6)
        W_hi = (S_rel ** 32.0) * (S_stack > 1e-4)
        W_hi_sum = np.maximum(np.sum(W_hi, axis=-1, keepdims=True), 1e-8)
        proj_det = np.sum(Det_stack * W_hi[..., None], axis=2) / W_hi_sum

        # Fallback limpio de passA en zonas nunca vistas
        lum_pa = 0.2126 * pa[..., 0] + 0.7152 * pa[..., 1] + 0.0722 * pa[..., 2]
        warm_mud = (pa[..., 0] > pa[..., 1] * 1.06) & (pa[..., 0] > pa[..., 2] * 1.12)
        slate_fb = np.stack([lum_pa * 0.90, lum_pa * 0.96, lum_pa * 0.94], axis=-1)
        pa_clean = np.where(warm_mud[..., None], 0.30 * pa + 0.70 * slate_fb, pa)

        fade_macro = _smoothstep(0.002, 0.030, S_max)
        base_low = proj_low * fade_macro[..., None] + pa_clean * (1.0 - fade_macro[..., None])
        det_macro = proj_det * fade_macro[..., None]

        # Relleno auténtico Quilt 2K para la pradera perimetral exterior no cubierta por cámaras macro
        out_meadow = (
            valid
            & (fade_macro < 0.999)
            & (P[..., 2] >= Z_BASE + 0.05)
            & (P[..., 2] <= Z_BASE + 11.5)
            & (np.abs(N[..., 2]) >= 0.68)
        )
        lum_b = 0.2126 * base_low[..., 0] + 0.7152 * base_low[..., 1] + 0.0722 * base_low[..., 2]
        sat_b = np.max(base_low, axis=-1) - np.min(base_low, axis=-1)
        not_snow = 1.0 - _smoothstep(0.66, 0.80, lum_b) * (1.0 - _smoothstep(0.10, 0.20, sat_b))
        quilt_w = np.where(out_meadow, (1.0 - fade_macro) * not_snow, 0.0).astype(np.float32)

        if np.any(quilt_w > 1e-3):
            # Warp senoidal suave para evitar periodicidad perceptible
            qx = (P[..., 0] + 1.4 * np.sin(P[..., 1] * 0.14)) * 25.0
            qy = (P[..., 1] + 1.4 * np.cos(P[..., 0] * 0.14)) * 25.0
            q_det_s = cv2.remap(
                q_det, np.mod(qx, qt_w).astype(np.float32), np.mod(qy, qt_h).astype(np.float32),
                cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP
            )
            q_low_s = cv2.remap(
                q_low, np.mod(qx, qt_w).astype(np.float32), np.mod(qy, qt_h).astype(np.float32),
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP
            )
            # Armonizar tono del quilt hacia tono verde-musgo natural en pradera exterior (sin exceso de gris passA)
            base_low = base_low * (1.0 - 0.65 * quilt_w[..., None]) + q_low_s * (0.65 * quilt_w[..., None])
            det_final = det_macro + q_det_s * quilt_w[..., None]
        else:
            det_final = det_macro

        strip_rgb = np.clip(base_low + det_final, 0.0, 1.0)
        albedo_out[y0:y1] = np.where(valid[..., None], strip_rgb, 0.0)
        print(f"  [Strip {y0}:{y1}] OK ({time.time() - t0:.1f}s)", flush=True)

    # AO de cavidad sutil (12%) + ligera ganancia de contraste local
    ao_mul = (1.0 - 0.12 * (1.0 - np.clip(ao_cav, 0.0, 1.0)))[..., None].astype(np.float32)
    albedo_out = np.clip(albedo_out * ao_mul, 0.0, 1.0)

    # Verificar Laplacian std en valle (valid & slope plana)
    pos_z = np.asarray(pos_mm[..., 2])
    nrm_z = np.asarray(nrm_mm[..., 2])
    valley_roi = (pos_z >= Z_BASE + 0.15) & (pos_z <= Z_BASE + 9.0) & (np.abs(nrm_z) >= 0.74)
    lum_full = 0.2126 * albedo_out[..., 0] + 0.7152 * albedo_out[..., 1] + 0.0722 * albedo_out[..., 2]
    lap_val = cv2.Laplacian(lum_full, cv2.CV_32F)
    lap_std = float(lap_val[valley_roi].std()) if np.any(valley_roi) else 0.0
    print(f"[v07 1B] Albedo 8K Laplacian std en valle = {lap_std:.5f} (v06 era 0.0433)", flush=True)

    # -----------------------------------------------------------------------
    # SUBFASE 1B-2: Esculpido Botánico 3D (Mini-Arbustos + Guijarros + Acantilado)
    # -----------------------------------------------------------------------
    print("[v07 1B-2] Esculpido botánico 3D real sobre malla de 65 µm (0 cráteres)...", flush=True)
    md = np.load(gbuf_dir / "mesh_data.npz")
    V = md["V"].astype(np.float64)
    N = md["N"].astype(np.float64)
    v_uv = md["v_uv"].astype(np.float64)
    is_ter_v = md["is_ter_v"].astype(bool)
    is_talus_v = md["is_talus_v"].astype(bool)
    e0_t = md["e0_ter"]
    e1_t = md["e1_ter"]
    n_verts = len(V)

    # Muestreo de albedo nítido en cada vértice
    fa_u = np.clip(v_uv[:, 0], 0.0, 1.0) * (res - 1)
    fa_v = np.clip(1.0 - v_uv[:, 1], 0.0, 1.0) * (res - 1)
    iu = np.clip(np.round(fa_u), 0, res - 1).astype(np.int32)
    iv = np.clip(np.round(fa_v), 0, res - 1).astype(np.int32)
    rgb = albedo_out[iv, iu].astype(np.float64)
    lum = 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]
    sat = np.max(rgb, axis=1) - np.min(rgb, axis=1)

    de, dv = _hex_distances_arr(V[:, 0], V[:, 1])
    safe_valley = (
        _smoothstep(ROAD_MARGIN + 0.85, ROAD_MARGIN + 2.35, de)
        * _smoothstep(PAD_RADIUS + 0.85, PAD_RADIUS + 2.35, dv)
        * _smoothstep(Z_BASE + 0.10, Z_BASE + 0.80, V[:, 2])
        * is_ter_v.astype(np.float64)
        * (~is_talus_v).astype(np.float64)
    )

    # Distancia exacta 3D a las rocas grandes de derrubio (slot 2) para no invadirlas ni distorsionar costuras
    if np.any(is_talus_v):
        talus_tree = cKDTree(V[is_talus_v])
        d_talus, _ = talus_tree.query(V, k=1, workers=-1)
        clear_talus = _smoothstep(0.18, 0.65, d_talus)
    else:
        clear_talus = np.ones(n_verts, dtype=np.float64)

    snow_mask = _smoothstep(0.66, 0.82, lum) * (1.0 - _smoothstep(0.10, 0.20, sat)) * _smoothstep(Z_BASE + 2.0, Z_BASE + 6.5, V[:, 2])

    # Suavizado rápido de grafo para estimar vecindario vegetal y relieve local
    deg = np.zeros(n_verts, dtype=np.float32)
    np.add.at(deg, e0_t, 1.0)
    np.add.at(deg, e1_t, 1.0)
    deg = np.maximum(deg, 1.0)

    green_dom = rgb[:, 1] - 0.5 * (rgb[:, 0] + rgb[:, 2])
    green_loc = green_dom.astype(np.float32).copy()
    for _ in range(12):
        acc = np.zeros(n_verts, dtype=np.float32)
        np.add.at(acc, e0_t, green_loc[e1_t])
        np.add.at(acc, e1_t, green_loc[e0_t])
        green_loc = np.where(is_ter_v, acc / deg, green_loc)

    # Detección multi-especie: verde oliva/esmeralda + cojines salvia/líquenes plateados + brezo ocre/rojizo
    emerald_bush = _smoothstep(0.010, 0.058, green_dom)
    sage_bush = (
        _smoothstep(0.006, 0.028, green_loc)
        * _smoothstep(-0.008, 0.014, rgb[:, 1] - rgb[:, 0] * 0.96)
        * _smoothstep(0.32, 0.44, lum)
        * (1.0 - _smoothstep(0.68, 0.80, lum))
    )
    heather_bush = (
        _smoothstep(0.055, 0.12, rgb[:, 0] - rgb[:, 2])
        * _smoothstep(0.16, 0.26, rgb[:, 1])
        * (1.0 - _smoothstep(0.45, 0.62, lum))
        * _smoothstep(0.004, 0.022, green_loc)
    )
    flat_meadow_mask = _smoothstep(0.64, 0.78, np.abs(N[:, 2])) * (1.0 - _smoothstep(Z_BASE + 12.0, Z_BASE + 17.5, V[:, 2]))
    veg_score = (
        np.clip(np.maximum(emerald_bush, np.maximum(0.88 * sage_bush, 0.80 * heather_bush)), 0.0, 1.0)
        * safe_valley
        * clear_talus
        * (1.0 - snow_mask)
        * flat_meadow_mask
    )

    # Contraste follaje local sobre grafo (para sub-flósculos de coliflor reales alineados con la foto)
    lum_sm8 = lum.astype(np.float32).copy()
    for _ in range(8):
        acc = np.zeros(n_verts, dtype=np.float32)
        np.add.at(acc, e0_t, lum_sm8[e1_t])
        np.add.at(acc, e1_t, lum_sm8[e0_t])
        lum_sm8 = np.where(is_ter_v, acc / deg, lum_sm8)
    foliage_contrast = np.clip((lum - lum_sm8) * 3.8, -0.35, 0.50)

    # Sembrado Poisson-Disk de 1,400–1,900 mini-arbustos en cojín
    cand_shrub = np.where((veg_score >= 0.24) & (safe_valley >= 0.35) & (clear_talus >= 0.55))[0]
    rng_s = np.random.default_rng(2027)
    r_cand_s = np.where(
        veg_score[cand_shrub] > 0.62,
        rng_s.uniform(0.85, 1.75, size=len(cand_shrub)),
        rng_s.uniform(0.65, 1.30, size=len(cand_shrub)),
    )
    sel_s_local = _poisson_select_2d(
        V[cand_shrub, :2], r_cand_s, veg_score[cand_shrub],
        min_spacing_factor=0.68, max_count=1850, seed=2027,
    )
    shrub_vi = cand_shrub[sel_s_local]
    shrub_xy = V[shrub_vi, :2]
    shrub_r = r_cand_s[sel_s_local]
    shrub_h = np.clip(0.48 + 0.44 * shrub_r + rng_s.uniform(-0.08, 0.14, size=len(shrub_vi)), 0.48, 1.28)
    print(f"  [MiniShrubs] Sembrados {len(shrub_vi)} mini-arbustos 3D (R=[{shrub_r.min():.2f},{shrub_r.max():.2f}]mm, H=[{shrub_h.min():.2f},{shrub_h.max():.2f}]mm)", flush=True)

    # Sembrado Poisson-Disk de 900–1,300 guijarros facetados de cauce en claros/pedregales
    lum_sm20 = lum.astype(np.float32).copy()
    for _ in range(20):
        acc = np.zeros(n_verts, dtype=np.float32)
        np.add.at(acc, e0_t, lum_sm20[e1_t])
        np.add.at(acc, e1_t, lum_sm20[e0_t])
        lum_sm20 = np.where(is_ter_v, acc / deg, lum_sm20)
    pebble_peak = np.clip((lum - lum_sm20) * 4.5 + 0.28 * (0.28 - np.abs(rgb[:, 1] - rgb[:, 0])), 0.0, 1.0)
    pebble_zone = (
        (veg_score < 0.18)
        & (np.abs(N[:, 2]) >= 0.74)
        & (V[:, 2] <= Z_BASE + 14.0)
        & (snow_mask < 0.25)
        & (clear_talus > 0.35)
        & (safe_valley > 0.35)
    )
    cand_peb = np.where(pebble_zone)[0]
    r_cand_p = rng_s.uniform(0.38, 0.90, size=len(cand_peb))
    sel_p_local = _poisson_select_2d(
        V[cand_peb, :2], r_cand_p, pebble_peak[cand_peb] + 0.15,
        min_spacing_factor=0.92, max_count=1200, seed=3037,
    )
    peb_vi = cand_peb[sel_p_local]
    peb_xy = V[peb_vi, :2]
    peb_r = r_cand_p[sel_p_local]
    peb_h = np.clip(0.24 + 0.38 * peb_r + rng_s.uniform(-0.04, 0.06, size=len(peb_vi)), 0.24, 0.58)
    peb_ang = rng_s.uniform(0.0, math.pi, size=len(peb_vi))
    print(f"  [Pebbles3D] Sembrados {len(peb_vi)} guijarros facetados (R=[{peb_r.min():.2f},{peb_r.max():.2f}]mm, H=[{peb_h.min():.2f},{peb_h.max():.2f}]mm)", flush=True)

    # Evaluar desplazamiento de mini-arbustos sobre toda la malla de valle
    eval_mask = safe_valley > 1e-4
    eval_idx = np.where(eval_mask)[0]
    Ve = V[eval_idx]

    shrub_disp = np.zeros(n_verts, dtype=np.float64)
    shrub_rn = np.full(n_verts, 9.0, dtype=np.float64)
    if len(shrub_vi) > 0:
        stree = cKDTree(shrub_xy)
        dists_s, idxs_s = stree.query(Ve[:, :2], k=4, distance_upper_bound=2.1, workers=-1)
        # Flósculos 3D tipo coliflor en frecuencia sub-milimétrica
        f1 = _value_noise_3d_fast(Ve[:, 0] * 3.8, Ve[:, 1] * 3.8, Ve[:, 2] * 3.8, seed=71)
        f2 = _value_noise_3d_fast(Ve[:, 0] * 7.4 + 5.1, Ve[:, 1] * 7.4 - 3.2, Ve[:, 2] * 7.4, seed=113)
        floret_mod = np.clip(0.76 + 0.22 * f1 + 0.12 * f2 + 0.16 * foliage_contrast[eval_idx], 0.52, 1.24)

        best_h = np.zeros(len(eval_idx), dtype=np.float64)
        best_rn = np.full(len(eval_idx), 9.0, dtype=np.float64)
        for k_i in range(4):
            id_k = idxs_s[:, k_i]
            v_k = id_k < len(shrub_vi)
            if not np.any(v_k):
                continue
            ik = np.where(v_k, id_k, 0)
            rn = np.where(v_k, dists_s[:, k_i] / shrub_r[ik], 9.0)
            dome = np.where(v_k & (rn < 1.0), shrub_h[ik] * (np.maximum(0.0, 1.0 - rn * rn) ** 0.60), 0.0)
            best_h = np.maximum(best_h, dome)
            best_rn = np.minimum(best_rn, rn)

        # Musgo tapizante positivo continuo en el suelo restante (+0..+0.32 mm)
        f_moss = 0.5 + 0.5 * _value_noise_3d_fast(Ve[:, 0] * 2.4, Ve[:, 1] * 2.4, Ve[:, 2] * 2.4, seed=43)
        carpet = np.clip(veg_score[eval_idx] * (0.14 + 0.18 * f_moss), 0.0, 0.32)
        shrub_disp[eval_idx] = np.maximum(best_h * floret_mod, carpet)
        shrub_rn[eval_idx] = best_rn

    # Evaluar desplazamiento positivo de guijarros facetados
    peb_disp = np.zeros(n_verts, dtype=np.float64)
    peb_rn = np.full(n_verts, 9.0, dtype=np.float64)
    if len(peb_vi) > 0:
        ptree = cKDTree(peb_xy)
        d_p, i_p = ptree.query(Ve[:, :2], k=1, distance_upper_bound=1.1, workers=-1)
        vp = i_p < len(peb_vi)
        if np.any(vp):
            ip_s = np.where(vp, i_p, 0)
            dx = Ve[:, 0] - peb_xy[ip_s, 0]
            dy = Ve[:, 1] - peb_xy[ip_s, 1]
            ang = peb_ang[ip_s]
            ca, sa = np.cos(ang), np.sin(ang)
            rx = (dx * ca + dy * sa) / peb_r[ip_s]
            ry = (-dx * sa + dy * ca) / (0.78 * peb_r[ip_s])
            # Facetas angulosas tipo canto rodado / laja rota
            rn_hex = np.maximum(np.abs(rx), 0.52 * np.abs(rx) + 0.85 * np.abs(ry))
            rn_oct = 0.72 * np.abs(rx + ry) + 0.72 * np.abs(rx - ry)
            r_fac = np.maximum(rn_hex, 0.88 * rn_oct)
            p_dome = np.where(vp & (r_fac < 1.0), peb_h[ip_s] * np.maximum(0.0, 1.0 - r_fac ** 1.45), 0.0)
            # Solo donde no domina un arbusto alto
            p_gate = 1.0 - _smoothstep(0.18, 0.45, shrub_disp[eval_idx])
            peb_disp[eval_idx] = p_dome * p_gate
            peb_rn[eval_idx] = np.where(vp, r_fac, 9.0)

    # Preservar relieve de escarpes rocosos altos (slope >= 48°, 0 vegetación) igual que v06
    lum_wide = lum.astype(np.float32).copy()
    for _ in range(38):
        acc = np.zeros(n_verts, dtype=np.float32)
        np.add.at(acc, e0_t, lum_wide[e1_t])
        np.add.at(acc, e1_t, lum_wide[e0_t])
        lum_wide = np.where(is_ter_v, acc / deg, lum_wide)

    lum_mid = lum.astype(np.float32).copy()
    for _ in range(6):
        acc = np.zeros(n_verts, dtype=np.float32)
        np.add.at(acc, e0_t, lum_mid[e1_t])
        np.add.at(acc, e1_t, lum_mid[e0_t])
        lum_mid = np.where(is_ter_v, acc / deg, lum_mid)

    crevice_m = np.clip((lum_wide - lum_mid) - 0.006, 0.0, 0.55) * 2.50
    crevice_u = np.clip((lum_mid - lum) - 0.004, 0.0, 0.45) * 2.20
    ridge_m = np.clip((lum_mid - lum_wide) * 2.00 + np.clip(lum - lum_mid, 0.0, 0.35) * 1.25, 0.0, 0.55)
    green_ex = np.clip(rgb[:, 1] - np.maximum(rgb[:, 0], rgb[:, 2]) * 0.96, 0.0, 0.25)
    no_veg = 1.0 - _smoothstep(0.010, 0.045, green_ex)
    elev_gate = _smoothstep(Z_BASE + 3.5, Z_BASE + 10.5, V[:, 2])
    cliff_gate = (1.0 - _smoothstep(0.42, 0.67, np.abs(N[:, 2]))) * no_veg * elev_gate
    cliff_disp = safe_valley * (1.0 - 0.90 * snow_mask) * cliff_gate * (0.58 * ridge_m - 0.72 * crevice_m - 0.52 * crevice_u)
    cliff_disp = np.clip(cliff_disp, -0.75, 0.42)

    # Dirección segura monótona +Z en valle (0 auto-intersecciones) y normal en acantilados
    N_sm = N.copy()
    for _ in range(4):
        for c_i in range(3):
            acc = np.zeros(n_verts, dtype=np.float32)
            np.add.at(acc, e0_t, N_sm[e1_t, c_i])
            np.add.at(acc, e1_t, N_sm[e0_t, c_i])
            N_sm[:, c_i] = np.where(is_ter_v, acc / deg, N_sm[:, c_i])
    N_sm /= np.maximum(np.linalg.norm(N_sm, axis=1, keepdims=True), 1e-8)
    D_val = N_sm * 0.22 + np.array([[0.0, 0.0, 0.78]], dtype=np.float64)
    D_val /= np.maximum(np.linalg.norm(D_val, axis=1, keepdims=True), 1e-8)

    val_pos_disp = np.clip(safe_valley * clear_talus * (1.0 - 0.90 * snow_mask) * (shrub_disp + peb_disp), 0.0, 1.45)
    V_new = V + D_val * val_pos_disp[:, None] + N * cliff_disp[:, None]
    V_new[is_ter_v, 2] = np.maximum(V_new[is_ter_v, 2], Z_BASE + 0.02)
    V_new = V_new.astype(np.float32)

    print(
        f"  [v07 Sculpt3D] Valle positivo=[0.000, {val_pos_disp.max():.3f}] mm | "
        f"Escarpes=[{cliff_disp.min():.3f}, {cliff_disp.max():.3f}] mm",
        flush=True,
    )

    # Hornear collar AO analítico en baked_albedo.png en la base de cada mini-arbusto/guijarro (y ligera oclusión de contacto)
    valid_8k = pos_z > 1.0
    vr, vc = np.where(valid_8k & (pos_z <= Z_BASE + 17.5) & (np.abs(nrm_z) >= 0.68))
    if len(vr) > 0:
        px_eval = np.column_stack([np.asarray(pos_mm[vr, vc, 0]), np.asarray(pos_mm[vr, vc, 1])])
        collar = np.ones(len(vr), dtype=np.float32)
        if len(shrub_vi) > 0:
            ds_t, is_t = stree.query(px_eval, k=1, distance_upper_bound=2.2, workers=-1)
            vs_t = is_t < len(shrub_vi)
            rns = np.where(vs_t, ds_t / shrub_r[np.where(vs_t, is_t, 0)], 9.0)
            # Anillo sutil en base [0.80 .. 1.16] del cojín
            ring_s = _smoothstep(0.75, 0.96, rns) * (1.0 - _smoothstep(0.96, 1.22, rns))
            collar *= (1.0 - 0.14 * ring_s).astype(np.float32)
        if len(peb_vi) > 0:
            dp_t, ip_t = ptree.query(px_eval, k=1, distance_upper_bound=1.2, workers=-1)
            vp_t = ip_t < len(peb_vi)
            rnp = np.where(vp_t, dp_t / peb_r[np.where(vp_t, ip_t, 0)], 9.0)
            ring_p = _smoothstep(0.72, 0.94, rnp) * (1.0 - _smoothstep(0.94, 1.18, rnp))
            collar *= (1.0 - 0.12 * ring_p).astype(np.float32)
        albedo_out[vr, vc, :] *= collar[:, None]

    # Guardar V_new.npy + baked_albedo.png (v07)
    np.save(gbuf_dir / "V_new.npy", V_new)
    alb_p = work_dir / "baked_albedo.png"
    cv2.imwrite(
        str(alb_p),
        cv2.cvtColor(np.clip(np.round(albedo_out * 255.0), 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_PNG_COMPRESSION), 6],
    )
    print(f"[v07 1B] Guardados {alb_p} y V_new.npy en {time.time() - t0:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# SUBFASE 1C: Blender Finalize (Carga V_new, Normal 8K + Roughness 4K + STL/GLB/Audit)
# ---------------------------------------------------------------------------
def run_blender_finalize(work_dir: Path, gbuf_dir: Path, render_samples: int = 32) -> None:
    import bpy

    sys.path.insert(0, str(REPO_ROOT / "tools" / "blender"))
    import catan_hero360_bake as cb
    import macro_bake_helpers as mbh

    t0 = time.time()
    blend_p = work_dir / "gpu_job_scene.blend"
    if not blend_p.exists():
        blend_p = work_dir / "passA_scene.blend"
    bpy.ops.wm.open_mainfile(filepath=str(blend_p))
    sc = bpy.context.scene
    mbh.configure_cycles_device(sc, "cpu")
    obj = bpy.data.objects.get("Catan_Hero360") or bpy.context.selected_objects[0]
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    V_new = np.load(gbuf_dir / "V_new.npy").astype(np.float64)
    obj.data.vertices.foreach_set("co", V_new.ravel())
    poly_mat = np.empty(len(obj.data.polygons), dtype=np.int32)
    obj.data.polygons.foreach_get("material_index", poly_mat)
    is_ter_face = poly_mat >= 1
    obj.data.polygons.foreach_set("use_smooth", is_ter_face.tolist())
    obj.data.update()
    print(f"[v07 1C] V_new cargado sobre malla ({len(V_new):,} verts) en {time.time() - t0:.1f}s", flush=True)

    alb_p = work_dir / "baked_albedo.png"
    plinth_p = work_dir / "plinth_albedo_1k.png"
    if not plinth_p.exists():
        plinth_p = cb.build_plinth_texture(work_dir)

    alb_top = cb._load_img_topdown_np(str(alb_p))
    hp_top = cb._compute_log_highpass_topdown(alb_top, rad=4)
    del alb_top

    # Hornear NORMAL 8K (4 samples CPU) + ROUGHNESS 4K (2 samples CPU) y asignar materiales finales
    cb._bake_normal_and_roughness_passB(
        sc, obj, work_dir, alb_p, plinth_p, hp_top, is_ter_face,
        map_n_res=8192, map_r_res=4096, samples=4,
    )

    master_stl = work_dir / "tile_master.stl"
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.wm.stl_export(filepath=str(master_stl), export_selected_objects=True, ascii_format=False)

    glb_path = work_dir / "tile_viewer.glb"
    bpy.ops.export_scene.gltf(filepath=str(glb_path), export_format="GLB", use_selection=True, export_apply=True)

    if render_samples > 0:
        cams = json.loads((work_dir / "cameras.json").read_text(encoding="utf-8"))
        cb.render_6views_audit(sc, obj, cams, work_dir, samples=render_samples)
    print(f"[v07 1C Done] STL={master_stl}, GLB={glb_path} en {time.time() - t0:.1f}s", flush=True)


def _run_subprocess(cmd: List[str]) -> None:
    proc = subprocess.Popen(
        cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    assert proc.stdout is not None
    for ln in proc.stdout:
        print(ln.rstrip(), flush=True)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Subproceso falló (rc={rc}): {' '.join(cmd[:5])}")


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser(description="Constructor Híbrido v07-shrubs8k (0 créditos Colab)")
    p.add_argument(
        "--mode",
        choices=["all", "blender-export-gbuf", "python-project-and-sculpt", "blender-finalize"],
        default="all",
    )
    p.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    p.add_argument("--gbuf-dir", default=str(DEFAULT_SCRATCH_DIR))
    p.add_argument("--res", type=int, default=8192)
    p.add_argument("--render-samples", type=int, default=32)
    p.add_argument("--skip-gbuf", action="store_true")
    args = p.parse_args(argv)

    work_dir = Path(args.work_dir).resolve()
    gbuf_dir = Path(args.gbuf_dir).resolve()

    if args.mode == "blender-export-gbuf":
        run_blender_export_gbuf(work_dir, gbuf_dir, res=args.res)
    elif args.mode == "python-project-and-sculpt":
        run_python_project_and_sculpt(work_dir, gbuf_dir)
    elif args.mode == "blender-finalize":
        run_blender_finalize(work_dir, gbuf_dir, render_samples=args.render_samples)
    else:
        sys.path.insert(0, str(REPO_ROOT / "tools"))
        import tool_paths as tp
        blender_exe = tp.find_blender()
        if not args.skip_gbuf:
            _run_subprocess([
                blender_exe, "-b", "--python-exit-code", "1", "--python", str(Path(__file__).resolve()),
                "--", "--mode", "blender-export-gbuf", "--work-dir", str(work_dir), "--gbuf-dir", str(gbuf_dir),
                "--res", str(args.res),
            ])
        run_python_project_and_sculpt(work_dir, gbuf_dir)
        _run_subprocess([
            blender_exe, "-b", "--python-exit-code", "1", "--python", str(Path(__file__).resolve()),
            "--", "--mode", "blender-finalize", "--work-dir", str(work_dir), "--gbuf-dir", str(gbuf_dir),
            "--render-samples", str(args.render_samples),
        ])


if __name__ == "__main__":
    main()
