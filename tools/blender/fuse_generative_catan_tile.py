"""
Fusión híbrida CAD + Malla Generativa 3D en Blender 5.3 headless:
Alinea automáticamente un modelo Image-to-3D (.glb) al hexágono Catán canónico,
entierra su zócalo artificial bajo Z_BASE=25mm, lo recorta al micrón con un troquel
B-Rep respetando carreteras (3.50mm) y 6 pads (Ø20mm), lo une de forma estanca a la
base build123d y genera renders PBR/Arcilla/Orto y comparación FRONT/BACK.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import addon_utils
import bmesh
import bpy
import mathutils
import numpy as np

R = 64.0000
APOTHEM = R * math.cos(math.radians(30.0))  # 55.425626 mm
Z_BASE = 25.0000


def _ensure_cycles_cpu(scene: bpy.types.Scene, samples: int = 16) -> None:
    try:
        addon_utils.enable("cycles")
    except Exception:
        pass
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    if hasattr(scene.cycles, "use_denoising"):
        try:
            scene.cycles.use_denoising = True
        except Exception:
            pass
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"


def _make_solid_material(name: str, rgba: Tuple[float, float, float, float], roughness: float = 0.65) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    if mat.node_tree is None:
        mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        nt.nodes.clear()
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = roughness
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.30
    return mat


def _pack_all_images() -> None:
    for img in bpy.data.images:
        if img.source == "FILE" and not img.packed_file:
            try:
                img.pack()
            except Exception:
                pass


def _boolean_op(target: bpy.types.Object, tool: bpy.types.Object, operation: str = "UNION") -> None:
    backup_mesh = target.data.copy()
    bpy.ops.object.select_all(action="DESELECT")
    target.select_set(True)
    bpy.context.view_layer.objects.active = target

    mod = target.modifiers.new(name=f"Bool_{operation}_{tool.name}", type="BOOLEAN")
    mod.operation = operation
    mod.object = tool
    solvers = [e.identifier for e in mod.bl_rna.properties["solver"].enum_items]
    used_manifold = "MANIFOLD" in solvers
    if used_manifold:
        mod.solver = "MANIFOLD"
    elif "EXACT" in solvers:
        mod.solver = "EXACT"
    bpy.ops.object.modifier_apply(modifier=mod.name)

    if used_manifold and len(target.data.polygons) < 100 and "EXACT" in solvers:
        print(f"[Boolean] MANIFOLD produjo {len(target.data.polygons)} caras en {operation}; reintentando con EXACT ...")
        bad_mesh = target.data
        target.data = backup_mesh
        bpy.data.meshes.remove(bad_mesh)
        mod2 = target.modifiers.new(name=f"BoolFallback_{operation}_{tool.name}", type="BOOLEAN")
        mod2.operation = operation
        mod2.object = tool
        mod2.solver = "EXACT"
        bpy.ops.object.modifier_apply(modifier=mod2.name)
    else:
        bpy.data.meshes.remove(backup_mesh)


def _join_meshes(objs: List[bpy.types.Object], name: str = "GenJoined") -> bpy.types.Object:
    mesh_objs = [o for o in objs if o.type == "MESH" and len(o.data.polygons) > 0]
    if not mesh_objs:
        raise RuntimeError("No hay objetos MESH tras importar el GLB generativo")

    bpy.ops.object.select_all(action="DESELECT")
    for o in mesh_objs:
        # Asegurar que el nombre de capa UV coincida antes de unir
        if o.data.uv_layers:
            o.data.uv_layers[0].name = "UVMap"
        else:
            o.data.uv_layers.new(name="UVMap")
        o.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objs[0]
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    if len(mesh_objs) > 1:
        bpy.ops.object.join()
    joined = bpy.context.active_object
    joined.name = name
    return joined


def _orient_and_fit_generative(
    gen_obj: bpy.types.Object,
    rot_z_deg: Optional[float],
    target_diameter_mm: float,
    target_peak_z_mm: float,
    z_seat_ratio: float,
    tilt_x_deg: float = 0.0,
    tilt_y_deg: float = 0.0,
) -> Dict[str, float]:
    """
    Reorienta, nivela inclinaciones opcionales en X/Y, alinea el hexágono pointy-top
    y asienta la base generativa bajo Z_BASE=25.00 mm.
    """
    mesh = gen_obj.data
    coords = np.array([v.co[:] for v in mesh.vertices], dtype=np.float32)

    ext = coords.max(axis=0) - coords.min(axis=0)
    cx = float(0.5 * (coords[:, 0].min() + coords[:, 0].max()))
    cy = float(0.5 * (coords[:, 1].min() + coords[:, 1].max()))
    z0 = float(coords[:, 2].min())
    coords[:, 0] -= cx
    coords[:, 1] -= cy
    coords[:, 2] -= z0

    if abs(tilt_x_deg) > 1e-4 or abs(tilt_y_deg) > 1e-4:
        rmat = np.array(
            mathutils.Euler((math.radians(tilt_x_deg), math.radians(tilt_y_deg), 0.0), "XYZ").to_matrix(),
            dtype=np.float32,
        )
        coords = coords @ rmat.T
        coords[:, 0] -= float(0.5 * (coords[:, 0].min() + coords[:, 0].max()))
        coords[:, 1] -= float(0.5 * (coords[:, 1].min() + coords[:, 1].max()))
        coords[:, 2] -= float(coords[:, 2].min())

    # Auto-rotación en Z buscando simetría de 6 vértices cada 60° (pointy-top = picos en 30° + 60°·k)
    r_xy = np.hypot(coords[:, 0], coords[:, 1])
    ang = np.degrees(np.arctan2(coords[:, 1], coords[:, 0])) % 360.0
    if rot_z_deg is None:
        # Evaluar correlación con los 6 vértices en 30+60k sobre el anillo exterior (percentil >= 85 radial)
        r_cut = np.percentile(r_xy, 80.0)
        outer_ang = ang[r_xy >= r_cut]
        outer_r = r_xy[r_xy >= r_cut]
        best_deg = 0.0
        best_score = -1e9
        for cand in np.linspace(-30.0, 30.0, 121):
            shifted = (outer_ang + cand) % 60.0
            # Vértice del hex pointy-top cae en 30° dentro del módulo 60°
            score = float(np.sum(outer_r * np.cos(np.radians((shifted - 30.0) * 6.0))))
            if score > best_score:
                best_score = score
                best_deg = float(cand)
        rot_used = best_deg
    else:
        rot_used = float(rot_z_deg)

    rad = math.radians(rot_used)
    ca, sa = math.cos(rad), math.sin(rad)
    rx = coords[:, 0] * ca - coords[:, 1] * sa
    ry = coords[:, 0] * sa + coords[:, 1] * ca
    coords[:, 0] = rx
    coords[:, 1] = ry

    # Calcular ancho plano a plano real del hexágono generado tras rotar
    # En hex pointy-top, los 6 lados tienen normales k*60° y W = 2*APOTHEM; la anchura en X tras recortar zócalo exterior debe igualar ~124-128 mm de vértice a vértice
    span_x = float(np.percentile(coords[:, 0], 99.5) - np.percentile(coords[:, 0], 0.5))
    span_y = float(np.percentile(coords[:, 1], 99.5) - np.percentile(coords[:, 1], 0.5))
    scale_xy = target_diameter_mm / max(span_x / math.cos(math.radians(30.0)), span_y, 1e-5)

    coords[:, 0] *= scale_xy
    coords[:, 1] *= scale_xy
    coords[:, 2] *= scale_xy

    # Recalcular altura z_seat donde empieza la pradera verde/zócalo para enterrar la base plástica IA en Z=24.3 mm
    z_max_scaled = float(coords[:, 2].max())
    # z_seat_ratio permite ajustar cuánto de la base IA queda bajo Z=25.00 mm (típicamente ~24-30% de z_max corresponde al grosor de la peana negra IA)
    z_seat = float(z_seat_ratio * z_max_scaled)
    relief_above = max(z_max_scaled - z_seat, 1.0)
    desired_relief = target_peak_z_mm - Z_BASE
    z_stretch = desired_relief / relief_above
    coords[:, 2] = (coords[:, 2] - z_seat) * z_stretch + (Z_BASE - 0.55)

    for i, v in enumerate(mesh.vertices):
        v.co = (float(coords[i, 0]), float(coords[i, 1]), float(coords[i, 2]))
    mesh.update()

    return {
        "rot_z_deg": round(rot_used, 2),
        "scale_xy": round(scale_xy, 4),
        "z_stretch": round(z_stretch, 4),
        "z_seat_local_mm": round(z_seat, 2),
        "raw_extents": [round(float(x), 3) for x in ext],
        "fitted_z_min": round(float(coords[:, 2].min()), 3),
        "fitted_z_max": round(float(coords[:, 2].max()), 3),
    }


def _keep_largest_component_bmesh(bm: bmesh.types.BMesh) -> int:
    visited = set()
    components: List[List[bmesh.types.BMVert]] = []
    for v_start in bm.verts:
        if v_start in visited:
            continue
        comp: List[bmesh.types.BMVert] = []
        stack = [v_start]
        visited.add(v_start)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for edge in cur.link_edges:
                other = edge.other_vert(cur)
                if other not in visited:
                    visited.add(other)
                    stack.append(other)
        components.append(comp)
    if len(components) > 1:
        largest = max(components, key=len)
        keep_set = set(largest)
        drop_verts = [v for v in bm.verts if v not in keep_set]
        bmesh.ops.delete(bm, geom=drop_verts, context="VERTS")
        print(f"[BFS] Eliminadas {len(components) - 1} micro-componentes desconectadas ({len(drop_verts)} verts).")
    return 1 if len(bm.verts) > 0 else 0


def _sanitize_generative_mesh(obj: bpy.types.Object) -> Dict[str, int]:
    """Solda costuras UV espaciales sin perder UVs, triangula, poda esquirlas y cierra huecos."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    # 1. Soldar primero vértices duplicados por cortes UV conservando las UVs por esquina (loop)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
    bm.faces.ensure_lookup_table()
    bmesh.ops.triangulate(bm, faces=list(bm.faces))
    bmesh.ops.dissolve_degenerate(bm, dist=1e-6, edges=list(bm.edges))
    orphan_verts = [v for v in bm.verts if not v.link_faces]
    if orphan_verts:
        bmesh.ops.delete(bm, geom=orphan_verts, context="VERTS")
    n_comps = _keep_largest_component_bmesh(bm)
    boundary_edges = [e for e in bm.edges if e.is_boundary]
    n_holes = len(boundary_edges)
    if boundary_edges:
        bmesh.ops.holes_fill(bm, edges=boundary_edges, sides=64)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    non_man = sum(1 for e in bm.edges if not e.is_manifold)
    boundary = sum(1 for e in bm.edges if e.is_boundary)
    # Fallback defensivo si quedara algún borde no-manifold aislado
    if non_man > 0:
        bad_faces = list({f for e in bm.edges if not e.is_manifold for f in e.link_faces})
        if bad_faces:
            bmesh.ops.delete(bm, geom=bad_faces, context="FACES")
            n_comps = _keep_largest_component_bmesh(bm)
            rem_b = [e for e in bm.edges if e.is_boundary]
            if rem_b:
                bmesh.ops.holes_fill(bm, edges=rem_b, sides=64)
            bmesh.ops.triangulate(bm, faces=list(bm.faces))
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            non_man = sum(1 for e in bm.edges if not e.is_manifold)
            boundary = sum(1 for e in bm.edges if e.is_boundary)
    print(f"[Sanitize] V={len(bm.verts)} F={len(bm.faces)} comps={n_comps} non_man={non_man} boundary={boundary} holes_filled={n_holes}")
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return {
        "raw_sanitized_components": n_comps,
        "raw_holes_filled": n_holes,
        "raw_non_manifold_after_sanitize": non_man,
        "raw_boundary_after_sanitize": boundary,
    }


def _clean_and_verify_bmesh(obj: bpy.types.Object) -> Dict[str, int]:
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    bmesh.ops.triangulate(bm, faces=list(bm.faces))
    bmesh.ops.dissolve_degenerate(bm, dist=1e-6, edges=list(bm.edges))
    orphan_verts = [v for v in bm.verts if not v.link_faces]
    if orphan_verts:
        bmesh.ops.delete(bm, geom=orphan_verts, context="VERTS")
    n_comps = _keep_largest_component_bmesh(bm)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    non_man = sum(1 for e in bm.edges if not e.is_manifold)
    boundary = sum(1 for e in bm.edges if e.is_boundary)
    if non_man > 0 or boundary > 0:
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        non_man = sum(1 for e in bm.edges if not e.is_manifold)
        boundary = sum(1 for e in bm.edges if e.is_boundary)
    verts = len(bm.verts)
    faces = len(bm.faces)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return {
        "vertices": verts,
        "faces": faces,
        "non_manifold_edges": non_man,
        "boundary_edges": boundary,
        "connected_components": n_comps,
    }


def _voxel_remesh_object(obj: bpy.types.Object, voxel_size: float = 0.22) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    mod = obj.modifiers.new(name="OpenVDBRemesh", type="REMESH")
    mod.mode = "VOXEL"
    mod.voxel_size = float(voxel_size)
    mod.adaptivity = 0.006
    mod.use_smooth_shade = False
    bpy.ops.object.modifier_apply(modifier=mod.name)


def _prepare_emit_bake_source(gen_raw: bpy.types.Object) -> None:
    """Convierte materiales PBR de entrada en Emission=1.0 puro para hornear albedo limpio."""
    for mat in gen_raw.data.materials:
        if mat is None:
            continue
        if not mat.use_nodes or mat.node_tree is None:
            mat.use_nodes = True
        nt = mat.node_tree
        out_node = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None)
        if out_node is None:
            out_node = nt.nodes.new("ShaderNodeOutputMaterial")
        bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
        emit = nt.nodes.new("ShaderNodeEmission")
        emit.inputs["Strength"].default_value = 1.0

        linked_socket = None
        if bsdf is not None:
            bc_in = bsdf.inputs.get("Base Color")
            if bc_in is not None and bc_in.is_linked:
                linked_socket = bc_in.links[0].from_socket
            elif bc_in is not None:
                emit.inputs["Color"].default_value = tuple(bc_in.default_value)
        if linked_socket is None:
            tex_node = next((n for n in nt.nodes if n.type == "TEX_IMAGE" and n.image is not None), None)
            if tex_node is not None:
                linked_socket = tex_node.outputs["Color"]

        if linked_socket is not None:
            nt.links.new(linked_socket, emit.inputs["Color"])
        for ln in list(out_node.inputs["Surface"].links):
            nt.links.remove(ln)
        nt.links.new(emit.outputs["Emission"], out_node.inputs["Surface"])


def _create_baked_material(bake_size: int = 2048) -> Tuple[bpy.types.Material, bpy.types.Image, bpy.types.ShaderNodeTexImage]:
    baked_img = bpy.data.images.new("TRELLIS2_Baked_Albedo", width=int(bake_size), height=int(bake_size), alpha=False)
    try:
        baked_img.colorspace_settings.name = "sRGB"
    except Exception:
        pass
    mat = bpy.data.materials.new(name="TRELLIS2_Baked_PBR")
    if mat.node_tree is None:
        mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out_node = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.80
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.25
    tex_node = nt.nodes.new("ShaderNodeTexImage")
    tex_node.image = baked_img
    tex_node.select = True
    nt.nodes.active = tex_node
    nt.links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])
    return mat, baked_img, tex_node


def _setup_lighting(scene: bpy.types.Scene) -> None:
    world = bpy.data.worlds.new("StudioWorld")
    scene.world = world
    if world.node_tree is None:
        world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.38, 0.40, 0.44, 1.0)
        bg.inputs["Strength"].default_value = 1.25

    suns = [
        ("KeySun", (55.0, -50.0, 82.0), 3.3, (1.0, 0.96, 0.91), math.radians(7.0)),
        ("FillSun", (-70.0, -30.0, 58.0), 1.3, (0.80, 0.90, 1.0), math.radians(18.0)),
        ("RimSun", (0.0, 80.0, 48.0), 1.5, (0.95, 0.98, 1.0), math.radians(12.0)),
        ("BottomSun", (12.0, -20.0, -85.0), 1.8, (0.94, 0.96, 1.0), math.radians(16.0)),
    ]
    for name, loc, energy, color, angle in suns:
        ld = bpy.data.lights.new(name=name, type="SUN")
        ld.energy = energy
        ld.color = color
        ld.angle = angle
        lo = bpy.data.objects.new(name, ld)
        lo.location = loc
        d = mathutils.Vector((0.0, 0.0, 26.0)) - mathutils.Vector(loc)
        lo.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        scene.collection.objects.link(lo)


def _render_views(
    scene: bpy.types.Scene,
    cam_obj: bpy.types.Object,
    views: List[Tuple[float, float, float, Tuple[float, float, float]]],
    out_png: Path,
    cols: int,
    rows: int,
    res_px: int,
    material_override: Optional[bpy.types.Material],
) -> None:
    scene.render.resolution_x = res_px
    scene.render.resolution_y = res_px
    bpy.context.view_layer.material_override = material_override
    tiles: List[np.ndarray] = []

    with tempfile.TemporaryDirectory() as td:
        tp_dir = Path(td)
        for idx, (cx, cy, cz, target) in enumerate(views):
            cam_obj.data.type = "PERSP"
            cam_obj.data.lens = 45.0
            cam_obj.location = (cx, cy, cz)
            d = mathutils.Vector(target) - mathutils.Vector((cx, cy, cz))
            cam_obj.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
            vp = tp_dir / f"v_{idx}.png"
            scene.render.filepath = str(vp)
            bpy.ops.render.render(write_still=True)
            im = bpy.data.images.load(str(vp), check_existing=False)
            arr = np.array(im.pixels[:], dtype=np.float32).reshape((res_px, res_px, 4))
            tiles.append(arr)
            bpy.data.images.remove(im)

    bpy.context.view_layer.material_override = None

    row_strips = []
    # En pixels Blender la fila 0 es abajo -> apilar filas en orden inverso para que row 0 aparezca arriba del PNG
    for r in range(rows - 1, -1, -1):
        row_strips.append(np.concatenate(tiles[r * cols:(r + 1) * cols], axis=1))
    canvas = np.concatenate(row_strips, axis=0)

    out_im = bpy.data.images.new(out_png.stem, width=cols * res_px, height=rows * res_px, alpha=True)
    out_im.pixels.foreach_set(canvas.ravel())
    out_im.filepath_raw = str(out_png)
    out_im.file_format = "PNG"
    out_im.save()
    bpy.data.images.remove(out_im)


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Fusión B-Rep Catán + Malla 3D Generativa en Blender 5.3")
    parser.add_argument("--base-stl", required=True)
    parser.add_argument("--cutter-stl", required=True)
    parser.add_argument("--gen-glb", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--rot-z-deg", default="auto", help="Rotación Z en grados o 'auto'")
    parser.add_argument("--tilt-x-deg", type=float, default=0.0, help="Nivelación opcional en eje X (grados)")
    parser.add_argument("--tilt-y-deg", type=float, default=0.0, help="Nivelación opcional en eje Y (grados)")
    parser.add_argument("--target-diameter-mm", type=float, default=124.0)
    parser.add_argument("--target-height-mm", type=float, default=60.0)
    parser.add_argument("--z-seat-ratio", type=float, default=0.28)
    parser.add_argument("--voxel-size", type=float, default=0.22)
    parser.add_argument("--bake-size", type=int, default=2048)
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args(argv)

    base_stl = Path(args.base_stl).resolve()
    cutter_stl = Path(args.cutter_stl).resolve()
    gen_glb = Path(args.gen_glb).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    _ensure_cycles_cpu(scene, samples=args.samples)
    _setup_lighting(scene)
    cam_data = bpy.data.cameras.new("AuditCam")
    cam_obj = bpy.data.objects.new("AuditCam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    # 1. Importar malla generativa 3D (.glb) conservando materiales y texturas PBR como fuente de bake
    bpy.ops.import_scene.gltf(filepath=str(gen_glb))
    gen_raw = _join_meshes(list(bpy.context.scene.objects), name="GenerativeRawSource")
    _pack_all_images()

    rot_val = None if str(args.rot_z_deg).lower() == "auto" else float(args.rot_z_deg)
    fit_info = _orient_and_fit_generative(
        gen_raw,
        rot_z_deg=rot_val,
        target_diameter_mm=args.target_diameter_mm,
        target_peak_z_mm=args.target_height_mm,
        z_seat_ratio=args.z_seat_ratio,
        tilt_x_deg=args.tilt_x_deg,
        tilt_y_deg=args.tilt_y_deg,
    )

    # Render previo 2x2 de la malla generativa cruda (frente -60°, derecha 30°, detrás 120°, cenital) antes del recorte
    z_mid_raw = 0.5 * (fit_info["fitted_z_min"] + fit_info["fitted_z_max"])
    views_raw = [
        (175.0 * math.cos(math.radians(-60)), 175.0 * math.sin(math.radians(-60)), z_mid_raw + 75.0, (0.0, 0.0, z_mid_raw)),
        (175.0 * math.cos(math.radians(30)), 175.0 * math.sin(math.radians(30)), z_mid_raw + 75.0, (0.0, 0.0, z_mid_raw)),
        (175.0 * math.cos(math.radians(120)), 175.0 * math.sin(math.radians(120)), z_mid_raw + 75.0, (0.0, 0.0, z_mid_raw)),
        (0.0, -25.0, z_mid_raw + 195.0, (0.0, 0.0, z_mid_raw)),
    ]
    _render_views(scene, cam_obj, views_raw, out_dir / "audit_raw_ai3d.png", cols=2, rows=2, res_px=512, material_override=None)

    # Clonar fuente raw para solidificación OpenVDB 100% estanca y ocultar fuente durante booleanos
    gen_solid = gen_raw.copy()
    gen_solid.data = gen_raw.data.copy()
    gen_solid.name = "GenerativeSolid"
    scene.collection.objects.link(gen_solid)
    gen_raw.hide_viewport = True
    gen_raw.hide_render = True

    raw_san = _sanitize_generative_mesh(gen_solid)
    _voxel_remesh_object(gen_solid, voxel_size=args.voxel_size)

    mat_baked, baked_img, tex_node = _create_baked_material(bake_size=args.bake_size)
    gen_solid.data.materials.clear()
    gen_solid.data.materials.append(mat_baked)

    # 2. Importar cutter B-Rep interior y recortar quirúrgicamente fuera de carril 3.5mm y 6 pads Ø20mm
    bpy.ops.wm.stl_import(filepath=str(cutter_stl))
    cutter_obj = bpy.context.selected_objects[0]
    cutter_obj.name = "CatanInnerCutter"
    _boolean_op(gen_solid, cutter_obj, operation="INTERSECT")
    bpy.data.objects.remove(cutter_obj, do_unlink=True)

    # 3. Importar base B-Rep exacta y registrar todos los slots de material antes de unir
    bpy.ops.wm.stl_import(filepath=str(base_stl))
    base_obj = bpy.context.selected_objects[0]
    base_obj.name = "CatanTileMaster"
    mat_plinth = _make_solid_material("Plinth_Slate", (0.032, 0.038, 0.048, 1.0), roughness=0.58)
    base_obj.data.materials.clear()
    base_obj.data.materials.append(mat_plinth)
    base_obj.data.materials.append(mat_baked)

    # 4. Unión booleana estanca Base B-Rep + Terreno Solidificado Recortado
    _boolean_op(base_obj, gen_solid, operation="UNION")
    bpy.data.objects.remove(gen_solid, do_unlink=True)
    topo = _clean_and_verify_bmesh(base_obj)

    # 5. Exportar Master STL (malla estanca de impresión)
    bpy.ops.object.select_all(action="DESELECT")
    base_obj.select_set(True)
    bpy.context.view_layer.objects.active = base_obj
    master_stl = out_dir / "tile_master.stl"
    viewer_glb = out_dir / "tile_viewer.glb"
    bpy.ops.wm.stl_export(filepath=str(master_stl), export_selected_objects=True, ascii_format=False)

    # 6. Desplegar UVs solo de las caras del terreno (slot 1) y hornear albedo puro (EMIT 1 muestra) desde gen_raw
    if not base_obj.data.uv_layers:
        base_obj.data.uv_layers.new(name="UVMap")
    base_obj.data.uv_layers.active_index = 0
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_mode(type="FACE")
    bpy.ops.mesh.select_all(action="DESELECT")
    bm_uv = bmesh.from_edit_mesh(base_obj.data)
    n_terrain_faces = 0
    for f in bm_uv.faces:
        if f.material_index == 1:
            f.select = True
            n_terrain_faces += 1
        else:
            f.select = False
    bmesh.update_edit_mesh(base_obj.data)
    print(f"[UV] Unwrapping {n_terrain_faces} terrain faces (slot 1) ...")
    bpy.ops.uv.smart_project(angle_limit=math.radians(66.0), island_margin=0.002, scale_to_bounds=True)
    bpy.ops.object.mode_set(mode="OBJECT")

    gen_raw.hide_viewport = False
    gen_raw.hide_render = False
    _prepare_emit_bake_source(gen_raw)
    mat_baked.node_tree.nodes.active = tex_node
    tex_node.select = True

    bpy.ops.object.select_all(action="DESELECT")
    gen_raw.select_set(True)
    base_obj.select_set(True)
    bpy.context.view_layer.objects.active = base_obj

    scene.cycles.samples = 1
    scene.render.bake.use_selected_to_active = True
    scene.render.bake.cage_extrusion = 2.0
    scene.render.bake.max_ray_distance = 4.5
    scene.render.bake.margin = 8
    bpy.ops.object.bake(type="EMIT")

    albedo_png = out_dir / "baked_albedo.png"
    baked_img.filepath_raw = str(albedo_png)
    baked_img.file_format = "PNG"
    baked_img.save()
    baked_img.pack()
    bpy.data.objects.remove(gen_raw, do_unlink=True)
    scene.cycles.samples = args.samples

    # 7. Exportar Viewer GLB con la textura horneada embebida
    bpy.ops.object.select_all(action="DESELECT")
    base_obj.select_set(True)
    bpy.context.view_layer.objects.active = base_obj
    bpy.ops.export_scene.gltf(
        filepath=str(viewer_glb),
        export_format="GLB",
        use_selection=True,
        export_apply=True,
    )

    # 6. Renders de auditoría (6 vistas PBR, 6 vistas Arcilla, Frente/Trasera PBR+Arcilla y Orto cenital)
    mat_clay = _make_solid_material("Clay_CAD", (0.58, 0.56, 0.54, 1.0), roughness=0.55)

    views_6 = [
        (168.0 * math.cos(math.radians(-60)), 168.0 * math.sin(math.radians(-60)), 112.0, (0.0, 0.0, 28.0)),
        (168.0 * math.cos(math.radians(30)), 168.0 * math.sin(math.radians(30)), 112.0, (0.0, 0.0, 28.0)),
        (168.0 * math.cos(math.radians(120)), 168.0 * math.sin(math.radians(120)), 112.0, (0.0, 0.0, 28.0)),
        (168.0 * math.cos(math.radians(210)), 168.0 * math.sin(math.radians(210)), 112.0, (0.0, 0.0, 28.0)),
        (0.0, -95.0, 185.0, (0.0, 0.0, 28.0)),
        (20.0, -40.0, -170.0, (0.0, 0.0, 8.0)),
    ]
    _render_views(scene, cam_obj, views_6, out_dir / "audit_6views_pbr.png", cols=3, rows=2, res_px=512, material_override=None)
    _render_views(scene, cam_obj, views_6, out_dir / "audit_6views_clay.png", cols=3, rows=2, res_px=512, material_override=mat_clay)

    # Comparación directa 2x2: Frente/Trasera PBR (arriba) + Frente/Trasera Arcilla CAD (abajo)
    fx, fy = 152.0 * math.cos(math.radians(-60)), 152.0 * math.sin(math.radians(-60))
    bx, by = 152.0 * math.cos(math.radians(120)), 152.0 * math.sin(math.radians(120))
    views_fb = [
        (fx, fy, 92.0, (0.0, 0.0, 29.0)),
        (bx, by, 92.0, (0.0, 0.0, 29.0)),
    ]
    with tempfile.TemporaryDirectory() as td_fb:
        tp_fb = Path(td_fb)
        pbr_tmp = tp_fb / "fb_pbr.png"
        clay_tmp = tp_fb / "fb_clay.png"
        _render_views(scene, cam_obj, views_fb, pbr_tmp, cols=2, rows=1, res_px=640, material_override=None)
        _render_views(scene, cam_obj, views_fb, clay_tmp, cols=2, rows=1, res_px=640, material_override=mat_clay)
        im_pbr = bpy.data.images.load(str(pbr_tmp), check_existing=False)
        im_clay = bpy.data.images.load(str(clay_tmp), check_existing=False)
        arr_pbr = np.array(im_pbr.pixels[:], dtype=np.float32).reshape((640, 1280, 4))
        arr_clay = np.array(im_clay.pixels[:], dtype=np.float32).reshape((640, 1280, 4))
        bpy.data.images.remove(im_pbr)
        bpy.data.images.remove(im_clay)
        canvas_fb = np.concatenate([arr_clay, arr_pbr], axis=0)
        out_fb = bpy.data.images.new("audit_front_back_pbr_clay", width=1280, height=1280, alpha=True)
        out_fb.pixels.foreach_set(canvas_fb.ravel())
        out_fb.filepath_raw = str(out_dir / "audit_front_back_pbr_clay.png")
        out_fb.file_format = "PNG"
        out_fb.save()
        bpy.data.images.remove(out_fb)

    # Ortográfico cenital
    cam_obj.data.type = "ORTHO"
    cam_obj.data.ortho_scale = 138.0
    cam_obj.location = (0.0, 0.0, 180.0)
    cam_obj.rotation_euler = (0.0, 0.0, math.radians(-30.0))
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024
    scene.render.filepath = str(out_dir / "audit_ortho.png")
    bpy.ops.render.render(write_still=True)

    xs = [v.co.x for v in base_obj.data.vertices]
    ys = [v.co.y for v in base_obj.data.vertices]
    zs = [v.co.z for v in base_obj.data.vertices]
    metrics = {
        **topo,
        **raw_san,
        "fit_info": fit_info,
        "bounds_mm": {
            "dx": round(max(xs) - min(xs), 4),
            "dy": round(max(ys) - min(ys), 4),
            "dz": round(max(zs) - min(zs), 4),
        },
        "master_stl": str(master_stl),
        "viewer_glb": str(viewer_glb),
    }
    (out_dir / "blender_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("HYBRID_BLENDER_METRICS:", json.dumps(metrics))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
