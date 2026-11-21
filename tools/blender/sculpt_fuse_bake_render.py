"""
Esculpido 3D en Blender 5.3 headless (estratos en normales de acantilados >= 45°),
fusión booleana estanca entre relieve/props 3D y zócalo B-Rep Catán (build123d),
mapeo UV/PBR y renderizado automático de auditoría (6 vistas PBR + 6 vistas Arcilla CAD + orto).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
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
ROAD_MARGIN = 3.50
PAD_RADIUS = 10.00


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


def _make_principled_material(
    name: str,
    base_color: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0),
    roughness: float = 0.65,
    image_path: Optional[Path] = None,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    if mat.node_tree is None:
        mat.use_nodes = True
    nt = mat.node_tree
    bsdf = None
    for n in nt.nodes:
        if n.type == "BSDF_PRINCIPLED":
            bsdf = n
            break
    if bsdf is None:
        nt.nodes.clear()
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    bsdf.inputs["Roughness"].default_value = roughness
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.35

    if image_path and image_path.exists():
        tex = nt.nodes.new("ShaderNodeTexImage")
        img = bpy.data.images.load(str(image_path), check_existing=True)
        try:
            img.pack()
        except Exception:
            pass
        tex.image = img
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = base_color
    return mat


def _hex_distances_xy(x: float, y: float) -> Tuple[float, float]:
    de = 999.0
    for k in range(6):
        ang = k * math.pi / 3.0
        de = min(de, APOTHEM - (x * math.cos(ang) + y * math.sin(ang)))
    dv = 999.0
    for k in range(6):
        ang = math.radians(30.0 + 60.0 * k)
        dv = min(dv, math.hypot(x - R * math.cos(ang), y - R * math.sin(ang)))
    return de, dv


def _bilinear_sample(arr: np.ndarray, x: float, y: float) -> float:
    h, w = arr.shape[:2]
    fx = max(0.0, min(1.0, (x + R) / (2.0 * R))) * (w - 1)
    fy = max(0.0, min(1.0, (y + R) / (2.0 * R))) * (h - 1)
    x0 = int(math.floor(fx))
    y0 = int(math.floor(fy))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    tx = fx - x0
    ty = fy - y0
    v00 = float(arr[y0, x0])
    v10 = float(arr[y0, x1])
    v01 = float(arr[y1, x0])
    v11 = float(arr[y1, x1])
    return (
        v00 * (1.0 - tx) * (1.0 - ty)
        + v10 * tx * (1.0 - ty)
        + v01 * (1.0 - tx) * ty
        + v11 * tx * ty
    )


def build_hex_terrain_cap(
    dem: np.ndarray,
    mat_terrain: bpy.types.Material,
    n_seg: int = 84,
    inset_mm: float = 0.65,
    z_bot: float = 20.0,
    z_flat_hidden: float = 24.45,
) -> Tuple[bpy.types.Object, int]:
    """
    Construye un prisma sólido hexagonal estanco (sin escalones ni caras coplanares
    con el zócalo B-Rep) y aplica esculpido 3D lateral a lo largo de normales en
    pendientes >= 45°.
    """
    r_cap = R - inset_mm
    verts_hex = [
        (r_cap * math.cos(math.radians(30.0 + 60.0 * k)), r_cap * math.sin(math.radians(30.0 + 60.0 * k)))
        for k in range(6)
    ]

    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap")

    top_vert_map: Dict[Tuple[int, int], bmesh.types.BMVert] = {}
    sector_grid: List[Dict[Tuple[int, int], bmesh.types.BMVert]] = []

    def get_top_vert(x: float, y: float) -> bmesh.types.BMVert:
        key = (int(round(x * 1000.0)), int(round(y * 1000.0)))
        v = top_vert_map.get(key)
        if v is None:
            dem_val = max(0.0, _bilinear_sample(dem, x, y))
            z = z_flat_hidden + dem_val
            v = bm.verts.new((x, y, z))
            top_vert_map[key] = v
        return v

    # Generar los 6 sectores triangulares equiláteros con malla baricéntrica uniforme
    for k in range(6):
        v1 = verts_hex[k]
        v2 = verts_hex[(k + 1) % 6]
        smap: Dict[Tuple[int, int], bmesh.types.BMVert] = {}
        for i in range(n_seg + 1):
            for j in range(n_seg + 1 - i):
                u = i / n_seg
                w = j / n_seg
                x = u * v1[0] + w * v2[0]
                y = u * v1[1] + w * v2[1]
                smap[(i, j)] = get_top_vert(x, y)
        sector_grid.append(smap)

        # Caras triangulares superiores orientadas hacia +Z
        for i in range(n_seg):
            for j in range(n_seg - i):
                a = smap[(i, j)]
                b = smap[(i + 1, j)]
                c = smap[(i, j + 1)]
                bm.faces.new((a, b, c))
                if i + j + 1 < n_seg:
                    d = smap[(i + 1, j + 1)]
                    bm.faces.new((b, d, c))

    bm.verts.ensure_lookup_table()
    bm.normal_update()

    # Esculpido 3D lateral a lo largo de normales en acantilados >= 45°
    cos45 = math.cos(math.radians(45.0))
    step_mm = r_cap / n_seg
    max_disp = 0.36 * step_mm
    cliff_displaced = 0
    for v in bm.verts:
        if v.co.z > (Z_BASE + 0.85) and abs(v.normal.z) < cos45:
            de, dv = _hex_distances_xy(v.co.x, v.co.y)
            if de > ROAD_MARGIN + 2.0 and dv > PAD_RADIUS + 2.0:
                hn_len = math.hypot(v.normal.x, v.normal.y)
                if hn_len > 1e-4:
                    nx, ny = v.normal.x / hn_len, v.normal.y / hn_len
                    strata = (
                        0.60 * math.sin(2.0 * v.co.z + 0.14 * v.co.x)
                        + 0.40 * math.cos(4.2 * v.co.z - 0.19 * v.co.y)
                    )
                    disp = max(-max_disp, min(max_disp, max_disp * strata))
                    v.co.x += nx * disp
                    v.co.y += ny * disp
                    cliff_displaced += 1

    # Anillo perimetral exterior de TerrainCap (sentido CCW visto desde arriba)
    rim_top: List[bmesh.types.BMVert] = []
    for k in range(6):
        smap = sector_grid[k]
        for j in range(n_seg):
            i = n_seg - j
            rim_top.append(smap[(i, j)])

    rim_bot = [bm.verts.new((v.co.x, v.co.y, z_bot)) for v in rim_top]
    bot_center = bm.verts.new((0.0, 0.0, z_bot))

    m_rim = len(rim_top)
    for idx in range(m_rim):
        nxt = (idx + 1) % m_rim
        # Pared lateral exterior (normal hacia afuera)
        bm.faces.new((rim_bot[idx], rim_bot[nxt], rim_top[nxt], rim_top[idx]))
        # Tapa inferior en z_bot (normal hacia -Z)
        bm.faces.new((bot_center, rim_bot[nxt], rim_bot[idx]))

    bm.normal_update()
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

    # Asignar coordenadas UV planares Y-up coherentes con albedo_base.png
    for face in bm.faces:
        for loop in face.loops:
            co = loop.vert.co
            u = max(0.0, min(1.0, (co.x + R) / (2.0 * R)))
            v = max(0.0, min(1.0, (co.y + R) / (2.0 * R)))
            loop[uv_layer].uv = (u, v)

    mesh_data = bpy.data.meshes.new("TerrainCapMesh")
    bm.to_mesh(mesh_data)
    bm.free()

    obj = bpy.data.objects.new("TerrainCap", mesh_data)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(mat_terrain)
    return obj, cliff_displaced


def build_demo_trees_object(
    dem: np.ndarray,
    slope_deg: Optional[np.ndarray],
    count: int,
    mat_foliage: bpy.types.Material,
    mat_bark: bpy.types.Material,
) -> Optional[bpy.types.Object]:
    """Genera N pinos 3D sólidos impresos-compatibles anclados al terreno sin colisiones."""
    if count <= 0:
        return None

    candidates: List[Tuple[float, float, float]] = []
    xs = np.linspace(-APOTHEM + 8.0, APOTHEM - 8.0, 44)
    ys = np.linspace(-R + 10.0, R - 10.0, 44)
    for y in ys:
        # Evitar colocar árboles dentro del cauce fluvial central (donde x ≈ 0 en zonas bajas)
        river_x = 7.5 * math.sin(float(y) / 17.0 + 0.35)
        for x in xs:
            de, dv = _hex_distances_xy(float(x), float(y))
            if de < ROAD_MARGIN + 7.5 or dv < PAD_RADIUS + 6.0:
                continue
            if abs(float(x) - river_x) < 9.5:
                continue
            z_rel = _bilinear_sample(dem, float(x), float(y))
            if z_rel < 2.0 or z_rel > 16.0:
                continue
            if slope_deg is not None:
                s = _bilinear_sample(slope_deg, float(x), float(y))
                if s > 27.0:
                    continue
            candidates.append((float(x), float(y), Z_BASE + z_rel))

    if not candidates:
        return None

    # Selección espaciada greedy (>= 14 mm entre árboles)
    chosen: List[Tuple[float, float, float]] = []
    # Ordenar determinísticamente dando preferencia a zonas intermedias bien visibles
    candidates.sort(key=lambda p: abs(p[0]) + abs(p[1]) * 0.8 + abs(p[2] - 31.0))
    for c in candidates:
        if all(math.hypot(c[0] - o[0], c[1] - o[1]) >= 14.0 for o in chosen):
            chosen.append(c)
            if len(chosen) >= count:
                break

    if not chosen:
        return None

    tree_objs: List[bpy.types.Object] = []
    for idx, (tx, ty, tz) in enumerate(chosen):
        scale = 0.88 + 0.14 * ((idx % 3) - 1)
        bm = bmesh.new()
        # Tronco sólido anclado 2.8 mm dentro del terreno (diámetro robusto >= 2.6 mm)
        bmesh.ops.create_cone(
            bm,
            cap_ends=True,
            cap_tris=True,
            segments=12,
            radius1=1.55 * scale,
            radius2=1.25 * scale,
            depth=6.5 * scale,
            matrix=mathutils.Matrix.Translation((0.0, 0.0, 0.45 * scale)),
        )
        m_trunk = bpy.data.meshes.new(f"TreeTrunkMesh_{idx}")
        bm.to_mesh(m_trunk)
        bm.free()
        o_trunk = bpy.data.objects.new(f"TreeTrunk_{idx}", m_trunk)
        o_trunk.data.materials.append(mat_bark)
        o_trunk.data.materials.append(mat_foliage)
        bpy.context.scene.collection.objects.link(o_trunk)

        # 3 niveles cónicos sólidos superpuestos con pendiente <= 42° imprimibles sin soporte
        cone_specs = [
            (4.8 * scale, 1.3 * scale, 4.6 * scale, 4.5 * scale),
            (3.9 * scale, 0.9 * scale, 4.2 * scale, 7.6 * scale),
            (2.8 * scale, 0.35 * scale, 4.0 * scale, 10.6 * scale),
        ]
        o_prev = o_trunk
        for c_i, (r1, r2, h_c, z_c) in enumerate(cone_specs):
            bmc = bmesh.new()
            bmesh.ops.create_cone(
                bmc,
                cap_ends=True,
                cap_tris=True,
                segments=14,
                radius1=r1,
                radius2=r2,
                depth=h_c,
                matrix=mathutils.Matrix.Translation((0.0, 0.0, z_c)),
            )
            mc = bpy.data.meshes.new(f"TreeConeMesh_{idx}_{c_i}")
            bmc.to_mesh(mc)
            bmc.free()
            oc = bpy.data.objects.new(f"TreeCone_{idx}_{c_i}", mc)
            oc.data.materials.append(mat_foliage)
            bpy.context.scene.collection.objects.link(oc)
            _boolean_op(o_prev, oc, operation="UNION")
            bpy.data.objects.remove(oc, do_unlink=True)

        o_prev.location = (tx, ty, tz)
        bpy.context.view_layer.update()
        # Hornear localización en vértices para boolean limpio posterior
        bm_bake = bmesh.new()
        bm_bake.from_mesh(o_prev.data)
        bm_bake.transform(o_prev.matrix_world)
        bm_bake.to_mesh(o_prev.data)
        bm_bake.free()
        o_prev.location = (0.0, 0.0, 0.0)
        tree_objs.append(o_prev)

    master_trees = tree_objs[0]
    for extra in tree_objs[1:]:
        _boolean_op(master_trees, extra, operation="UNION")
        bpy.data.objects.remove(extra, do_unlink=True)
    master_trees.name = "DemoTrees3D"
    return master_trees


def _boolean_op(target: bpy.types.Object, tool: bpy.types.Object, operation: str = "UNION") -> None:
    bpy.ops.object.select_all(action="DESELECT")
    target.select_set(True)
    bpy.context.view_layer.objects.active = target

    mod = target.modifiers.new(name=f"Bool_{tool.name}", type="BOOLEAN")
    mod.operation = operation
    mod.object = tool
    solvers = [e.identifier for e in mod.bl_rna.properties["solver"].enum_items]
    if "MANIFOLD" in solvers:
        mod.solver = "MANIFOLD"
    elif "EXACT" in solvers:
        mod.solver = "EXACT"
    bpy.ops.object.modifier_apply(modifier=mod.name)


def _check_bmesh_topology(obj: bpy.types.Object) -> Dict[str, int]:
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    non_man = sum(1 for e in bm.edges if not e.is_manifold)
    boundary = sum(1 for e in bm.edges if e.is_boundary)
    verts = len(bm.verts)
    faces = len(bm.faces)
    bm.free()
    return {
        "vertices": verts,
        "faces": faces,
        "non_manifold_edges": non_man,
        "boundary_edges": boundary,
    }


def _setup_studio_lighting(scene: bpy.types.Scene) -> None:
    world = bpy.data.worlds.new("StudioWorld")
    scene.world = world
    if world.node_tree is None:
        world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.14, 0.17, 0.21, 1.0)
        bg.inputs["Strength"].default_value = 1.1

    # Luces SUN independientes de la distancia (el modelo está en mm: 128 Blender units)
    sun_specs = [
        ("KeySun", (50.0, -45.0, 80.0), 3.4, (1.0, 0.96, 0.90), math.radians(6.0)),
        ("FillSun", (-65.0, -35.0, 55.0), 1.3, (0.78, 0.89, 1.0), math.radians(18.0)),
        ("RimSun", (0.0, 75.0, 45.0), 1.6, (0.95, 0.98, 1.0), math.radians(10.0)),
        ("BottomSun", (12.0, -20.0, -80.0), 1.8, (0.94, 0.96, 1.0), math.radians(15.0)),
    ]
    for name, loc, energy, color, angle in sun_specs:
        ld = bpy.data.lights.new(name=name, type="SUN")
        ld.energy = energy
        ld.color = color
        ld.angle = angle
        lo = bpy.data.objects.new(name, ld)
        lo.location = loc
        direction = mathutils.Vector((0.0, 0.0, 26.0)) - mathutils.Vector(loc)
        lo.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        scene.collection.objects.link(lo)


def _render_mosaic_6views(
    scene: bpy.types.Scene,
    cam_obj: bpy.types.Object,
    out_png: Path,
    tmp_dir: Path,
    res_px: int = 512,
    material_override: Optional[bpy.types.Material] = None,
) -> None:
    scene.render.resolution_x = res_px
    scene.render.resolution_y = res_px
    view_layer = bpy.context.view_layer
    view_layer.material_override = material_override

    # 5 ángulos elevados alrededor del hexágono + 1 vista inferior mostrando hueco imán
    views = [
        (165.0 * math.cos(math.radians(35)), 165.0 * math.sin(math.radians(35)), 118.0, (0.0, 0.0, 28.0)),
        (165.0 * math.cos(math.radians(145)), 165.0 * math.sin(math.radians(145)), 118.0, (0.0, 0.0, 28.0)),
        (165.0 * math.cos(math.radians(215)), 165.0 * math.sin(math.radians(215)), 118.0, (0.0, 0.0, 28.0)),
        (165.0 * math.cos(math.radians(325)), 165.0 * math.sin(math.radians(325)), 118.0, (0.0, 0.0, 28.0)),
        (0.0, -90.0, 185.0, (0.0, 0.0, 26.0)),
        (25.0, -45.0, -165.0, (0.0, 0.0, 8.0)),
    ]

    tiles: List[np.ndarray] = []
    for idx, (cx, cy, cz, target) in enumerate(views):
        cam_obj.data.type = "PERSP"
        cam_obj.data.lens = 42.0
        cam_obj.location = (cx, cy, cz)
        direction = mathutils.Vector(target) - mathutils.Vector((cx, cy, cz))
        cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        vp = tmp_dir / f"v_{idx}.png"
        scene.render.filepath = str(vp)
        bpy.ops.render.render(write_still=True)

        im = bpy.data.images.load(str(vp), check_existing=False)
        arr = np.array(im.pixels[:], dtype=np.float32).reshape((res_px, res_px, 4))
        tiles.append(arr)
        bpy.data.images.remove(im)

    view_layer.material_override = None

    # Canvas 3x2 en coordenadas Blender (row 0 abajo -> vistas 3,4,5; row 1 arriba -> vistas 0,1,2)
    top_row = np.concatenate([tiles[0], tiles[1], tiles[2]], axis=1)
    bot_row = np.concatenate([tiles[3], tiles[4], tiles[5]], axis=1)
    mosaic = np.concatenate([bot_row, top_row], axis=0)

    out_im = bpy.data.images.new(out_png.stem, width=3 * res_px, height=2 * res_px, alpha=True)
    out_im.pixels.foreach_set(mosaic.ravel())
    out_im.filepath_raw = str(out_png)
    out_im.file_format = "PNG"
    out_im.save()
    bpy.data.images.remove(out_im)


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Blender 5.3 Headless Sculpt + Boolean Fuse + Render")
    parser.add_argument("--base-stl", required=True)
    parser.add_argument("--dem-npy", required=True)
    parser.add_argument("--albedo-png", required=True)
    parser.add_argument("--slope-npy", default=None)
    parser.add_argument("--props-glb", default=None)
    parser.add_argument("--demo-trees", type=int, default=0)
    parser.add_argument("--grid-seg", type=int, default=80)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    base_stl = Path(args.base_stl).resolve()
    dem_npy = Path(args.dem_npy).resolve()
    albedo_png = Path(args.albedo_png).resolve()
    slope_npy = Path(args.slope_npy).resolve() if args.slope_npy else None
    props_glb = Path(args.props_glb).resolve() if args.props_glb else None
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    _ensure_cycles_cpu(scene, samples=args.samples)

    # 1. Materiales PBR (color lineal en Blender equivalente al sRGB #363B42 del bisel)
    mat_plinth = _make_principled_material("Plinth_Slate", base_color=(0.036, 0.043, 0.054, 1.0), roughness=0.68)
    mat_terrain = _make_principled_material("Terrain_Albedo", roughness=0.78, image_path=albedo_png)
    mat_foliage = _make_principled_material("Tree_Foliage", base_color=(0.12, 0.31, 0.14, 1.0), roughness=0.76)
    mat_bark = _make_principled_material("Tree_Bark", base_color=(0.24, 0.15, 0.09, 1.0), roughness=0.88)
    mat_clay = _make_principled_material("Clay_CAD", base_color=(0.72, 0.70, 0.67, 1.0), roughness=0.62)

    # 2. Importar base B-Rep exacta y registrar TODOS los slots de material antes de booleanos
    bpy.ops.wm.stl_import(filepath=str(base_stl))
    base_obj = bpy.context.selected_objects[0]
    base_obj.name = "CatanTileMaster"
    base_obj.data.materials.clear()
    base_obj.data.materials.append(mat_plinth)
    base_obj.data.materials.append(mat_terrain)
    base_obj.data.materials.append(mat_foliage)
    base_obj.data.materials.append(mat_bark)
    if not base_obj.data.uv_layers:
        base_obj.data.uv_layers.new(name="UVMap")

    # 3. Construir prisma hexagonal superior + esculpido 3D lateral >=45°
    dem = np.load(dem_npy).astype(np.float32)
    slope = np.load(slope_npy).astype(np.float32) if slope_npy and slope_npy.exists() else None
    terrain_obj, cliff_verts_sculpted = build_hex_terrain_cap(dem, mat_terrain, n_seg=args.grid_seg)

    # 4. Unir terreno con base B-Rep preservando exactamente carril 3.5mm, pads Ø20mm e imán Ø10.2mm
    _boolean_op(base_obj, terrain_obj, operation="UNION")
    bpy.data.objects.remove(terrain_obj, do_unlink=True)

    # 5. Árboles 3D sólidos opcionales (y/o props GLB externos)
    trees_obj = build_demo_trees_object(dem, slope, args.demo_trees, mat_foliage, mat_bark)
    trees_placed = 0 if trees_obj is None else args.demo_trees
    if trees_obj is not None:
        _boolean_op(base_obj, trees_obj, operation="UNION")
        bpy.data.objects.remove(trees_obj, do_unlink=True)

    if props_glb and props_glb.exists():
        bpy.ops.import_scene.gltf(filepath=str(props_glb))
        for pobj in list(bpy.context.selected_objects):
            if pobj.type == "MESH" and pobj != base_obj:
                _boolean_op(base_obj, pobj, operation="UNION")
                bpy.data.objects.remove(pobj, do_unlink=True)

    # Asegurar UVs planares intactas en toda la malla fusionada y validar bmesh
    bm_uv = bmesh.new()
    bm_uv.from_mesh(base_obj.data)
    uv_l = bm_uv.loops.layers.uv.get("UVMap") or bm_uv.loops.layers.uv.new("UVMap")
    for face in bm_uv.faces:
        for loop in face.loops:
            co = loop.vert.co
            u = max(0.0, min(1.0, (co.x + R) / (2.0 * R)))
            v = max(0.0, min(1.0, (co.y + R) / (2.0 * R)))
            loop[uv_l].uv = (u, v)
    bm_uv.to_mesh(base_obj.data)
    bm_uv.free()
    base_obj.data.update()

    topo = _check_bmesh_topology(base_obj)

    # 6. Exportar Master STL y Viewer GLB
    bpy.ops.object.select_all(action="DESELECT")
    base_obj.select_set(True)
    bpy.context.view_layer.objects.active = base_obj

    master_stl = out_dir / "tile_master.stl"
    viewer_glb = out_dir / "tile_viewer.glb"
    bpy.ops.wm.stl_export(filepath=str(master_stl), export_selected_objects=True, ascii_format=False)
    bpy.ops.export_scene.gltf(
        filepath=str(viewer_glb),
        export_format="GLB",
        use_selection=True,
        export_apply=True,
    )

    # 7. Iluminación y renders de auditoría
    _setup_studio_lighting(scene)
    cam_data = bpy.data.cameras.new("AuditCam")
    cam_obj = bpy.data.objects.new("AuditCam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    with tempfile.TemporaryDirectory() as tmp_d:
        tmp_path = Path(tmp_d)
        _render_mosaic_6views(scene, cam_obj, out_dir / "audit_6views_pbr.png", tmp_path, res_px=480, material_override=None)
        _render_mosaic_6views(scene, cam_obj, out_dir / "audit_6views_clay.png", tmp_path, res_px=480, material_override=mat_clay)

    # Render ortográfico cenital
    cam_obj.data.type = "ORTHO"
    cam_obj.data.ortho_scale = 138.0
    cam_obj.location = (0.0, 0.0, 180.0)
    cam_obj.rotation_euler = (0.0, 0.0, 0.0)
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024
    scene.render.filepath = str(out_dir / "audit_ortho.png")
    bpy.ops.render.render(write_still=True)

    xs = [v.co.x for v in base_obj.data.vertices]
    ys = [v.co.y for v in base_obj.data.vertices]
    zs = [v.co.z for v in base_obj.data.vertices]
    metrics = {
        **topo,
        "cliff_vertices_3d_sculpted": cliff_verts_sculpted,
        "demo_trees_requested": args.demo_trees,
        "trees_placed": trees_placed,
        "bounds_mm": {
            "dx": round(max(xs) - min(xs), 4),
            "dy": round(max(ys) - min(ys), 4),
            "dz": round(max(zs) - min(zs), 4),
        },
        "master_stl": str(master_stl),
        "viewer_glb": str(viewer_glb),
    }
    (out_dir / "blender_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("BLENDER_METRICS:", json.dumps(metrics))


if __name__ == "__main__":
    main()
