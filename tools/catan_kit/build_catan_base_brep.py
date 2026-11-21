"""
Generador B-Rep paramétrico exacto de la base hexagonal canónica de Catán
mediante build123d (kernel OpenCASCADE), con exportación STEP + STL estanco
y validación dimensional estricta.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from build123d import (
    Align,
    BuildPart,
    BuildSketch,
    Cylinder,
    Locations,
    Mode,
    Polygon,
    RegularPolygon,
    chamfer,
    export_step,
    export_stl,
    extrude,
    make_face,
)
import trimesh

R = 64.0000
APOTHEM = R * math.cos(math.radians(30.0))  # 55.42562584 mm
W_EXPECTED = 2.0 * APOTHEM                  # 110.85125168 mm
D_EXPECTED = 2.0 * R                        # 128.000000 mm
Z_BASE = 25.0000
CHAMFER_BOTTOM = 0.80
MAG_RADIUS = 5.10   # Ø10.20 mm
MAG_DEPTH = 5.20
MAG_CHAMFER = 0.40


def build_catan_base(magnet_shape: str = "circle"):
    """Construye la base hexagonal pointy-top sólida B-Rep con hueco inferior de imán."""
    hex_pts = [
        (R * math.cos(math.radians(30.0 + 60.0 * k)), R * math.sin(math.radians(30.0 + 60.0 * k)))
        for k in range(6)
    ]

    with BuildPart() as part:
        with BuildSketch():
            Polygon(*hex_pts, align=None)
        extrude(amount=Z_BASE)

        # Bisel perimetral inferior contra pata de elefante (solo borde exterior Z=0)
        bottom_outer_edges = [
            e for e in part.edges()
            if abs(e.center().Z) < 1e-3 and math.hypot(e.center().X, e.center().Y) > 20.0
        ]
        chamfer(bottom_outer_edges, length=CHAMFER_BOTTOM)

        # Alojamiento inferior centrado para imán 10x5 mm
        with Locations((0.0, 0.0, 0.0)):
            if magnet_shape == "hex":
                # Hexágono con apotema 5.10 mm (el imán Ø10 entra justo dejando esquinas para pegamento)
                circ_r = MAG_RADIUS / math.cos(math.radians(30.0))
                with BuildSketch() as mag_sk:
                    RegularPolygon(radius=circ_r, side_count=6, major_radius=True)
                extrude(mag_sk.sketch, amount=MAG_DEPTH, mode=Mode.SUBTRACT)
            else:
                Cylinder(
                    radius=MAG_RADIUS,
                    height=MAG_DEPTH,
                    align=(Align.CENTER, Align.CENTER, Align.MIN),
                    mode=Mode.SUBTRACT,
                )

        # Micro-chaflán de entrada en boca del hueco del imán (Z=0, radio interior)
        mag_rim_edges = [
            e for e in part.edges()
            if abs(e.center().Z) < 1e-3 and math.hypot(e.center().X, e.center().Y) < 10.0
        ]
        if mag_rim_edges:
            chamfer(mag_rim_edges, length=MAG_CHAMFER)

    solid = part.part
    valid_ok = solid.is_valid() if callable(solid.is_valid) else bool(solid.is_valid)
    assert valid_ok, "El sólido B-Rep generado no superó solid.is_valid"

    bb = solid.bounding_box()
    dx = float(bb.max.X - bb.min.X)
    dy = float(bb.max.Y - bb.min.Y)
    dz = float(bb.max.Z - bb.min.Z)
    assert abs(dx - W_EXPECTED) <= 0.01, f"Ancho X inesperado: {dx:.4f} vs {W_EXPECTED:.4f}"
    assert abs(dy - D_EXPECTED) <= 0.01, f"Largo Y inesperado: {dy:.4f} vs {D_EXPECTED:.4f}"
    assert abs(dz - Z_BASE) <= 0.01, f"Altura Z inesperada: {dz:.4f} vs {Z_BASE:.4f}"
    assert solid.volume > 200000.0, f"Volumen inesperado: {solid.volume}"
    return solid, {"dx_mm": dx, "dy_mm": dy, "dz_mm": dz, "volume_mm3": float(solid.volume)}


def build_catan_inner_cutter(road_margin: float = 3.50, pad_radius: float = 10.00, z_min: float = 20.0, z_max: float = 120.0):
    """Construye el troquel sólido B-Rep del área escultórica interior (respetando carril 3.5mm y 6 pads Ø20mm)."""
    inner_apothem = APOTHEM - road_margin
    inner_r = inner_apothem / math.cos(math.radians(30.0))
    inner_pts = [
        (inner_r * math.cos(math.radians(30.0 + 60.0 * k)), inner_r * math.sin(math.radians(30.0 + 60.0 * k)))
        for k in range(6)
    ]
    h_cutter = z_max - z_min
    with BuildPart() as part:
        with Locations((0.0, 0.0, z_min)):
            with BuildSketch():
                Polygon(*inner_pts, align=None)
            extrude(amount=h_cutter)
        pad_pts = [
            (R * math.cos(math.radians(30.0 + 60.0 * k)), R * math.sin(math.radians(30.0 + 60.0 * k)), z_min)
            for k in range(6)
        ]
        with Locations(*pad_pts):
            Cylinder(
                radius=pad_radius,
                height=h_cutter + 2.0,
                align=(Align.CENTER, Align.CENTER, Align.MIN),
                mode=Mode.SUBTRACT,
            )
    cutter = part.part
    valid_ok = cutter.is_valid() if callable(cutter.is_valid) else bool(cutter.is_valid)
    assert valid_ok, "Troquel interior B-Rep inválido"
    return cutter


def main() -> None:
    parser = argparse.ArgumentParser(description="Constructor B-Rep build123d de base hexagonal Catán")
    parser.add_argument("--out-dir", required=True, help="Directorio de salida")
    parser.add_argument("--magnet-shape", choices=["circle", "hex"], default="circle", help="Geometría del hueco de imán")
    parser.add_argument("--prefix", default="catan_base_brep", help="Prefijo de archivos")
    parser.add_argument("--export-cutter", action="store_true", help="Exportar además catan_inner_cutter.stl")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    solid, stats = build_catan_base(magnet_shape=args.magnet_shape)
    step_path = out_dir / f"{args.prefix}.step"
    stl_path = out_dir / f"{args.prefix}.stl"
    metrics_path = out_dir / "catan_base_metrics.json"

    export_step(solid, str(step_path))
    export_stl(solid, str(stl_path), tolerance=0.01, angular_tolerance=0.08)

    if args.export_cutter:
        cutter = build_catan_inner_cutter()
        cutter_stl = out_dir / "catan_inner_cutter.stl"
        export_stl(cutter, str(cutter_stl), tolerance=0.01, angular_tolerance=0.06)
        c_mesh = trimesh.load_mesh(str(cutter_stl), force="mesh")
        assert c_mesh.is_watertight and int(c_mesh.euler_number) == 2, "Cutter STL no es estanco"

    mesh = trimesh.load_mesh(str(stl_path), force="mesh")
    assert mesh.is_watertight, "STL exportado desde OCCT no es watertight"
    assert int(mesh.euler_number) == 2, f"Euler inesperado en base STL: {mesh.euler_number}"
    degen = int((mesh.area_faces < 1e-10).sum())
    assert degen == 0, f"Caras degeneradas en base STL: {degen}"

    metrics = {
        "magnet_shape": args.magnet_shape,
        "occt_valid": True,
        **stats,
        "stl_faces": int(len(mesh.faces)),
        "stl_vertices": int(len(mesh.vertices)),
        "stl_watertight": bool(mesh.is_watertight),
        "stl_euler_chi": int(mesh.euler_number),
        "stl_degenerate_faces": degen,
        "step_path": str(step_path),
        "stl_path": str(stl_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
