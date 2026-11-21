---
name: parametric-brep-cad
description: >-
  Diseño CAD paramétrico exacto con build123d (kernel OCCT B-Rep, exportación
  STEP + STL) y OpenSCAD Nightly con backend Manifold + librería BOSL2. Usar para
  piezas mecánicas, carcasas, encajes, roscas, imanes, soportes, organizadores
  Gridfinity y bases exactas de losetas Catán.
---

# CAD Paramétrico B-Rep (`build123d`) y CSG Rápido (`OpenSCAD + BOSL2`)

Utiliza esta skill siempre que el objeto necesite **medidas milimétricas exactas, chaflanes limpios, filetes constantes, agujeros cilíndricos perfectos, roscas o exportación `.step` editable**.

## 1. Cuándo Elegir `build123d` vs. `OpenSCAD Nightly + BOSL2`

- **`build123d` (Python OCCT B-Rep — Por defecto para ingeniería y bases Catán):**
  - Exporta **STEP** (sólidos analíticos exactos sin facetas) además de STL teselado fino.
  - Perfecto para filetes/chaflanes complejos (`fillet`, `chamfer`), barridos (`sweep`), secciones (`loft`), aserciones topológicas (`solid.is_valid`, `solid.volume`, `solid.bounding_box()`) e integración directa con scripts Python y servidor MCP `build123d`.
- **`OpenSCAD Nightly` (`--backend=Manifold`) + `BOSL2` (`C:\Users\migue\Documents\OpenSCAD\libraries\BOSL2`):**
  - Ideal para librerías mecánicas rápidas: roscas estándar (`threading.scad`), bisagras print-in-place, rejillas, snap-fits, `cuboid(..., rounding=..., chamfer=...)`, cajas parametrizadas y compilaciones en milisegundos vía `--backend=Manifold` o servidor MCP `openscad`.

---

## 2. Flujo con Servidores MCP Activos

- **`build123d` MCP (`build123d-mcp`):**
  - Permite ejecutar código `build123d`, inspeccionar medidas (`measure`), validar topología (`validate`, `analyze_printability`) y renderizar vistas previas (`render_view`) en una sesión interactiva.
- **`openscad` MCP (`openscad-mcp`):**
  - Configurado con `OPENSCAD_PATH` hacia `openscad.com` (`2026.11.21` con `has_manifold_backend: true`) y `OPENSCADPATH` hacia `BOSL2`.
  - Dispone de `check_openscad`, `get_libraries`, `render`, `measure` y `export_model` (STL/3MF).

---

## 3. Reglas de Diseño B-Rep Imprimible (FDM)

Consulta la tabla completa en [cad-invariants.md](./references/cad-invariants.md). Reglas clave:

1. **Pata de elefante:** Todo contorno de apoyo en cama que encaje dentro de otra pieza o requiera arista inferior limpia debe llevar `chamfer` de `0.6–0.8 mm` a `45°` en la base `Z = 0`.
2. **Alojamientos de imanes y tornillos:**
   - Imán neodimio `10×5 mm` → hueco `Ø10.20 × 5.20 mm` (o perfil hexagonal circunscrito anti-rebose) + micro-chaflán de entrada `0.4 mm`.
   - Orificios pasantes M3 → `Ø3.40 mm`; inserto térmico M3 → `Ø4.00 mm` (prof. `5.0 mm`); tornillo autorroscante M3 → `Ø2.85 mm`.
3. **Aserciones OCCT obligatorias antes de exportar:**
   ```python
   assert solid.is_valid, "Sólido B-Rep inválido"
   bb = solid.bounding_box()
   ```
4. **Script de referencia Catán B-Rep:**
   ```powershell
   python tools/catan_kit/build_catan_base_brep.py --magnet-shape circle --out-dir <dir_salida>
   ```
   Genera `catan_base_brep.step`, `catan_base_brep.stl` y `catan_base_metrics.json` verificados contra `110.8513 × 128.0000 × 25.0000 ± 0.01 mm`.
5. **Compilación directa OpenSCAD + BOSL2 por CLI:**
   ```powershell
   openscad.com --backend=Manifold -o salida.stl modelo.scad
   ```
