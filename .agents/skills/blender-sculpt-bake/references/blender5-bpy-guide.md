# Guía Verificada de la API `bpy` — Blender 5.3.0 (Maker3D Windows)

Notas comprobadas directamente sobre el ejecutable instalado en `C:\Users\migue\AppData\Local\Programs\Blender5.3\blender-5.3.0-windows-x64\blender.exe`.

## 1. CLI Headless y Comportamiento en Windows

- **Usar siempre `--background` (`-b`):** En invocaciones automatizadas pasa siempre `--background` antes de `--python` o `--python-expr` (incluyendo `--background --version` si se consulta desde subprocess con timeout generoso en primer arranque).
- **Addon MCP:** `blender_mcp` está instalado en `%APPDATA%\Blender Foundation\Blender\5.3\scripts\addons\blender_mcp.py` y activado en `userpref.blend` junto con `cycles`.

## 2. Operadores I/O de Mallas Verificados

| Formato | Importar en Blender 5.3 | Exportar en Blender 5.3 |
|---|---|---|
| **STL** | `bpy.ops.wm.stl_import(filepath=...)` | `bpy.ops.wm.stl_export(filepath=..., export_selected_objects=True, ascii_format=False)` |
| **glTF / GLB** | `bpy.ops.import_scene.gltf(filepath=...)` | `bpy.ops.export_scene.gltf(filepath=..., export_format='GLB', use_selection=True, export_apply=True)` |

*(Nota: los viejos operadores `bpy.ops.import_mesh.stl` y `export_mesh.stl` fueron retirados; usar exclusivamente `bpy.ops.wm.stl_import` y `bpy.ops.wm.stl_export`).*

## 3. Materiales PBR (`Principled BSDF`) en Blender 5.3

- `Material.use_nodes` emite `DeprecationWarning` (prevista su eliminación en 6.0); en 5.3 los materiales ya traen árbol de nodos o admiten `if hasattr(mat, "use_nodes") and not mat.use_nodes: mat.use_nodes = True`.
- Nombres exactos verificados de sockets de entrada de **`Principled BSDF`**:
  - `'Base Color'`, `'Metallic'`, `'Roughness'`, `'IOR'`, `'Alpha'`, `'Normal'`, `'Specular IOR Level'`, `'Coat Weight'`, `'Coat Roughness'`, `'Emission Color'`, `'Emission Strength'`.
  - *(Atención: en versiones recientes no se llama `'Specular'`, sino `'Specular IOR Level'`.)*

## 4. Renderizado Headless Verificado (`CYCLES` CPU)

- Sin GPU NVIDIA CUDA ni monitor OpenGL activo, **`CYCLES` con `device = 'CPU'`** renderiza imágenes PNG de forma totalmente determinista y estable en `--background`:
  ```python
  scene.render.engine = 'CYCLES'
  scene.cycles.device = 'CPU'
  scene.cycles.samples = 16
  if hasattr(scene.cycles, "use_denoising"):
      scene.cycles.use_denoising = True
  ```
- Para renders `Clay` rápidos y nítidos en auditoría topológica, asignar material override neutro mate (`Roughness = 0.65`, `Base Color = (0.72, 0.70, 0.66, 1.0)`) usando `view_layer.material_override = clay_mat` temporalmente sobre `CYCLES` CPU.

## 5. Trampas Verificadas de Materiales y Python Embebido (Blender 5.3)

- **Preservar `material_index` al limpiar slots:** llamar a `obj.data.materials.clear()` o `me.materials.clear()` reinicia a `0` todos los índices de material por polígono. Guardar siempre antes con `poly_mat = np.empty(len(me.polygons), dtype=np.int32); me.polygons.foreach_get("material_index", poly_mat)` y restaurar con `foreach_set("material_index", poly_mat.tolist())` tras añadir los nuevos materiales.
- **Sin `Pillow` (`PIL`) en el Python de Blender:** el intérprete embebido de Blender `5.3` no trae `PIL`. Cualquier ensamblado de mosaicos PNG de auditoría debe ejecutarse en Python del sistema (o protegerse con `try/except ImportError` dentro del script de Blender).
