---
name: generative-3d-mesh
description: >-
  Generación de mallas 3D orgánicas y figurativas a partir de arte conceptual de
  Nano Banana (image-generator) conectado mediante tools/generative/cloud_3d_bridge.py
  a modelos Image-to-3D nativos en la nube (Hugging Face ZeroGPU gratuito:
  TRELLIS.2, Hunyuan3D, Hi3DGen; o APIs opcionales fal.ai y Replicate).
---

# Generación de Mallas 3D Nativas (`Nano Banana` → `Cloud 3D Bridge`)

Esta skill reemplaza la extrusión ciega `2.5D` para **figuras, miniaturas, árboles, criaturas, rocas complejas con voladizos, estatuas, bustos y props arquitectónicos** que requieren geometría 3D completa de 360 grados.

> **Hardware local:** Este equipo dispone de iGPU AMD Radeon 860M (sin GPU NVIDIA CUDA discreta). Por ello, la inferencia neural Image-to-3D pesada se delega limpiamente mediante `tools/generative/cloud_3d_bridge.py`.

---

## 1. Flujo en 4 Pasos (`Concept → 3D Mesh → Clean → Fuse`)

### Paso 1: Render de Referencia Limpio con Nano Banana (`image-generator`)
Genera una imagen **óptima para reconstrucción Image-to-3D** antes de llamar al puente:
- **Fondo neutro liso** (blanco puro o gris uniforme, sin suelo ni sombras arrojadas largas).
- **Objeto único completo y centrado**, en ángulo 3/4 ligeramente elevado o vista frontal ortográfica limpia, sin recortes en los bordes.
- **Sin texto, cotas ni marcas de agua**.
- Inspecciona la imagen con `read_file` antes de enviarla a 3D.

### Paso 2: Conversión Image-to-3D vía `cloud_3d_bridge.py`
Comprueba primero qué proveedores están listos:
```powershell
python tools/generative/cloud_3d_bridge.py --doctor
```

- **Opción gratuita por defecto (Hugging Face ZeroGPU via `gradio_client`):**
  ```powershell
  python tools/generative/cloud_3d_bridge.py `
    --provider hf-zero `
    --model trellis2 `
    --image <ruta_imagen.png> `
    --out <ruta_salida.glb>
  ```
  Modelos soportados en `--provider hf-zero`: `trellis2`, `hunyuan3d`, `hi3dgen` (ver detalles en [cloud-3d-workflows.md](./references/cloud-3d-workflows.md)).
- **Opción rápida / sin cola con clave API en `.env` (`fal.ai` / `Replicate`):**
  ```powershell
  python tools/generative/cloud_3d_bridge.py `
    --provider fal `
    --model trellis `
    --image <ruta_imagen.png> `
    --out <ruta_salida.glb>
  ```

### Paso 3: Normalización, Cierre y Escala Física (Nunca usar la malla cruda directamente)
Las mallas generadas por IA suelen llegar en escala normalizada (`~1 unidad`), rotadas `Y-up`, con caras flotantes o paredes finas:
1. Importa el `.glb` en Blender 5.3 (`blender-sculpt-bake`) o repáralo con `tools/audit/mesh_and_print_audit.py --repair`.
2. Orienta a `Z-up`, apoya la base sobre `Z = 0` (o fúndela dentro de la superficie base CAD), escala a milímetros reales y refuerza salientes débiles `< 1.2 mm` con `Voxel Remesh` + `Sculpt` en Blender antes de imprimir.

### Paso 4: Fusión con Base CAD y Auditoría
Si el activo forma parte de una loseta Catán o peana de miniatura, únelo de forma estanca a la base B-Rep usando `tools/blender/sculpt_fuse_bake_render.py --props-glb <asset.glb>` y audítalo con Bambu Studio CLI.
