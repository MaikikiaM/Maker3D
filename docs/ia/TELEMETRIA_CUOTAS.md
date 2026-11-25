# Telemetría de Cuotas y Rendimiento — Maker3D

Registro estructurado y verificable del consumo real medido en cada plataforma durante el desarrollo del proyecto (para planificar lotes, elegir modelo y evitar agotar límites antes de cerrar una iteración).

## 1. Generación de Imágenes (`Gemini Desktop App` / API `gemini-3.1-flash-image`)

| Fecha | Flujo / Carga de Trabajo | Volumen | Consumo Medido | Media por Imagen / Notas |
| :--- | :--- | :--- | :--- | :--- |
| `24/11/2026` | **Gemini Desktop App (`Gemini.exe`)** — Método **3 imágenes** de entrada (`_pbr` + `_clay` + `patch_B2_150_nb.jpg` como ref.) + prompt `v05-macro8k` del PDF (`Prompts_Macro_Montana_Hero360_v05.pdf`), **100% a la primera** sin reintentos. | **`15` imágenes** (`15/16` parches; `B2_150` conservado) | **`91%` del uso disponible** | **`~6.07%` / generación** (`1` sola ronda sin fallos consume prácticamente el cupo completo). |
| `22/11/2026` | **API subagente `image-generator` (`gemini-3.1-flash-image`)** — Entradas de 2 imágenes (`2048×2048` `_pbr` + `_clay`). | `9` aceptadas + reintentos | **`100%` (`429 RESOURCE_EXHAUSTED`)** | Alcanzó límite de cuota antes de completar el bloque 3/4. |

**Regla operativa derivada:** Con el método de 3 imágenes + PDF bien afinado, **`15` vistas completas caben justas en ~1 ciclo (`91%`) siempre que vayan a la primera**. No gastar tiros sueltos en el chat ni re-tirar vistas que ya cumplen el mínimo (`B2_150` / `B3_210` `hp_std ~ 0.126–0.131`).

---

## 2. Agentes y Modelos de Razonamiento (`Antigravity` — Suscripción Google AI Pro)

| Fecha | Modelo / Modo | Consumo Medido por Interacción Inicial | Notas Prácticas |
| :--- | :--- | :--- | :--- |
| `24/11/2026` | **`Neon (Alto)`** | **`8%`** en la primera interacción | Reserva para decisiones críticas puntuales o evita usarlo turno a turno en chats largos con imágenes. |
| `24/11/2026` | **`Hydrogen (Alto)`** (`gemini-4-hydrogen-high`) | **`6%`** en la primera interacción (**`25%` menos** que Neon Alto) | Suficiente y probado en `Maker3D` (`F0–F5.3` + `Colab L4`). Guardar las `14–16` imágenes en `macro/nb/` y procesarlas en lote tras `/compact` o chat nuevo para no multiplicar el `6%`. |

---

## 3. Cómputo GPU Remoto (`Google Colab Pro` — Google AI Pro)

| Fecha | Recurso (`tools/colab/colab_api_runner.py`) | Especificaciones Verificadas | Notas |
| :--- | :--- | :--- | :--- |
| `24/11/2026` | **`L4` (`SHAPE_STANDARD`, prueba smoke)** | `NVIDIA L4` (`23,034 MiB` VRAM, `55,530,568 kB` / `~53 GiB` RAM) | Prioridad por defecto (`OPTIX` en Blender `5.3.0`) con apagado automático `DeleteRuntime` al acabar. Elegibles también: `A100 (HIGHMEM/STD)`, `G4 (STD)`, `T4 (HIGHMEM/STD)`. |
| `24/11/2026` | **`L4` (`--stage passB 8K` completo, `6.15M Δ`)** | Upload `449.8 MB` en `~1m 58s` (8 hilos); Blender `5.3.0 OPTIX` bake `8K` + `Sculpt3D` + `6` PBR + `6` Clay + Ortho en **`19m 03s`** (pico `21.59 GiB` RAM, `2.63 GiB` VRAM, `52 °C`); download `541.7 MB` en **`29s`** (8 hilos). | `DeleteRuntime` verificó `0` runtimes activos al terminar. Regla operativa: refrescar token OAuth automáticamente en `_api_headers` para ejecuciones `>= 60 min`. |
| `25/11/2026` | **`L4` (`v06 --stage gpu-bake-only` Zero-Idle, `6.15M Δ`)** | **F1 PC (`0` créd):** `3m 09s` (`16/16` RGB+HP + `VisCache` `0.40 MB` + tar `427.3 MB`). **F2 Drive PC (`0` créd):** `22s`. **F3 `L4 OPTIX`:** `gpu_elapsed_s = 728.0s` (`12m 08s`: Drive pull `50s` + Blender `36s` + `4×4 EMIT 8K` `5m 13s` + `AO 8K` `3m 01s` + `NORMAL 8K/ROUGH 4K` `1m 18s` + push `202 MB` `16s`); **`14m 10s` totales de runtime** (`~37%` menos vs tanda limpia `v05` y `~62%` menos vs total `v05`). **F4 PC (`0` créd):** `4m 33s` (`sculpt` + `STL`/`GLB` + renders). | `DeleteRuntime` ejecutado **antes** de descargar (`ACTIVE RUNTIMES: []` verificado en API). `0` crédito desperdiciado en red local ni CPU `BVHTree`/`STL`. |

### Auditoría de Eficiencia y Arquitectura Zero-Idle GPU (`v06+`)
- **Diagnóstico de desperdicio medido en `v05`:**
  1. Subida HTTP Base64 de `449.8 MB` y descarga de `541.7 MB` ejecutadas con la `NVIDIA L4` ya encendida (y un fallo previo de `timeout=30` en `POST /api/kernels` consumió `~14 min` de `L4` al `0%` GPU).
  2. El bucle mono-hilo CPU de `bvh.ray_cast` (`~12–15s × 16` cámaras), `sculpt_3d` (96 pasadas sobre `3.07M` vértices), exportación `STL` (`293 MB`) y `GLB` (`399 MB`) corrieron dentro del runtime GPU manteniendo la `L4` al `0%` (`~16.8 W`) casi todo el tramo.
- **Presupuesto objetivo con Arquitectura Zero-Idle de 4 Fases (`tools/colab/drive_staging.py` + `run_v05_passb_autonomous.py`):**
  1. **Fase 1 (`prep-local` PC):** alineación OpenCV + caché exacta `BVHTree` (`macro_vis_cache.npz` con `np.packbits`, `~1.2 MB`) + `gpu_job_scene.blend` -> **`0` créditos**.
  2. **Fase 2 (`Drive staging` PC, GPU apagada):** subida resumable a `Maker3D_Colab_Staging` (`drive.file`) -> **`0` créditos**.
  3. **Fase 3 (`L4` efímera `--stage gpu-bake-only`):** descarga interna Drive (`< 3s`) + exclusivamente pases `Cycles OptiX` (`~90–150s` GPU activa) + subida PNGs (`~75 MB`, `< 2s`) + `DeleteRuntime` inmediato -> **`~2–2.5 min` facturables (~88% de ahorro vs `v05`)**.
  4. **Fase 4 (`finalize-local` PC, GPU ya apagada):** descarga desde Drive + borrado temporal Drive + `sculpt_3d` anti-cráter + `STL`/`GLB`/auditoría -> **`0` créditos**.
