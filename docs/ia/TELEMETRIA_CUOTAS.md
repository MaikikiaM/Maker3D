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
| `24/11/2026` | **`L4` (`SHAPE_STANDARD`, headless OAuth2)** | `NVIDIA L4` (`23,034 MiB` VRAM, `55,530,568 kB` / `~53 GiB` RAM) | Prioridad por defecto (`OPTIX` en Blender `5.3.0`) con apagado automático `DeleteRuntime` al acabar. Elegibles también: `A100 (HIGHMEM/STD)`, `G4 (STD)`, `T4 (HIGHMEM/STD)`. |
