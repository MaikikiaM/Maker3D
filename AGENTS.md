# Protocolo Operativo: Maker3D (Diseño Paramétrico e Impresión 3D)

Este proyecto opera bajo el Protocolo Global definido en `~/.gemini/config/AGENTS.md`.

---

## 1. Cabecera Temporal Obligatoria
Toda respuesta del asistente debe iniciar con:
```markdown
⏱️ **[Prompt: HH:MM:SS | Respuesta: HH:MM:SS]**
```

---

## 2. Memoria en Dos Niveles
- `docs/ia/ESTADO.md`: Resumen ejecutivo vivo de máximo 50 líneas (leer al inicio de cada sesión).
- `registro.txt`: Bitácora continua indexable e inmutable en la raíz del proyecto (`SESIÓN: DD/MM/AAAA - TÍTULO`).
  - **Regla obligatoria de prompts literales:** Cada entrada DEBE comenzar por `- Prompts literales del usuario:` copiando **palabra por palabra y sin resumir** cada mensaje enviado por el usuario con su fecha/hora (`  * [DD/MM/AAAA HH:MM]: "<texto literal exacto>"`).
  - **Prohibido sustituir por objetivos:** Nunca parafrasear ni reemplazar el prompt del usuario por `- Objetivo:`. Si se incluye un `- Objetivo:`, sólo puede aparecer *después* de `- Prompts literales del usuario:`. Tras compactación, recuperar los mensajes literales desde `transcript_full.jsonl` / `transcript.jsonl`.

---

## 3. Reglas Específicas de Maker3D
- **Enrutamiento y Skills Maker3D 2.0:** Usar siempre `.agents/skills/maker3d-router/SKILL.md` y `.agents/rules/maker3d-architecture.md` para clasificar cada nuevo diseño hacia el motor adecuado (`parametric-brep-cad`, `terrain-geology-engine`, `generative-3d-mesh`, `blender-sculpt-bake`, `print-audit-and-webgl`), manteniendo intacto `.agents/skills/nanobanana-3d-sculpt/SKILL.md` para las 8 losetas legacy de Catán. Verificar entorno con `python tools/env_check.py`.
- **Auditoría e Imprimibilidad:** Toda pieza entregada requiere validación estanca (`watertight=True`, `chi=2`, `degen=0`), inspección visual de renders PBR/Arcilla y validación geométrica con `tools/audit/mesh_and_print_audit.py` (**sin laminar con Bambu Studio por defecto**; usar `--bambu-slice` solo si el usuario pide explícitamente G-code o tiempos de laminado).
- **Visores HTML móviles:** Todo visor en `Visores-HTML/` debe generarse con compresión `Meshopt + WebP` (`tools/webgl/optimize_and_build_viewer.py`), mantenerse `< 50 MB` (objetivo `< 20 MB`) y verificarse sin errores WebGL/consola en navegador headless.
- **Filtro Git estricto:** Nunca subir binarios pesados (`*.stl`, `*.3mf`, `*.obj`, `*.glb`, `*.step`, `*.blend`, `*.npz`, `*.npy`, carpetas `versiones/` locales, `node_modules/`, `tools/smoke_test_out/`) ni artefactos `> 50 MB` ni secretos (`.env`) a GitHub; preservarlos localmente y en Google Drive.

---

## 4. Sincronización Proactiva con GitHub
- Tras concluir cada tarea funcional o corrección, ejecutar:
  ```powershell
  git add -A
  git commit -m "<tipo>(<alcance>): <descripción concisa en español>"
  git push origin main
  ```
