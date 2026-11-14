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
- **Skill de esculpido:** Seguir `.agents/skills/nanobanana-3d-sculpt/SKILL.md` para biomas/modelos orgánicos con Nano Banana, tangentes analíticas, suavizado Taubin y validación de mallas estancas (`watertight=True`, `chi=2`).
- **Visores HTML móviles:** Todo visor en `Visores-HTML/` debe mantenerse `< 50 MB` y verificado sin errores WebGL/consola en navegador headless.
- **Filtro Git estricto:** Nunca subir binarios pesados (`*.stl`, `*.3mf`, `*.obj`, `*.npz`, `*.npy`, carpetas `versiones/` locales) ni artefactos `> 50 MB` a GitHub; preservarlos localmente y en Google Drive.

---

## 4. Sincronización Proactiva con GitHub
- Tras concluir cada tarea funcional o corrección, ejecutar:
  ```powershell
  git add -A
  git commit -m "<tipo>(<alcance>): <descripción concisa en español>"
  git push origin main
  ```
