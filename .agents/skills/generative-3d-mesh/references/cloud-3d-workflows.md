# Workflows Image-to-3D en la Nube (`cloud_3d_bridge.py`)

## 1. Proveedores Soportados

### `hf-zero` (Gratuito, sin clave obligatoria)
Usa `gradio_client` contra Spaces públicos con GPU H200/A100 ZeroGPU en Hugging Face. Si un Space está temporalmente saturado o en cola, puedes reintentar con `--model hunyuan3d` o `--model hi3dgen`, o añadir un `HF_TOKEN` gratuito en `.env` para ampliar cuota de ZeroGPU.

- **`trellis2`**: Recomendado para props arquitectónicos, miniaturas estilizadas y texturas PBR limpias.
- **`hunyuan3d`**: Muy sólido en personajes, criaturas y formas orgánicas completas.
- **`hi3dgen`**: Especializado en alta nitidez geométrica y bajorrelieves/esculturas con detalles finos.

### `fal` (Comercial rápido por uso, requiere `FAL_KEY` en `.env`)
- Latencia muy baja y sin colas públicas. Soporta `trellis`, `hunyuan3d` y `meshy`.
- Configura `FAL_KEY=...` en `.env` (partiendo de `.env.example`, nunca subas `.env` a Git).

### `replicate` (Requiere `REPLICATE_API_TOKEN` en `.env`)
- Ejecuta modelos Trellis/Hunyuan versionados vía API de Replicate.

## 2. Buenas Prácticas de Entrada (Prompting para `image-generator`)

Para maximizar la fidelidad geométrica en Image-to-3D:
1. Pide **silueta cerrada y base sólida** (evita capas finas volando, espadas ultrafinas sin grosor o partículas sueltas).
2. Usa iluminación suave de estudio sin alto contraste de sombras negras duras (ayuda a que el modelo 3D estime el albedo sin manchar cavidades ni confundir sombras con huecos).
3. Para árboles/edificios que van sobre Catán, genera props con **troncos/muros robustos (`>= 1.8 mm` al escalar a 25–45 mm de alto)**.
