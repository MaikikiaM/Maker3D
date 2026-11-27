"""
Servidor de telemetría en vivo (http://127.0.0.1:8799) estilo Colab Pro + Mission Control
para supervisar F5.3 -> F10 de cualquier loseta/bioma en tiempo real.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

REPO_ROOT = Path(__file__).resolve().parents[2]
TILES_ROOT = REPO_ROOT / "Tableros-Catan" / "losetas"

GALLERY_ITEMS = [
    ("Macro Grid 16 (F5.3)", "macro/macro_grid_16.jpg"),
    ("Albedo Final 8K", "baked_albedo.png"),
    ("Normal Final 8K", "baked_normal.png"),
    ("Roughness 4K", "baked_roughness.png"),
    ("AO Cavidad 8K", "macro/ao_cavity_8k.png"),
    ("Auditoría 6 Vistas PBR", "audit_6views_pbr.png"),
    ("Auditoría 6 Vistas Arcilla CAD", "audit_6views_clay.png"),
    ("Auditoría Ortográfica", "audit_ortho.png"),
]

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Maker3D — Monitor En Vivo Colab NVIDIA L4</title>
<style>
  :root {
    --bg: #0b0f17; --card: #131a26; --border: #233044; --text: #e5edf7;
    --muted: #8ea0b8; --blue: #3b82f6; --green: #22c55e; --amber: #f59e0b; --purple: #a855f7;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); padding: 18px; }
  .wrap { max-width: 1360px; margin: 0 auto; }
  .header { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px;
            background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 14px 18px; margin-bottom: 14px; }
  .title { font-size: 18px; font-weight: 700; }
  .sub { font-size: 13px; color: var(--muted); margin-top: 3px; }
  .badge { padding: 6px 12px; border-radius: 999px; font-size: 12px; font-weight: 700; background: #1e293b; border: 1px solid var(--border); }
  .badge.on { background: rgba(34,197,94,0.14); border-color: #22c55e; color: #86efac; }
  .badge.off { background: rgba(148,163,184,0.14); border-color: #64748b; color: #cbd5e1; }
  .badge.done { background: rgba(59,130,246,0.18); border-color: #3b82f6; color: #93c5fd; }
  .pbar-wrap { background: #0d131d; border: 1px solid var(--border); border-radius: 999px; height: 14px; overflow: hidden; margin-bottom: 14px; }
  .pbar { height: 100%; width: 0%; background: linear-gradient(90deg, #2563eb, #22c55e); transition: width 0.35s ease; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; margin-bottom: 14px; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 14px; }
  .ctitle { font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); margin-bottom: 6px; display: flex; justify-content: space-between; }
  .cval { font-size: 20px; font-weight: 700; margin-bottom: 6px; }
  .csub { font-size: 12px; color: var(--muted); margin-bottom: 8px; }
  .mini-bar { height: 7px; background: #0c111b; border-radius: 999px; overflow: hidden; margin-bottom: 8px; }
  .mini-fill { height: 100%; width: 0%; transition: width 0.3s ease; }
  svg.spark { width: 100%; height: 54px; background: #0b1019; border-radius: 6px; border: 1px solid #1b2638; }
  .panels { display: grid; grid-template-columns: 1.15fr 1fr; gap: 12px; }
  @media (max-width: 980px) { .panels { grid-template-columns: 1fr; } }
  pre.console { background: #070a10; border: 1px solid var(--border); border-radius: 8px; padding: 12px; height: 430px;
                overflow-y: auto; font-family: 'Consolas', monospace; font-size: 12px; line-height: 1.42; color: #cbd5e1; margin: 0; white-space: pre-wrap; word-break: break-word; }
  .gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 10px; max-height: 430px; overflow-y: auto; }
  .gitem { background: #0b1019; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
  .gitem img { width: 100%; height: 150px; object-fit: cover; display: block; cursor: pointer; }
  .gcap { padding: 6px 8px; font-size: 11px; color: var(--muted); }
</style>
</head>
<body>
<div class="wrap">
  <div class="header">
    <div>
      <div class="title">Maker3D Mission Control — <span id="phase">Iniciando...</span></div>
      <div class="sub">Detalle: <span id="detail">Preparando canalización</span> &nbsp;|&nbsp; Progreso: <strong id="pct">0%</strong> &nbsp;|&nbsp; Transcurrido: <strong id="elapsed">00:00</strong></div>
    </div>
    <div id="rtBadge" class="badge off">RUNTIME: DESCONECTADO</div>
  </div>

  <div class="pbar-wrap"><div id="pbar" class="pbar"></div></div>

  <div class="grid">
    <div class="card">
      <div class="ctitle"><span>RAM Sistema Colab</span><span id="ramPct">-</span></div>
      <div class="cval" id="ramVal">-- / -- GiB</div>
      <div class="csub" id="ramSub">Esperando telemetría de kernel...</div>
      <div class="mini-bar"><div id="ramBar" class="mini-fill" style="background: var(--blue);"></div></div>
      <svg id="ramSpark" class="spark" viewBox="0 0 260 54"></svg>
    </div>

    <div class="card">
      <div class="ctitle"><span>GPU &amp; VRAM (NVIDIA L4)</span><span id="gpuUtil">-</span></div>
      <div class="cval" id="vramVal">-- / -- GiB</div>
      <div class="csub" id="gpuSub">Temp: -- °C &nbsp;|&nbsp; Potencia: -- W</div>
      <div class="mini-bar"><div id="vramBar" class="mini-fill" style="background: var(--green);"></div></div>
      <svg id="gpuSpark" class="spark" viewBox="0 0 260 54"></svg>
    </div>

    <div class="card">
      <div class="ctitle"><span>Espacio en Disco (/content)</span><span id="diskPct">-</span></div>
      <div class="cval" id="diskVal">-- / -- GiB</div>
      <div class="csub" id="diskSub">Libre: -- GiB</div>
      <div class="mini-bar"><div id="diskBar" class="mini-fill" style="background: var(--amber);"></div></div>
      <svg id="diskSpark" class="spark" viewBox="0 0 260 54"></svg>
    </div>

    <div class="card">
      <div class="ctitle"><span>Transferencia &amp; Cycles OptiX</span><span id="xferPct">-</span></div>
      <div class="cval" id="xferVal">0.0 / 0.0 MB</div>
      <div class="csub" id="cyclesSub">Samples: -- &nbsp;|&nbsp; GPU: --</div>
      <div class="mini-bar"><div id="xferBar" class="mini-fill" style="background: var(--purple);"></div></div>
      <svg id="xferSpark" class="spark" viewBox="0 0 260 54"></svg>
    </div>
  </div>

  <div class="panels">
    <div class="card">
      <div class="ctitle"><span>Consola en Directo</span><span id="logLines">0 líneas</span></div>
      <pre id="console" class="console">Esperando salida del proceso...</pre>
    </div>
    <div class="card">
      <div class="ctitle"><span>Galería Automática de Texturas y Renders</span><span id="galCount">0 archivos</span></div>
      <div id="gallery" class="gallery"></div>
    </div>
  </div>
</div>

<script>
const HIST = { ram: [], gpu: [], disk: [], xfer: [] };
function pushHist(arr, v) {
  if (typeof v !== 'number' || Number.isNaN(v)) return;
  arr.push(Math.max(0, Math.min(100, v)));
  if (arr.length > 60) arr.shift();
}
function drawSpark(id, arr, color) {
  const svg = document.getElementById(id);
  if (!arr.length) { svg.innerHTML = ''; return; }
  const pts = arr.map((v, i) => {
    const x = arr.length === 1 ? 260 : (i / (arr.length - 1)) * 260;
    const y = 50 - (v / 100) * 44;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
  svg.innerHTML = `<polyline fill="none" stroke="${color}" stroke-width="2" points="${pts}" />`;
}
function fmtTime(s) {
  const m = Math.floor(s / 60);
  const r = Math.floor(s % 60);
  return String(m).padStart(2, '0') + ':' + String(r).padStart(2, '0');
}
async function tick() {
  try {
    const res = await fetch('/api/state?_=' + Date.now());
    const d = await res.json();
    const t = d.telemetry || {};
    document.getElementById('phase').textContent = t.phase || 'Esperando inicio';
    document.getElementById('detail').textContent = t.detail || '';
    const pct = Number(t.progress_pct || 0);
    document.getElementById('pct').textContent = pct.toFixed(0) + '%';
    document.getElementById('pbar').style.width = pct + '%';
    document.getElementById('elapsed').textContent = fmtTime(t.elapsed_s || 0);

    const rb = document.getElementById('rtBadge');
    const st = t.runtime_status || 'INACTIVO';
    rb.textContent = st;
    rb.className = 'badge ' + (t.completed ? 'done' : (t.runtime_active ? 'on' : 'off'));

    const ramU = Number(t.ram_used_gib || 0), ramT = Number(t.ram_total_gib || 0);
    const ramP = ramT > 0 ? (ramU / ramT) * 100 : 0;
    document.getElementById('ramVal').textContent = ramT > 0 ? `${ramU.toFixed(1)} / ${ramT.toFixed(1)} GiB` : '-- / -- GiB';
    document.getElementById('ramPct').textContent = ramT > 0 ? ramP.toFixed(0) + '%' : '-';
    document.getElementById('ramBar').style.width = ramP + '%';
    document.getElementById('ramSub').textContent = t.runtime_active ? `Libre: ${Math.max(0, ramT - ramU).toFixed(1)} GiB` : 'Runtime en espera o apagado';

    const vrU = Number(t.vram_used_gib || 0), vrT = Number(t.vram_total_gib || 0);
    const gpuU = Number(t.gpu_util_pct || 0);
    const vrP = vrT > 0 ? (vrU / vrT) * 100 : 0;
    document.getElementById('vramVal').textContent = vrT > 0 ? `${vrU.toFixed(2)} / ${vrT.toFixed(1)} GiB` : '-- / -- GiB';
    document.getElementById('gpuUtil').textContent = vrT > 0 ? `GPU ${gpuU.toFixed(0)}%` : '-';
    document.getElementById('vramBar').style.width = vrP + '%';
    document.getElementById('gpuSub').textContent = `Temp: ${t.gpu_temp_c ?? '--'} °C | Potencia: ${t.gpu_power_w ?? '--'} W`;

    const dkU = Number(t.disk_used_gib || 0), dkT = Number(t.disk_total_gib || 0), dkF = Number(t.disk_free_gib || 0);
    const dkP = dkT > 0 ? (dkU / dkT) * 100 : 0;
    document.getElementById('diskVal').textContent = dkT > 0 ? `${dkU.toFixed(1)} / ${dkT.toFixed(1)} GiB` : '-- / -- GiB';
    document.getElementById('diskPct').textContent = dkT > 0 ? dkP.toFixed(0) + '%' : '-';
    document.getElementById('diskBar').style.width = dkP + '%';
    document.getElementById('diskSub').textContent = dkT > 0 ? `Libre en /content: ${dkF.toFixed(1)} GiB` : 'Esperando disco remoto...';

    const xfU = Number(t.xfer_done_mb || 0), xfT = Number(t.xfer_total_mb || 0);
    const xfP = xfT > 0 ? (xfU / xfT) * 100 : 0;
    document.getElementById('xferVal').textContent = `${xfU.toFixed(1)} / ${xfT.toFixed(1)} MB`;
    document.getElementById('xferPct').textContent = xfT > 0 ? xfP.toFixed(0) + '%' : '-';
    document.getElementById('xferBar').style.width = xfP + '%';
    document.getElementById('cyclesSub').textContent = `Blender: ${t.blender_stage || '--'} | GPU: ${t.gpu_name || '--'}`;

    if (ramT > 0) pushHist(HIST.ram, ramP);
    if (vrT > 0) pushHist(HIST.gpu, gpuU);
    if (dkT > 0) pushHist(HIST.disk, dkP);
    pushHist(HIST.xfer, pct);
    drawSpark('ramSpark', HIST.ram, '#3b82f6');
    drawSpark('gpuSpark', HIST.gpu, '#22c55e');
    drawSpark('diskSpark', HIST.disk, '#f59e0b');
    drawSpark('xferSpark', HIST.xfer, '#a855f7');

    const lines = d.console_lines || [];
    const pre = document.getElementById('console');
    const atBottom = (pre.scrollHeight - pre.scrollTop - pre.clientHeight) < 40;
    pre.textContent = lines.join('\n') || 'Esperando salida...';
    document.getElementById('logLines').textContent = lines.length + ' líneas';
    if (atBottom) pre.scrollTop = pre.scrollHeight;

    const gal = document.getElementById('gallery');
    const imgs = d.gallery || [];
    document.getElementById('galCount').textContent = imgs.length + ' archivos';
    const sig = imgs.map(i => i.path + ':' + i.mtime).join('|');
    if (gal.dataset.sig !== sig) {
      gal.dataset.sig = sig;
      gal.innerHTML = imgs.map(i => `
        <div class="gitem">
          <a href="/files/${i.path}?t=${i.mtime}" target="_blank">
            <img src="/files/${i.path}?t=${i.mtime}" alt="${i.label}">
          </a>
          <div class="gcap"><strong>${i.label}</strong> (${(i.size_kb/1024).toFixed(2)} MB)</div>
        </div>
      `).join('');
    }
  } catch (e) {}
}
tick();
setInterval(tick, 1000);
</script>
</body>
</html>
"""


class MonitorHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        return

    def _send_json(self, obj: dict, code: int = 200) -> None:
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        try:
            self._handle_get()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return

    def _handle_get(self) -> None:
        path = unquote(self.path.split("?", 1)[0])
        if path == "/" or path == "/index.html":
            raw = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        if path == "/api/state":
            tel = {}
            if TELEMETRY_FILE.exists():
                try:
                    tel = json.loads(TELEMETRY_FILE.read_text(encoding="utf-8"))
                except Exception:
                    pass
            lines = []
            if CONSOLE_FILE.exists():
                try:
                    lines = CONSOLE_FILE.read_text(encoding="utf-8", errors="replace").splitlines()[-120:]
                except Exception:
                    pass
            gal = []
            for label, rel in GALLERY_ITEMS:
                fp = (WORK_DIR / rel).resolve()
                if fp.exists() and fp.is_file():
                    st = fp.stat()
                    gal.append({
                        "label": label,
                        "path": rel,
                        "mtime": int(st.st_mtime),
                        "size_kb": round(st.st_size / 1024.0, 1),
                    })
            self._send_json({"telemetry": tel, "console_lines": lines, "gallery": gal})
            return

        if path.startswith("/files/"):
            rel = path[len("/files/"):].lstrip("/")
            fp = (WORK_DIR / rel).resolve()
            if not str(fp).startswith(str(WORK_DIR)) or not fp.exists() or not fp.is_file():
                self.send_error(404)
                return
            mime, _ = mimetypes.guess_type(str(fp))
            raw = fp.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mime or "application/octet-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        self.send_error(404)


def main() -> None:
    global WORK_DIR, TELEMETRY_FILE, CONSOLE_FILE
    p = argparse.ArgumentParser(description="Monitor en vivo Colab NVIDIA L4")
    p.add_argument("--tile", default=None, help="Bioma en Tableros-Catan/losetas/<tile>/work")
    p.add_argument("--work-dir", default=None, help="Ruta explícita al directorio work/ de la loseta")
    p.add_argument("--port", type=int, default=8799)
    args = p.parse_args()
    if args.work_dir:
        WORK_DIR = Path(args.work_dir).resolve()
    elif args.tile:
        WORK_DIR = (TILES_ROOT / args.tile / "work").resolve()
    else:
        p.error("Debes especificar --tile <bioma> o --work-dir <ruta>.")
    TELEMETRY_FILE = WORK_DIR / "live_telemetry.json"
    CONSOLE_FILE = WORK_DIR / "live_console.log"
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), MonitorHandler)
    print(f"[LiveMonitor] Sirviendo telemetría de {WORK_DIR} en http://localhost:{args.port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
