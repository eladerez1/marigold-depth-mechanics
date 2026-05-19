#!/usr/bin/env python3
"""Interactive denoising trajectory viewer (PCA features + depth per timestep).

Precompute frames:
  python scripts/export_denoise_trajectory.py --nyu_index 0 --n_steps 10 --gpu 0

On laptop:
  ssh -L 8766:localhost:8766 dgx04
  open http://localhost:8766
"""

from __future__ import annotations

import argparse
import html
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIZ_ROOT = ROOT / "results" / "denoise_viz"
VIZ_ROOT = DEFAULT_VIZ_ROOT


def _load_manifest() -> dict:
    path = VIZ_ROOT / "manifest.json"
    if not path.exists():
        return {"samples": []}
    return json.loads(path.read_text())


def _load_sample_meta(sample_id: str) -> dict | None:
    meta_path = VIZ_ROOT / sample_id / "meta.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text())


def build_page(sample_id: str | None, step_idx: int) -> bytes:
    manifest = _load_manifest()
    samples = manifest.get("samples", [])
    if not samples:
        body = "<p>No trajectories yet. Run <code>scripts/export_denoise_trajectory.py</code> on GPU.</p>"
        return _html_shell("Denoise viewer", body).encode()

    if sample_id is None:
        sample_id = samples[0]["id"]
    meta = _load_sample_meta(sample_id)
    if meta is None:
        body = f"<p>Missing meta for {html.escape(sample_id)}</p>"
        return _html_shell("Denoise viewer", body).encode()

    steps = meta["steps"]
    n = len(steps)
    step_idx = max(0, min(step_idx, n - 1))
    step = steps[step_idx]
    step_dir = step["dir"]
    t_val = step["timestep"]

    opts = ""
    for s in samples:
        sel = " selected" if s["id"] == sample_id else ""
        cap = html.escape(s.get("caption", s["id"]))
        opts += f'<option value="{html.escape(s["id"])}"{sel}>{html.escape(s["id"])} — {cap}</option>'

    sample_path = VIZ_ROOT / sample_id
    gt_block = ""
    if (sample_path / "gt_depth.png").exists():
        gt_block = (
            f'<figure><figcaption>GT depth</figcaption>'
            f'<img src="/viz/{sample_id}/gt_depth.png" /></figure>'
        )

    hook = html.escape(meta.get("hook_layer", "?"))
    doc = f"""
<p class="hint">PCA on <code>{hook}</code> · shared color scale across timesteps (PnP style)</p>
<div class="controls">
  <label>Sample <select id="sampleSel">{opts}</select></label>
  <label>Step <input type="range" id="stepSlider" min="0" max="{n - 1}" value="{step_idx}" />
    <span id="stepLabel">{step_idx + 1}/{n}</span></label>
  <span class="ts">t = {t_val}</span>
  <button type="button" id="prevBtn">←</button>
  <button type="button" id="nextBtn">→</button>
</div>
<div class="row">
  <figure><figcaption>Input RGB</figcaption>
    <img src="/viz/{sample_id}/rgb.jpg" /></figure>
  <figure><figcaption>PCA features</figcaption>
    <img id="pcaImg" src="/viz/{sample_id}/{step_dir}/pca.png" /></figure>
  <figure><figcaption>Decoded depth (latent → VAE)</figcaption>
    <img id="depthImg" src="/viz/{sample_id}/{step_dir}/depth.png" /></figure>
  {gt_block}
</div>
<script>
const meta = {json.dumps(meta)};
let stepIdx = {step_idx};
const sampleId = {json.dumps(sample_id)};

function showStep(i) {{
  stepIdx = Math.max(0, Math.min(i, meta.steps.length - 1));
  const s = meta.steps[stepIdx];
  document.getElementById('stepSlider').value = stepIdx;
  document.getElementById('stepLabel').textContent = (stepIdx + 1) + '/' + meta.steps.length;
  document.querySelector('.ts').textContent = 't = ' + s.timestep;
  document.getElementById('pcaImg').src = '/viz/' + sampleId + '/' + s.dir + '/pca.png?' + Date.now();
  document.getElementById('depthImg').src = '/viz/' + sampleId + '/' + s.dir + '/depth.png?' + Date.now();
}}

document.getElementById('stepSlider').oninput = (e) => showStep(parseInt(e.target.value, 10));
document.getElementById('prevBtn').onclick = () => showStep(stepIdx - 1);
document.getElementById('nextBtn').onclick = () => showStep(stepIdx + 1);
document.getElementById('sampleSel').onchange = (e) => {{
  window.location.search = '?sample=' + encodeURIComponent(e.target.value) + '&step=0';
}};
document.onkeydown = (e) => {{
  if (e.key === 'ArrowLeft') showStep(stepIdx - 1);
  if (e.key === 'ArrowRight') showStep(stepIdx + 1);
}};
</script>
"""
    return _html_shell("Denoise trajectory · PCA + depth", doc).encode()


def _html_shell(title: str, inner: str) -> str:
    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8"/>
<title>{html.escape(title)}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 1.5rem; background: #0f1117; color: #e8e8e8; }}
  h1 {{ color: #9fd4ff; font-size: 1.4rem; }}
  .hint {{ color: #aaa; font-size: 0.9rem; }}
  .controls {{ display: flex; flex-wrap: wrap; gap: 1rem; align-items: center; margin: 1rem 0; }}
  .controls label {{ display: flex; align-items: center; gap: 0.5rem; }}
  input[type=range] {{ width: 220px; }}
  .ts {{ font-weight: 600; color: #ffb86c; min-width: 5rem; }}
  .row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; }}
  figure {{ margin: 0; background: #1a1d27; border-radius: 8px; padding: 0.75rem; }}
  figcaption {{ font-size: 0.85rem; color: #9aa; margin-bottom: 0.5rem; }}
  img {{ width: 100%; border-radius: 4px; display: block; }}
  button {{ background: #2a3148; color: #fff; border: none; padding: 0.4rem 0.8rem; border-radius: 4px; cursor: pointer; }}
  a {{ color: #7eb8ff; }}
</style>
</head><body>
<h1>Denoising trajectory viewer</h1>
<p><a href="http://localhost:8765/">← Summary dashboard (8765)</a></p>
{inner}
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        print(f"[denoise-viewer] {self.address_string()} {fmt % args}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            qs = parse_qs(parsed.query)
            sample = qs.get("sample", [None])[0]
            step = int(qs.get("step", ["0"])[0])
            body = build_page(sample, step)
            self._send(200, "text/html; charset=utf-8", body)
            return

        if parsed.path.startswith("/viz/"):
            rel = parsed.path[len("/viz/") :]
            fp = (VIZ_ROOT / rel).resolve()
            if not str(fp).startswith(str(VIZ_ROOT.resolve())) or not fp.is_file():
                self.send_error(404)
                return
            data = fp.read_bytes()
            ctype = mimetypes.guess_type(str(fp))[0] or "application/octet-stream"
            self._send(200, ctype, data)
            return

        self.send_error(404)

    def _send(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--viz_root", type=str, default=str(DEFAULT_VIZ_ROOT))
    args = p.parse_args()
    global VIZ_ROOT
    VIZ_ROOT = Path(args.viz_root)
    server = HTTPServer((args.host, args.port), Handler)
    print(f"Denoise viewer: http://{args.host}:{args.port}  (data: {VIZ_ROOT})")
    server.serve_forever()


if __name__ == "__main__":
    main()
