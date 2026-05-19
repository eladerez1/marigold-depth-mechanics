#!/usr/bin/env python3
"""Serve experiment figures + CSV summaries for browser viewing on laptop via SSH tunnel.

On laptop:
  ssh -L 8765:localhost:8765 dgx04
  open http://localhost:8765
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _read_csv_table(path: Path, max_rows: int = 30) -> str:
    if not path.exists():
        return "<p><em>No data yet</em></p>"
    rows = list(csv.DictReader(path.open()))
    if not rows:
        return "<p><em>Empty CSV</em></p>"
    cols = list(rows[0].keys())
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = ""
    for row in rows[:max_rows]:
        body += "<tr>" + "".join(f"<td>{html.escape(str(row[c]))}</td>" for c in cols) + "</tr>"
    extra = f"<p><em>… {len(rows) - max_rows} more rows</em></p>" if len(rows) > max_rows else ""
    return f"<table><tr>{head}</tr>{body}</table>{extra}"


def build_page() -> bytes:
    fig_dir = RESULTS / "figures"
    figures = sorted(fig_dir.glob("*.png")) if fig_dir.exists() else []
    fig_html = ""
    for p in figures:
        rel = p.relative_to(ROOT)
        fig_html += (
            f'<section class="card"><h3>{html.escape(p.name)}</h3>'
            f'<img src="/file/{rel.as_posix()}" loading="lazy" /></section>'
        )

    tables = [
        ("Exp01 — weight delta", RESULTS / "exp01" / "weight_delta_by_layer.csv"),
        ("Exp02 — probing matrix", RESULTS / "exp02" / "probing_matrix.csv"),
        ("Exp03 — timestep curves", RESULTS / "exp03" / "timestep_curves.csv"),
        ("Exp04 — CKA", RESULTS / "exp04" / "cka_matrix.csv"),
    ]
    tbl_html = ""
    for title, path in tables:
        tbl_html += f'<section class="card"><h3>{title}</h3>{_read_csv_table(path)}</section>'

    status_block = ""
    status_path = RESULTS / "run_status.json"
    if status_path.exists():
        st = json.loads(status_path.read_text())
        status_block = f"<pre class='status'>{html.escape(json.dumps(st, indent=2))}</pre>"

    doc = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8"/>
<meta http-equiv="refresh" content="30"/>
<title>Marigold Depth Mechanics</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 1.5rem; background: #0f1117; color: #e6e6e6; }}
  h1 {{ color: #7eb8ff; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 1rem; }}
  .card {{ background: #1a1d27; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }}
  img {{ max-width: 100%; border-radius: 4px; }}
  table {{ border-collapse: collapse; font-size: 12px; width: 100%; }}
  th, td {{ border: 1px solid #333; padding: 4px 8px; }}
  th {{ background: #252836; }}
  .status {{ background: #111; padding: 1rem; overflow: auto; font-size: 12px; }}
</style>
</head><body>
<h1>Marigold depth mechanics — live results</h1>
<p>Auto-refresh 30s · Tunnel: <code>ssh -L 8765:localhost:8765 dgx04</code></p>
<p><a href="http://localhost:8766/" style="color:#7eb8ff">Denoising PCA viewer (port 8766)</a> — run <code>export_denoise_trajectory.py</code> first</p>
{status_block}
<h2>Figures</h2>
<div class="grid">{fig_html or "<p>No figures yet.</p>"}</div>
<h2>Tables</h2>
{tbl_html}
</body></html>"""
    return doc.encode()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        print(f"[viewer] {self.address_string()} {fmt % args}")

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            body = build_page()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/file/"):
            rel = self.path[len("/file/") :]
            fp = (ROOT / rel).resolve()
            if not str(fp).startswith(str(ROOT.resolve())) or not fp.is_file():
                self.send_error(404)
                return
            data = fp.read_bytes()
            ctype = mimetypes.guess_type(str(fp))[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_error(404)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", type=str, default="0.0.0.0")
    args = p.parse_args()
    server = HTTPServer((args.host, args.port), Handler)
    print(f"Serving {ROOT} at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
