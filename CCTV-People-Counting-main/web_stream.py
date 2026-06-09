"""
web_stream.py
Flask web server — single-page dashboard with:
  - Live MJPEG video (combined_view from main.py)
  - Grafana panels embedded as iframes
"""

import cv2
import threading
import time
import os
from flask import Flask, Response

app = Flask(__name__)

_frame_lock = threading.Lock()
_latest_frame = None

GRAFANA_PORT = os.getenv("GRAFANA_PORT", "3000")

# Grafana panel URLs are built in JS using window.location.hostname
# so they work regardless of which IP you use to open the page.
HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>CCTV People Counting</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{background:#0f1117;color:#e0e0e0;font-family:'Segoe UI',sans-serif}}
    header{{background:#1a1d27;border-bottom:2px solid #2e7dff;padding:14px 24px;
            display:flex;align-items:center;gap:12px}}
    header h1{{font-size:1.3rem;color:#4da6ff;letter-spacing:1px}}
    .dot{{width:10px;height:10px;border-radius:50%;background:#22cc66;
          animation:blink 1.5s infinite}}
    @keyframes blink{{0%,100%{{opacity:1}}50%{{opacity:.3}}}}
    .grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:16px;
           max-width:1600px;margin:0 auto}}
    .video-panel{{grid-column:1;grid-row:1/3;background:#1a1d27;
                  border:1px solid #2e3a5c;border-radius:10px;overflow:hidden;
                  display:flex;flex-direction:column}}
    .stats-row{{grid-column:2;grid-row:1;display:grid;
                grid-template-columns:repeat(2,1fr);gap:12px}}
    .graphs-row{{grid-column:2;grid-row:2}}
    .panel{{background:#1a1d27;border:1px solid #2e3a5c;border-radius:10px;
            overflow:hidden;display:flex;flex-direction:column}}
    .panel-header{{padding:10px 14px;background:#1f2435;
                   border-bottom:1px solid #2e3a5c;font-size:.78rem;
                   text-transform:uppercase;letter-spacing:1px;color:#8899bb}}
    img#stream{{width:100%;height:100%;object-fit:contain;
                display:block;min-height:380px}}
    iframe{{width:100%;border:none;display:block}}
    .stat-iframe{{height:130px}}
    .graph-iframe{{height:280px}}
    @media(max-width:900px){{
      .grid{{grid-template-columns:1fr}}
      .video-panel,.stats-row,.graphs-row{{grid-column:1}}
      .video-panel{{grid-row:1}}.stats-row{{grid-row:2}}.graphs-row{{grid-row:3}}
    }}
  </style>
</head>
<body>
<header><div class="dot"></div>
  <h1>🎥 CCTV People Counting — Live Dashboard</h1>
</header>
<div class="grid">
  <div class="video-panel panel">
    <div class="panel-header">📷 Live CCTV Stream</div>
    <img id="stream" src="/video_feed" alt="Loading stream..."/>
  </div>
  <div class="stats-row">
    <div class="panel">
      <div class="panel-header">🔲 ROI Count</div>
      <iframe id="if1" class="stat-iframe" title="ROI Count"></iframe>
    </div>
    <div class="panel">
      <div class="panel-header">📏 Line Crossings</div>
      <iframe id="if2" class="stat-iframe" title="Line Crossings"></iframe>
    </div>
    <div class="panel">
      <div class="panel-header">👥 Total Occupancy</div>
      <iframe id="if3" class="stat-iframe" title="Occupancy"></iframe>
    </div>
    <div class="panel">
      <div class="panel-header">🎬 Frame Per Second</div>
      <iframe id="if4" class="stat-iframe" title="WiFi Signal"></iframe>
    </div>
  </div>
  <div class="graphs-row panel">
    <div class="panel-header">📈 People per Hour (last 24h)</div>
    <iframe id="if5" class="graph-iframe" title="Hourly Graph"></iframe>
  </div>
</div>
<script>
  const gBase = 'http://' + window.location.hostname + ':{GRAFANA_PORT}';
  const dash  = '/d-solo/people/people-counting?orgId=1&theme=dark&refresh=5s';
  document.getElementById('if1').src = gBase + dash + '&panelId=1';
  document.getElementById('if2').src = gBase + dash + '&panelId=2';
  document.getElementById('if3').src = gBase + dash + '&panelId=3';
  document.getElementById('if4').src = gBase + dash + '&panelId=4&refresh=10s';
  document.getElementById('if5').src = gBase + dash + '&panelId=5&refresh=1m';
  const img = document.getElementById('stream');
  img.onerror = () => setTimeout(() => {{ img.src = '/video_feed?' + Date.now(); }}, 2000);
</script>
</body></html>"""


# ── Public API — called from main.py ──────────────────────────────────────────

def update_frame(frame):
    """Call every frame from main.py instead of (or alongside) cv2.imshow()."""
    global _latest_frame
    with _frame_lock:
        _latest_frame = frame.copy()


# ── Flask routes ───────────────────────────────────────────────────────────────

def _generate_mjpeg():
    while True:
        with _frame_lock:
            frame = _latest_frame
        if frame is None:
            time.sleep(0.05)
            continue
        ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + buf.tobytes() + b'\r\n')
        time.sleep(1 / 25)


@app.route('/')
def index():
    return HTML


@app.route('/video_feed')
def video_feed():
    return Response(_generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/health')
def health():
    return {"status": "ok"}, 200


# ── Start ──────────────────────────────────────────────────────────────────────

def start_server(host='0.0.0.0', port=8080):
    """Start Flask in a background daemon thread. Call once from main.py."""
    t = threading.Thread(
        target=lambda: app.run(host=host, port=port,
                               threaded=True, use_reloader=False),
        daemon=True, name="FlaskWebStream"
    )
    t.start()
    print(f"[WebStream] Dashboard → http://{host}:{port}")
