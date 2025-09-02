# ... (semua import, konfigurasi, class, workers, dsb. tetap) ...

app = Flask(__name__)

INDEX_HTML = """
<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Dashboard - 4 Video Streams</title>
  <style>
    body { margin:0; background:#0b0b0b; color:#eaeaea; font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Helvetica Neue', Arial; }
    header { padding:16px 24px; font-weight:700; font-size:20px; background:#111; border-bottom:1px solid #222; display:flex; justify-content:space-between; align-items:center; }
    header a { color:#9ad; text-decoration:none; font-weight:600; }
    .grid { display:grid; grid-template-columns: repeat(2, 1fr); gap:8px; padding:8px; }
    .card { background:#111; border:1px solid #222; border-radius:12px; overflow:hidden; }
    .card h3 { margin:0; padding:10px 12px; font-size:14px; border-bottom:1px solid #222; background:#0f0f0f; display:flex; justify-content:space-between; align-items:center; }
    .frame { width:100%; aspect-ratio: 1280/1000; background:#000; display:block; }
    .link { color:#8ecaff; text-decoration:none; font-size:12px; }
    footer { text-align:center; padding:12px; font-size:12px; color:#999; }
    @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <span>🚦 Pelanggaran Lalu Lintas — Dashboard 4 Kamera</span>
    <a href="/">Beranda</a>
  </header>
  <div class="grid">
    <div class="card">
      <h3>Stream 1 <a class="link" href="{{ url_for('detail', idx=0) }}">lihat detail →</a></h3>
      <a href="{{ url_for('detail', idx=0) }}" title="Buka detail Stream 1">
        <img class="frame" src="{{ url_for('video_feed', idx=0) }}" alt="stream 1" />
      </a>
    </div>
    <div class="card">
      <h3>Stream 2 <a class="link" href="{{ url_for('detail', idx=1) }}">lihat detail →</a></h3>
      <a href="{{ url_for('detail', idx=1) }}" title="Buka detail Stream 2">
        <img class="frame" src="{{ url_for('video_feed', idx=1) }}" alt="stream 2" />
      </a>
    </div>
    <div class="card">
      <h3>Stream 3 <a class="link" href="{{ url_for('detail', idx=2) }}">lihat detail →</a></h3>
      <a href="{{ url_for('detail', idx=2) }}" title="Buka detail Stream 3">
        <img class="frame" src="{{ url_for('video_feed', idx=2) }}" alt="stream 3" />
      </a>
    </div>
    <div class="card">
      <h3>Stream 4 <a class="link" href="{{ url_for('detail', idx=3) }}">lihat detail →</a></h3>
      <a href="{{ url_for('detail', idx=3) }}" title="Buka detail Stream 4">
        <img class="frame" src="{{ url_for('video_feed', idx=3) }}" alt="stream 4" />
      </a>
    </div>
  </div>
  <footer>Model: {{ model_name }} | Device: {{ device }} | Refreshing via MJPEG</footer>
</body>
</html>
"""

# ====== HALAMAN DETAIL PER STREAM ======
DETAIL_HTML = """
<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Detail Stream {{ idx+1 }}</title>
  <style>
    body { margin:0; background:#0b0b0b; color:#eaeaea; font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Helvetica Neue', Arial; }
    header { padding:16px 24px; font-weight:700; font-size:20px; background:#111; border-bottom:1px solid #222; display:flex; justify-content:space-between; align-items:center; }
    a { color:#8ecaff; text-decoration:none; }
    .wrap { padding:12px; max-width:1300px; margin:0 auto; }
    .video { width:100%; max-width:1280px; aspect-ratio: 1280/1000; background:#000; border-radius:12px; overflow:hidden; border:1px solid #222; }
    .meta { margin-top:12px; font-size:14px; color:#bbb; }
    .back { font-size:14px; }
  </style>
</head>
<body>
  <header>
    <span>🎥 Detail Stream {{ idx+1 }}</span>
    <a class="back" href="/">← Kembali ke Dashboard</a>
  </header>
  <div class="wrap">
    <img class="video" src="{{ url_for('video_feed', idx=idx) }}" alt="detail stream {{ idx+1 }}" />
    <div class="meta">Sumber: {{ source }} | Model: {{ model_name }} | Device: {{ device }}</div>
  </div>
</body>
</html>
"""

@app.route("/")
def index():
    model_name = os.path.basename(MODEL_PATH)
    return render_template_string(INDEX_HTML, model_name=model_name, device=DEVICE)

@app.route("/detail/<int:idx>")
def detail(idx: int):
    if idx < 0 or idx >= len(workers):
        return "Invalid stream index", 404
    model_name = os.path.basename(MODEL_PATH)
    source = str(VIDEO_SOURCES[idx])
    return render_template_string(DETAIL_HTML, idx=idx, source=source, model_name=model_name, device=DEVICE)

def mjpeg_generator(worker: StreamWorker):
    boundary = b"--frame"
    while True:
        frame = worker.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        yield b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"

@app.route("/stream/<int:idx>")
def video_feed(idx: int):
    if idx < 0 or idx >= len(workers):
        return "Invalid stream index", 404
    return Response(mjpeg_generator(workers[idx]), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/video_feed")
def video_feed_alias():
    idx = int(request.args.get("idx", 0))
    return video_feed(idx)

# ... (_shutdown dan main tetap sama) ...
