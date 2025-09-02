from __future__ import annotations
import os
import cv2
import time
import queue
import threading
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np

import torch
from flask import Flask, Response, render_template_string, request, url_for
from ultralytics import YOLO
import easyocr

# =============================
# ==== KONFIGURASI ============
# =============================
# Ganti dengan path model YOLO Anda
MODEL_PATH = r"C:\\Users\\Nico Marvels\\belajar\\data windu\\frame - Original\\yolo_dataset\\runs\\detect\\my_yolo_training\\weights\\best.pt"

# Ganti dengan 4 sumber video Anda (bisa file, URL RTSP, atau indeks kamera)
VIDEO_SOURCES = [
    r"C:\\Users\\Nico Marvels\\belajar\\data windu\\video2\\video1.mp4",
    r"C:\\Users\\Nico Marvels\\belajar\\data windu\\video2\\video2.mp4",
    r"C:\\Users\\Nico Marvels\\belajar\\data windu\\video2\\video3.mp4",
    r"C:\\Users\\Nico Marvels\\belajar\\data windu\\video2\\video4.mp4",
]

# Folder untuk menyimpan frame pelanggaran
SAVE_VIOLATION_FRAMES_DIR = r"C:\\Users\\Nico Marvels\\belajar\\data windu\\video2\\pelanggaran_frames"
os.makedirs(SAVE_VIOLATION_FRAMES_DIR, exist_ok=True)

# Nama kelas sesuai dengan model Anda
CLASS_NAMES = [
    "car", "driver_buckled", "driver_unbuckled", "driver_unknown", "kaca",
    "motor_1_helmet", "motor_1_nohelmet", "motor_2_helmet", "motor_2_nohelmet",
    "motor_more_2", "passanger_buckled", "passanger_unbuckled",
    "passanger_unknown", "plat_nomor"
]

# Kelas yang dianggap sebagai pelanggaran
VIOLATION_MOTOR = {"motor_1_nohelmet", "motor_2_nohelmet", "motor_more_2"}
VIOLATION_CAR = {"driver_unbuckled", "passanger_unbuckled"}

# =======================================
# ==== Inisialisasi Model & Perangkat ===
# =======================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Perangkat yang digunakan: {DEVICE}")

print("Memuat model YOLO...")
MODEL = YOLO(MODEL_PATH)
MODEL.to(DEVICE)
print("Model YOLO berhasil dimuat.")

print("Memuat model OCR...")
OCR_READER = easyocr.Reader(['en'], gpu=(DEVICE == 'cuda'))
print("Model OCR berhasil dimuat.")

# Variabel global untuk menyimpan data deteksi dari setiap stream
latest_detections_data = {}
data_lock = threading.Lock()

# ========================================
# ==== Struktur Data & Logika Inti =======
# ========================================

@dataclass
class VehicleInfo:
    """Menyimpan informasi terstruktur untuk satu kendaraan."""
    vehicle_type: str = "Unknown"
    license_plate: str = "N/A"
    conditions: List[str] = field(default_factory=list)
    note: str = "-"
    box: Tuple[int, int, int, int] = (0, 0, 0, 0)

    @property
    def condition_str(self) -> str:
        return ", ".join(sorted(self.conditions)) if self.conditions else "N/A"

def is_box_inside(inner_box, outer_box):
    """Mengecek apakah 'inner_box' berada di dalam 'outer_box'."""
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box
    return ix1 >= ox1 and iy1 >= oy1 and ix2 <= ox2 and iy2 <= oy2

def process_frame_details(frame: np.ndarray, results) -> List[VehicleInfo]:
    """
    Memproses hasil deteksi untuk mengelompokkan informasi per kendaraan
    dan melakukan OCR pada plat nomor.
    """
    detected_vehicles = []
    if not results or not results[0].boxes:
        return []

    all_boxes = results[0].boxes

    vehicle_boxes = []
    attribute_boxes = []
    for box in all_boxes:
        cls_id = int(box.cls[0])
        label = CLASS_NAMES[cls_id]
        xyxy = tuple(map(int, box.xyxy[0].tolist()))
        
        if label == "car" or label.startswith("motor_"):
            vehicle_boxes.append({'label': label, 'box': xyxy})
        elif label in ["driver_buckled", "driver_unbuckled", "passanger_buckled", "passanger_unbuckled", "plat_nomor"]:
            attribute_boxes.append({'label': label, 'box': xyxy})

    for vehicle in vehicle_boxes:
        v_label, v_box = vehicle['label'], vehicle['box']
        info = VehicleInfo(box=v_box)
        has_violation = False

        if v_label == "car":
            info.vehicle_type = "Mobil"
        elif v_label.startswith("motor_"):
            info.vehicle_type = "Motor"
            if v_label in VIOLATION_MOTOR:
                has_violation = True
            if "motor_1_nohelmet" in v_label: info.conditions.append("Motor 1 nohelmet")
            elif "motor_1_helmet" in v_label: info.conditions.append("Motor 1 helmet")
            if "motor_2_helmet" in v_label: info.conditions.append("Motor 2 helmet")
            elif "motor_more_2" in v_label: info.conditions.append("Motor more 2")

        for attr in attribute_boxes:
            a_label, a_box = attr['label'], attr['box']
            if is_box_inside(a_box, v_box):
                if a_label == 'plat_nomor':
                    plate_img = frame[a_box[1]:a_box[3], a_box[0]:a_box[2]]
                    ocr_results = OCR_READER.readtext(plate_img, detail=0, paragraph=True)
                    if ocr_results:
                        info.license_plate = " ".join(ocr_results).upper()
                
                if info.vehicle_type == "Mobil":
                    if a_label == "driver_buckled": info.conditions.append("Driver Buckled")
                    elif a_label == "driver_unbuckled": 
                        info.conditions.append("Driver Unbuckled")
                        has_violation = True
                    elif a_label == "passanger_buckled": info.conditions.append("Passenger Buckled")
                    elif a_label == "passanger_unbuckled": 
                        info.conditions.append("Passenger Unbuckled")
                        has_violation = True
        
        if has_violation:
            info.note = "Violation"
        detected_vehicles.append(info)
    return detected_vehicles

# =============================
# ==== Worker Video Stream ====
# =============================
class StreamWorker:
    def __init__(self, source: any, name: str):
        self.source = source
        self.name = name
        self.cap = cv2.VideoCapture(source)
        self.q: "queue.Queue[bytes]" = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        while not self.stop_event.is_set():
            if not self.cap.isOpened():
                print(f"Mencoba menyambungkan kembali ke {self.name}...")
                time.sleep(2)
                self.cap = cv2.VideoCapture(self.source)
                continue

            ok, frame = self.cap.read()
            if not ok:
                self.cap.release()
                continue
            
            try:
                results = MODEL.predict(frame, conf=0.45, verbose=False, device=DEVICE)
            except Exception as e:
                print(f"[ERROR] Gagal melakukan inferensi pada {self.name}: {e}")
                continue

            vehicle_details = process_frame_details(frame.copy(), results)
            with data_lock:
                latest_detections_data[self.name] = vehicle_details
            
            annotated_frame = results[0].plot()
            
            total_violations = sum(1 for v in vehicle_details if v.note == "Violation")
            cv2.putText(annotated_frame, f"PELANGGARAN: {total_violations}", (20, 60), 
                        cv2.FONT_HERSHEY_DUPLEX, 2, (36, 36, 255), 3)

            ok, jpg = cv2.imencode('.jpg', annotated_frame)
            if ok:
                if self.q.full():
                    try: self.q.get_nowait()
                    except queue.Empty: pass
                self.q.put(jpg.tobytes())
        self.cap.release()

    def get_frame(self) -> Optional[bytes]:
        try: return self.q.get(timeout=1.0)
        except queue.Empty: return None

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive(): self.thread.join(timeout=2)
        if self.cap.isOpened(): self.cap.release()

workers: List[StreamWorker] = [StreamWorker(src, f"stream{i+1}") for i, src in enumerate(VIDEO_SOURCES)]
for w in workers: w.start()

# =============================
# ==== Aplikasi Flask =========
# =============================
app = Flask(__name__)

INDEX_HTML = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <title>Dashboard Pelanggaran Lalu Lintas</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #0b0b0b; color: #eaeaea; margin: 0; }
        header { background-color: #1a1a1a; padding: 20px; text-align: center; font-size: 24px; font-weight: bold; border-bottom: 2px solid #333; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 15px; padding: 15px; }
        .card a { text-decoration: none; color: inherit; }
        .card { background-color: #1e1e1e; border-radius: 12px; overflow: hidden; border: 1px solid #333; transition: transform 0.2s, box-shadow 0.2s; }
        .card:hover { transform: translateY(-5px); box-shadow: 0 8px 15px rgba(0,0,0,0.4); }
        .card h3 { margin: 0; padding: 12px 15px; background-color: #2a2a2a; }
        .card img { width: 100%; display: block; background-color: #000; aspect-ratio: 16/10; }
    </style>
</head>
<body>
    <header>Dashboard Monitoring Pelanggaran Lalu Lintas</header>
    <div class="grid">
        {% for i in range(num_streams) %}
        <div class="card">
            <a href="{{ url_for('detail_page', idx=i) }}">
                <h3>Stream {{ i + 1 }}</h3>
                <img src="{{ url_for('video_feed', idx=i) }}" alt="Stream {{ i + 1 }}">
            </a>
        </div>
        {% endfor %}
    </div>
</body>
</html>
"""

### PERUBAHAN DI SINI: Tata letak DETAIL_HTML diubah agar video di atas dan tabel di bawah ###
DETAIL_HTML = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <title>Detail Stream {{ stream_idx + 1 }}</title>
    <meta http-equiv="refresh" content="3">
    <style>
        body { font-family: 'Segoe UI', sans-serif; background-color: #f0f2f5; margin: 0; padding: 0; }
        .header { background-color: #fff; padding: 15px 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); display: flex; align-items: center; }
        .header a { text-decoration: none; color: #333; font-size: 18px; font-weight: bold; }
        .header h1 { margin: 0; margin-left: 20px; font-size: 22px; }
        .main-content {
            width: 90%;
            max-width: 1200px;
            margin: 20px auto; /* Membuat konten berada di tengah */
        }
        .video-container {
            background-color: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .video-container img {
            width: 100%;
            border-radius: 8px;
        }
        .data-container {
            margin-top: 25px; /* Jarak antara video dan tabel */
        }
        .data-container h2, .video-container h2 {
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
            margin-top: 0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            overflow: hidden; /* Agar border-radius terlihat di tabel */
        }
        th, td {
            padding: 14px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #4CAF50;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .note-violation {
            color: #D32F2F;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="header">
        <a href="{{ url_for('index') }}">&laquo; Kembali ke Dashboard</a>
        <h1>Detail Stream {{ stream_idx + 1 }}</h1>
    </div>
    <div class="main-content">
        <div class="video-container">
            <h2>Live View</h2>
            <img src="{{ url_for('video_feed', idx=stream_idx) }}" alt="Live stream">
        </div>
        <div class="data-container">
            <h2>Hasil Deteksi</h2>
            <table>
                <thead>
                    <tr><th>Vehicle</th><th>Plat Nomor</th><th>Condition</th><th>Note</th></tr>
                </thead>
                <tbody>
                    {% if detections %}
                        {% for vehicle in detections %}
                        <tr>
                            <td>{{ vehicle.vehicle_type }}</td>
                            <td>{{ vehicle.license_plate }}</td>
                            <td>{{ vehicle.condition_str }}</td>
                            <td class="{{ 'note-violation' if vehicle.note == 'Violation' else '' }}">{{ vehicle.note }}</td>
                        </tr>
                        {% endfor %}
                    {% else %}
                        <tr><td colspan="4" style="text-align:center; padding: 20px;">Tidak ada kendaraan terdeteksi.</td></tr>
                    {% endif %}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(INDEX_HTML, num_streams=len(workers))

@app.route('/detail/<int:idx>')
def detail_page(idx: int):
    if not (0 <= idx < len(workers)): return "Indeks stream tidak valid", 404
    stream_name = f"stream{idx+1}"
    with data_lock:
        detection_data = latest_detections_data.get(stream_name, [])
    return render_template_string(DETAIL_HTML, stream_idx=idx, detections=detection_data)

def mjpeg_generator(worker: StreamWorker):
    while True:
        frame = worker.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'

@app.route("/video_feed")
def video_feed():
    idx = int(request.args.get("idx", 0))
    if not (0 <= idx < len(workers)): return "Indeks stream tidak valid", 404
    return Response(mjpeg_generator(workers[idx]), mimetype='multipart/x-mixed-replace; boundary=frame')

def shutdown_server():
    print("Mematikan semua worker...")
    for w in workers: w.stop()

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=False)
    finally:
        shutdown_server()