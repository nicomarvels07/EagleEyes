import os
import cv2
import torch
import numpy as np
import threading
import time  # <-- untuk throttle FPS
from flask import Flask, render_template, Response, jsonify, abort
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# =============================
# Konfigurasi
# =============================
app = Flask(__name__)

# Tuning streaming
DETECT_EVERY = 50       # lakukan deteksi setiap 50 frame
TARGET_STREAM_FPS = 5   # stream ~5 fps agar tidak terlalu cepat

# Ganti dengan path folder video Anda
VIDEO_FOLDER = r"C:\Users\Nico Marvels\belajar\data windu\video2"
# Secara otomatis mendeteksi semua file video di dalam folder
try:
    VIDEO_FILES = sorted([f for f in os.listdir(VIDEO_FOLDER) if f.lower().endswith(('.mp4', '.mov', '.avi'))])
    if not VIDEO_FILES:
        raise FileNotFoundError
except FileNotFoundError:
    print(f"Error: Folder '{VIDEO_FOLDER}' tidak ditemukan atau kosong.")
    print("Pastikan Anda sudah membuat folder 'videos' dan menaruh file video di dalamnya.")
    exit()
    
# Tambahkan lokasi sesuai dengan jumlah video Anda
VIDEO_LOCATIONS = [
    "JPO Pondok Bambu",
    "JPO Lenteng Agung",
    "JPO Tanjung Barat",
    "JPO JORR Pesanggrahan"
]

# Path model (pastikan berada di folder yang sama dengan script ini)
ANGKA_MODEL_PATH = "angka_classifier.h5"
HURUF_MODEL_PATH = "huruf_classifier.h5"
YOLO_DETECT_MODEL_PATH = "best.pt"
YOLO_PLATE_MODEL_PATH = "plat_nomor.pt"

# Label dan Kelas
plat_angka = list("0123456789")
plat_huruf = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
violation_motor_labels = {"motor_1_nohelmet", "motor_2_nohelmet", "motor_more_2"}
violation_car_labels = {"driver_unbuckled", "passanger_unbuckled"}
unknown_car_labels = {"driver_unknown", "passanger_unknown"}
class_names = [
    "car", "driver_buckled", "driver_unbuckled", "driver_unknown", "kaca",
    "motor_1_helmet", "motor_1_nohelmet", "motor_2_helmet", "motor_2_nohelmet",
    "motor_more_2", "passanger_buckled", "passanger_unbuckled",
    "passanger_unknown", "plat_nomor"
]

# =============================
# Pemuatan Model
# =============================
print("Memuat model, mohon tunggu...")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
try:
    model_plat_angka = load_model(ANGKA_MODEL_PATH)
    model_plat_huruf = load_model(HURUF_MODEL_PATH)
    model_contains = YOLO(YOLO_PLATE_MODEL_PATH)
    model_detect = YOLO(YOLO_DETECT_MODEL_PATH)
    model_detect.to(DEVICE)
    print(f"Model berhasil dimuat pada perangkat: {DEVICE}")
except Exception as e:
    raise RuntimeError(f"Gagal memuat salah satu model: {e}")

# Variabel global untuk menyimpan data deteksi terbaru
latest_detection_data = {}
data_lock = threading.Lock()

# =============================
# Fungsi Inti & Pembantu
# =============================
def is_inside(parent_box, child_box):
    px1, py1, px2, py2 = parent_box
    cx1, cy1, cx2, cy2 = child_box
    return px1 <= cx1 and py1 <= cy1 and px2 >= cx2 and py2 >= cy2

def check_platenumber(plate_crop):
    plate_resized = cv2.resize(plate_crop, (400, 200), interpolation=cv2.INTER_CUBIC)
    results = model_contains(plate_resized, verbose=False)[0]
    
    char_boxes = []
    for box in results.boxes:
        x1, _, _, _ = map(int, box.xyxy[0].tolist())
        char_img = plate_resized[int(box.xyxy[0][1]):int(box.xyxy[0][3]), int(box.xyxy[0][0]):int(box.xyxy[0][2])]
        if char_img.size > 0:
            char_boxes.append((x1, char_img))
    
    char_boxes.sort(key=lambda item: item[0])

    plate_text = ""
    for _, char_img in char_boxes:
        img_prep = cv2.resize(char_img, (64, 64)).astype('float32') / 255.0
        img_prep = np.expand_dims(img_prep, axis=0)
        
        pred_angka = model_plat_angka.predict(img_prep, verbose=0)[0]
        pred_huruf = model_plat_huruf.predict(img_prep, verbose=0)[0]

        if np.max(pred_angka) > np.max(pred_huruf):
            plate_text += plat_angka[np.argmax(pred_angka)]
        else:
            plate_text += plat_huruf[np.argmax(pred_huruf)]
            
    return plate_text if plate_text else "N/A"

def analyze_frame(frame_bgr):
    objects_output = []
    annotated_frame = frame_bgr.copy()
    
    yolo_out = model_detect(annotated_frame, verbose=False)[0]
    detections = []
    for box in yolo_out.boxes:
        try:
            cls_id = int(box.cls[0])
            label = class_names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append({"label": label, "box": (x1, y1, x2, y2)})
        except (IndexError, KeyError):
            continue

    for obj in [d for d in detections if d["label"] == "car" or d["label"].startswith("motor")]:
        parent_box = obj["box"]
        children = [d for d in detections if d is not obj and is_inside(parent_box, d["box"])]
        
        plate_text = "N/A"
        conditions = []
        vehicle_type = "Mobil" if obj["label"] == "car" else "Motor"
        
        if vehicle_type == "Motor" and obj["label"] in violation_motor_labels:
            conditions.append(obj["label"])
        
        for ch in children:
            if vehicle_type == "Mobil" and (ch["label"] in violation_car_labels or ch["label"] in unknown_car_labels):
                conditions.append(ch["label"])
            if ch["label"] == "plat_nomor":
                x1, y1, x2, y2 = ch["box"]
                crop = frame_bgr[y1:y2, x1:x2]
                if crop.size > 0:
                    plate_text = check_platenumber(crop)

        status = "AMAN"
        if any(c in violation_car_labels or c in violation_motor_labels for c in conditions):
            status = "PELANGGARAN"
        elif any(c in unknown_car_labels for c in conditions):
            status = "TIDAK DIKETAHUI"
        
        objects_output.append({
            "vehicle": vehicle_type,
            "plate": plate_text,
            "condition": ", ".join(conditions) if conditions else "Aman",
            "note": status,
        })
        
        # Warna teks & kotak sesuai status
        # PELANGGARAN = merah; TIDAK DIKETAHUI = oranye/kuning; AMAN = hijau
        color = (0, 0, 255) if status == "PELANGGARAN" else ((255, 165, 0) if status == "TIDAK DIKETAHUI" else (0, 255, 0))
        cv2.rectangle(annotated_frame, (parent_box[0], parent_box[1]), (parent_box[2], parent_box[3]), color, 3)
        cv2.putText(
            annotated_frame,
            f"{vehicle_type} - {status}",
            (parent_box[0], max(0, parent_box[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,  # <- teks ikut warna status, pelanggaran = merah
            2
        )
        
    return objects_output, annotated_frame

# =============================
# Routes / Endpoint Flask
# =============================
@app.route("/")
def index():
    items = list(enumerate(VIDEO_FILES))
    return render_template("index.html", items=items, video_location=VIDEO_LOCATIONS)

@app.route("/detail/<int:idx>")
def detail(idx):
    if not (0 <= idx < len(VIDEO_FILES)):
        abort(404)
    title = VIDEO_LOCATIONS[idx] if idx < len(VIDEO_LOCATIONS) else VIDEO_FILES[idx]
    return render_template("detail.html", idx=idx, title=title)

@app.route("/video_feed/<int:idx>")
def video_feed(idx):
    """Route ini hanya untuk thumbnail di halaman utama."""
    try:
        video_path = os.path.join(VIDEO_FOLDER, VIDEO_FILES[idx])
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret: abort(500)
        _, buffer = cv2.imencode('.jpg', frame)
        return Response(buffer.tobytes(), mimetype='image/jpeg')
    except (IndexError, FileNotFoundError):
        abort(404)

@app.route("/detail_feed/<int:idx>")
def detail_feed(idx):
    """Streaming video yang dianalisis untuk halaman detail."""
    def generate():
        video_path = os.path.join(VIDEO_FOLDER, VIDEO_FILES[idx])
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        frame_count = 0
        last_annotated = None
        sleep_each = 1.0 / max(1, TARGET_STREAM_FPS)

        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # lakukan deteksi tiap N frame agar hemat & stabil
            if frame_count % DETECT_EVERY == 0:
                objects, annotated_frame = analyze_frame(frame)
                with data_lock:
                    latest_detection_data[idx] = objects

                last_annotated = cv2.resize(annotated_frame, (1024, 600))
                out_img = last_annotated
            else:
                # gunakan frame beranotasi terakhir supaya kotak & teks tetap terlihat
                out_img = last_annotated if last_annotated is not None else cv2.resize(frame, (1024, 600))

            ok, buffer = cv2.imencode('.jpg', out_img)
            if ok:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            frame_count += 1
            # perlambat stream supaya label terbaca
            time.sleep(sleep_each)

        cap.release()
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# PENTING: Nama endpoint ini harus sama dengan yang dipanggil di JavaScript
@app.route("/detection_data/<int:idx>")
def detection_data(idx):
    """Menyediakan data deteksi dalam format JSON."""
    with data_lock:
        data = latest_detection_data.get(idx, [])
    return jsonify(data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
