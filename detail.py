import os
import cv2
import time
import torch
import numpy as np
import threading
from flask import Flask, render_template, Response, jsonify, abort
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# =============================
# Konfigurasi
# =============================
app = Flask(__name__)

# Ganti dengan path folder video Anda dan pastikan folder 'videos' ada
VIDEO_FOLDER = "videos" 
try:
    VIDEO_FILES = sorted([f for f in os.listdir(VIDEO_FOLDER) if f.lower().endswith(('.mp4', '.mov', '.avi'))])
    if not VIDEO_FILES: raise FileNotFoundError
except FileNotFoundError:
    print(f"Error: Folder '{VIDEO_FOLDER}' tidak ditemukan atau kosong.")
    exit()

# Sesuaikan lokasi dengan jumlah video Anda
VIDEO_LOCATIONS = ["JPO PONDOK BAMBU", "JPO LENTENG AGUNG", "JPO TANJUNG BARAT", "JPO JORR PESANGRAHAN"]

# Path model (letakkan di folder yang sama dengan script ini)
MODEL_DETECT_PATH = r"C:\Users\Nico Marvels\belajar\data windu\frame - Original\yolo_dataset\runs\detect\my_yolo_training\weights\best.pt"
MODEL_PLAT_DETECT_PATH = "plat_nomor.pt"
MODEL_HURUF_PATH = "huruf_classifier.h5"
MODEL_ANGKA_PATH = "angka_classifier.h5"

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
print("Memuat model...")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
try:
    detect_model = YOLO(MODEL_DETECT_PATH).to(DEVICE)
    if os.path.exists(MODEL_PLAT_DETECT_PATH): model_contains = YOLO(MODEL_PLAT_DETECT_PATH).to(DEVICE)
    if os.path.exists(MODEL_HURUF_PATH): model_plat_huruf = load_model(MODEL_HURUF_PATH)
    if os.path.exists(MODEL_ANGKA_PATH): model_plat_angka = load_model(MODEL_ANGKA_PATH)
    print(f"Model dimuat di: {DEVICE}")
except Exception as e:
    raise RuntimeError(f"Gagal memuat model: {e}")

# Variabel global untuk data deteksi
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
    if 'model_contains' not in globals(): return "N/A"
    plate_resized = cv2.resize(plate_crop, (400, 200), interpolation=cv2.INTER_CUBIC)
    results = model_contains(plate_resized, verbose=False)[0]
    char_boxes = sorted([(int(b.xyxy[0][0]), plate_resized[int(b.xyxy[0][1]):int(b.xyxy[0][3]), int(b.xyxy[0][0]):int(b.xyxy[0][2])]) for b in results.boxes if b.xyxy[0][2] - b.xyxy[0][0] > 0], key=lambda item: item[0])
    plate_text = ""
    for _, char_img in char_boxes:
        img_prep = np.expand_dims(cv2.resize(char_img, (64, 64)).astype('float32') / 255.0, axis=0)
        pred_angka = model_plat_angka.predict(img_prep, verbose=0)[0]
        pred_huruf = model_plat_huruf.predict(img_prep, verbose=0)[0]
        plate_text += plat_angka[np.argmax(pred_angka)] if np.max(pred_angka) > np.max(pred_huruf) else plat_huruf[np.argmax(pred_huruf)]
    return plate_text if plate_text else "N/A"

def analyze_frame(frame_bgr):
    objects_for_json, annotated_frame = [], frame_bgr.copy()
    yolo_out = detect_model(annotated_frame, verbose=False)[0]
    detections = []
    for box in yolo_out.boxes:
        try:
            detections.append({"label": class_names[int(box.cls[0])], "box": tuple(map(int, box.xyxy[0]))})
        except: continue
    
    for obj in [d for d in detections if d["label"] == "car" or d["label"].startswith("motor")]:
        parent_box, children = obj["box"], [d for d in detections if d is not obj and is_inside(obj["box"], d["box"])]
        plate_text, conditions = "N/A", []
        vehicle_type = "Mobil" if obj["label"] == "car" else "Motor"
        
        if vehicle_type == "Motor" and obj["label"] in violation_motor_labels: conditions.append(obj["label"])
        for ch in children:
            if vehicle_type == "Mobil" and (ch["label"] in violation_car_labels or ch["label"] in unknown_car_labels): conditions.append(ch["label"])
            if ch["label"] == "plat_nomor" and ch["box"][3] > ch["box"][1]: plate_text = check_platenumber(frame_bgr[ch["box"][1]:ch["box"][3], ch["box"][0]:ch["box"][2]])

        status = "AMAN"
        if any(c in violation_car_labels or c in violation_motor_labels for c in conditions): status = "PELANGGARAN"
        elif any(c in unknown_car_labels for c in conditions): status = "TIDAK DIKETAHUI"
        
        objects_for_json.append({"vehicle": vehicle_type, "plate": plate_text, "condition": ", ".join(conditions) if conditions else "Aman", "note": status})
        
        color = (0, 0, 255) if status == "PELANGGARAN" else ((255, 165, 0) if status == "TIDAK DIKETAHUI" else (0, 255, 0))
        label_text = f"{plate_text} - {status}"
        (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated_frame, (parent_box[0], parent_box[1]), (parent_box[2], parent_box[3]), color, 2)
        cv2.rectangle(annotated_frame, (parent_box[0], parent_box[1] - h - 10), (parent_box[0] + w, parent_box[1]), color, -1)
        cv2.putText(annotated_frame, label_text, (parent_box[0], parent_box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return objects_for_json, annotated_frame

# =============================
# Routes / Endpoint Flask
# =============================
@app.route("/")
def index():
    return render_template("index.html", items=list(enumerate(VIDEO_FILES)), video_location=VIDEO_LOCATIONS)

@app.route("/detail/<int:idx>")
def detail(idx):
    if not (0 <= idx < len(VIDEO_FILES)): abort(404)
    return render_template("detail.html", idx=idx, title=VIDEO_LOCATIONS[idx] if idx < len(VIDEO_LOCATIONS) else VIDEO_FILES[idx])

@app.route("/video_feed/<int:idx>")
def video_feed(idx):
    """Hanya untuk thumbnail di halaman utama."""
    try:
        cap = cv2.VideoCapture(os.path.join(VIDEO_FOLDER, VIDEO_FILES[idx]))
        ret, frame = cap.read()
        cap.release()
        if not ret: abort(500)
        _, buffer = cv2.imencode('.jpg', frame)
        return Response(buffer.tobytes(), mimetype='image/jpeg')
    except: abort(404)

@app.route("/detail_feed/<int:idx>")
def detail_feed(idx):
    """Streaming video yang sudah dianalisis."""
    def generate():
        cap = cv2.VideoCapture(os.path.join(VIDEO_FOLDER, VIDEO_FILES[idx]))
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
            if frame_count % 20 == 0:
                json_data, annotated_frame = analyze_frame(frame)
                with data_lock:
                    latest_detection_data[idx] = json_data
                _, buffer = cv2.imencode('.jpg', cv2.resize(annotated_frame, (1280, 720)))
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            frame_count += 1
        cap.release()
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# INI BAGIAN YANG BARU DITAMBAHKAN UNTUK MENGATASI ERROR 404
@app.route("/detection_data/<int:idx>")
def detection_data(idx):
    """Menyediakan data deteksi dalam format JSON untuk tabel."""
    with data_lock:
        data = latest_detection_data.get(idx, [])
    return jsonify(data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)