import os
import cv2
import time
import torch
import numpy as np
from flask import Flask, render_template, Response, abort
from ultralytics import YOLO

# (Opsional) diamkan log TensorFlow
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# === Jika Anda butuh TF untuk classifier huruf/angka ===
# Pastikan paketnya terpasang. Jika tidak diperlukan, Anda bisa comment 3 baris di bawah.
from tensorflow.keras.models import load_model

# === Initialize Flask App ===
app = Flask(__name__)

# === Cek Device ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device yang digunakan:", device)
if device == 'cuda':
    try:
        print("GPU name:", torch.cuda.get_device_name(0))
        print("Memori GPU tersedia:", round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2), "GB")
    except Exception:
        pass

# === Path Model dan Direktori ===
# GANTI path sesuai milik Anda
MODEL_DETECT_PATH = r"C:\Users\Nico Marvels\belajar\data windu\frame - Original\yolo_dataset\runs\detect\my_yolo_training\weights\best.pt"
MODEL_PLAT_DETECT_PATH = r"plat_nomor.pt"            # YOLO untuk potongan karakter plat
MODEL_HURUF_PATH = r"huruf_classifier.h5"            # Keras huruf
MODEL_ANGKA_PATH = r"angka_classifier.h5"            # Keras angka

VIDEO_FOLDER = r"C:\Users\Nico Marvels\belajar\data windu\video2"
video_files = ["VID_20250328_125544.mp4", "IMG_8533.mov", "tj.mov", "video4.mp4"]
video_location = ["JPO PONDOK BAMBU","JPO LENTANG AGUNG","JPO TANJUNG BARAT","JPO JORR PESANGRAHAN"] 

# === Kelas Pelanggaran dan Label ===
violation_motor_labels = {"motor_1_nohelmet", "motor_2_nohelmet", "motor_more_2"}
violation_car_labels = {"driver_unbuckled", "passanger_unbuckled"}
unknown_car_labels = {"driver_unknown", "passanger_unknown"}

class_names = [
    "car", "driver_buckled", "driver_unbuckled", "driver_unknown", "kaca",
    "motor_1_helmet", "motor_1_nohelmet", "motor_2_helmet", "motor_2_nohelmet",
    "motor_more_2", "passanger_buckled", "passanger_unbuckled",
    "passanger_unknown", "plat_nomor"
]

# === Load Models (sekali saja) ===
detect_model = YOLO(MODEL_DETECT_PATH).to(device)

# Model pembaca plat (huruf/angka + YOLO berisi potongan karakter)
# Jika tidak ada / belum dilatih, comment blok ini dan fungsinya akan skip
model_contains = None
model_plat_huruf = None
model_plat_angka = None
try:
    if os.path.exists(MODEL_PLAT_DETECT_PATH):
        model_contains = YOLO(MODEL_PLAT_DETECT_PATH).to(device)
    if os.path.exists(MODEL_HURUF_PATH):
        model_plat_huruf = load_model(MODEL_HURUF_PATH)
    if os.path.exists(MODEL_ANGKA_PATH):
        model_plat_angka = load_model(MODEL_ANGKA_PATH)
except Exception as e:
    print("Peringatan: model pembaca plat belum siap ->", e)

plat_angka = list("0123456789")
plat_huruf = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["~"]


# === Util: cek child di dalam parent bbox ===
def is_inside(parent_box, child_box):
    px1, py1, px2, py2 = parent_box
    cx1, cy1, cx2, cy2 = child_box
    return (cx1 >= px1 and cy1 >= py1 and cx2 <= px2 and cy2 <= py2)


# === Util: OCR sangat sederhana untuk isi plat ===
def check_platenumber(plate_crop):
    """
    Menggunakan YOLO 'model_contains' untuk mendeteksi potongan karakter,
    lalu klasifikasikan dengan model huruf/angka. Jika model tidak tersedia,
    akan mengembalikan string kosong.
    """
    if model_contains is None or model_plat_huruf is None or model_plat_angka is None:
        return ""

    # Normalisasi ukuran agar lebih stabil
    plate_crop = cv2.resize(plate_crop, (400, 200), interpolation=cv2.INTER_CUBIC)
    result_contains = model_contains(plate_crop)[0]
    detected_boxes = []

    for box_contain in result_contains.boxes:
        x1c, y1c, x2c, y2c = map(int, box_contain.xyxy[0].tolist())
        contain_image = plate_crop[y1c:y2c, x1c:x2c]
        detected_boxes.append((x1c, y1c, x2c, y2c, contain_image))

    # Urut dari kiri ke kanan
    detected_boxes.sort(key=lambda b: b[0])

    text_contain = ""
    count_box = 1
    pos_box = 1
    x2_old = 0

    for (x1c, y1c, x2c, y2c, contain_image) in detected_boxes:
        # resize ke 64x64 dan normalisasi untuk classifier
        ci = cv2.resize(contain_image, (64, 64)).astype("float32") / 255.0
        ci = np.expand_dims(ci, axis=0)

        # Logika kasar: huruf di awal, lalu angka, dst.
        if count_box == 1:
            pred = model_plat_huruf.predict(ci, verbose=0)
            char = plat_huruf[int(np.argmax(pred))]
            x2_old = x2c
            pos_box = 1
        elif count_box == 2:
            if x1c > x2_old:
                pred = model_plat_angka.predict(ci, verbose=0)
                char = plat_angka[int(np.argmax(pred))]
                pos_box = 2
            else:
                pred = model_plat_huruf.predict(ci, verbose=0)
                char = plat_huruf[int(np.argmax(pred))]
                pos_box = 1
            x2_old = x2c
        else:
            # setelah pola awal, sederhana saja:
            if pos_box == 2 and x1c > x2_old:
                pred = model_plat_huruf.predict(ci, verbose=0)
                char = plat_huruf[int(np.argmax(pred))]
                pos_box = 3
            else:
                pred = model_plat_angka.predict(ci, verbose=0)
                char = plat_angka[int(np.argmax(pred))]
            x2_old = x2c

        text_contain += char
        count_box += 1

    return text_contain


# === Stream sederhana: hanya deteksi & count (MJPEG) ===
def gen_frames(video_file):
    video_path = os.path.join(VIDEO_FOLDER, video_file)
    if not os.path.exists(video_path):
        abort(404, description=f"Video tidak ditemukan: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        abort(500, description=f"Gagal membuka video: {video_path}")

    save_folder = os.path.join(VIDEO_FOLDER, "pelanggaran_frames")
    os.makedirs(save_folder, exist_ok=True)

    total_car = total_motor = total_car_violation = total_motor_violation = 0

    no_grad_ctx = torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad()
    with no_grad_ctx:
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % 50 == 0:
                results = detect_model.predict(frame, conf=0.5, verbose=False)
                annotated = frame.copy()

                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        label = class_names[cls_id] if 0 <= cls_id < len(class_names) else f"cls_{cls_id}"
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                        is_violation = (label in violation_motor_labels) or (label in violation_car_labels)
                        color = (0, 0, 255) if is_violation else (0, 255, 0)

                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(annotated, f"{label} {conf:.2f}",
                                    (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        # Simpan frame jika pelanggaran
                        if is_violation:
                            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                            frame_name = f"{os.path.splitext(video_file)[0]}_frame{frame_idx}.jpg"
                            cv2.imwrite(os.path.join(save_folder, frame_name), annotated)

                        # Hitung mobil & motor
                        if label == "car":
                            total_car += 1
                        elif label.startswith("motor_"):
                            total_motor += 1

                        # Hitung pelanggaran
                        if label in violation_car_labels:
                            total_car_violation += 1
                        elif label in violation_motor_labels:
                            total_motor_violation += 1

                # Overlay totals
                cv2.putText(annotated, f"Mobil: {total_car}, Violation: {total_car_violation}",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(annotated, f"Motor: {total_motor}, Violation: {total_motor_violation}",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                annotated = cv2.resize(annotated, (800, 600))

                ok, buf = cv2.imencode('.jpg', annotated)
                if not ok:
                    frame_num += 1
                    continue

                frame_bytes = buf.tobytes()
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n\r\n" +
                       frame_bytes + b"\r\n")

            frame_num += 1

    cap.release()


# === Stream DETAIL: analitik lengkap (status, plat, dsb) ===
def gen_detail_frames(video_file):
    video_path = os.path.join(VIDEO_FOLDER, video_file)
    if not os.path.exists(video_path):
        abort(404, description=f"Video tidak ditemukan: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        abort(500, description=f"Gagal membuka video: {video_path}")

    no_grad_ctx = torch.inference_mode() if hasattr(torch, "inference_mode") else torch.no_grad()
    with no_grad_ctx:
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            if frame_num % 20 != 0:
                continue

            # Deteksi semua objek
            results = detect_model(frame)[0]
            detections = []
            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = class_names[cls_id] if 0 <= cls_id < len(class_names) else f"cls_{cls_id}"
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append({"label": label, "box": (x1, y1, x2, y2)})

            # Analitik per objek utama (car/motor)
            overlay_lines = []
            counter = 1

            for obj in detections:
                if obj["label"] == "car" or obj["label"].startswith("motor"):
                    children = [d for d in detections if d is not obj and is_inside(obj["box"], d["box"])]

                    violation_details = []
                    text_contain = ""  # default jika tidak ada plat

                    if obj["label"] == "car":
                        # cek violation & unknown untuk car
                        for ch in children:
                            if ch["label"] in violation_car_labels:
                                violation_details.append(ch["label"])
                            elif ch["label"] in unknown_car_labels:
                                violation_details.append(ch["label"])
                            if ch["label"] == "plat_nomor":
                                x1, y1, x2, y2 = ch["box"]
                                plate_crop = frame[y1:y2, x1:x2]
                                text_contain = check_platenumber(plate_crop)

                    else:  # motor
                        if obj["label"] in violation_motor_labels:
                            violation_details.append(obj["label"])
                        for ch in children:
                            if ch["label"] == "plat_nomor":
                                x1, y1, x2, y2 = ch["box"]
                                plate_crop = frame[y1:y2, x1:x2]
                                text_contain = check_platenumber(plate_crop)

                    status = "VIOLATION" if violation_details else "OK"
                    color = (0, 0, 255) if status == "VIOLATION" else (0, 255, 0)
                    label_obj = "car" if obj["label"] == "car" else "motor"

                    # Teks ringkas untuk overlay & list
                    line = f"{counter}. {label_obj} - {text_contain} - {status}"
                    if violation_details:
                        line += f" ({', '.join(violation_details)})"
                    overlay_lines.append(line)
                    counter += 1

                    # Gambar bbox
                    x1, y1, x2, y2 = obj["box"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame, f"{label_obj} - {text_contain} - {status}",
                                (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Tempel overlay list di kiri atas
            y0 = 40
            for line in overlay_lines[:8]:
                cv2.putText(frame, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y0 += 28

            frame_resized = cv2.resize(frame, (1024, 600))
            ok, buf = cv2.imencode(".jpg", frame_resized)
            if not ok:
                continue
            frame_bytes = buf.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n\r\n" +
                   frame_bytes + b"\r\n")

    cap.release()

# ===================== ROUTES =====================

# @app.route("/")
# def index():
#     # kirim (index, filename) agar ditampilkan di UI & bisa diklik
#     items = list(enumerate(video_files))
#     locations=list(enumerate(video_location))
#     return render_template("index.html", items=items, locations=locations)
@app.route("/")
def index():
    items = list(enumerate(video_files))
    return render_template("index.html", items=items, video_location=video_location)

@app.route("/video_feed/<int:idx>")
def video_feed(idx):
    if idx < 0 or idx >= len(video_files):
        abort(404, description="Index video di luar jangkauan.")
    headers = {"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
               "Pragma": "no-cache"}
    return Response(gen_frames(video_files[idx]),
                    mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers=headers)

# === Halaman DETAIL yang bisa diklik dari index ===
@app.route("/detail/<int:idx>")
def detail(idx):
    if idx < 0 or idx >= len(video_files):
        abort(404, description="Index video di luar jangkauan.")
    return render_template("detail.html", idx=idx, filename=video_files[idx]) #, filelocation=video_location[idx]

# === Stream analitik untuk halaman detail ===
@app.route("/detail_feed/<int:idx>")
def detail_feed(idx):
    if idx < 0 or idx >= len(video_files):
        abort(404, description="Index video di luar jangkauan.")
    headers = {"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
               "Pragma": "no-cache"}
    return Response(gen_detail_frames(video_files[idx]),
                    mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers=headers)

if __name__ == "__main__":
    # Hindari reloader ganda yang bisa memicu proses 2x
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, use_reloader=False)