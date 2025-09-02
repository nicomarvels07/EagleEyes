import os
import io
import time
import base64
import threading
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify, Response, send_file, render_template_string
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# =============================
# Configuration
# =============================
# Allow overriding model paths via environment variables
ANGKA_MODEL_PATH = os.environ.get("ANGKA_MODEL", "angka_classifier.h5")
HURUF_MODEL_PATH = os.environ.get("HURUF_MODEL", "huruf_classifier.h5")
YOLO_DETECT_MODEL_PATH = os.environ.get("YOLO_DETECT_MODEL", "best.pt")
YOLO_PLATE_MODEL_PATH = os.environ.get("YOLO_PLATE_MODEL", "plat_nomor.pt")

VIDEO_FOLDER = os.environ.get("VIDEO_FOLDER", "./videos")
VIDEO_FILES_ENV = os.environ.get("VIDEO_FILES", "")  # comma-separated
FRAME_STRIDE_DEFAULT = int(os.environ.get("FRAME_STRIDE", "20"))  # process every Nth frame

# =============================
# Flask App
# =============================
app = Flask(__name__)

# =============================
# Labels & Classes
# =============================
plat_angka = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
plat_huruf = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '~']

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
# Device Setup
# =============================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GPU_INFO = None
if DEVICE == 'cuda':
    GPU_INFO = {
        "name": torch.cuda.get_device_name(0),
        "total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
    }

# =============================
# Load Models (once)
# =============================
# Protect inference/model loading with a single lock to be safe in threaded servers
_infer_lock = threading.Lock()

try:
    model_plat_angka = load_model(ANGKA_MODEL_PATH)
    model_plat_huruf = load_model(HURUF_MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed loading Keras models: {e}")

try:
    model_contains = YOLO(YOLO_PLATE_MODEL_PATH)  # detects characters within plate crop
except Exception as e:
    raise RuntimeError(f"Failed loading plate YOLO model: {e}")

try:
    model_detect = YOLO(YOLO_DETECT_MODEL_PATH)   # main detector (car/motor/driver/passenger/plate)
    model_detect.to(DEVICE)
except Exception as e:
    raise RuntimeError(f"Failed loading main YOLO detect model: {e}")

# =============================
# Utility Functions
# =============================

def is_inside(parent_box: Tuple[int, int, int, int], child_box: Tuple[int, int, int, int]) -> bool:
    px1, py1, px2, py2 = parent_box
    cx1, cy1, cx2, cy2 = child_box
    return (cx1 >= px1 and cy1 >= py1 and cx2 <= px2 and cy2 <= py2)


def _predict_huruf(img64: np.ndarray) -> str:
    # img64: (64, 64, 3) float32 [0-1]
    preds = model_plat_huruf.predict(np.expand_dims(img64, 0), verbose=0)
    return plat_huruf[int(np.argmax(preds))]


def _predict_angka(img64: np.ndarray) -> str:
    preds = model_plat_angka.predict(np.expand_dims(img64, 0), verbose=0)
    return plat_angka[int(np.argmax(preds))]


def _prep_64x64(bgr_img: np.ndarray) -> np.ndarray:
    img = cv2.resize(bgr_img, (64, 64))
    img = img.astype('float32') / 255.0
    return img


def process_contain(detected_boxes: List[Tuple[int, int, int, int, np.ndarray]]) -> str:
    """Mimics original sequencing logic for plate substring - letters/digits ordering."""
    text_contain = ""
    count_box = 1
    pos_box = 1
    x2_old = 0

    for (x1_contain, y1_contain, x2_contain, y2_contain, contain_image) in detected_boxes:
        img64 = _prep_64x64(contain_image)

        if x1_contain > 0:
            if count_box == 1:
                char = _predict_huruf(img64)
                x2_old = x2_contain
            elif count_box == 2:
                if x1_contain > x2_old:
                    char = _predict_angka(img64)
                else:
                    char = _predict_huruf(img64)
                pos_box = 2
                x2_old = x2_contain
            elif pos_box == 2:
                if x1_contain > x2_old:
                    char = _predict_huruf(img64)
                    pos_box = 3
                else:
                    char = _predict_angka(img64)
                x2_old = x2_contain
            elif pos_box == 3:
                char = _predict_huruf(img64)
                x2_old = 0
            else:
                char = ""

            text_contain += char
            count_box += 1

    return text_contain


def check_platenumber(plate_crop_bgr: np.ndarray) -> str:
    # Resize for more stable char boxes (same as original)
    plate_crop = cv2.resize(plate_crop_bgr, (400, 200), interpolation=cv2.INTER_CUBIC)

    # Run YOLO that finds character "contains" inside the plate
    with _infer_lock:
        results = model_contains(plate_crop, verbose=False)[0]

    detected_boxes: List[Tuple[int, int, int, int, np.ndarray]] = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        contain_image = plate_crop[y1:y2, x1:x2]
        detected_boxes.append((x1, y1, x2, y2, contain_image))

    # Sort from left to right
    detected_boxes.sort(key=lambda b: b[0])
    text_contain = process_contain(detected_boxes)
    return text_contain


def draw_box_with_label(frame: np.ndarray, box: Tuple[int, int, int, int], label: str, color: Tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    cv2.putText(frame, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


def analyze_frame(frame_bgr: np.ndarray) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Run detection on a single BGR frame and return (objects, annotated_frame)."""
    objects_output: List[Dict[str, Any]] = []
    annotated = frame_bgr.copy()

    # Run main detector
    with _infer_lock:
        yolo_out = model_detect(annotated, verbose=False)[0]

    # Collect detections
    detections = []
    for box in yolo_out.boxes:
        cls_id = int(box.cls[0])
        if 0 <= cls_id < len(class_names):
            label = class_names[cls_id]
        else:
            label = f"cls_{cls_id}"
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        detections.append({
            "label": label,
            "box": (x1, y1, x2, y2)
        })

    # Analyze top-level objects (car / motor)
    counter = 1
    for obj in detections:
        lbl = obj["label"]
        if lbl == "car" or lbl.startswith("motor"):
            parent_box = obj["box"]
            # children inside the parent
            children = [d for d in detections if d is not obj and is_inside(parent_box, d["box"])]

            text_contain = ""
            violation_details: List[str] = []
            status = "OK"
            color = (0, 255, 0)
            label_obj = "car" if lbl == "car" else "motor"

            if label_obj == "car":
                for ch in children:
                    ch_lbl = ch["label"]
                    if ch_lbl in violation_car_labels:
                        violation_details.append(ch_lbl)
                    elif ch_lbl in unknown_car_labels:
                        violation_details.append(ch_lbl)
                    if ch_lbl == "plat_nomor":
                        x1, y1, x2, y2 = ch["box"]
                        crop = frame_bgr[y1:y2, x1:x2]
                        if crop.size > 0:
                            text_contain = check_platenumber(crop)

            else:  # motor
                if lbl in violation_motor_labels:
                    violation_details.append(lbl)
                for ch in children:
                    if ch["label"] == "plat_nomor":
                        x1, y1, x2, y2 = ch["box"]
                        crop = frame_bgr[y1:y2, x1:x2]
                        if crop.size > 0:
                            text_contain = check_platenumber(crop)

            if violation_details:
                status = "VIOLATION"
                color = (0, 0, 255)
            elif any(v in unknown_car_labels for v in violation_details):
                status = "UNKNOWN"
                color = (255, 0, 0)

            draw_box_with_label(
                annotated,
                parent_box,
                f"{label_obj} - {text_contain} - {status}" + (f" ({', '.join(violation_details)})" if violation_details else ""),
                color,
            )

            objects_output.append({
                "idx": counter,
                "type": label_obj,
                "plate": text_contain,
                "status": status,
                "violations": violation_details,
                "box": parent_box,
            })
            counter += 1

    return objects_output, annotated


def _jpeg_bytes(bgr_img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode('.jpg', bgr_img)
    if not ok:
        raise ValueError("Failed to encode JPEG")
    return buf.tobytes()


def _b64_jpeg(bgr_img: np.ndarray) -> str:
    return base64.b64encode(_jpeg_bytes(bgr_img)).decode('utf-8')


# =============================
# Routes
# =============================
@app.get("/")
def index():
    return render_template_string(
        """
        <!doctype html>
        <html>
            <head>
                <meta charset="utf-8" />
                <title>Plate & Violation Detector</title>
                <style>
                    body { font-family: system-ui, sans-serif; margin: 24px; }
                    .row { display:flex; gap:24px; align-items:flex-start; }
                    .card { border:1px solid #ddd; padding:16px; border-radius:12px; }
                    img { max-width:100%; height:auto; border-radius:12px; }
                    input, button { padding: 10px; border-radius: 8px; border: 1px solid #ccc; }
                </style>
            </head>
            <body>
                <h1>Plate & Violation Detector (Flask)</h1>
                <p>Use the form to test <code>/detect</code> on a single image, or open the MJPEG <code>/video-stream</code> with a local file path.</p>
                <div class="row">
                    <div class="card">
                        <h3>Image Detect</h3>
                        <form id="f" method="post" action="/detect" enctype="multipart/form-data" target="_blank">
                            <input type="file" name="image" accept="image/*" required>
                            <label><input type="checkbox" name="return_image" value="1"> return annotated image</label>
                            <button type="submit">Run</button>
                        </form>
                    </div>
                    <div class="card">
                        <h3>Video Stream (server-side file)</h3>
                        <form onsubmit="event.preventDefault(); const p=document.getElementById('p').value; window.open('/video-stream?path='+encodeURIComponent(p),'_blank');">
                            <input id="p" type="text" placeholder="/absolute/or/relative/path.mp4" size="40">
                            <button type="submit">Open stream</button>
                        </form>
                        <p>Or set env <code>VIDEO_FOLDER</code> and <code>VIDEO_FILES</code> to default playlist.</p>
                    </div>
                </div>
            </body>
        </html>
        """
    )


@app.get("/health")
def health():
    return jsonify({
        "device": DEVICE,
        "gpu": GPU_INFO,
        "models": {
            "angka": os.path.abspath(ANGKA_MODEL_PATH),
            "huruf": os.path.abspath(HURUF_MODEL_PATH),
            "yolo_detect": os.path.abspath(YOLO_DETECT_MODEL_PATH),
            "yolo_plate": os.path.abspath(YOLO_PLATE_MODEL_PATH),
        }
    })


@app.post("/detect")
def detect_image():
    if 'image' not in request.files:
        return jsonify({"error": "missing file field 'image'"}), 400

    file = request.files['image']
    data = np.frombuffer(file.read(), np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        return jsonify({"error": "failed to decode image"}), 400

    objects, annotated = analyze_frame(bgr)

    resp: Dict[str, Any] = {"objects": objects}
    if request.form.get('return_image') == '1' or request.args.get('return_image') == '1':
        resp["annotated_jpeg_base64"] = _b64_jpeg(annotated)

    return jsonify(resp)


@app.get("/video-stream")
def video_stream():
    """MJPEG stream of processed frames from a video file (server-side path) or env playlist."""
    path = request.args.get('path')
    stride = int(request.args.get('stride', FRAME_STRIDE_DEFAULT))

    play_list: List[str] = []
    if path and os.path.exists(path):
        play_list = [path]
    else:
        if VIDEO_FILES_ENV:
            play_list = [p.strip() for p in VIDEO_FILES_ENV.split(',') if p.strip()]
            play_list = [p if os.path.isabs(p) else os.path.join(VIDEO_FOLDER, p) for p in play_list]
            play_list = [p for p in play_list if os.path.exists(p)]
        if not play_list:
            return jsonify({"error": "No valid video path. Pass ?path=/path/to/file.mp4 or set VIDEO_FILES."}), 400

    def gen():
        for vid_path in play_list:
            cap = cv2.VideoCapture(vid_path)
            if not cap.isOpened():
                continue
            frame_num = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_num += 1
                if frame_num % max(1, stride) != 0:
                    continue
                objects, annotated = analyze_frame(frame)
                # Downscale for streaming
                annotated = cv2.resize(annotated, (1024, int(1024 * annotated.shape[0] / annotated.shape[1])))
                jpg = _jpeg_bytes(annotated)
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            cap.release()

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


# =============================
# CLI Entrypoint
# =============================
if __name__ == "__main__":
    # Example: FLASK_ENV=development python flask_plate_detection.py
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG", "0") == "1")
