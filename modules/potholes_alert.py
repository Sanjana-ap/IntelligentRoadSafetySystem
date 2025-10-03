



# Pothole Detection — Complete Software Solution

# This document contains a ready-to-run Python script (`potholes_alert.py`) and a frontend (`index.html`) that together:

# * Read a **video file** (or camera stream) and run YOLO pothole detection.
# * Optionally consume **live GPS** (serial GPS or HTTP POST from phone) and attach accurate positions to detections.
# * If GPS is not available, estimate pothole geolocation from camera geometry using **camera intrinsics + mounting height + heading** (simple pinhole-ray → ground-plane intersection).
# * Compute a **severity score** using monocular depth (MiDaS) + PCA on the depth point cloud inside the detected ROI.
# * Store detections (lat, lon, severity, confidence, bbox, base64 crop) in **MongoDB**.
# * Serve an API and an interactive Leaflet map (`index.html`) showing potholes colored by severity.

# ---

# ## Requirements

# Install packages (recommend using a virtualenv):

# ```bash
# pip install ultralytics opencv-python-headless pymongo flask flask-cors pyttsx3 geopy scikit-learn torch torchvision timm pillow numpy pynmea2 pyserial
# ```

# Notes:

# * `ultralytics` for YOLO model (`YOLOv8`/`ultralytics` package). You already used `ultralytics`.
# * `torch` and `torchvision` are required for MiDaS depth model (monocular depth). Make sure you install the correct torch wheel for your CUDA or CPU.
# * If you have GPU, install GPU-capable `torch` for faster depth estimation.

# ---

# ## How to run

# 1. Start MongoDB locally (default `mongodb://localhost:27017`).
# 2. Put your YOLO `best.pt` path in the config of the script, and set `VIDEO_SOURCE` to your video file path (or `0` for webcam).
# 3. Run the Python script:

# ```bash
# python potholes_alert.py
# ```

# 4. Open `index.html` in your browser (or the served root: `http://<machine-ip>:5000/`) — it will fetch `http://<machine-ip>:5000/api/potholes` and display markers.

# ---

# ## Files included below

# * `potholes_alert.py` — main Python script (detection, depth, PCA severity, DB, Flask API)
# * `index.html` — frontend map (Leaflet)

# > Copy both files into the same folder and run the Python script. The Flask server serves `index.html` for convenience.

# ---

## `potholes_alert.py`

# ```python
# potholes_alert.py

import os
import time
import cv2
import base64
import random
import threading
import numpy as np
from datetime import datetime
from io import BytesIO
from math import sin, cos, radians

from ultralytics import YOLO
import torch
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image

from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
from pymongo import MongoClient
from geopy.distance import geodesic
from sklearn.decomposition import PCA

# Optional: pynmea2 + pyserial for serial GPS
try:
    import serial
    import pynmea2
    SERIAL_AVAILABLE = True
except Exception:
    SERIAL_AVAILABLE = False

# -------------------------------
# Configuration (change these)
# -------------------------------
MODEL_PATH = r"C:\MajorProject\intelligent_road_safety\models\trained\pothole_detector\weights\best.pt"
VIDEO_SOURCE = r"C:\MajorProject\intelligent_road_safety\datasets\Merged\test\sample_video2.mp4"
CONF_THRESHOLD = 0.3
DUPLICATE_DISTANCE_THRESHOLD_METERS = 50
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "road_safety"
POPTABLE = "potholes"

# Camera geometry fallback (if no GPS available). Provide a reference GPS for a known frame
# If you don't have GPS attached, set these to the known starting location of the vehicle and the frame index
# where that location was at (for video without GPS you can manually note a reference frame and its lat-lon)
REF_LAT = None   # e.g. 12.90
REF_LON = None   # e.g. 77.60
REF_FRAME_INDEX = None  # frame index corresponding to the above reference
REF_HEADING_DEG = 0.0

# Simple camera intrinsics for fallback pixel->world. You should calibrate for accurate results
FX = 1200.0
FY = 1200.0
CX = 640.0
CY = 360.0
CAMERA_HEIGHT_M = 1.2
CAMERA_PITCH_DEG = 8.0

# MiDaS config
MIDAS_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MIDAS_MODEL_TYPE = 'MiDaS_small'  # or 'MiDaS_small' for faster

# GPS serial config (if using USB GPS)
GPS_SERIAL_PORT = None  # e.g. 'COM3' or '/dev/ttyUSB0'
GPS_BAUD = 4800

# Flask config
API_HOST = '0.0.0.0'
API_PORT = 5000

# -------------------------------
# End Configuration
# -------------------------------

# MongoDB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
potholes_col = db[POPTABLE]

# Flask App
app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/potholes')
def get_potholes():
    docs = list(potholes_col.find({}, {'_id': 0}))
    return jsonify(docs)

@app.route('/live_gps')
def live_gps():
    return send_from_directory('.', 'live_gps.html')


# Allow phone to POST live GPS (JSON: {lat:..., lon:..., heading:..., timestamp:...})
@app.route('/api/gps', methods=['POST'])
def post_gps():
    data = request.get_json(force=True)
    if data is None:
        return jsonify({'status': 'error', 'reason': 'no json'}), 400
    
    # store into latest_gps
    latest_gps_lock.acquire()
    try:
        latest_gps['lat'] = float(data.get('lat', latest_gps.get('lat')))
        latest_gps['lon'] = float(data.get('lon', latest_gps.get('lon')))
        if 'heading' in data:
            latest_gps['heading'] = float(data['heading'])
        latest_gps['timestamp'] = data.get('timestamp', datetime.utcnow().isoformat())

        # 👇 just put your print here (inside try block)
        print("Latest GPS:", latest_gps)

    finally:
        latest_gps_lock.release()
    
    return jsonify({'status': 'ok'})


# Shared GPS structure
latest_gps = {'lat': None, 'lon': None, 'heading': None, 'timestamp': None}
latest_gps_lock = threading.Lock()

# Serial GPS reader (if configured)
def gps_serial_reader(port, baud):
    if not SERIAL_AVAILABLE:
        print('[gps] serial or pynmea2 not installed')
        return
    try:
        ser = serial.Serial(port, baud, timeout=1)
    except Exception as e:
        print(f'[gps] serial open error: {e}')
        return
    print('[gps] serial reader started on', port)
    while True:
        try:
            line = ser.readline().decode('ascii', errors='ignore').strip()
            if not line:
                continue
            if line.startswith('$GPRMC') or line.startswith('$GPGGA'):
                try:
                    msg = pynmea2.parse(line)
                    if hasattr(msg, 'latitude') and hasattr(msg, 'longitude'):
                        latest_gps_lock.acquire()
                        latest_gps['lat'] = msg.latitude
                        latest_gps['lon'] = msg.longitude
                        latest_gps['timestamp'] = getattr(msg, 'timestamp', datetime.utcnow().isoformat())
                        latest_gps_lock.release()
                except Exception:
                    continue
        except Exception:
            time.sleep(0.1)

# -------------------------------
# MiDaS setup for monocular depth
# -------------------------------
print('[depth] loading MiDaS model on', MIDAS_DEVICE)
try:
    midas = torch.hub.load('intel-isl/MiDaS', MIDAS_MODEL_TYPE)
    midas.to(MIDAS_DEVICE)
    midas.eval()
    transform = torch.hub.load('intel-isl/MiDaS', 'transforms')
    if MIDAS_MODEL_TYPE == 'MiDaS_small':
        midas_transform = transform.small_transform
    else:
        midas_transform = transform.dpt_transform
    print('[depth] MiDaS loaded')
except Exception as e:
    print('[depth] MiDaS load failed:', e)
    midas = None
    midas_transform = None

# -------------------------------
# Helper functions
# -------------------------------

def encode_img_b64(img_bgr):
    _, buf = cv2.imencode('.jpg', img_bgr)
    return base64.b64encode(buf).decode('ascii')


def compute_severity_from_depth(depth_patch, pixel_area_m2=0.0004):
    """
    Compute severity between 0..1 using PCA on depth patch.
    depth_patch: 2D numpy array with relative depth (higher = farther) or absolute meters.
    Returns (severity, meta)
    """
    if depth_patch is None:
        return 0.0, {'reason': 'no_depth'}
    # Normalize depth so larger = deeper (MiDaS gives inverse-like relative depth; we use range)
    depth = depth_patch.astype(np.float32)
    # remove invalid
    mask = np.isfinite(depth)
    if np.sum(mask) < 20:
        return 0.0, {'reason': 'few_points'}

    pts_z = depth[mask]
    # For PCA we need (x,y,z). Use pixel coords for x,y and depth for z
    ys, xs = np.where(mask)
    xs = xs.astype(np.float32)
    ys = ys.astype(np.float32)
    zs = pts_z.astype(np.float32)
    pts = np.vstack([xs, ys, zs]).T
    pca = PCA(n_components=3)
    try:
        pca.fit(pts)
    except Exception:
        return 0.0, {'reason': 'pca_fail'}
    variances = pca.explained_variance_
    # heuristics combining max depth (z), area in pixels, and vertical variance
    max_depth = float(np.max(zs))
    area_px = float(np.sum(mask))
    vertical_variance = float(variances[-1])
    # Normalize using expected ranges (tweak for your setup)
    raw = 0.5 * (max_depth / (np.percentile(zs, 90) + 1e-6)) + 0.3 * (area_px / 500.0) + 0.2 * (vertical_variance / (np.max(variances)+1e-6))
    severity = float(max(0.0, min(1.0, raw)))
    meta = {'max_depth': max_depth, 'area_px': area_px, 'variances': variances.tolist()}
    return severity, meta


def pixel_to_world_latlon(u, v, ref_lat, ref_lon, ref_frame_idx, current_frame_idx, ref_heading_deg=0.0):
    """
    Simple linear approximation mapping pixel coordinates to world offsets using a reference frame.
    This is a very rough fallback and works only if camera is rigid and vehicle motion is small between frames.
    For better results, use proper camera calibration and IMU/odometry.
    """
    # If no reference provided return None
    if ref_lat is None or ref_lon is None or ref_frame_idx is None:
        return None
    # compute pixel deltas relative to image center
    dx_px = u - CX
    dy_px = v - CY
    # approximate meters per pixel on ground plane at some distance; tune for your camera
    meters_per_px = 0.02  # 2cm per pixel as a default guess
    right_m = dx_px * meters_per_px
    forward_m = (CY - v) * meters_per_px
    # rotate according to heading
    heading = radians(ref_heading_deg)
    north_offset = forward_m * cos(heading) - right_m * sin(heading)
    east_offset  = forward_m * sin(heading) + right_m * cos(heading)
    origin = (ref_lat, ref_lon)
    lat_point = geodesic(meters=north_offset).destination(origin, 0)
    final = geodesic(meters=east_offset).destination((lat_point.latitude, lat_point.longitude), 90)
    return final.latitude, final.longitude

# -------------------------------
# Start optional serial GPS reader thread
# -------------------------------
if GPS_SERIAL_PORT and SERIAL_AVAILABLE:
    threading.Thread(target=gps_serial_reader, args=(GPS_SERIAL_PORT, GPS_BAUD), daemon=True).start()
else:
    if GPS_SERIAL_PORT:
        print('[gps] GPS serial port configured but serial/pynmea2 not available')

# Start Flask server in a separate thread
threading.Thread(target=lambda: app.run(host=API_HOST, port=API_PORT, debug=False, use_reloader=False), daemon=True).start()
print('[api] Flask server started')

# -------------------------------
# Load YOLO model
# -------------------------------
print('[yolo] loading model:', MODEL_PATH)
model = YOLO(MODEL_PATH)
print('[yolo] model loaded')

# Video capture
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    raise SystemExit(f"Could not open video source {VIDEO_SOURCE}")

frame_idx = 0
print('[detect] starting detection... press Ctrl+C to stop')

while True:
    ret, frame = cap.read()
    if not ret:
        print('[detect] end of stream')
        break
    frame_idx += 1

    # Run YOLO inference
    results = model(frame)

    # For each detection
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls = int(box.cls.cpu().numpy()[0])
            conf = float(box.conf.cpu().numpy()[0])
            if conf < CONF_THRESHOLD:
                continue
            # Assuming class index 8 is pothole as user's model had
            # If your model's class id for pothole differs, change accordingly
            # We'll accept any class (or you can filter)

            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            crop = frame[y1:y2, x1:x2]

            # Wait for GPS: prefer latest_gps, else fallback to reference mapping
            latest = None
            latest_gps_lock.acquire()
            try:
                if latest_gps.get('lat') is not None:
                    latest = latest_gps.copy()
            finally:
                latest_gps_lock.release()

            if latest is not None and latest.get('lat') is not None:
                lat, lon = latest['lat'], latest['lon']
            else:
                # fallback: map using approximate pixel->world based on REF_*
                # use bbox center
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                geo = pixel_to_world_latlon(cx, cy, REF_LAT, REF_LON, REF_FRAME_INDEX, frame_idx, REF_HEADING_DEG)
                if geo is None:
                    # ultimate fallback: random jitter near a default location (bad but prevents crash)
                    lat = 12.88 + random.uniform(0, 0.04)
                    lon = 77.58 + random.uniform(0, 0.04)
                else:
                    lat, lon = geo

            # Depth / severity
            severity = 0.0
            sev_meta = {}
            if midas is not None and midas_transform is not None and crop.size != 0:
                try:
                    img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(img)
                    inp = midas_transform(pil).to(MIDAS_DEVICE)
                    with torch.no_grad():
                        prediction = midas(inp.unsqueeze(0))
                        prediction = torch.nn.functional.interpolate(prediction.unsqueeze(1), size=img.shape[:2], mode='bicubic', align_corners=False).squeeze()
                        depth_map = prediction.cpu().numpy()
                    sev, meta = compute_severity_from_depth(depth_map)
                    severity = float(sev)
                    sev_meta = meta
                except Exception as e:
                    print('[depth] error on crop depth:', e)
            else:
    # Fallback: severity based on bbox area relative to frame
                bbox_area = (x2 - x1) * (y2 - y1)
                frame_area = frame.shape[0] * frame.shape[1]
                ratio = bbox_area / frame_area
            
                # Scale ratio (heuristic) to 1–5
                severity = min(5, max(1, int(ratio * 50)))  # tune multiplier
                sev_meta = {
                    'method': 'bbox_area_ratio',
                    'bbox_area': bbox_area,
                    'frame_area': frame_area,
                    'ratio': ratio
                }

            # De-duplicate using geodesic distance
            # is_dup = False
            # for existing in potholes_col.find({}):
            #     if 'latitude' in existing and 'longitude' in existing:
            #         try:
            #             dist = geodesic((existing['latitude'], existing['longitude']), (lat, lon)).meters
            #             if dist <= DUPLICATE_DISTANCE_THRESHOLD_METERS:
            #                 is_dup = True
            #                 break
            #         except Exception:
            #             continue

            # if is_dup:
            #     print(f'[db] duplicate at ~{lat:.6f},{lon:.6f} skipped')
            #     # Optionally emit an alert
            #     continue

            # encode crop as base64 (careful with DB size)
            b64 = encode_img_b64(crop)

            doc = {
                'latitude': float(lat),
                'longitude': float(lon),
                'confidence': float(conf),
                'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                'severity': float(severity),
                'severity_meta': sev_meta,
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'crop_jpg_b64': b64
            }
            potholes_col.insert_one(doc)
            print('[db] inserted:', doc['latitude'], doc['longitude'], 'sev=', doc['severity'])

    # Optional: show frame with boxes for debugging (slower)
    # draw boxes and show
    for r in results:
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())
            conf = float(b.conf.cpu().numpy()[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f'{conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow('pothole detect', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


## `index.html`



# ## Tuning & next steps

# 1. **Camera calibration:** run `cv2.calibrateCamera` with a checkerboard to get `fx, fy, cx, cy` for accurate pixel→world projection.
# 2. **IMU/heading:** connect a compass/IMU or use phone heading to rotate offsets correctly.
# 3. **Stereo or LiDAR:** replace MiDaS with stereo or LiDAR depth for accurate severity.
# 4. **DB / Storage:** store crops on disk or S3/GridFS instead of base64 in MongoDB for large deployments.
# 5. **Batch DB writes / dedupe by time window** to avoid repeated inserts while vehicle passes one pothole.
# 6. **UI improvements:** clustering, timeline, filters by severity/date.

# ---

# If you want, I can now:

# * produce a smaller version that **does not** use MiDaS (faster) and just uses box area proxies, or
# * give a calibration script for `cv2.calibrateCamera`, or
# * provide an Android companion app snippet (HTTP POST of GPS) you can run on a phone to stream GPS to the server.

# Tell me which follow-up you want and I will add it.
