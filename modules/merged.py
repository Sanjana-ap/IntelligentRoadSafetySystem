import os
import time
import cv2
import base64
import random
import threading
import numpy as np
from datetime import datetime
from math import sqrt, cos, sin, radians
from PIL import Image

import torch
from ultralytics import YOLO
from torchvision.transforms import Compose, Resize, ToTensor
from sklearn.decomposition import PCA
from geopy.distance import geodesic
from pymongo import MongoClient
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
from deep_sort_realtime.deepsort_tracker import DeepSort

# -------------------------------
# Configuration
# -------------------------------
MODEL_PATH = r"C:\MajorProject\intelligent_road_safety\models\trained\pothole_detector\weights\best.pt"
VIDEO_SOURCE = r"C:\MajorProject\intelligent_road_safety\datasets\Merged\test\sample_video2.mp4"
OUTPUT_VIDEO = r"C:\MajorProject\intelligent_road_safety\outputs\final_result.avi"
CONF_THRESHOLD = 0.3
SCALE_M_PER_PIXEL = 0.05  # for speed estimation
VEHICLE_CLASSES = list(range(8))  # 0–7
POTHOLE_CLASS = 8  # class index of pothole
VEHICLE_LINE_Y_RATIO = 0.5  # middle of frame
LINE_OFFSET = 10  # tolerance for counting
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "road_safety"
POPTABLE = "potholes"
DUPLICATE_DISTANCE_THRESHOLD_METERS = 50

# GPS fallback (if no live GPS)
REF_LAT = None
REF_LON = None
REF_FRAME_IDX = None
REF_HEADING_DEG = 0.0

# Camera intrinsics for fallback
FX = 1200.0
FY = 1200.0
CX = 640.0
CY = 360.0
CAMERA_HEIGHT_M = 1.2
CAMERA_PITCH_DEG = 8.0

# MiDaS setup
MIDAS_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MIDAS_MODEL_TYPE = 'MiDaS_small'

# Flask config
API_HOST = '0.0.0.0'
API_PORT = 5000

# -------------------------------
# MongoDB setup
# -------------------------------
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
potholes_col = db[POPTABLE]

# -------------------------------
# Flask setup
# -------------------------------
app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/potholes')
def get_potholes():
    docs = list(potholes_col.find({}, {'_id': 0}))
    return jsonify(docs)

@app.route('/api/gps', methods=['POST'])
def post_gps():
    data = request.get_json(force=True)
    if not data:
        return jsonify({'status': 'error', 'reason': 'no json'}), 400
    latest_gps_lock.acquire()
    try:
        latest_gps['lat'] = float(data.get('lat', latest_gps.get('lat')))
        latest_gps['lon'] = float(data.get('lon', latest_gps.get('lon')))
        if 'heading' in data:
            latest_gps['heading'] = float(data['heading'])
        latest_gps['timestamp'] = data.get('timestamp', datetime.utcnow().isoformat())
    finally:
        latest_gps_lock.release()
    return jsonify({'status': 'ok'})

latest_gps = {'lat': None, 'lon': None, 'heading': None, 'timestamp': None}
latest_gps_lock = threading.Lock()

# -------------------------------
# Helper functions
# -------------------------------
def encode_img_b64(img_bgr):
    _, buf = cv2.imencode('.jpg', img_bgr)
    return base64.b64encode(buf).decode('ascii')

def compute_severity_from_depth(depth_patch, pixel_area_m2=0.0004):
    if depth_patch is None:
        return 0.0, {'reason': 'no_depth'}
    depth = depth_patch.astype(np.float32)
    mask = np.isfinite(depth)
    if np.sum(mask) < 20:
        return 0.0, {'reason': 'few_points'}
    ys, xs = np.where(mask)
    xs = xs.astype(np.float32)
    ys = ys.astype(np.float32)
    zs = depth[mask].astype(np.float32)
    pts = np.vstack([xs, ys, zs]).T
    pca = PCA(n_components=3)
    try:
        pca.fit(pts)
    except Exception:
        return 0.0, {'reason': 'pca_fail'}
    variances = pca.explained_variance_
    max_depth = float(np.max(zs))
    area_px = float(np.sum(mask))
    vertical_variance = float(variances[-1])
    raw = 0.5*(max_depth/(np.percentile(zs,90)+1e-6)) + 0.3*(area_px/500.0) + 0.2*(vertical_variance/(np.max(variances)+1e-6))
    severity = float(max(0.0,min(1.0,raw)))
    meta = {'max_depth': max_depth,'area_px': area_px,'variances': variances.tolist()}
    return severity, meta

def pixel_to_world_latlon(u, v, ref_lat, ref_lon, ref_frame_idx, current_frame_idx, ref_heading_deg=0.0):
    if ref_lat is None or ref_lon is None or ref_frame_idx is None:
        return None
    dx_px = u - CX
    dy_px = v - CY
    meters_per_px = 0.02
    right_m = dx_px * meters_per_px
    forward_m = (CY - v) * meters_per_px
    heading = radians(ref_heading_deg)
    north_offset = forward_m * cos(heading) - right_m * sin(heading)
    east_offset  = forward_m * sin(heading) + right_m * cos(heading)
    origin = (ref_lat, ref_lon)
    lat_point = geodesic(meters=north_offset).destination(origin,0)
    final = geodesic(meters=east_offset).destination((lat_point.latitude, lat_point.longitude),90)
    return final.latitude, final.longitude

# -------------------------------
# Load models
# -------------------------------
print('[yolo] loading model...')
model = YOLO(MODEL_PATH)
print('[yolo] model loaded.')

print('[depth] loading MiDaS model...')
try:
    midas = torch.hub.load('intel-isl/MiDaS', MIDAS_MODEL_TYPE)
    midas.to(MIDAS_DEVICE).eval()
    transform = torch.hub.load('intel-isl/MiDaS', 'transforms')
    midas_transform = transform.small_transform if MIDAS_MODEL_TYPE=='MiDaS_small' else transform.dpt_transform
    print('[depth] MiDaS loaded')
except Exception as e:
    print('[depth] failed:', e)
    midas = None
    midas_transform = None

# -------------------------------
# Vehicle Tracker
# -------------------------------
tracker = DeepSort(max_age=30)
last_positions = {}
vehicle_count = 0
crossed_ids = set()

# -------------------------------
# Run Flask server
# -------------------------------
threading.Thread(target=lambda: app.run(host=API_HOST, port=API_PORT, debug=False, use_reloader=False), daemon=True).start()
print('[api] Flask server started.')

# -------------------------------
# Video processing
# -------------------------------
cap = cv2.VideoCapture(VIDEO_SOURCE)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_width, frame_height))
line_y = int(frame_height * VEHICLE_LINE_Y_RATIO)
frame_idx = 0
print('[detect] starting detection...')

while True:
    ret, frame = cap.read()
    if not ret:
        print('[detect] end of stream')
        break
    frame_idx += 1

    results = model(frame)
    dets_for_tracker = []

    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls = int(box.cls.cpu().numpy()[0])
            conf = float(box.conf.cpu().numpy()[0])
            if conf < CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            # Vehicles
            if cls in VEHICLE_CLASSES:
                dets_for_tracker.append(([x1, y1, x2-x1, y2-y1], conf, cls))
            # Potholes
            elif cls == POTHOLE_CLASS:
                crop = frame[y1:y2, x1:x2]
                latest = None
                latest_gps_lock.acquire()
                try:
                    if latest_gps.get('lat') is not None:
                        latest = latest_gps.copy()
                finally:
                    latest_gps_lock.release()
                if latest is not None:
                    lat, lon = latest['lat'], latest['lon']
                else:
                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    geo = pixel_to_world_latlon(cx, cy, REF_LAT, REF_LON, REF_FRAME_IDX, frame_idx, REF_HEADING_DEG)
                    lat, lon = geo if geo is not None else (12.88+random.uniform(0,0.04),77.58+random.uniform(0,0.04))
                severity, sev_meta = 0.0, {}
                if midas is not None and midas_transform is not None and crop.size != 0:
                    try:
                        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        pil = Image.fromarray(img)
                        inp = midas_transform(pil).to(MIDAS_DEVICE)
                        with torch.no_grad():
                            prediction = midas(inp.unsqueeze(0))
                            prediction = torch.nn.functional.interpolate(prediction.unsqueeze(1), size=img.shape[:2], mode='bicubic', align_corners=False).squeeze()
                            depth_map = prediction.cpu().numpy()
                        severity, sev_meta = compute_severity_from_depth(depth_map)
                    except:
                        severity, sev_meta = 0.0, {'reason':'depth_fail'}
                else:
                    bbox_area = (x2-x1)*(y2-y1)
                    frame_area = frame.shape[0]*frame.shape[1]
                    ratio = bbox_area/frame_area
                    severity = min(5,max(1,int(ratio*50)))
                    sev_meta = {'method':'bbox_area_ratio','bbox_area':bbox_area,'frame_area':frame_area,'ratio':ratio}

                doc = {'latitude':float(lat),'longitude':float(lon),'confidence':float(conf),
                       'timestamp':datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                       'severity':float(severity),'severity_meta':sev_meta,
                       'bbox':[x1,y1,x2,y2],'crop_jpg_b64':encode_img_b64(crop)}
                potholes_col.insert_one(doc)
                print('[db] inserted pothole', doc['latitude'], doc['longitude'],'sev=',doc['severity'])

    # Update tracker
    tracks = tracker.update_tracks(dets_for_tracker, frame=frame)
    for track in tracks:
        if not track.is_confirmed():
            continue
        x1, y1, x2, y2 = track.to_ltrb()
        track_id = track.track_id
        cx, cy = (x1+x2)//2, (y1+y2)//2
        # Speed
        if track_id in last_positions:
            prev_x, prev_y = last_positions[track_id]
            pixel_dist = sqrt((cx-prev_x)**2 + (cy-prev_y)**2)
            speed = pixel_dist*SCALE_M_PER_PIXEL*fps*3.6
        else:
            speed = 0.0
        last_positions[track_id] = (cx, cy)
        # Counting
        if abs(cy-line_y)<LINE_OFFSET and track_id not in crossed_ids:
            vehicle_count += 1
            crossed_ids.add(track_id)
        # Draw vehicle
        cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
        cv2.putText(frame,f"ID {track_id} | {speed:.1f} km/h",(int(x1),int(y1)-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    # Draw line + count
    cv2.line(frame,(0,line_y),(frame.shape[1],line_y),(0,0,255),2)
    cv2.putText(frame,f"Vehicle Count: {vehicle_count}",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)

    out.write(frame)
    cv2.imshow('Intelligent Road Safety', frame)
    if cv2.waitKey(1)&0xFF in [27,ord('q')]:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
