
import cv2
from ultralytics import YOLO
import winsound
import pyttsx3
import os
import random
from flask import Flask, jsonify, send_from_directory
from threading import Thread
from pymongo import MongoClient
from datetime import datetime
from flask_cors import CORS
from geopy.distance import geodesic  # To calculate real distance in meters

# -------------------------------
# Configuration
# -------------------------------
MODEL_PATH = r"C:\MajorProject\intelligent_road_safety\models\trained\pothole_detector\weights\best.pt"
VIDEO_SOURCE = r"C:\MajorProject\intelligent_road_safety\datasets\Merged\test\sample_video.mp4"
CONF_THRESHOLD = 0.3
DUPLICATE_DISTANCE_THRESHOLD_METERS = 50  # Avoid inserting potholes within 50 meters

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["road_safety"]
potholes_col = db["potholes"]

# Flask app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

@app.route("/api/potholes", methods=["GET"])
def get_potholes():
    data = list(potholes_col.find({}, {"_id": 0}))
    return jsonify(data)

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

def run_api():
    app.run(host="0.0.0.0", port=5000, debug=False)

Thread(target=run_api, daemon=True).start()

# -------------------------------
# YOLO + Alerts
# -------------------------------
engine = pyttsx3.init()
engine.setProperty("rate", 150)

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print(f"❌ Could not open video source {VIDEO_SOURCE}")
    exit()

print("✅ Real-time detection started. Press 'q' to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ End of video stream")
        break

    pothole_detected = False

    results = model(frame, stream=True)
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls == 8 and conf > CONF_THRESHOLD:  # pothole class
                pothole_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"Pothole {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # 🚩 Use real GPS data here instead of random values
                # Example placeholder values (replace with GPS sensor input)
                lat = 12.88 + random.uniform(0, 0.04)
                lon = 77.58 + random.uniform(0, 0.04)

                # Check for existing pothole within 50 meters
                is_duplicate = False
                for existing in potholes_col.find({}):
                    existing_coords = (existing["latitude"], existing["longitude"])
                    new_coords = (lat, lon)
                    distance = geodesic(existing_coords, new_coords).meters
                    if distance <= DUPLICATE_DISTANCE_THRESHOLD_METERS:
                        is_duplicate = True
                        print(f"⚠️ Duplicate pothole detected (within {distance:.1f} meters), skipping insert.")
                        break

                if not is_duplicate:
                    pothole_data = {
                        "latitude": lat,
                        "longitude": lon,
                        "confidence": conf,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    potholes_col.insert_one(pothole_data)
                    print(f"📍 New pothole saved: {pothole_data}")

                # Play alert if known pothole is nearby
                if is_duplicate:
                    winsound.Beep(1500, 400)
                    engine.say("Warning! Known pothole ahead!")
                    engine.runAndWait()

    # Alert if pothole detected in frame
    if pothole_detected:
        print("⚠️ ALERT: Pothole detected in video!")
        winsound.Beep(1000, 500)
        engine.say("Pothole ahead, slow down!")
        engine.runAndWait()

    # Show frame in real-time
    cv2.imshow("Pothole Detection (Real-Time)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
engine.stop()
