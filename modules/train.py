# train.py
from ultralytics import YOLO
import os

# ------------------------------
# Paths
# ------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "../datasets/merged/dataset.yaml")  # adjust if running from modules/
PRETRAINED_MODEL = os.path.join(PROJECT_ROOT, "../models/yolov8n.pt")    # pretrained YOLOv8
SAVE_DIR = os.path.join(PROJECT_ROOT, "../models/trained")               # where to save trained weights

os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------------------
# Training configuration
# ------------------------------
EPOCHS = 50
IMG_SIZE = 640

BATCH_SIZE = 16  # adjust based on your GPU/CPU memory

# ------------------------------
# Start training
# ------------------------------
def main():
    print("🚀 Starting YOLOv8 training...")
    print(f"Dataset: {DATA_PATH}")
    print(f"Pretrained model: {PRETRAINED_MODEL}")
    print(f"Saving weights to: {SAVE_DIR}\n")

    model = YOLO(PRETRAINED_MODEL)

    model.train(
        data=DATA_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=SAVE_DIR,
        name="pothole_detector",
        exist_ok=True  # overwrite if folder exists
    )

    print("\n✅ Training completed!")
    print(f"Best weights saved at: {os.path.join(SAVE_DIR, 'pothole_detector', 'weights', 'best.pt')}")

if __name__ == "__main__":
    main()
