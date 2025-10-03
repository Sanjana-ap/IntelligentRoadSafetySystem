from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
import math


def run_intelligent_road_safety(video_path, output_path="outputs/final_result.avi", scale=0.05):
    """
    Intelligent Road Safety System
    - Object detection (YOLOv8)
    - Tracking (DeepSORT)
    - Speed estimation
    - Vehicle counting
    """

    # Load YOLO model
    model = YOLO("yolov8n.pt")

    # Initialize DeepSORT
    tracker = DeepSort(max_age=30)

    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(
        output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4)))
    )

    # Vehicle counting line
    line_y = int(cap.get(4) * 0.5)  # middle of the frame
    offset = 10  # tolerance
    vehicle_count = 0
    crossed_ids = set()

    # For speed estimation
    last_positions = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = model(frame)
        detections = results[0].boxes.data.cpu().numpy()

        # Convert detections for DeepSORT
        dets_for_tracker = []
        for x1, y1, x2, y2, conf, cls in detections:
            dets_for_tracker.append(([x1, y1, x2 - x1, y2 - y1], conf, int(cls)))

        # Update tracker
        tracks = tracker.update_tracks(dets_for_tracker, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            x1, y1, x2, y2 = track.to_ltrb()
            track_id = track.track_id
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)  # center

            # Speed estimation
            if track_id in last_positions:
                prev_x, prev_y = last_positions[track_id]
                pixel_dist = math.sqrt((cx - prev_x) ** 2 + (cy - prev_y) ** 2)
                meters_per_frame = pixel_dist * scale
                speed = meters_per_frame * fps * 3.6  # km/h
            else:
                speed = 0.0

            last_positions[track_id] = (cx, cy)

            # Vehicle counting (crossing line)
            if abs(cy - line_y) < offset and track_id not in crossed_ids:
                vehicle_count += 1
                crossed_ids.add(track_id)

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Display ID + Speed
            cv2.putText(
                frame,
                f"ID {track_id} | {speed:.1f} km/h",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # Draw counting line
        cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)

        # Show vehicle count
        cv2.putText(
            frame,
            f"Vehicle Count: {vehicle_count}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            3,
        )

        out.write(frame)
        cv2.imshow("Intelligent Road Safety System", frame)

        if cv2.waitKey(1) & 0xFF in [27, ord("q")]:  # ESC or q to quit
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_intelligent_road_safety(r"C:\MajorProject\intelligent_road_safety\datasets\traffic_scene.mp4", r"outputs/final_result.avi")