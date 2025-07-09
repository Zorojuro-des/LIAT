import cv2
import json
from ultralytics import YOLO
from tqdm import tqdm
import os

def extract_relative_position(bbox, frame_width, frame_height):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return [cx / frame_width, cy / frame_height]

def run_yolo_on_video(video_path, model_path, output_json_path):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    frame_data = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for frame_idx in tqdm(range(frame_count), desc=f"Processing {os.path.basename(video_path)}"):
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, verbose=False)[0]
        detections = []

        for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            label = model.names[int(cls)]
            bbox = box.tolist()
            rel_pos = extract_relative_position(bbox, width, height)

            detections.append({
                "class": label,
                "bbox": bbox,
                "confidence": float(conf),
                "relative_position": rel_pos
            })

        frame_data.append({
            "frame": frame_idx,
            "detections": detections
        })

    with open(output_json_path, "w") as f:
        json.dump(frame_data, f, indent=2)

    cap.release()
    print(f"[✓] Done: {output_json_path}")

# Example usage:
if __name__ == "__main__":
    run_yolo_on_video("broadcast.mp4", "best.pt", "detections_view_a.json")
    run_yolo_on_video("tacticam.mp4", "best.pt", "detections_view_b.json")
