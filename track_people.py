import cv2
from ultralytics import YOLO
import numpy as np

# ✅ Load YOLOv8n model
model = YOLO("yolov8n.pt")

# ✅ Load video
video_path = r"C:\Users\Hp\Downloads\peron_id\pervideo.mp4"
cap = cv2.VideoCapture(video_path)

cv2.namedWindow("YOLOv8 Person Detection", cv2.WINDOW_NORMAL)

# ✅ Tracker dict {id: [x, y]}
trackers = {}
next_id = 1
distance_threshold = 50  # Adjust for more or less strict matching

def get_centroid(x1, y1, x2, y2):
    return ((x1 + x2) // 2, (y1 + y2) // 2)

while True:
    ret, frame = cap.read()

    # ✅ Restart video if it ends
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        trackers.clear()
        next_id = 1
        continue

    frame = cv2.resize(frame, (640, 360))
    results = model(frame, conf=0.5, verbose=False)[0]

    current_centroids = []
    boxes = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        if label == "person":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2, y2))
            current_centroids.append(get_centroid(x1, y1, x2, y2))

    matched_ids = {}
    used_ids = set()

    # Match current centroids to previous trackers
    for i, centroid in enumerate(current_centroids):
        min_dist = float("inf")
        matched_id = None
        for tid, prev_centroid in trackers.items():
            if tid in used_ids:
                continue
            dist = np.linalg.norm(np.array(centroid) - np.array(prev_centroid))
            if dist < min_dist and dist < distance_threshold:
                min_dist = dist
                matched_id = tid

        if matched_id is not None:
            matched_ids[i] = matched_id
            trackers[matched_id] = centroid
            used_ids.add(matched_id)
        else:
            matched_ids[i] = next_id
            trackers[next_id] = centroid
            next_id += 1

    # ✅ Draw boxes and consistent labels
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        person_id = matched_ids[i]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(frame, f'Person {person_id}', (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    cv2.imshow("YOLOv8 Person Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
