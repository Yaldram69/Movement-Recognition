import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8 model (use yolov8n.pt for speed, yolov8s.pt for better accuracy)
model = YOLO("yolov8m.pt")

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)

# Open webcam
cap = cv2.VideoCapture(0)

# Store previous object positions
previous_positions = {}

# Define movement threshold (pixels)
MOVEMENT_THRESHOLD = 10

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection
    results = model(frame)[0]

    # Extract bounding boxes, confidence scores, and class IDs
    detections = []
    for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        class_id = int(cls.item())
        detections.append(([x1, y1, x2, y2], conf.item(), class_id))

    # Update tracker with detections
    tracks = tracker.update_tracks(detections, frame=frame)

    # Store detected moving objects
    moving_objects = []

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id  # Unique ID for the tracked object
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        # Get class ID and name
        class_id = track.det_class
        class_name = model.names[class_id]

        # Calculate object center
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Check if the object has moved significantly
        if track_id in previous_positions:
            prev_x, prev_y = previous_positions[track_id]
            movement = np.sqrt((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2)

            if movement > MOVEMENT_THRESHOLD:
                moving_objects.append((x1, y1, x2, y2, class_name))
        else:
            # If first appearance, add object to moving list
            moving_objects.append((x1, y1, x2, y2, class_name))

        # Update previous position
        previous_positions[track_id] = (center_x, center_y)

    # Draw bounding boxes only for moving objects
    for x1, y1, x2, y2, class_name in moving_objects:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, class_name, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show output frame
    cv2.imshow("Moving Object Tracker", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
