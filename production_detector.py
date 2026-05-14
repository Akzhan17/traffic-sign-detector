# production_detector.py - READY TO USE!
from ultralytics import YOLO
import cv2

# ============================================
# COMPLETE LABEL MAPPING (Based on your testing)
# ============================================
LABELS = {
    # Confirmed working
    0: "Information",
    16: "No Vehicles",
    23: "Traffic Sign",
    33: "Traffic Sign",

    # Speed limits
    7: "20 km/h",
    8: "30 km/h",
    9: "40 km/h",
    10: "60 km/h",
    11: "50 km/h",
    12: "60 km/h",

    # Warning & prohibition
    1: "No Left Turn",
    2: "No Passing",
    21: "Bus Zone",
    22: "Pedestrian Crossing",
    25: "Parking",
    26: "Children Crossing",
    31: "Two-way Traffic",
    34: "Roundabout",
    35: "Straight or Right",
    36: "Straight Only",
    37: "Straight Only",
    38: "Yield",
    39: "Priority Road",
    41: "Children Crossing",
    42: "Road Work",

    # Direction
    24: "Ahead Only",

    # STOP
    40: "STOP",
}

# Color mapping for different sign types
SIGN_COLORS = {
    "STOP": (0, 0, 255),  # Red
    "km/h": (255, 0, 0),  # Blue for speed
    "Pedestrian": (255, 255, 0),  # Cyan
    "Children": (255, 255, 0),  # Cyan
    "Yield": (0, 255, 255),  # Yellow
    "Parking": (128, 0, 128),  # Purple
    "Road Work": (0, 165, 255),  # Orange
    "Roundabout": (0, 165, 255),  # Orange
}


def get_color(label):
    """Get color based on sign type"""
    for key, color in SIGN_COLORS.items():
        if key in label:
            return color
    return (0, 255, 0)  # Default green


def get_label(class_id):
    """Get label for class ID"""
    return LABELS.get(class_id, f"Unknown_{class_id}")


# ============================================
# MAIN DETECTION LOOP
# ============================================

print("🚦 TRAFFIC SIGN DETECTION SYSTEM")
print("=" * 50)
print(f"📋 Loaded {len(LABELS)} sign types")
print("\nControls:")
print("  'q' - Quit")
print("  's' - Save screenshot")
print("  '+' - Increase confidence")
print("  '-' - Decrease confidence")
print("  'l' - Show detected labels")
print("=" * 50)

# Load model
model = YOLO('model/best.pt')

# Start webcam
cap = cv2.VideoCapture(0)
confidence = 0.25
detected_history = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run detection
    results = model(frame, conf=confidence)

    current_detections = []

    # Draw results
    if results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # Get label
            label = get_label(cls)
            color = get_color(label)

            # Track detection
            current_detections.append(label)

            # Draw bounding box (thicker for STOP)
            thickness = 3 if label == "STOP" else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Draw label with background
            text = f"{label} ({conf:.0%})"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + text_w, y1), color, -1)
            cv2.putText(frame, text, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Update history
    if current_detections:
        detected_history = (detected_history + current_detections)[-10:]

    # Draw UI
    y_offset = 30
    cv2.putText(frame, f"Signs: {len(current_detections)}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_offset += 30

    cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_offset += 30

    # Show recent detections
    if detected_history:
        cv2.putText(frame, "Recent:", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y_offset += 20
        for i, sign in enumerate(set(detected_history[-5:])):
            cv2.putText(frame, f"  • {sign}", (10, y_offset + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # Display
    cv2.imshow('Traffic Sign Detection System', frame)

    # Handle keys
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        timestamp = cv2.getTickCount()
        filename = f"detection_{len(current_detections)}_signs.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 Saved: {filename}")
    elif key == ord('+') or key == ord('='):
        confidence = min(0.9, confidence + 0.05)
        print(f"Confidence: {confidence:.2f}")
    elif key == ord('-') or key == ord('_'):
        confidence = max(0.1, confidence - 0.05)
        print(f"Confidence: {confidence:.2f}")
    elif key == ord('l'):
        print("\n📋 Detected signs in this frame:")
        for sign in set(current_detections):
            print(f"  - {sign}")

cap.release()
cv2.destroyAllWindows()
print("\n✅ Detection stopped")