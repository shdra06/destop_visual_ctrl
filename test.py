import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -----------------------
# CONFIG
# -----------------------
MODEL_PATH = "hand_landmarker.task"

# -----------------------
# Create Hand Landmarker
# -----------------------
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
    running_mode=vision.RunningMode.VIDEO
)

landmarker = vision.HandLandmarker.create_from_options(options)

# -----------------------
# Gesture Geometry Logic
# -----------------------

def finger_folded(lm, tip, pip):
    return lm[tip].y > lm[pip].y


def is_thumb_down_all_closed(lm):
    thumb_down = lm[4].y > lm[2].y
    fingers_closed = (
        finger_folded(lm,8,6) and
        finger_folded(lm,12,10) and
        finger_folded(lm,16,14) and
        finger_folded(lm,20,18)
    )
    return thumb_down and fingers_closed


def is_fist(lm):
    return (
        finger_folded(lm,8,6) and
        finger_folded(lm,12,10) and
        finger_folded(lm,16,14) and
        finger_folded(lm,20,18)
    )

# -----------------------
# MAIN
# -----------------------
cap = cv2.VideoCapture(0)
start_time = time.time()

print("🚀 Test Started")
print("Class 2 = Brightness")
print("Config A: Thumb Down (closed)")
print("Config B: Fist")
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp = int((time.time() - start_time) * 1000)

    result = landmarker.detect_for_video(mp_image, timestamp)

    text = "Idle"
    color = (0, 0, 255)

    if result.hand_landmarks:
        lm = result.hand_landmarks[0]

        if is_thumb_down_all_closed(lm):
            text = "Class 2 - Config A (Thumb Down)"
            color = (0, 255, 0)

        elif is_fist(lm):
            text = "Class 2 - Config B (Fist)"
            color = (255, 255, 0)

    cv2.putText(frame, text,
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                color,
                3)

    cv2.imshow("Two Config Same Label Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
