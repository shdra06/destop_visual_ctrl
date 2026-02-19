import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time

# Import MediaPipe Tasks
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Local imports
from utils import normalize_landmarks

# Config
MODEL_FILE = 'gesture_model.h5'
# Labels MUST match train_model.py encoding
# The original LABELS list is kept as it's used for display,
# and new CLASSES and COLORS are introduced for potential future use or specific visualization.
LABELS = ["Volume (0)", "Bright UP (1)", "Bright DOWN (2)", "Idle (3)"]

def main():
    print("Loading model...") # Load Model
    try:
        model = tf.keras.models.load_model(MODEL_FILE)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Updated Classes
    CLASSES = ['Volume', 'Brightness_Up', 'Brightness_Down', 'Show_Desktop', 'Idle']
    COLORS = [(0, 255, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255), (0, 0, 255)] 
    # Green, Cyan, Yellow, Magenta, Red

    # Initialize MediaPipe
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5, # Lower for testing
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.VIDEO)
    
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    start_time = time.time()

    print("Starting Gesture Tester...")
    print("Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for mirror view (same as training/main)
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.time() - start_time) * 1000)

        try:
            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
        except:
            continue

        probs = None
        predicted_idx = -1

        if detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]
            
            # Draw landmarks
            for lm in hand_landmarks:
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

            # Predict
            input_data = np.array([normalize_landmarks(hand_landmarks)])
            prediction = model.predict(input_data, verbose=0)
            probs = prediction[0]
            predicted_idx = np.argmax(probs)

        # Visualization
        h, w = frame.shape[:2]
        
        # Draw status
        if probs is not None:
            # Draw predicted class
            text = f"Pred: {LABELS[predicted_idx]} ({probs[predicted_idx]*100:.1f}%)"
            cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw Probability Bars
            bar_x = 10
            bar_y = 80
            bar_h = 20
            bar_w_max = 200
            
            for i, prob in enumerate(probs):
                label_text = LABELS[i] if i < len(LABELS) else f"Class {i}"
                
                # Bar background
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w_max, bar_y + bar_h), (50, 50, 50), -1)
                # Bar fill
                fill_w = int(bar_w_max * prob)
                color = (0, 255, 255) if i == predicted_idx else (200, 200, 200)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
                
                # Text
                cv2.putText(frame, f"{label_text}: {prob:.2f}", (bar_x + bar_w_max + 10, bar_y + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                bar_y += 30
        else:
            cv2.putText(frame, "No Hand Detected", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Gesture Tester", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
