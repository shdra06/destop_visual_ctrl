import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time

import json

# Load classes from config
CONFIG_FILE = 'gestures_config.json'
try:
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
        CLASSES = config.get("gestures", [])
except Exception as e:
    print(f"Error loading config: {e}")
    CLASSES = ['Volume', 'Bright_Up', 'Bright_Down', 'Show_Desktop'] # Fallback

DATA_DIR = 'data'
SAMPLES_PER_CLASS = 1000 # Good amount for stability

# Import MediaPipe Tasks
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# File to store data
DATA_FILE = 'gesture_data.csv'

# Check if file exists, if not create with header
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        # 21 landmarks * 3 coordinates (x, y, z) + label = 64 columns
        header = []
        for i in range(21):
            header.extend([f'x{i}', f'y{i}', f'z{i}'])
        header.append('label')
        writer.writerow(header)

def normalize_landmarks(landmarks):
    # Convert list of objects to numpy array
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    
    # Relative usage: subtract wrist (landmark 0)
    base_x, base_y, base_z = coords[0]
    coords[:, 0] -= base_x
    coords[:, 1] -= base_y
    coords[:, 2] -= base_z
    
    # Scale invariance: divide by max absolute value
    max_val = np.max(np.abs(coords))
    if max_val > 0:
        coords /= max_val
        
    return coords.flatten().tolist()

def main():
    # Create an HandLandmarker object.
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
        running_mode=vision.RunningMode.VIDEO)

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)

        print("--- GESTURE DATA COLLECTION ---")
        print("--- GESTURE DATA COLLECTION ---")
        print(f"Press keys 0-{len(CLASSES)-1} to record {SAMPLES_PER_CLASS} samples per class:")
        
        for i, gesture in enumerate(CLASSES):
            print(f"  [{i}] {gesture}")
            
        print("Press 'q' to quit.")
        
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Flip frame for mirror view
            frame = cv2.flip(frame, 1)
            # Convert to MediaPipe Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Timestamp required for VIDEO mode
            timestamp_ms = int((time.time() - start_time) * 1000)
            
            try:
                detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
            except Exception as e:
                print(f"Detection error: {e}")
                continue
            
            # Visualization
            if detection_result.hand_landmarks:
                # We need to draw manually or use a helper. 
                # For simplicity in this script, we'll just draw circles for joints.
                for hand_landmarks in detection_result.hand_landmarks:
                    for lm in hand_landmarks:
                        h, w, c = frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            cv2.imshow('Collect Data', frame)
            
            # Key Handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            # Dynamic key check based on number of classes
            if 0 <= key - ord('0') < len(CLASSES):
                if detection_result.hand_landmarks:
                    label = int(chr(key))
                    # Get landmarks for the first hand
                    landmarks = detection_result.hand_landmarks[0]
                    normalized_landmarks = normalize_landmarks(landmarks)
                    
                    # Append label
                    normalized_landmarks.append(label)
                    
                    # Save to CSV
                    with open(DATA_FILE, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(normalized_landmarks)
                    
                    print(f"Recorded sample for class {label}: {CLASSES[label]}")
                else:
                    print("No hand detected! Cannot record.")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
