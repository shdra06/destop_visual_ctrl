import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time

import json


CONFIG_FILE = 'gestures_config.json'
try:
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
        CLASSES = config.get("gestures", [])
except Exception as e:
    print(f"Error loading config: {e}")
    CLASSES = ['Volume', 'Bright_Up', 'Bright_Down', 'Show_Desktop'] 

DATA_DIR = 'data'
SAMPLES_PER_CLASS = 1000 


from mediapipe.tasks import python
from mediapipe.tasks.python import vision


DATA_FILE = 'gesture_data.csv'


if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        
        header = []
        for i in range(21):
            header.extend([f'x{i}', f'y{i}', f'z{i}'])
        header.append('label')
        writer.writerow(header)

def normalize_landmarks(landmarks):
    
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    
    
    base_x, base_y, base_z = coords[0]
    coords[:, 0] -= base_x
    coords[:, 1] -= base_y
    coords[:, 2] -= base_z
    
    
    max_val = np.max(np.abs(coords))
    if max_val > 0:
        coords /= max_val
        
    return coords.flatten().tolist()

def main():
    
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
        
        print(f"Press keys 0-{len(CLASSES)-1} to record {SAMPLES_PER_CLASS} samples per class:")
        
        SUGGESTED_POSES = {
            "Volume": "Pinch (👌) (Index+Thumb Closed)",
            "Bright_Up": "Thumb Up (👍) (Hand Closed)",
            "Bright_Down": "Thumb Down (👎) (Hand Closed)",
            "Show_Desktop": "Rock (🤘) / Thumb + 2 Fingers",
            "Idle": "No Command / Random Motion / Resting (🚫)",
            "Victory": "Peace Sign (✌️) (Index + Middle Finger Up)"
        }

        for i, gesture in enumerate(CLASSES):
            desc = SUGGESTED_POSES.get(gesture, "Custom Pose")
            print(f"  [{i}] {gesture}: {desc}")
            
        print("Press 'q' to quit.")
        
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            
            frame = cv2.flip(frame, 1)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            
            timestamp_ms = int((time.time() - start_time) * 1000)
            
            try:
                detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
            except Exception as e:
                print(f"Detection error: {e}")
                continue
            
            
            if detection_result.hand_landmarks:
                
                for hand_landmarks in detection_result.hand_landmarks:
                    for lm in hand_landmarks:
                        h, w, c = frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            cv2.imshow('Collect Data', frame)
            
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            
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
