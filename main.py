import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
from collections import deque
import pyautogui
import threading
import json
import math

# Import Performance Module
from performance import setup_performance, VideoStream, FPSMonitor
# Import Gesture Logic Module
from gesture_control import SystemController, GestureParams

# Run Optimizations
setup_performance()

# Import MediaPipe Tasks
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Local imports
from utils import normalize_landmarks

# --- Configuration ---
CONFIG_FILE = 'gestures_config.json'
try:
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
        CLASSES = config.get("gestures", [])
        MOUSE_MODE = config.get("mouse_mode", False)
except Exception as e:
    CLASSES = ['Volume', 'Bright_Up', 'Bright_Down', 'Show_Desktop']
    MOUSE_MODE = False

MODEL_FILE = 'gesture_model.h5'
SMOOTHING_BUFFER = 5

def main():
    # Optimizations
    pyautogui.PAUSE = 0
    pyautogui.FAILSAFE = False # We handle boundaries manually or via try-except
    
    # Initialize Controllers
    sys_ctrl = SystemController()
    gesture_params = GestureParams(CLASSES)

    # Load Model (Only needed for Gesture Mode)
    model = None
    if not MOUSE_MODE:
        try:
            model = tf.keras.models.load_model(MODEL_FILE)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    
    # Initialize MediaPipe
    # Mouse Mode requires 2 hands
    num_hands = 2 if MOUSE_MODE else 1
    
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=num_hands,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
        running_mode=vision.RunningMode.VIDEO)
    
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        cap = VideoStream(0).start()
        fps_monitor = FPSMonitor()
        start_time = time.time()
        
        # State variables (Gesture Mode)
        prev_y = None
        gesture_buffer = deque(maxlen=SMOOTHING_BUFFER)
        current_gesture_state = None 
        gesture_hold_start_time = 0
        is_active = False
        
        # State variables (Mouse Mode)
        screen_w, screen_h = pyautogui.size()
        prev_x, prev_y_mouse = 0, 0
        smoothing_alpha = 0.2
        already_clicked = False

        print("Starting Control System...")
        print(f"Mode: {'MOUSE MODE (2 Hands)' if MOUSE_MODE else 'GESTURE MODE (1 Hand)'}")
        print("Press 'q' to quit.")

        while True:
            frame = cap.read()
            if frame is None:
                break
                
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            fps = fps_monitor.update()
            cv2.putText(frame, f"FPS: {fps}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            timestamp_ms = int((time.time() - start_time) * 1000)
        
            try:
                detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
            except Exception as e:
                continue

            # Draw Landmarks
            if detection_result.hand_landmarks:
                for hand_landmarks in detection_result.hand_landmarks:
                    for lm in hand_landmarks:
                        h, w, c = frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

            # --- MOUSE MODE LOGIC ---
            if MOUSE_MODE:
                cursor_hand = None
                click_hand = None
                
                if detection_result.hand_landmarks and detection_result.handedness:
                    for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
                        # Handedness label (Left/Right)
                        # Note: MediaPipe mirrors handedness if flip is not handled correctly, 
                        # but since we flipped image, Right hand should be "Right".
                        # Let's trust the label for now.
                        label = detection_result.handedness[i][0].category_name
                        
                        if label == "Right":
                            cursor_hand = hand_landmarks
                        elif label == "Left":
                            click_hand = hand_landmarks
                
                # 1. Cursor Control (Right Hand)
                if cursor_hand:
                    lm = cursor_hand[8] # Index Tip
                    # Screen Mapping
                    target_x = int(lm.x * screen_w)
                    target_y = int(lm.y * screen_h)
                    
                    # Smoothing
                    cur_x = prev_x + smoothing_alpha * (target_x - prev_x)
                    cur_y = prev_y_mouse + smoothing_alpha * (target_y - prev_y_mouse)
                    
                    try:
                        pyautogui.moveTo(cur_x, cur_y)
                    except pyautogui.FailSafeException:
                        pass
                        
                    prev_x, prev_y_mouse = cur_x, cur_y
                    cv2.putText(frame, "Cursor: Active", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                # 2. Click Control (Left Hand)
                if click_hand:
                    # Pinch Detection (Thumb 4, Index 8)
                    x1, y1 = click_hand[4].x, click_hand[4].y
                    x2, y2 = click_hand[8].x, click_hand[8].y
                    distance = math.hypot(x2 - x1, y2 - y1)
                    
                    # Visual feedback for pinch
                    h, w, c = frame.shape
                    cx, cy = int(x1 * w), int(y1 * h)
                    cv2.putText(frame, f"Dist: {distance:.3f}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                    if distance < 0.05: # Pinch Threshold
                        if not already_clicked:
                            pyautogui.click()
                            already_clicked = True
                            cv2.putText(frame, "CLICK!", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        already_clicked = False

            # --- GESTURE MODE LOGIC ---
            else: 
                gesture_text = "Idle"
                color = (0, 0, 255)
                progress_bar_val = 0

                if detection_result.hand_landmarks:
                    hand_landmarks = detection_result.hand_landmarks[0]
                    
                    # Inference
                    input_data = np.array([normalize_landmarks(hand_landmarks)])
                    prediction = model.predict(input_data, verbose=0)
                    class_id = np.argmax(prediction)
                    confidence = prediction[0][class_id]
                    
                    threshold = gesture_params.get_confidence_threshold(class_id)
                    
                    if confidence > threshold:
                        gesture_buffer.append(class_id)
                    else:
                        gesture_buffer.append(-1)
                    
                    if len(gesture_buffer) == SMOOTHING_BUFFER:
                        detected_state = max(set(gesture_buffer), key=gesture_buffer.count)
                        
                        # State Machine
                        if detected_state != -1 and detected_state != gesture_params.IDLE_ID:
                            if detected_state == current_gesture_state:
                                elapsed = time.time() - gesture_hold_start_time
                                duration = gesture_params.get_hold_duration(detected_state)
                                
                                if elapsed >= duration:
                                    is_active = True
                                    progress_bar_val = 1.0
                                else:
                                    is_active = False
                                    if duration > 0:
                                        progress_bar_val = elapsed / duration
                                        gesture_text = f"Holding... {duration - elapsed:.1f}s"
                                    else:
                                        progress_bar_val = 1.0
                                        gesture_text = "Active"
                                    color = (0, 165, 255)
                            else:
                                current_gesture_state = detected_state
                                gesture_hold_start_time = time.time()
                                is_active = False
                                prev_y = hand_landmarks[8].y # Reset reference
                                gesture_text = "New Gesture Detected"
                        else:
                            current_gesture_state = None
                            is_active = False
                            prev_y = None
                            gesture_text = "Idle"

                        # Execution
                        if is_active:
                            color = (0, 255, 0)
                            
                            # Volume
                            if current_gesture_state == gesture_params.VOL_ID:
                                current_y = hand_landmarks[8].y
                                if prev_y is not None:
                                    delta_y = current_y - prev_y
                                    
                                    # Use Helper for Logic
                                    steps, speed_text = gesture_params.get_volume_steps(delta_y)
                                    
                                    if steps > 0:
                                        # Y increases down
                                        if delta_y < 0: # Up
                                            sys_ctrl.change_volume(1, steps)
                                            gesture_text = "Active: Volume Up"
                                        else:
                                            sys_ctrl.change_volume(-1, steps)
                                            gesture_text = "Active: Volume Down"
                                    else:
                                        gesture_text = "Active: Volume Mode"
                                
                                prev_y = current_y

                            # Brightness
                            elif current_gesture_state == gesture_params.B_UP_ID:
                                sys_ctrl.change_brightness(1)
                                gesture_text = "Active: Brightness Up"
                            
                            elif current_gesture_state == gesture_params.B_DOWN_ID:
                                sys_ctrl.change_brightness(-1)
                                gesture_text = "Active: Brightness Down"

                            # Desktop
                            elif current_gesture_state == gesture_params.DESK_ID:
                                if sys_ctrl.toggle_desktop():
                                    gesture_text = "Active: Desktop Toggle"
                                    is_active = False
                                    current_gesture_state = None
                            
                            else:
                                gesture_name = CLASSES[current_gesture_state]
                                gesture_text = f"Active: {gesture_name}"
                                # No system action for new/custom gestures by default
                                is_active = False

                else:
                    current_gesture_state = None
                    is_active = False

                # Visual Feedback
                cv2.putText(frame, f"{gesture_text}", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                
                if not is_active and current_gesture_state is not None:
                     bar_w = 200
                     cv2.rectangle(frame, (10, 60), (10 + int(bar_w * progress_bar_val), 80), color, -1)
                     cv2.rectangle(frame, (10, 60), (10 + bar_w, 80), (255, 255, 255), 2)

            # Common Show
            cv2.imshow('Gesture Control', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
