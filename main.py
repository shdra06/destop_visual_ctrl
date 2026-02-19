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
        HEADLESS = config.get("headless", False)
        ALWAYS_ON_TOP = config.get("always_on_top", True)
except Exception as e:
    CLASSES = ['Volume', 'Bright_Up', 'Bright_Down', 'Show_Desktop']
    MOUSE_MODE = False
    HEADLESS = False
    ALWAYS_ON_TOP = True

MODEL_FILE = 'gesture_model.h5'
SMOOTHING_BUFFER = 5

def is_victory_gesture(landmarks):
    """
    Detects 'Victory' (Peace) sign: Index and Middle extended, others curled.
    """
    # Landmarks: 8 (Index Tip), 6 (Index PIP), 12 (Middle Tip), 10 (Middle PIP)
    # 16 (Ring Tip), 14 (Ring PIP), 20 (Pinky Tip), 18 (Pinky PIP)
    
    # Check Index and Middle UP (Tip higher than PIP, y is inverted)
    index_up = landmarks[8].y < landmarks[6].y
    middle_up = landmarks[12].y < landmarks[10].y
    
    # Check Ring and Pinky DOWN (Tip lower/same as PIP)
    ring_down = landmarks[16].y > landmarks[14].y
    pinky_down = landmarks[20].y > landmarks[18].y
    
    return index_up and middle_up and ring_down and pinky_down

def main():
    # Optimizations
    pyautogui.PAUSE = 0
    pyautogui.FAILSAFE = False # We handle boundaries manually or via try-except
    
    # Initialize Controllers
    sys_ctrl = SystemController()
    gesture_params = GestureParams(CLASSES)

    # Load Model (Only needed for Gesture Mode)
    model = None
    # We load model once, even if MOUSE_MODE is enabled initially, just in case we switch.
    # Actually, if we switch to Gesture Mode, we need model. 
    # Let's load it if not loaded.
    try:
        model = tf.keras.models.load_model(MODEL_FILE)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        # Proceed, but gesture mode won't work well
    
    # Initialize Video Stream ONCE
    cap = VideoStream(0).start()
    fps_monitor = FPSMonitor()
    
    global MOUSE_MODE # Use global to update config if needed or just local variable
    # Actually locally is fine, we update config file too.
    
    try:
        while True: # Outer Loop for Mode Switching
            
            # Initialize MediaPipe based on current Mode
            num_hands = 2 if MOUSE_MODE else 1
            print(f"Initializing MediaPipe for {num_hands} hand(s)...")

            base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=num_hands,
                min_hand_detection_confidence=0.5, # Lowered for speed
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                running_mode=vision.RunningMode.VIDEO)
            
            # Helper for Mode Switch
            mode_switch_start_time = 0
            MODE_SWITCH_DELAY = 1.0 
            should_break = False # Flag to break inner loop
            should_exit = False # Flag to exit program

            with vision.HandLandmarker.create_from_options(options) as landmarker:
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
                smoothing_alpha = 0.5
                already_clicked = False

                print("Active.")
                print(f"Mode: {'MOUSE MODE (2 Hands)' if MOUSE_MODE else 'GESTURE MODE (1 Hand)'}")
                print(f"Window: {'HEADLESS' if HEADLESS else ('ALWAYS ON TOP' if ALWAYS_ON_TOP else 'NORMAL')}")
                print("Press 'q' (or Ctrl+C in Headless) to quit.")

                while True: # Inner Loop (Frame Processing)
                    frame = cap.read()
                    if frame is None:
                        should_exit = True
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
                    
                    # --- VICTORY GESTURE CHECK (Mode Switch) ---
                    victory_detected = False
                    if detection_result.hand_landmarks:
                        # Check all detected hands
                        for hl in detection_result.hand_landmarks:
                            if is_victory_gesture(hl):
                                victory_detected = True
                                break # Found one
                    
                    if victory_detected:
                        if mode_switch_start_time == 0:
                            mode_switch_start_time = time.time()
                        
                        elapsed = time.time() - mode_switch_start_time
                        remaining = MODE_SWITCH_DELAY - elapsed
                        
                        msg = "Hold Victory to Switch..." if remaining > 0 else "Switching..."
                        cv2.putText(frame, msg, (int(frame.shape[1]/2)-100, int(frame.shape[0]/2)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
                        
                        # Draw Progress Bar
                        bar_w = 200
                        progress = min(1.0, elapsed / MODE_SWITCH_DELAY)
                        cv2.rectangle(frame, (220, 250), (220 + int(bar_w * progress), 270), (255, 0, 255), -1)
                        cv2.rectangle(frame, (220, 250), (220 + bar_w, 270), (255, 255, 255), 2)

                        if elapsed >= MODE_SWITCH_DELAY:
                            MOUSE_MODE = not MOUSE_MODE
                            # Update Config File for persistence (optional)
                            try:
                                with open(CONFIG_FILE, 'r') as f:
                                    c = json.load(f)
                                c["mouse_mode"] = MOUSE_MODE
                                with open(CONFIG_FILE, 'w') as f:
                                    json.dump(c, f, indent=4)
                            except: pass
                            
                            # Visual Feedback before break
                            cv2.putText(frame, f"SWITCHED TO {'MOUSE' if MOUSE_MODE else 'GESTURE'}!", 
                                       (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
                            if not HEADLESS:
                                cv2.imshow('Gesture Control', frame)
                                cv2.waitKey(500) # Pause briefly to show message
                            should_break = True
                            break # Break Inner Loop
                    else:
                        mode_switch_start_time = 0


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
                                
                                if label == "Left":
                                    cursor_hand = hand_landmarks
                                elif label == "Right":
                                    click_hand = hand_landmarks
                        
                        # 1. Cursor Control (Left Hand)
                        if cursor_hand:
                            lm = cursor_hand[8] # Index Tip
                            
                            # --- Mouse Sensitivity / Gain Logic ---
                            # Map a smaller area of the camera to the full screen to increase speed
                            h, w, c = frame.shape
                            margin = 50 # Pixels from edge (Balanced: Reachable corners but stable)
                            
                            # Clamp raw coordinates to the active area
                            lm_x = max(margin, min(w - margin, int(lm.x * w)))
                            lm_y = max(margin, min(h - margin, int(lm.y * h)))
                            
                            # Map to Screen Coordinates
                            target_x = np.interp(lm_x, [margin, w - margin], [0, screen_w])
                            target_y = np.interp(lm_y, [margin, h - margin], [0, screen_h])
                            
                            # Smoothing (Alpha 0.4 = Good balance of speed and smoothness)
                            smoothing_alpha = 0.4 
                            cur_x = prev_x + smoothing_alpha * (target_x - prev_x)
                            cur_y = prev_y_mouse + smoothing_alpha * (target_y - prev_y_mouse)
                            
                            try:
                                pyautogui.moveTo(cur_x, cur_y)
                            except pyautogui.FailSafeException:
                                pass
                                
                            prev_x, prev_y_mouse = cur_x, cur_y
                            
                            # Visual Guide for Active Area
                            cv2.rectangle(frame, (margin, margin), (w - margin, h - margin), (0, 255, 255), 1)
                            cv2.putText(frame, "Cursor: Active", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        
                        # 2. Click Control (Right Hand)
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
                            # Check model
                            if model:
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
                                gesture_text = "Model Not Loaded"

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
                    if not HEADLESS:
                        window_name = 'Gesture Control'
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                        
                        if ALWAYS_ON_TOP:
                            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

                        cv2.imshow(window_name, frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            should_exit = True
                            break
                    else:
                        # Headless Mode: Sleep to prevent 100% CPU usage
                        time.sleep(0.03) 
                
                # Inner loop broke (Switch or Exit)
                if should_exit:
                    break
                if should_break:
                    continue # Re-init

            # Outer loop continues (Re-init)
            if should_exit:
                break
            
    finally:       
        cap.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
