import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
from collections import deque
import pyautogui
import subprocess
import threading

# Import MediaPipe Tasks
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Local imports
from utils import normalize_landmarks

# --- Configuration ---
MODEL_FILE = 'gesture_model.h5'
CONFIDENCE_THRESHOLD = 0.8  # Higher threshold for stability
MOVEMENT_THRESHOLD = 0.02 # Sensitivity for Volume movement
SMOOTHING_BUFFER = 5

# --- Volume Control (PowerShell WScript.Shell) ---
def change_volume_worker(direction, steps=1):
    """
    Worker to send volume keys via PowerShell.
    steps: Number of times to press the key (simulating variable speed).
    """
    try:
        # [char]175 is Volume Up, [char]174 is Volume Down
        char_code = 175 if direction == 1 else 174
        
        # Construct command to repeat the key press 'steps' times
        # Using a simple loop in PowerShell is more efficient than spawning multiple processes
        ps_script = f"""
        $obj = New-Object -ComObject WScript.Shell
        for ($i=0; $i -lt {steps}; $i++) {{
            $obj.SendKeys([char]{char_code})
            Start-Sleep -Milliseconds 10 
        }}
        """
        # Collapse to single line for subprocess
        cmd = ps_script.replace('\n', ';')
        
        subprocess.run(["powershell", "-Command", cmd], 
                       stdout=subprocess.DEVNULL, 
                       stderr=subprocess.DEVNULL, 
                       creationflags=subprocess.CREATE_NO_WINDOW)
    except Exception as e:
        print(f"Volume Error: {e}")

def change_volume_step(direction, steps=1):
    """
    Change volume by spawning a thread.
    direction: 1 (Up), -1 (Down)
    steps: Number of increments (determines speed)
    """
    threading.Thread(target=change_volume_worker, args=(direction, steps), daemon=True).start()


import subprocess
import threading

# --- Brightness Control (PowerShell WMI) ---
current_brightness = 50

def init_brightness():
    global current_brightness
    try:
        # Get current brightness via PowerShell
        cmd = "(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightness).CurrentBrightness"
        # Startup can be sync
        result = subprocess.check_output(["powershell", "-Command", cmd], creationflags=subprocess.CREATE_NO_WINDOW)
        current_brightness = int(result.decode().strip())
        print(f"Initial Brightness: {current_brightness}")
    except Exception as e:
        print(f"Brightness Init Error: {e}")
        current_brightness = 50

def set_brightness_worker(value):
    """Worker function to run PowerShell command in a thread"""
    try:
        cmd = f"(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1,{value})"
        subprocess.run(["powershell", "-Command", cmd], 
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL,
                       creationflags=subprocess.CREATE_NO_WINDOW)
    except Exception as e:
        print(f"Set Brightness Error: {e}")

def change_brightness_step(direction):
    global current_brightness
    
    step = 10 # Larger step since rate limit is high
    current_brightness += direction * step
    current_brightness = max(0, min(100, current_brightness))
    
    # Run in thread to avoid lagging the video loop
    threading.Thread(target=set_brightness_worker, args=(current_brightness,), daemon=True).start()
    print(f"Action: Brightness -> {current_brightness}")

def main():
    # Load Model
    try:
        model = tf.keras.models.load_model(MODEL_FILE)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Initialize Brightness
    init_brightness()

    # Initialize MediaPipe - SINGLE HAND
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
        running_mode=vision.RunningMode.VIDEO)
    
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    start_time = time.time()
    
    # State variables
    prev_y = None
    gesture_buffer = deque(maxlen=SMOOTHING_BUFFER)
    
    # Timer Logic
    # Timer Logic
    current_gesture_state = None 
    gesture_hold_start_time = 0
    is_active = False
    HOLD_DURATION = 3.0 # User requested 3 seconds
    last_brightness_update = 0 
    last_volume_update = 0
    last_desktop_toggle = 0
    
    print("Starting gesture control...")
    print(f"HOLD a gesture for {HOLD_DURATION}s to ACTIVATE it.")
    print("Commands:")
    print("  - Pinch (Class 0)    -> Volume Mode (Hold 3s, then Drag Up/Down)")
    print("  - Index UP (Class 1) -> Brightness Up")
    print("  - Index DOWN (Class 2)-> Brightness Down")
    print("  - Open Hand (Class 3)-> Toggle Desktop")
    print("  - Low Confidence     -> Idle")
    print("Press 'q' to quit.")

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
            continue

        gesture_text = "Idle"
        color = (0, 0, 255) # Red
        progress_bar_val = 0

        if detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]
            
            # Draw
            for lm in hand_landmarks:
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

            # Preprocess & Predict
            input_data = np.array([normalize_landmarks(hand_landmarks)])
            prediction = model.predict(input_data, verbose=0)
            class_id = np.argmax(prediction)
            confidence = prediction[0][class_id]
            
            # 0=Vol, 1=B_Up, 2=B_Down, 3=Desktop
            if confidence > CONFIDENCE_THRESHOLD:
                gesture_buffer.append(class_id)
            else:
                gesture_buffer.append(4) # Internal "Idle" state
            
            # Majority Voting
            if len(gesture_buffer) == SMOOTHING_BUFFER:
                detected_state = max(set(gesture_buffer), key=gesture_buffer.count)
                
                # State Machine
                if detected_state != 4: # If NOT Idle
                    if detected_state == current_gesture_state:
                        # Holding same gesture
                        elapsed = time.time() - gesture_hold_start_time
                        
                        if elapsed >= HOLD_DURATION:
                            is_active = True
                            progress_bar_val = 1.0
                        else:
                            is_active = False
                            progress_bar_val = elapsed / HOLD_DURATION
                            gesture_text = f"Holding... {HOLD_DURATION - elapsed:.1f}s"
                            color = (0, 165, 255) # Orange during hold
                            
                    else:
                        # New gesture detected
                        current_gesture_state = detected_state
                        gesture_hold_start_time = time.time()
                        is_active = False
                        prev_y = hand_landmarks[8].y # Reset reference
                        gesture_text = "New Gesture Detected"
                else:
                    # Idle detected
                    current_gesture_state = None
                    is_active = False
                    prev_y = None
                    gesture_text = "Idle"

                # Execution Logic (Only if Active)
                if is_active:
                    color = (0, 255, 0) # Green
                    
                    # Class 0: Volume Mode (Pinch & Drag)
                    if current_gesture_state == 0:
                        gesture_text = "ACTIVE: Volume Mode (Move UP/DOWN)"
                        
                        current_y = hand_landmarks[8].y # Index tip
                        if prev_y is not None:
                            delta_y = current_y - prev_y
                            # Sensitivity check
                            speed = abs(delta_y)
                            if speed > MOVEMENT_THRESHOLD:
                                if time.time() - last_volume_update > 0.1: # More responsive (10Hz)
                                    # Determine steps based on speed (delta_y magnitude)
                                    # User req: Traversing 0-100 without leaving frame
                                    # Increased sensitivity:
                                    # Slow: 3 steps (~6%)
                                    # Normal: 8 steps (~16%)
                                    # Fast: 15 steps (~30%)
                                    
                                    if speed < 0.04:
                                        steps = 3
                                        speed_text = "SLOW"
                                    elif speed < 0.08:
                                        steps = 8
                                        speed_text = "NORMAL"
                                    else:
                                        steps = 15
                                        speed_text = "FAST"

                                    # Y increases downwards.
                                    if delta_y < 0: # Moving UP
                                        change_volume_step(1, steps)
                                        gesture_text = f"Vol UP ({speed_text})"
                                    else: # Moving DOWN
                                        change_volume_step(-1, steps)
                                        gesture_text = f"Vol DOWN ({speed_text})"
                                        
                                    last_volume_update = time.time()
                        prev_y = current_y

                    # Class 1: Brightness UP
                    elif current_gesture_state == 1:
                        gesture_text = "ACTIVE: Bright UP"
                        if time.time() - last_brightness_update > 0.5: 
                            change_brightness_step(1)
                            last_brightness_update = time.time()

                    # Class 2: Brightness DOWN
                    elif current_gesture_state == 2:
                        gesture_text = "ACTIVE: Bright DOWN"
                        if time.time() - last_brightness_update > 0.5:
                            change_brightness_step(-1)
                            last_brightness_update = time.time()

                    # Class 3: Show Desktop
                    elif current_gesture_state == 3:
                        gesture_text = "ACTIVE: Show Desktop"
                        if time.time() - last_desktop_toggle > 2.0: # Debounce
                            # Toggle Desktop via PowerShell
                            subprocess.Popen(["powershell", "-Command", "(New-Object -ComObject Shell.Application).ToggleDesktop()"], 
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
                            last_desktop_toggle = time.time()
                            is_active = False # Reset after toggle
                            current_gesture_state = None # Force reset to avoid spamming
                            
                    # Internal Idle (4) or unknown
                    else:
                        gesture_text = "Idle"
                        is_active = False

        else:
             # No hand
             current_gesture_state = None
             is_active = False

        # Visual Feedback (Status + simple bar)
        cv2.putText(frame, f"{gesture_text}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        
        # Draw Progress Bar
        if not is_active and current_gesture_state is not None:
             bar_w = 200
             cv2.rectangle(frame, (10, 60), (10 + int(bar_w * progress_bar_val), 80), color, -1)
             cv2.rectangle(frame, (10, 60), (10 + bar_w, 80), (255, 255, 255), 2)

        cv2.imshow('Gesture Control', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
