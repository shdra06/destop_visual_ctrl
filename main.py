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

import os
import psutil
from gesture_control import SystemController, GestureParams, MouseController, GestureProcessor

def setup_performance():
    try:
        p = psutil.Process(os.getpid())
        p.nice(psutil.HIGH_PRIORITY_CLASS)
        print("Process set to HIGH PRIORITY to prevent background lag.")
    except Exception as e:
        print(f"Failed to set high priority: {e}")

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU Detected: {len(gpus)} device(s). Using GPU for inference.")
        except RuntimeError as e:
            print(f"GPU Config Error: {e}")
    else:
        print("No GPU detected. Using CPU.")

class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True 
        t.start()
        return self

    def update(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()
            if not self.grabbed:
                self.stop()
                break
            time.sleep(0.001) 

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

class FPSMonitor:
    def __init__(self):
        self.prev_time = 0
        self.new_time = 0
        self.fps = 0

    def update(self):
        self.new_time = time.time()
        diff = self.new_time - self.prev_time
        if diff > 0:
            self.fps = 1 / diff
        self.prev_time = self.new_time
        return int(self.fps)

    def get(self):
        return int(self.fps)

def normalize_landmarks(landmarks):
    if hasattr(landmarks[0], 'x'):
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    else:
        coords = np.array(landmarks)
        
    base_x, base_y, base_z = coords[0]
    coords[:, 0] -= base_x
    coords[:, 1] -= base_y
    coords[:, 2] -= base_z
    
    max_val = np.max(np.abs(coords))
    if max_val > 0:
        coords /= max_val
        
    return coords.flatten().tolist()

setup_performance()

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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

    index_up = landmarks[8].y < landmarks[6].y
    middle_up = landmarks[12].y < landmarks[10].y

    ring_down = landmarks[16].y > landmarks[14].y
    pinky_down = landmarks[20].y > landmarks[18].y

    return index_up and middle_up and ring_down and pinky_down

def main():

    try:
        import psutil, os
        p = psutil.Process(os.getpid())
        p.nice(psutil.HIGH_PRIORITY_CLASS)
        print("Process priority elevated to HIGH to prevent background lag.")
    except Exception as e:
        print(f"Could not elevate process priority: {e}")

    pyautogui.PAUSE = 0
    pyautogui.FAILSAFE = False

    sys_ctrl = SystemController()
    gesture_params = GestureParams(CLASSES)

    model = None

    try:
        model = tf.keras.models.load_model(MODEL_FILE)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

    cap = VideoStream(0).start()
    fps_monitor = FPSMonitor()

    global MOUSE_MODE

    try:
        while True:

            num_hands = 2 if MOUSE_MODE else 1
            print(f"Initializing MediaPipe for {num_hands} hand(s)...")

            base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=num_hands,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                running_mode=vision.RunningMode.VIDEO)

            mode_switch_start_time = 0
            MODE_SWITCH_DELAY = 1.0
            should_break = False
            should_exit = False

            with vision.HandLandmarker.create_from_options(options) as landmarker:
                start_time = time.time()

                screen_w, screen_h = pyautogui.size()
                mouse_ctrl = MouseController(screen_w, screen_h)
                gesture_processor = GestureProcessor(sys_ctrl, gesture_params, CLASSES, SMOOTHING_BUFFER)

                print("Active.")
                print(f"Mode: {'MOUSE MODE (2 Hands)' if MOUSE_MODE else 'GESTURE MODE (1 Hand)'}")
                print(f"Window: {'HEADLESS' if HEADLESS else ('ALWAYS ON TOP' if ALWAYS_ON_TOP else 'NORMAL')}")
                print("Press 'q' (or Ctrl+C in Headless) to quit.")

                while True:
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

                    victory_detected = False
                    if detection_result.hand_landmarks:

                        for hl in detection_result.hand_landmarks:
                            if model and "Victory" in CLASSES:
                                input_data = np.array([normalize_landmarks(hl)])
                                prediction = model(input_data, training=False).numpy()
                                class_id = np.argmax(prediction)
                                confidence = prediction[0][class_id]

                                if class_id == CLASSES.index("Victory") and confidence > 0.85:
                                    victory_detected = True
                                    break
                            else:

                                if is_victory_gesture(hl):
                                    victory_detected = True
                                    break

                    if victory_detected:
                        if mode_switch_start_time == 0:
                            mode_switch_start_time = time.time()

                        elapsed = time.time() - mode_switch_start_time
                        remaining = MODE_SWITCH_DELAY - elapsed

                        msg = "Hold Victory to Switch..." if remaining > 0 else "Switching..."
                        cv2.putText(frame, msg, (int(frame.shape[1]/2)-100, int(frame.shape[0]/2)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

                        bar_w = 200
                        progress = min(1.0, elapsed / MODE_SWITCH_DELAY)
                        cv2.rectangle(frame, (220, 250), (220 + int(bar_w * progress), 270), (255, 0, 255), -1)
                        cv2.rectangle(frame, (220, 250), (220 + bar_w, 270), (255, 255, 255), 2)

                        if elapsed >= MODE_SWITCH_DELAY:
                            MOUSE_MODE = not MOUSE_MODE

                            try:
                                with open(CONFIG_FILE, 'r') as f:
                                    c = json.load(f)
                                c["mouse_mode"] = MOUSE_MODE
                                with open(CONFIG_FILE, 'w') as f:
                                    json.dump(c, f, indent=4)
                            except: pass

                            cv2.putText(frame, f"SWITCHED TO {'MOUSE' if MOUSE_MODE else 'GESTURE'}!",
                                       (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
                            if not HEADLESS:
                                cv2.imshow('Gesture Control', frame)
                                cv2.waitKey(500)
                            should_break = True
                            break
                    else:
                        mode_switch_start_time = 0

                    if detection_result.hand_landmarks:
                        for hand_landmarks in detection_result.hand_landmarks:
                            for lm in hand_landmarks:
                                h, w, c = frame.shape
                                cx, cy = int(lm.x * w), int(lm.y * h)
                                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

                    if MOUSE_MODE:
                        cursor_hand = None
                        click_hand = None

                        if detection_result.hand_landmarks and detection_result.handedness:
                            for i, hand_landmarks in enumerate(detection_result.hand_landmarks):

                                label = detection_result.handedness[i][0].category_name

                                if label == "Left":
                                    cursor_hand = hand_landmarks
                                elif label == "Right":
                                    click_hand = hand_landmarks

                        if cursor_hand:
                            h, w, c = frame.shape
                            mouse_ctrl.process_cursor(cursor_hand, w, h)
                            cv2.rectangle(frame, (mouse_ctrl.margin_x, mouse_ctrl.margin_y), (w - mouse_ctrl.margin_x, h - mouse_ctrl.margin_y), (0, 255, 255), 1)
                            cv2.putText(frame, "Cursor: Active", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                        if click_hand:
                            h, w, c = frame.shape
                            texts = mouse_ctrl.process_click_and_scroll(click_hand, w, h, model, gesture_params, normalize_landmarks)
                            for txt, pos, color in texts:
                                cv2.putText(frame, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                    else:
                        gesture_text = "Idle"
                        color = (0, 0, 255)
                        progress_bar_val = 0

                        if detection_result.hand_landmarks:
                            hand_landmarks = detection_result.hand_landmarks[0]
                            gesture_text, color, progress_bar_val = gesture_processor.process_gesture(hand_landmarks, model, normalize_landmarks)
                            
                        cv2.putText(frame, f"{gesture_text}", (10, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

                        if progress_bar_val > 0 and progress_bar_val < 1.0:
                             bar_w = 200
                             cv2.rectangle(frame, (10, 60), (10 + int(bar_w * progress_bar_val), 80), color, -1)
                             cv2.rectangle(frame, (10, 60), (10 + bar_w, 80), (255, 255, 255), 2)

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

                        time.sleep(0.03)

                if should_exit:
                    break
                if should_break:
                    continue

            if should_exit:
                break

    finally:
        cap.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

