import time
import threading
import subprocess
import math
import numpy as np
import ctypes
import pyautogui
import collections
class SystemController:
    def __init__(self):
        self.current_brightness = 50
        self.last_volume_update = 0
        self.last_brightness_update = 0
        self.last_desktop_toggle = 0

        self.init_brightness()

    def init_brightness(self):
        try:
            cmd = "(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightness).CurrentBrightness"
            result = subprocess.check_output(["powershell", "-Command", cmd], creationflags=subprocess.CREATE_NO_WINDOW)
            self.current_brightness = int(result.decode().strip())
            print(f"Initial Brightness: {self.current_brightness}")
        except Exception as e:
            print(f"Brightness Init Error: {e}")
            self.current_brightness = 50

    def _set_brightness_worker(self, value):
        try:
            cmd = f"(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1,{value})"
            subprocess.run(["powershell", "-Command", cmd],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        except Exception as e:
            print(f"Set Brightness Error: {e}")

    def change_brightness(self, direction):
        if time.time() - self.last_brightness_update < 0.2:
            return

        step = 10
        self.current_brightness += direction * step
        self.current_brightness = max(0, min(100, self.current_brightness))

        threading.Thread(target=self._set_brightness_worker, args=(self.current_brightness,), daemon=True).start()
        self.last_brightness_update = time.time()
        return self.current_brightness

    def _change_volume_worker(self, direction, steps=1):
        try:
            char_code = 175 if direction == 1 else 174
            ps_script = f"""
            $obj = New-Object -ComObject WScript.Shell
            for ($i=0; $i -lt {steps}; $i++) {{
                $obj.SendKeys([char]{char_code})
                Start-Sleep -Milliseconds 10
            }}
            """
            cmd = ps_script.replace('\n', ';')
            subprocess.run(["powershell", "-Command", cmd],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        except Exception as e:
            print(f"Volume Error: {e}")

    def change_volume(self, direction, steps=1):

        if time.time() - self.last_volume_update < 0.1:
            return

        threading.Thread(target=self._change_volume_worker, args=(direction, steps), daemon=True).start()
        self.last_volume_update = time.time()

    def toggle_desktop(self):
        if time.time() - self.last_desktop_toggle < 2.0:
            return False

        user32 = ctypes.windll.user32
        user32.keybd_event(0x5B, 0, 0, 0)
        user32.keybd_event(0x44, 0, 0, 0)
        user32.keybd_event(0x44, 0, 2, 0)
        user32.keybd_event(0x5B, 0, 2, 0)

        self.last_desktop_toggle = time.time()
        return True

    def toggle_media(self):
        if time.time() - getattr(self, 'last_media_toggle', 0) < 1.0:
            return False

        user32 = ctypes.windll.user32
        user32.keybd_event(0xB3, 0, 0, 0)
        user32.keybd_event(0xB3, 0, 2, 0)

        self.last_media_toggle = time.time()
        return True

class GestureParams:
    def __init__(self, config_classes):
        self.classes = config_classes

        self.VOL_ID = self._get_id("Volume")
        self.B_UP_ID = self._get_id("Bright_Up")
        self.B_DOWN_ID = self._get_id("Bright_Down")
        self.DESK_ID = self._get_id("Show_Desktop")
        self.IDLE_ID = self._get_id("Idle")

        self.PAUSE_ID = self._get_id("pause track")

    def _get_id(self, name):
        try:
            return self.classes.index(name)
        except ValueError:
            return -999

    def get_hold_duration(self, class_id):
        if class_id == self.VOL_ID: return 0.5
        if class_id == self.B_UP_ID: return 0.5
        if class_id == self.B_DOWN_ID: return 0.5
        if class_id == self.DESK_ID: return 1.0
        if class_id == self.PAUSE_ID: return 0.5
        return 1.0

    def get_confidence_threshold(self, class_id):
        if class_id == self.VOL_ID: return 0.8
        if class_id == self.B_UP_ID: return 0.9
        if class_id == self.B_DOWN_ID: return 0.9
        if class_id == self.DESK_ID: return 0.9
        if class_id == self.PAUSE_ID: return 0.8
        if class_id == self.IDLE_ID: return 0.8
        return 0.8

    def get_volume_steps(self, delta_y):
        MOVEMENT_THRESHOLD = 0.005
        speed = abs(delta_y)

        if speed <= MOVEMENT_THRESHOLD:
            return 0, "STATIC"

        if speed < 0.02:
            return 2, "FINE"
        elif speed < 0.05:
            return 5, "MED"
        else:
            return 10, "FAST"

class MouseController:
    def __init__(self, screen_w, screen_h):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.prev_x = 0
        self.prev_y_mouse = 0
        self.already_clicked = False
        self.last_thumbs_scroll_time = 0
        self.smoothing_alpha = 0.15
        self.margin_x = 120
        self.margin_y = 100

    def process_cursor(self, cursor_hand, frame_w, frame_h):
        lm = cursor_hand[8]
        lm_x = max(self.margin_x, min(frame_w - self.margin_x, int(lm.x * frame_w)))
        lm_y = max(self.margin_y, min(frame_h - self.margin_y, int(lm.y * frame_h)))

        target_x = np.interp(lm_x, [self.margin_x, frame_w - self.margin_x], [0, self.screen_w])
        target_y = np.interp(lm_y, [self.margin_y, frame_h - self.margin_y], [0, self.screen_h])

        cur_x = self.prev_x + self.smoothing_alpha * (target_x - self.prev_x)
        cur_y = self.prev_y_mouse + self.smoothing_alpha * (target_y - self.prev_y_mouse)

        if abs(cur_x - self.prev_x) > 1.0 or abs(cur_y - self.prev_y_mouse) > 1.0:
            try:
                pyautogui.moveTo(cur_x, cur_y)
            except pyautogui.FailSafeException:
                pass

        self.prev_x, self.prev_y_mouse = cur_x, cur_y
        return cur_x, cur_y

    def process_click_and_scroll(self, click_hand, frame_w, frame_h, model, gesture_params, normalize_fn):
        return_texts = []
        is_thumbs_scrolling = False
        thumbs_class_id = -1

        input_data = np.array([normalize_fn(click_hand)])
        prediction = model(input_data, training=False).numpy() if model else None

        if prediction is not None:
            class_id = np.argmax(prediction)
            confidence = prediction[0][class_id]
            b_up_id = getattr(gesture_params, 'B_UP_ID', -999)
            b_down_id = getattr(gesture_params, 'B_DOWN_ID', -999)

            if class_id in (b_up_id, b_down_id) and confidence > 0.8:
                is_thumbs_scrolling = True
                thumbs_class_id = class_id

        if is_thumbs_scrolling:
            current_time = time.time()
            if current_time - self.last_thumbs_scroll_time > 0.5:
                scroll_amt = 300 if thumbs_class_id == getattr(gesture_params, 'B_UP_ID', -999) else -300
                try:
                    pyautogui.scroll(scroll_amt)
                    self.last_thumbs_scroll_time = current_time
                    dir_str = "UP" if scroll_amt > 0 else "DOWN"
                    return_texts.append((f"THUMB SCROLL {dir_str}", (10, 80), (255, 165, 0)))
                except Exception:
                    pass
            else:
                return_texts.append(("HOLD TO SCROLL...", (10, 80), (255, 255, 0)))
        else:
            x1, y1 = click_hand[4].x, click_hand[4].y
            x2, y2 = click_hand[8].x, click_hand[8].y
            distance = math.hypot(x2 - x1, y2 - y1)

            cx, cy = int(x1 * frame_w), int(y1 * frame_h)
            return_texts.append((f"Dist: {distance:.3f}", (cx, cy), (255, 255, 0)))

            if distance < 0.05:
                if not self.already_clicked:
                    try:
                        pyautogui.click()
                    except Exception:
                        pass
                    self.already_clicked = True
                    return_texts.append(("CLICK!", (10, 80), (0, 0, 255)))
            else:
                self.already_clicked = False
                
        return return_texts

class GestureProcessor:
    def __init__(self, sys_ctrl, gesture_params, classes_list, smoothing_buffer_size=5):
        self.sys_ctrl = sys_ctrl
        self.gesture_params = gesture_params
        self.classes = classes_list
        self.gesture_buffer = collections.deque(maxlen=smoothing_buffer_size)
        self.current_gesture_state = None
        self.gesture_hold_start_time = 0
        self.is_active = False
        self.prev_y = None
        
    def process_gesture(self, hand_landmarks, model, normalize_fn):
        gesture_text = "Idle"
        color = (0, 0, 255)
        progress_bar_val = 0
        
        if not model:
            return "Model Not Loaded", color, 0

        input_data = np.array([normalize_fn(hand_landmarks)])
        prediction = model(input_data, training=False).numpy()
        class_id = np.argmax(prediction)
        confidence = prediction[0][class_id]
        threshold = self.gesture_params.get_confidence_threshold(class_id)

        if confidence > threshold:
            self.gesture_buffer.append(class_id)
        else:
            self.gesture_buffer.append(-1)

        if len(self.gesture_buffer) == self.gesture_buffer.maxlen:
            detected_state = max(set(self.gesture_buffer), key=self.gesture_buffer.count)

            if detected_state != -1 and detected_state != self.gesture_params.IDLE_ID:
                if detected_state == self.current_gesture_state:
                    elapsed = time.time() - self.gesture_hold_start_time
                    duration = self.gesture_params.get_hold_duration(detected_state)

                    if elapsed >= duration:
                        self.is_active = True
                        progress_bar_val = 1.0
                    else:
                        self.is_active = False
                        if duration > 0:
                            progress_bar_val = elapsed / duration
                            gesture_text = f"Holding... {duration - elapsed:.1f}s"
                        else:
                            progress_bar_val = 1.0
                            gesture_text = "Active"
                        color = (0, 165, 255)
                else:
                    self.current_gesture_state = detected_state
                    self.gesture_hold_start_time = time.time()
                    self.is_active = False
                    self.prev_y = hand_landmarks[8].y
                    gesture_text = "New Gesture Detected"
            else:
                self.current_gesture_state = None
                self.is_active = False
                self.prev_y = None
                gesture_text = "Idle"

            if self.is_active:
                color = (0, 255, 0)
                
                if self.current_gesture_state == self.gesture_params.VOL_ID:
                    current_y = hand_landmarks[8].y
                    if self.prev_y is not None:
                        delta_y = current_y - self.prev_y
                        steps, speed_text = self.gesture_params.get_volume_steps(delta_y)
                        if steps > 0:
                            if delta_y < 0:
                                self.sys_ctrl.change_volume(1, steps)
                                gesture_text = "Active: Volume Up"
                            else:
                                self.sys_ctrl.change_volume(-1, steps)
                                gesture_text = "Active: Volume Down"
                        else:
                            gesture_text = "Active: Volume Mode"
                    self.prev_y = current_y

                elif self.current_gesture_state == self.gesture_params.B_UP_ID:
                    self.sys_ctrl.change_brightness(1)
                    gesture_text = "Active: Brightness Up"

                elif self.current_gesture_state == self.gesture_params.B_DOWN_ID:
                    self.sys_ctrl.change_brightness(-1)
                    gesture_text = "Active: Brightness Down"

                elif self.current_gesture_state == self.gesture_params.DESK_ID:
                    if self.sys_ctrl.toggle_desktop():
                        gesture_text = "Active: Desktop Toggle"
                        self.is_active = False
                        self.current_gesture_state = None

                elif getattr(self.gesture_params, 'PAUSE_ID', -999) != -999 and self.current_gesture_state == self.gesture_params.PAUSE_ID:
                    if self.sys_ctrl.toggle_media():
                        gesture_text = "Active: Play/Pause"
                        self.is_active = False
                        self.current_gesture_state = None
                    else:
                        gesture_text = "Active: Media Cooldown"

                else:
                    gesture_name = self.classes[self.current_gesture_state]
                    gesture_text = f"Active: {gesture_name}"
                    self.is_active = False
                    
        return gesture_text, color, progress_bar_val

