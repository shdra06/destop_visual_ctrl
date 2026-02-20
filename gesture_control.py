import time
import threading
import subprocess
import math
import numpy as np
import ctypes

# OS Interaction Helpers
class SystemController:
    def __init__(self):
        self.current_brightness = 50
        self.last_volume_update = 0
        self.last_brightness_update = 0
        self.last_desktop_toggle = 0
        
        # Initialize Brightness
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
        if time.time() - self.last_brightness_update < 0.2: # Rate limit
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
        # We handle rate limiting in the logic layer usually, but checks here act as safeguard
        if time.time() - self.last_volume_update < 0.1:
            return
            
        threading.Thread(target=self._change_volume_worker, args=(direction, steps), daemon=True).start()
        self.last_volume_update = time.time()

    def toggle_desktop(self):
        if time.time() - self.last_desktop_toggle < 2.0:
            return False
            
        user32 = ctypes.windll.user32
        user32.keybd_event(0x5B, 0, 0, 0) # Win Down
        user32.keybd_event(0x44, 0, 0, 0) # D Down
        user32.keybd_event(0x44, 0, 2, 0) # D Up
        user32.keybd_event(0x5B, 0, 2, 0) # Win Up
        
        self.last_desktop_toggle = time.time()
        return True

    def toggle_media(self):
        if time.time() - getattr(self, 'last_media_toggle', 0) < 1.0: # 1 second cooldown
            return False
            
        user32 = ctypes.windll.user32
        user32.keybd_event(0xB3, 0, 0, 0) # VK_MEDIA_PLAY_PAUSE Down
        user32.keybd_event(0xB3, 0, 2, 0) # VK_MEDIA_PLAY_PAUSE Up
        
        self.last_media_toggle = time.time()
        return True

# Logic & Params Helper
class GestureParams:
    def __init__(self, config_classes):
        self.classes = config_classes
        # IDs
        self.VOL_ID = self._get_id("Volume")
        self.B_UP_ID = self._get_id("Bright_Up")
        self.B_DOWN_ID = self._get_id("Bright_Down")
        self.DESK_ID = self._get_id("Show_Desktop")
        self.IDLE_ID = self._get_id("Idle")
        # Custom gestures
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
