import cv2
import threading
import time
import psutil
import os
import tensorflow as tf

def setup_performance():
    """Sets process priority to High and configures TensorFlow GPU memory growth."""
    # 1. Set Process to High Priority
    try:
        p = psutil.Process(os.getpid())
        p.nice(psutil.HIGH_PRIORITY_CLASS)
        print("🚀 Process set to HIGH PRIORITY to prevent background lag.")
    except Exception as e:
        print(f"⚠️ Failed to set high priority: {e}")

    # 2. Configure TensorFlow for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU Detected: {len(gpus)} device(s). Using GPU for inference.")
        except RuntimeError as e:
            print(f"❌ GPU Config Error: {e}")
    else:
        print("ℹ️ No GPU detected. Using CPU.")

class VideoStream:
    """Threaded Video Stream to prevent IO blocking and reduce input lag."""
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        # Start the thread to read frames from the video stream
        t = threading.Thread(target=self.update, args=())
        t.daemon = True # Daemonize thread
        t.start()
        return self

    def update(self):
        # Keep looping infinitely until the thread is stopped
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()
            if not self.grabbed:
                self.stop()
                break
            time.sleep(0.001) # Minimal delay to prevent CPU hogging

    def read(self):
        # Return the most recent frame
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

class FPSMonitor:
    """Simple class to track and calculate FPS."""
    def __init__(self):
        self.prev_time = 0
        self.new_time = 0
        self.fps = 0

    def update(self):
        self.new_time = time.time()
        # Avoid division by zero
        diff = self.new_time - self.prev_time
        if diff > 0:
            self.fps = 1 / diff
        self.prev_time = self.new_time
        return int(self.fps)

    def get(self):
        return int(self.fps)
