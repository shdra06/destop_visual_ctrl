import tensorflow as tf
import mediapipe as mp
import os

print("--- TensorFlow Diagnosis ---")
print(f"TensorFlow Version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs Available: {len(gpus)}")
for gpu in gpus:
    print(f"  - {gpu}")

print("\n--- MediaPipe Diagnosis ---")
print(f"MediaPipe Version: {mp.__version__}")

from mediapipe.tasks import python
try:
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task', delegate=python.BaseOptions.Delegate.GPU)
    print("MediaPipe GPU Delegate: APPEARS SUPPORTED (Configuration accepted)")
except Exception as e:
    print(f"MediaPipe GPU Delegate: NOT SUPPORTED / ERROR: {e}")

print("\n--- System Enviroment ---")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
