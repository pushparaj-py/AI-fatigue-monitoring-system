import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, "face_landmarker.task")

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

if not os.path.exists(model_path):
    print(f"Downloading face_landmarker model to {model_path}...", flush=True)
    urllib.request.urlretrieve(MODEL_URL, model_path)
    print("Download complete!", flush=True)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.IMAGE
)

landmarker = vision.FaceLandmarker.create_from_options(options)
print("FaceLandmarker created successfully!", flush=True)
landmarker.close()
