import os
from ultralytics import YOLO
from PIL import Image

# Absolute project root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(
    ROOT_DIR, "models", "yolo", "krishigpt_disease_cls.pt"
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"YOLO model not found at {MODEL_PATH}")

model = YOLO(MODEL_PATH)

def predict_disease(image):
    if not isinstance(image, Image.Image):
        image = Image.open(image)

    results = model(image)
    probs = results[0].probs

    disease = results[0].names[probs.top1]
    confidence = float(probs.top1conf)

    return disease, round(confidence * 100, 2)
