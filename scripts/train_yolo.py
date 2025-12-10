from ultralytics import YOLO
import os

ROOT = os.path.dirname(os.path.dirname(__file__))
DATASET = os.path.join(ROOT, "data", "disease_dataset", "data.yaml")
OUT = os.path.join(ROOT, "models", "yolo")
os.makedirs(OUT, exist_ok=True)

model = YOLO("yolov8n-cls.pt")

model.train(
    data=DATASET,
    epochs=50,
    imgsz=224,
    batch=16,
    device=0
)

print("Training done! Copy best.pt to models/yolo/")
