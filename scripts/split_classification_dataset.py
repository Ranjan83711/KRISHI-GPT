import os
import shutil
import random

ROOT = os.path.dirname(os.path.dirname(__file__))  # KRISHI-GPT root

# FIXED PATH
DATASET = os.path.join(ROOT, "data", "raw_disease_dataset")

OUTPUT = os.path.join(ROOT, "data", "disease_dataset_split")

print("ROOT =", ROOT)
print("DATASET =", DATASET)
print("OUTPUT =", OUTPUT)

os.makedirs(OUTPUT, exist_ok=True)

classes = [c for c in os.listdir(DATASET) if os.path.isdir(os.path.join(DATASET, c))]

print("Found classes:", classes)

train_ratio = 0.8

for cls in classes:
    cls_path = os.path.join(DATASET, cls)
    images = [f for f in os.listdir(cls_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    
    if not images:
        print("⚠️ No images in class:", cls)
        continue

    random.shuffle(images)
    train_len = int(len(images) * train_ratio)

    train_imgs = images[:train_len]
    val_imgs = images[train_len:]

    os.makedirs(os.path.join(OUTPUT, "train", cls), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT, "val", cls), exist_ok=True)

    for img in train_imgs:
        shutil.copy(os.path.join(cls_path, img), os.path.join(OUTPUT, "train", cls, img))

    for img in val_imgs:
        shutil.copy(os.path.join(cls_path, img), os.path.join(OUTPUT, "val", cls, img))

print("Done! Check:", OUTPUT)
