import os
import shutil
import random

SOURCE_DIR = r"krishigpt\data\raw_disease_dataset"
TARGET_DIR = r"krishigpt\data\disease_dataset"

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.2
TEST_SPLIT = 0.1

os.makedirs(os.path.join(TARGET_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(TARGET_DIR, "val"), exist_ok=True)
os.makedirs(os.path.join(TARGET_DIR, "test"), exist_ok=True)

classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]

for cls in classes:
    src = os.path.join(SOURCE_DIR, cls)
    images = [f for f in os.listdir(src) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(images)

    n = len(images)
    train_n = int(n * TRAIN_SPLIT)
    val_n = int(n * VAL_SPLIT)

    train_imgs = images[:train_n]
    val_imgs = images[train_n:train_n+val_n]
    test_imgs = images[train_n+val_n:]

    for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
        split_dir = os.path.join(TARGET_DIR, split_name, cls)
        os.makedirs(split_dir, exist_ok=True)
        for img in split_imgs:
            shutil.copy(os.path.join(src, img), os.path.join(split_dir, img))

    print(f"{cls}: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")

print("Dataset split completed successfully!")
