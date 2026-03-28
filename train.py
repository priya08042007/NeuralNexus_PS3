import os
import random
import shutil
from ultralytics import YOLO

# ==============================
# STEP 1: SPLIT DATA (70/15/15)
# ==============================

base = "final_dataset"

train_img = os.path.join(base, "train/images")
train_lbl = os.path.join(base, "train/labels")

val_img = os.path.join(base, "val/images")
val_lbl = os.path.join(base, "val/labels")

test_img = os.path.join(base, "test/images")
test_lbl = os.path.join(base, "test/labels")

os.makedirs(test_img, exist_ok=True)
os.makedirs(test_lbl, exist_ok=True)

images = os.listdir(train_img)
random.shuffle(images)

n = len(images)

test_split = int(0.15 * n)
val_split = int(0.15 * n)

test_files = images[:test_split]
val_files = images[test_split:test_split + val_split]

# Move to test
for file in test_files:
    shutil.move(os.path.join(train_img, file), os.path.join(test_img, file))
    shutil.move(
        os.path.join(train_lbl, file.replace(".jpg", ".txt")),
        os.path.join(test_lbl, file.replace(".jpg", ".txt"))
    )

# Move to val
for file in val_files:
    shutil.move(os.path.join(train_img, file), os.path.join(val_img, file))
    shutil.move(
        os.path.join(train_lbl, file.replace(".jpg", ".txt")),
        os.path.join(val_lbl, file.replace(".jpg", ".txt"))
    )

print("✅ Dataset split completed")

# ==============================
# STEP 2: CREATE data.yaml
# ==============================

yaml_content = f"""
train: {base}/train/images
val: {base}/val/images
test: {base}/test/images

nc: 4
names: ['fiber', 'fragment', 'film', 'pellet']
"""

with open("data.yaml", "w") as f:
    f.write(yaml_content)

print("✅ data.yaml created")

# ==============================
# STEP 3: TRAIN YOLOv8
# ==============================

model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640
)

print("✅ Training complete!")
print("📁 Model saved at: runs/detect/train/weights/best.pt")