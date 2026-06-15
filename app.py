#Step 1
!pip -q install ultralytics
!pip -q install opencv-python
!pip -q install pandas
!pip -q install scikit-learn
!pip -q install pyyaml
!pip -q install tqdm

#Step 2 =import
import os
import cv2
import yaml
import shutil
import random
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split

from ultralytics import YOLO

random.seed(42)

#Step 3
from google.colab import drive
drive.mount('/content/drive')

ROOT = "/content/drive/MyDrive/testing1"

IMAGE_DIR = os.path.join(ROOT,"images")
LABEL_DIR = os.path.join(ROOT,"labels")

OUTPUT_DATASET = "/content/student_posture_dataset"


#Step 4=class mapping
ROOT = "/content/drive/MyDrive/testing1"

IMAGE_DIR = os.path.join(ROOT, "images")
LABEL_DIR = os.path.join(ROOT, "labels")

OUTPUT_DATASET = "/content/student_posture_dataset"

CLASSES = [
    "handrise",
    "look_forward",
    "read",
    "sleep",
    "stand",
    "turn_head",
    "using_device",
    "write"
]

print("Images Folder Exists :", os.path.exists(IMAGE_DIR))
print("Labels Folder Exists :", os.path.exists(LABEL_DIR))

print("Images Count :", len(os.listdir(IMAGE_DIR)))
print("Labels Count :", len(os.listdir(LABEL_DIR)))


#Step 5=dataset validation
valid_images = []

for img in os.listdir(IMAGE_DIR):

    if not img.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):
        continue

    label_path = os.path.join(
        LABEL_DIR,
        Path(img).stem + ".txt"
    )

    if os.path.exists(label_path):
        valid_images.append(img)

print("Valid Pairs:", len(valid_images))

#Step 6=data split
train_imgs, temp_imgs = train_test_split(
    valid_images,
    test_size=0.30,
    random_state=42
)

val_imgs, test_imgs = train_test_split(
    temp_imgs,
    test_size=0.50,
    random_state=42
)

print("Train:", len(train_imgs))
print("Val:", len(val_imgs))
print("Test:", len(test_imgs))

import os

print(os.path.exists(IMAGE_DIR))
print(os.path.exists(LABEL_DIR))

print(IMAGE_DIR)
print(LABEL_DIR)

#Step 7=create yolo folder
for split in ["train", "valid", "test"]:

    os.makedirs(
        os.path.join(
            OUTPUT_DATASET,
            split,
            "images"
        ),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(
            OUTPUT_DATASET,
            split,
            "labels"
        ),
        exist_ok=True
    )

#Step 8=copy file
def copy_data(files, split):

    for img in files:

        shutil.copy2(
            os.path.join(
                IMAGE_DIR,
                img
            ),
            os.path.join(
                OUTPUT_DATASET,
                split,
                "images",
                img
            )
        )

        label_file = Path(img).stem + ".txt"

        shutil.copy2(
            os.path.join(
                LABEL_DIR,
                label_file
            ),
            os.path.join(
                OUTPUT_DATASET,
                split,
                "labels",
                label_file
            )
        )

copy_data(train_imgs, "train")
copy_data(val_imgs, "valid")
copy_data(test_imgs, "test")

print("Dataset Ready")

#Step 9= create data.yaml
yaml_data = {

    "path": OUTPUT_DATASET,

    "train": "train/images",

    "val": "valid/images",

    "test": "test/images",

    "names": CLASSES
}

yaml_path = os.path.join(
    OUTPUT_DATASET,
    "data.yaml"
)

with open(yaml_path, "w") as f:
    yaml.dump(yaml_data, f)

print(yaml_path)


#step 10=label quality check
from ultralytics import YOLO

model = YOLO("yolo11s.pt")

results = model.train(

    data=yaml_path,

    epochs=35,

    imgsz=960,

    batch=8,

    workers=2,

    cache=False,

    amp=True,

    optimizer="AdamW",

    lr0=0.001,

    patience=8,

    fliplr=0.5,

    mosaic=0.3,

    hsv_h=0.015,
    hsv_s=0.3,
    hsv_v=0.2,

    project="Student_Posture",

    name="Fast_Model",

    exist_ok=True
)

#Step 11=model training
best_model_path = os.path.join(
    results.save_dir,
    "weights",
    "best.pt"
)

print(best_model_path)

best_model = YOLO(best_model_path)

#Step 12=validation
results = best_model.predict(

    source=os.path.join(
        OUTPUT_DATASET,
        "test/images"
    ),

    imgsz=1280,

    conf=0.25,

    augment=True,

    save=True,

    save_txt=True,

    save_conf=True
)


# Step 14=export csv

rows = []

test_folder = os.path.join(
    OUTPUT_DATASET,
    "test/images"
)

for img_path in Path(test_folder).glob("*"):

    result = best_model.predict(
        str(img_path),
        conf=0.25,
        verbose=False
    )[0]

    for box in result.boxes:

        cls = int(box.cls[0])

        conf = float(box.conf[0])

        rows.append({

            "image": img_path.name,

            "posture": result.names[cls],

            "confidence": round(conf,4)
        })

pred_df = pd.DataFrame(rows)

pred_df.to_csv(
    "student_posture_predictions.csv",
    index=False
)

pred_df.head()

#Step 15=inferece on new image
best_model.predict(
    source="/content/student_posture_dataset/test/images",
    imgsz=640,
    conf=0.25,
    save=True
)

metrics = best_model.val()

print("Precision :", metrics.box.mp)
print("Recall    :", metrics.box.mr)
print("mAP50     :", metrics.box.map50)
print("mAP50-95  :", metrics.box.map)

metrics = best_model.val()

print(f"Precision : {metrics.box.mp*100:.2f}%")
print(f"Recall    : {metrics.box.mr*100:.2f}%")
print(f"mAP50     : {metrics.box.map50*100:.2f}%")
print(f"mAP50-95  : {metrics.box.map*100:.2f}%")
