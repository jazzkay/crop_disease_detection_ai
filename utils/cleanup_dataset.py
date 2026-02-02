import os
import shutil

# Path to training folder
DATA_PATH = "data/train"

# Classes we want to keep
KEEP_CLASSES = [
    "cotton_anthracnose",
    "cotton_bacterial_blight",
    "cotton_leaf_curl",
    "cotton_healthy",

    "rice_blast",
    "rice_bacterial_leaf_blight",
    "rice_brown_spot",
    "rice_tungro",

    "maize_gray_leaf_spot",
    "maize_ear_rot",
    "maize_fall_armyworm",
    "maize_healthy"
]

for folder in os.listdir(DATA_PATH):
    folder_path = os.path.join(DATA_PATH, folder)

    if folder not in KEEP_CLASSES:
        print("Deleting:", folder)
        shutil.rmtree(folder_path)

print("Dataset cleanup completed.")
