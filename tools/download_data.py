import kagglehub
import os
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.parent

# Create the "input" directory if it doesn't exist
input_dir = project_root / "input"
input_dir.mkdir(exist_ok=True)

labels_path = kagglehub.dataset_download("shreydan/kitti-dataset-yolo-format")
images_path = kagglehub.dataset_download("klemenko/kitti-dataset")

os.symlink(labels_path, input_dir / "labels", target_is_directory=True)
os.symlink(images_path, input_dir / "images", target_is_directory=True)
