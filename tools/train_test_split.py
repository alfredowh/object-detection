from pathlib import Path
import json
import numpy as np
import os


def is_running_on_kaggle():
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def get_paths():
    if is_running_on_kaggle():
        base_dir = Path(__file__).parent.parent / "kaggle" / "input" / "kitti-dataset"
        img_path = base_dir / "data_object_image_2" / "training" / "image_2"
        label_path = Path("/kaggle/input/kitti-dataset-yolo-format/labels")
        classes_path = Path("/kaggle/input/kitti-dataset-yolo-format/classes.json")

        return img_path, label_path, classes_path

    base_dir = Path(__file__).parent.parent / "input"
    img_path = base_dir / "images" / "data_object_image_2" / "training" / "image_2"
    label_path = base_dir / "labels"
    classes_path = label_path / "classes.json"

    return img_path, label_path, classes_path


# Paths to image and label data (for loading the KITTI dataset within Kaggle or locally)
img_path, label_path, classes_path = get_paths()

# Loading the classes present in the dataset
with open(classes_path, "r") as f:
    classes = json.load(f)
print(f"Classes : {classes}")

# Sorting images and labels to ensure alignment
ims = sorted(img_path.glob("*"))
labels = sorted(label_path.glob("*"))
pairs = list(zip(ims, labels))

# Dataset shuffle for randomized train/test split
seed = 42  # For reproducibility
random_state = np.random.RandomState(seed)
random_state.shuffle(pairs)

# Calculating the test size (10%)
test_size = int(0.1 * len(pairs))
splits = {}

# Creating 3 distinct splits
for i in range(3):
    # Select the test set for this split
    test_set = pairs[i * test_size : (i + 1) * test_size]
    # Select the training set (remaining data)
    train_set = pairs[: i * test_size] + pairs[(i + 1) * test_size :]
    splits[f"split{i + 1}"] = {"train": train_set, "test": test_set}

# Verifying sizes of each split
for key, value in splits.items():
    train_size = len(value["train"])
    test_size = len(value["test"])
    print(f"{key} - Train Size: {train_size}, Test Size: {test_size}")
    assert train_size + test_size == len(
        pairs
    ), "Train and test sizes do not add up to total pairs"

# Verifying distinct test sets
for i in range(3):
    for j in range(i + 1, 3):
        assert not set(splits[f"split{i + 1}"]["test"]).intersection(
            set(splits[f"split{j + 1}"]["test"])
        ), f"Test sets for split{i + 1} and split{j + 1} overlap"
