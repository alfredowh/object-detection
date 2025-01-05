import os
from pathlib import Path
import cv2
from typing import Union, List, Optional, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import random

kitti_classes = {
    "Car": 0,
    "Pedestrian": 1,
    "Van": 2,
    "Cyclist": 3,
    "Truck": 4,
    "Misc": 5,
    "Tram": 6,
    "Person_sitting": 7,
}

kitti_id_to_class = {v: n for n, v in kitti_classes.items()}


def search_objects(
    classes: Union[List[str], str],
    data_path: str = "../../input",
    is_gazebo: bool = False,
) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
    data_path = Path(data_path)

    if is_gazebo:
        label_path = data_path
    else:
        label_path = (
            data_path / "images" / "data_object_label_2" / "training" / "label_2"
        )

    files = [
        f for f in os.listdir(label_path) if os.path.isfile(os.path.join(label_path, f))
    ]

    count_obj = {}
    obj_files = {}

    if isinstance(classes, str):
        classes = [classes]
        count_obj[classes] = 0
        obj_files[classes] = set()
    if isinstance(classes, list):
        for c in classes:
            count_obj[c] = 0
            obj_files[c] = set()

    for file in files:

        label_file = os.path.join(label_path, file)

        with open(label_file, "r") as text:
            labels = text.readlines()
            if len(labels) > 0:
                for label in labels:
                    label = label.split()

                    # Count object
                    if is_gazebo:
                        c = kitti_id_to_class[int(label[0])]
                    else:
                        c = label[0]
                    if c in classes:
                        count_obj[c] += 1
                        obj_files[c].add(file)

    return count_obj, obj_files


def show_img(
    img_path: str,
    files: List,
    label_path: str,
    objects: Optional[Union[list[str], str]] = None,
    img_size: Tuple[int] = (1224, 370),
    figsize: Tuple[int] = (20, 5),
    scale_factor: float = 1,
    plot_examples: Optional[int] = None,
    is_gazebo: bool = False,
) -> None:
    img_path = Path(img_path)

    extension = ".jpeg" if is_gazebo else ".png"

    if isinstance(plot_examples, int):
        files = list(files)
        n = int(plot_examples / 2)
        total_indices = n * 2

        plt.figure(figsize=figsize)
        for i in range(total_indices):
            file_idx = random.randint(0, len(files) - 1)
            img = cv2.imread(
                os.path.join(
                    img_path, f"{os.path.splitext(files[file_idx])[0]}{extension}"
                )
            )
            if img is None:
                print(
                    f"Failed to open {img_path}/{os.path.splitext(files[file_idx])[0]}{extension}"
                )
            else:
                print(
                    f"Opened {img_path}/{os.path.splitext(files[file_idx])[0]}{extension}"
                )

            img = draw_bbox(img, label_path, files[file_idx], objects, is_gazebo)

            resized_img = cv2.resize(
                img, (int(img_size[0] * scale_factor), int(img_size[1] * scale_factor))
            )

            plt.subplot(n, n, i + 1)
            plt.imshow(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
        plt.tight_layout()
        plt.show()

    else:
        for file in files:
            # Read the image
            img = cv2.imread(
                os.path.join(img_path, f"{os.path.splitext(file)[0]}{extension}")
            )
            if img is None:
                print(
                    f"Failed to open {img_path}/{os.path.splitext(file)[0]}{extension}"
                )
            else:
                print(f"Opened {img_path}/{os.path.splitext(file)[0]}{extension}")

                img = draw_bbox(img, label_path, file, objects, is_gazebo)

                # Display the image
                resized_img = cv2.resize(
                    img,
                    (int(img_size[0] * scale_factor), int(img_size[1] * scale_factor)),
                )

                cv2.imshow(f"{file}", resized_img)
                # Wait for a key press and close the window
                cv2.waitKey(0)
                cv2.destroyAllWindows()


def get_unique_color(class_name: str):
    # Convert class name to a unique color using hashing
    np.random.seed(hash(class_name) % 2**32)  # Ensure consistent color for each class
    return tuple(np.random.randint(0, 256, 3).tolist())


def draw_bbox(
    img,
    label_path,
    file,
    objects: Optional[Union[list[str], str]] = None,
    is_gazebo: bool = False,
):

    label_file = os.path.join(label_path, f"{os.path.splitext(file)[0]}.txt")
    with open(label_file, "r") as text:
        labels = text.readlines()

    class_colors = {}

    for label in labels:
        label = label.strip().split()

        if is_gazebo:
            x_center, y_center, width, height = map(float, label[1:])
            class_name = kitti_id_to_class[int(label[0])]

            img_height, img_width = img.shape[:-1]

            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height

            x1 = x_center - (width / 2)
            y1 = y_center - (height / 2)
            x2 = x_center + (width / 2)
            y2 = y_center + (height / 2)
        else:
            label[4:8] = map(float, label[4:8])  # convert to float
            label = [label[0], label[4], label[5], label[6], label[7]]

            class_name, x1, y1, x2, y2 = label

        if isinstance(objects, list):
            if class_name not in objects:
                continue
        if isinstance(objects, str):
            if class_name == objects:
                continue

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        if class_name not in class_colors:
            class_colors[class_name] = get_unique_color(class_name)

        cv2.rectangle(
            img, (x1, y1), (x2, y2), color=class_colors[class_name], thickness=4
        )

        # Put the class ID as text
        cv2.putText(
            img,
            f"{class_name}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            class_colors[class_name],
            2,
        )

    return img


def get_bbox_areas(
    label_dir: str, image_dir: str, is_gazebo: bool = False
) -> Dict[int, np.ndarray]:
    class_areas = {}

    if is_gazebo:
        extension = ".jpeg"
    else:
        extension = ".png"

    for label_file in os.listdir(label_dir):
        file_path = os.path.join(label_dir, label_file)
        img = cv2.imread(
            os.path.join(image_dir, f"{os.path.splitext(label_file)[0]}{extension}")
        )
        with open(file_path, "r") as f:
            for line in f:
                data = line.strip().split()
                class_id = int(data[0])
                bbox_width_norm = float(data[3])
                bbox_height_norm = float(data[4])

                bbox_width = bbox_width_norm * img.shape[1]
                bbox_height = bbox_height_norm * img.shape[0]

                area = bbox_width * bbox_height
                if area > 0:  # Avoid invalid or zero-area bounding boxes
                    if class_id not in class_areas:
                        class_areas[class_id] = []
                    class_areas[class_id].append(area)

    for class_id in class_areas:
        class_areas[class_id] = np.array(class_areas[class_id])

    return class_areas


def plot_area_distribution(
    class_areas: Dict[int, np.ndarray],
    class_names: Dict[int, str],
    is_gazebo: bool = False,
):
    num_classes = len(class_areas)
    if is_gazebo:
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    else:
        fig, axes = plt.subplots(int(num_classes / 2), 2, figsize=(10, 10))
    fig.suptitle("Distributions of Boounding Box Area")
    axes = axes.flatten()

    if num_classes == 1:
        axes = [axes]

    for i, (class_id, areas) in enumerate(class_areas.items()):
        axes[i].hist(
            areas / 1e3, bins=50, color="skyblue", alpha=0.7, edgecolor="black"
        )
        class_name = class_names[class_id] if class_names else f"Class {class_id}"
        axes[i].set_title(f"{class_name}", fontsize=14)
        axes[i].set_xlabel(r"Bbox Area ($pixels^2 . 10^3$)", fontsize=12)
        axes[i].set_ylabel("Frequency", fontsize=12)
        axes[i].grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()
