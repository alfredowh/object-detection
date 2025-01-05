from pathlib import Path
from tqdm import tqdm
import shutil
from typing import Optional
import os


def copytree_with_suffix(src: str, dst: str, suffix: str):
    os.makedirs(dst, exist_ok=True)

    for root, dirs, files in os.walk(src):
        relative_path = os.path.relpath(root, src)
        dest_dir = os.path.join(dst, relative_path)
        os.makedirs(dest_dir, exist_ok=True)

        for file in files:
            src_file = os.path.join(root, file)
            # Add the suffix to the filename
            file_name, file_ext = os.path.splitext(file)
            new_file_name = f"{file_name}{suffix}{file_ext}"
            dst_file = os.path.join(dest_dir, new_file_name)
            shutil.copy2(src_file, dst_file)


def copy_split(
    data: list[tuple],
    destination: str,
    separate_labels: bool = False,
    label_folder_name: Optional[str] = "labels",
    img_folder_name: Optional[str] = "images",
):
    data_path = Path(destination).resolve()
    data_path.mkdir(exist_ok=True)

    if separate_labels:
        (data_path / img_folder_name).mkdir(exist_ok=True)
        (data_path / label_folder_name).mkdir(exist_ok=True)

        for t_img, t_lb in tqdm(data):
            im_path = data_path / img_folder_name / t_img.name
            lb_path = data_path / label_folder_name / t_lb.name
            shutil.copy(t_img, im_path)
            shutil.copy(t_lb, lb_path)
    else:
        for t_img, t_lb in tqdm(data):
            im_path = data_path / t_img.name
            lb_path = data_path / t_lb.name
            shutil.copy(t_img, im_path)
            shutil.copy(t_lb, lb_path)


def create_data_yaml(train_path: str, test_path: str, classes: list, output_path: str):
    yaml_file = "names:\n"
    yaml_file += "\n".join(f"- {c}" for c in classes)
    yaml_file += f"\nnc: {len(classes)}"
    yaml_file += f"\ntrain: {str(train_path)}\nval: {str(test_path)}"
    with open(output_path, "w") as f:
        f.write(yaml_file)
