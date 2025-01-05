import matplotlib.pyplot as plt
from typing import Literal, Dict, List, Tuple, Union
import os
import numpy as np
import torch
from scipy import interpolate
import torch.utils.benchmark as benchmark
import json
from pathlib import Path


def parse_evaluation_results(
    result_path: str,
    classes: dict,
    model_name: Literal["yolov7", "vision_lstm", "detr"],
    is_coco: bool = False,
) -> Tuple[
    Dict[str, Dict[str, List]],
    Dict[str, List],
    Dict[str, List[List]],
    Dict[str, List[List]],
]:

    files = [
        f
        for f in os.listdir(result_path)
        if os.path.isfile(os.path.join(result_path, f)) and model_name in f
    ]
    print(f"{files} are found!")

    if is_coco:
        iter_num = 4  # by COCO pretrained result
    else:
        iter_num = 8  # by KITTI finetuned result

    ap_scores = {}  # Average precision scores for each class
    precisions = {}  # Precisions for PR-Curve
    recalls = {}  # Recalls for PR-Curve
    detections = (
        {}
    )  # Containing values for precall and recall calculation on inference settings

    for file_idx, file in enumerate(files):
        with open(result_path / file, "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if line == "# Number of ground-truth objects per class\n":
                for j in range(iter_num):  # number of total classes of KITTI dataset
                    l = lines[i + j + 1].strip().replace(":", "").split(" ")
                    if l[0] == "":
                        break
                    elif l[0] not in list(classes.keys()):
                        continue
                    elif detections.get(l[0], -1) == -1:
                        detections[l[0]] = {}
                        if detections[l[0]].get("total_gt", -1) == -1:
                            detections[l[0]]["total_gt"] = [int(l[1])]
                    elif detections[l[0]].get("total_gt", -1) == -1:
                        detections[l[0]]["total_gt"] = [int(l[1])]
                    else:
                        detections[l[0].strip(":")]["total_gt"].append(int(l[1]))
            elif line == "# Number of detected objects per class\n":
                for j in range(iter_num):  # number of total classes of KITTI dataset
                    if len(lines) <= i + j + 1:
                        break
                    l = (
                        lines[i + j + 1]
                        .strip()
                        .replace("(", "")
                        .replace(")", "")
                        .replace(",", "")
                        .split(" ")
                    )

                    if l[0].strip(":") not in list(classes.keys()):
                        continue
                    if detections.get(l[0].strip(":"), -1) == -1:
                        detections[l[0].strip(":")] = {}
                    if detections[l[0].strip(":")].get("total_dt", -1) == -1:
                        detections[l[0].strip(":")]["total_dt"] = []
                        for k in range(len(files)):
                            detections[l[0].strip(":")]["total_dt"].append(None)  # Init
                        detections[l[0].strip(":")]["total_dt"][file_idx] = int(l[1])
                    else:
                        detections[l[0].strip(":")]["total_dt"][file_idx] = int(l[1])

                    if detections[l[0].strip(":")].get("tp", -1) == -1:
                        detections[l[0].strip(":")]["tp"] = []
                        for k in range(len(files)):
                            detections[l[0].strip(":")]["tp"].append(None)  # Init
                        detections[l[0].strip(":")]["tp"][file_idx] = int(
                            l[2].replace("tp:", "")
                        )
                    else:
                        detections[l[0].strip(":")]["tp"][file_idx] = int(
                            l[2].replace("tp:", "")
                        )

                    if detections[l[0].strip(":")].get("fp", -1) == -1:
                        detections[l[0].strip(":")]["fp"] = []
                        for k in range(len(files)):
                            detections[l[0].strip(":")]["fp"].append(None)  # Init
                        detections[l[0].strip(":")]["fp"][file_idx] = int(
                            l[3].replace("fp:", "")
                        )
                    else:
                        detections[l[0].strip(":")]["fp"][file_idx] = int(
                            l[3].replace("fp:", "")
                        )

    for i, file in enumerate(files):
        with open(result_path / file, "r") as f:
            lines = f.readlines()

        for c in classes.keys():
            current_class = None
            for line in lines:
                if f"{c} AP" in line and "=" in line:  # Average precision for a class
                    parts = line.split("=")
                    class_name = parts[1].strip().split(" ")[0]
                    ap_score = float(parts[0].strip().replace("%", "")) / 100
                    if ap_scores.get(class_name, -1) == -1:
                        ap_scores[class_name] = []
                    ap_scores[class_name].append(ap_score)
                    current_class = class_name
                elif "Precision" in line and current_class:
                    precision = [
                        float(p.strip().strip("'"))
                        for p in line.split(":")[1].strip(" []\n").split(",")
                    ]
                    if precisions.get(current_class, -1) == -1:
                        precisions[current_class] = []
                    precisions[current_class].append(precision)
                elif "Recall" in line and current_class:
                    recall = [
                        float(r.strip().strip("'"))
                        for r in line.split(":")[1].strip(" []\n").split(",")
                    ]
                    if recalls.get(current_class, -1) == -1:
                        recalls[current_class] = []
                    recalls[current_class].append(recall)

                if (
                    c in ap_scores.keys()
                    and c in precisions.keys()
                    and c in recalls.keys()
                ):
                    if i == 0:
                        break
                    elif (
                        len(ap_scores[c]) == i + 1
                        and len(precisions[c]) == i + 1
                        and len(recalls[c]) == i + 1
                    ):
                        break

    return detections, ap_scores, precisions, recalls


def plot_pr(
    detections: Dict[str, Dict[str, List]],
    model_name: str,
    iou_thres: float,
    conf_thres: float,
) -> None:
    classes = list(detections.keys())
    tps = np.array([detections[cls]["tp"] for cls in classes])
    fps = np.array([detections[cls]["fp"] for cls in classes])
    # total_dt = np.array([detections[cls]["total_dt"] for cls in classes.keys()])
    total_gt = np.array([detections[cls]["total_gt"] for cls in classes])
    precisions = tps / (tps + fps)  # tps / total_dt
    recalls = tps / total_gt  # tps / (tps + fns)

    plt.figure(figsize=(16, 6))
    plt.suptitle(
        f"Precision and Recall for {model_name} at IoU: {iou_thres:.2f} and Confidence: {conf_thres:.2f} on 3 Training Runs"
    )

    if len(tps[0]) > 1:
        precision_means = np.mean(precisions, axis=1)
        precision_stds = np.std(precisions, axis=1)
        recall_means = np.mean(recalls, axis=1)
        recall_stds = np.std(recalls, axis=1)

        plt.subplot(1, 2, 1)
        plt.ylabel("Precision")
        plt.ylim([0, 1])
        plt.grid(ls="--", axis="y", alpha=0.5)
        plt.xticks(rotation=90)
        plt.bar(
            classes,
            precision_means,
            yerr=precision_stds,
            capsize=3,
            color="skyblue",
            alpha=0.8,
            edgecolor="black",
        )
        plt.title(
            f"Precision = {np.mean(precision_means) * 100:.2f}% ± {np.std(precision_means):.2f}",
            size=10,
        )

        plt.subplot(1, 2, 2)
        plt.ylabel("Recall")
        plt.ylim([0, 1])
        plt.grid(ls="--", axis="y", alpha=0.5)
        plt.xticks(rotation=90)
        plt.bar(
            classes,
            recall_means,
            yerr=recall_stds,
            capsize=3,
            color="skyblue",
            alpha=0.8,
            edgecolor="black",
        )
        plt.title(
            f"Recall = {np.mean(recall_means) * 100:.2f}% ± {np.std(recall_means):.2f}",
            size=10,
        )
        plt.show()

    elif len(list(detections.values())[0]["tp"]) == 1:
        plt.subplot(1, 2, 1)
        precision = np.mean(precisions)
        plt.bar(classes, precisions, color="skyblue")
        plt.title(f"{model_name} Precision = {precision * 100:.2f}%", size=10)
        plt.show()

    else:
        print("Detections dictionary is empty!")


def plot_ap(ap_scores: Dict[str, List], model_name: str) -> None:
    classes = list(ap_scores.keys())
    scores = list(ap_scores.values())

    plt.figure(figsize=(8, 4))
    plt.ylabel("Average Precision (AP)")
    plt.ylim([0, 1])
    plt.grid(ls="--", axis="y", alpha=0.5)
    plt.xticks(rotation=90)

    if len(scores[0]) > 1:
        ap_means = [np.mean(ap_scores[cls]) for cls in classes]
        ap_stds = [np.std(ap_scores[cls]) for cls in classes]
        plt.bar(
            classes,
            ap_means,
            yerr=ap_stds,
            capsize=3,
            color="skyblue",
            alpha=0.8,
            edgecolor="black",
        )
        plt.title(
            f"Average Precisions of {model_name} on {len(scores[0])} training runs\n mAP@.5 = {np.mean(ap_means)*100:.2f}% ± {np.std(ap_means):.2f}",
            size=10,
        )
        plt.show()

    elif len(scores[0]) == 1:
        mAP = sum(ap_scores.values()) / len(ap_scores.values())
        plt.bar(classes, scores, color="skyblue")
        plt.title(f"{model_name} mAP@.5 = {mAP*100:.2f}%", size=10)
        plt.show()

    else:
        print("AP scores dictionary is empty!")


def plot_precision_recall_curves(
    precisions: Dict[str, List[List]],
    recalls: Dict[str, List[List]],
    separate: bool = False,
    show_auc: bool = True,
    force_single_result: bool = False,
) -> None:
    num_results = len(precisions[list(precisions.keys())[0]])
    single_result = num_results == 1

    if not single_result and not separate and not force_single_result:
        raise ValueError(
            "Argument 'separate' has to be True and 'show_auc' has to be False for multiple results evaluation"
        )

    if single_result or force_single_result:
        if separate:
            fig, axes = plt.subplots(
                2, 4, figsize=(15, 7), gridspec_kw={"wspace": 0.4, "hspace": 0.5}
            )
            fig.suptitle("Precision Recall Curve")
            axes = axes.flatten()
            for i, class_name in enumerate(precisions.keys()):
                axes[i].plot(
                    recalls[class_name][0],
                    precisions[class_name][0],
                    label=f"{class_name}",
                    linewidth=2,
                )
                (
                    axes[i].fill_between(
                        recalls[class_name][0], precisions[class_name][0], alpha=0.2
                    )
                    if show_auc
                    else None
                )
                axes[i].set_xlabel("Recall")
                axes[i].set_ylabel("Precision")
                axes[i].set_xlim([0.0, 1.0])
                axes[i].set_ylim([0.0, 1.05])
                axes[i].set_title(f"{class_name}")
        else:
            plt.figure(figsize=(6, 5))
            for class_name in precisions.keys():
                plt.plot(
                    recalls[class_name][0],
                    precisions[class_name][0],
                    label=f"{class_name}",
                    linewidth=2,
                )
                (
                    plt.fill_between(
                        recalls[class_name][0], precisions[class_name][0], alpha=0.2
                    )
                    if show_auc
                    else None
                )

            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Precision Recall Curve")
            plt.legend(loc="lower left")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.show()

    else:
        fig, axes = plt.subplots(
            2, 4, figsize=(15, 7), gridspec_kw={"wspace": 0.4, "hspace": 0.5}
        )
        fig.suptitle(f"Precision Recall Curve on {num_results} training runs")
        axes = axes.flatten()

        for i, class_name in enumerate(precisions.keys()):
            max_recall = max(max(rec) for rec in recalls[class_name])
            common_recall = np.linspace(0, max_recall, 50)

            # Interpolate precision values to the common recall grid
            precision_interpolated = []
            for recall, precision in zip(recalls[class_name], precisions[class_name]):
                f = interpolate.interp1d(
                    recall, precision, kind="linear", fill_value="extrapolate"
                )
                precision_interpolated.append(f(common_recall))

            # Convert the list of interpolated precision values into a numpy array for easy manipulation
            precision_interpolated = np.array(precision_interpolated)

            # Calculate mean and standard deviation across the interpolated precision values
            precision_mean = np.mean(precision_interpolated, axis=0)
            precision_std = np.std(precision_interpolated, axis=0)

            # Plotting the mean precision-recall curve
            axes[i].plot(
                common_recall,
                precision_mean,
                label="Mean Precision",
                color="blue",
                lw=3,
            )

            (
                axes[i].fill_between(common_recall, precision_mean, alpha=0.8)
                if show_auc
                else None
            )

            # Adding standard deviation as a shaded area
            axes[i].fill_between(
                common_recall,
                precision_mean - precision_std,
                precision_mean + precision_std,
                color="skyblue",
                alpha=0.3,
                label="Std Dev",
            )

            # Plotting individual runs for reference (interpolated)
            for j, run in enumerate(precision_interpolated, start=1):
                axes[i].plot(
                    common_recall, run, linestyle="--", alpha=0.4, label=f"Run {j}"
                )

            # Customizing the plot
            axes[i].set_title(f"{class_name}")
            axes[i].set_xlim([0.0, 1.0])
            axes[i].set_ylim([0.0, 1.05])
            axes[i].set_xlabel("Recall")
            axes[i].set_ylabel("Precision")


# TODO
def plot_f1_curves(
    precisions: Dict[str, List[List]],
    recalls: Dict[str, List[List]],
) -> None:
    num_results = len(precisions[list(precisions.keys())[0]])
    single_result = num_results == 1

    if single_result:
        raise ValueError("TODO")
    else:

        plt.title(f"F1 Curve on {num_results} Training Runs")

        for i, class_name in enumerate(precisions.keys()):
            max_recall = max(max(rec) for rec in recalls[class_name])
            common_recall = np.linspace(0, max_recall, 100)

            # Interpolate precision values to the common recall grid
            precision_interpolated = []
            for recall, precision in zip(recalls[class_name], precisions[class_name]):
                f = interpolate.interp1d(
                    recall, precision, kind="linear", fill_value="extrapolate"
                )
                precision_interpolated.append(f(common_recall))

            # Convert the list of interpolated precision values into a numpy array for easy manipulation
            precision_interpolated = np.array(precision_interpolated)

            # Calculate mean and standard deviation across the interpolated precision values
            f1 = (
                2
                * precision_interpolated
                * common_recall
                / (precision_interpolated + common_recall + 1e-16)
            )
            f1_mean = np.mean(f1, axis=0)
            f1_std = np.std(f1, axis=0)

            # Plotting the mean precision-recall curve
            plt.plot(
                common_recall,
                f1_mean,
                label=f"{class_name}",
            )

            # Adding standard deviation as a shaded area
            plt.fill_between(
                common_recall,
                f1_mean - f1_std,
                f1_mean + f1_std,
                color="skyblue",
                alpha=0.3,
                # label="Std Dev",
            )

            # Customizing the plot
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xlabel("Confidence")
            plt.legend()
            plt.ylabel("F1")
        plt.show()


def measure_inference_time(
    model: torch.nn.Module,
    input_size: Tuple[int],
    device: torch.device,
    iteration: int = 1000,
    half: bool = False,
) -> torch.utils.benchmark.utils.common.Measurement:
    model = model.to(device)
    model.eval()

    b, ch, h, w = input_size  # Batch, input channel, height, width
    dummy_input = (
        torch.randn(b, ch, h, w, device=device).half()
        if half
        else torch.randn(b, ch, h, w, device=device)
    )

    # Warm up the model (to account for CUDA initialization)
    with torch.no_grad():  # Disable gradient computation
        for _ in range(3):  # Run some warm-up iterations
            _ = model(dummy_input)

    # Create timer benchmark
    timer = benchmark.Timer(
        stmt="model(dummy_input)", globals={"model": model, "dummy_input": dummy_input}
    )

    # Measure inference time
    result = timer.timeit(iteration)

    return result


def init_metrics_json(result_path: str) -> Dict[str, Dict]:
    result_path = Path(result_path)
    if os.path.exists(result_path):
        try:
            with open(result_path / "metrics.json", "r+") as f:
                metrics = json.load(f)
                if metrics.get("model", -1) == -1:
                    return {"model": {}, "hardware": {}}
                return metrics
        except json.JSONDecodeError:
            with open(result_path / "metrics.json", "r+") as f:
                return {"model": {}, "hardware": {}}
    else:
        with open(result_path / "metrics.json", "w") as f:
            return {"model": {}, "hardware": {}}


def save_json(data, file_path):
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=2)


# Add or update model information in the JSON data
def add_model_metrics(
    data: Dict[str, Dict],
    model_name: str,
    inference_time_cpu: float,
    inference_time_gpu: float,
    num_parameters: int,
    memory_usage: float,
) -> None:
    model_data = {
        "inference_time_gpu": inference_time_gpu,
        "inference_time_cpu": inference_time_cpu,
        "num_parameter": num_parameters,
        "memory_usage": memory_usage,
    }

    # Add metrics in the 'model' section
    data["model"][model_name] = model_data


def save_json(data, file_path):
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=2)


def coco2kitti(
    folder_path: Union[str, Path], convert_type: Literal["groundtruth", "detection"]
):
    folder_path = Path(folder_path)
    coco_classes = [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

    coco_id_to_class = {}
    for class_id, c in enumerate(coco_classes):
        coco_id_to_class[class_id] = c
    kitti_class_to_id = {
        "Car": 0,
        "Pedestrian": 1,
        "Van": 2,
        "Cyclist": 3,
        "Truck": 4,
        "Misc": 5,
        "Tram": 6,
        "Person_sitting": 7,
    }
    kitti_id_to_class = {v: n for n, v in kitti_class_to_id.items()}

    coco_to_kitti = {
        "car": "Car",
        "person": "Pedestrian",
        "truck": "Truck",
        "train": "Tram",
    }

    files = [
        f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]

    if convert_type == "detection":
        for file in files:
            label_file = os.path.join(folder_path, file)
            new_labels = []
            with open(label_file, "r") as text:
                labels = text.readlines()
            for label in labels:
                label = label.split()
                if coco_id_to_class[int(label[0])] in [
                    "car",
                    "person",
                    "truck",
                    "train",
                ]:
                    remaped_name = coco_to_kitti[coco_id_to_class[int(label[0])]]
                    label[0] = str(kitti_class_to_id[remaped_name])
                    label = " ".join(label)
                    new_labels.append(label + "\n")
            with open(label_file, "w") as text:
                text.writelines(new_labels)

    elif convert_type == "groundtruth":
        for file in files:
            label_file = os.path.join(folder_path, file)
            new_labels = []
            with open(label_file, "r") as text:
                labels = text.readlines()
            for label in labels:
                label = label.split()
                if kitti_id_to_class[int(label[0])] in [
                    "Car",
                    "Pedestrian",
                    "Truck",
                    "Tram",
                ]:
                    label = " ".join(label)
                    new_labels.append(label + "\n")
            with open(label_file, "w") as text:
                text.writelines(new_labels)
    else:
        raise ValueError("Select convert_type!")
