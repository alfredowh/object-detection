import matplotlib.pyplot as plt
from typing import Dict, List
import numpy as np
from matplotlib.cm import get_cmap
import math


def plot_pr(
    detections: Dict[str, Dict[str, Dict[str, List]]],
    iou_thres: List[float],
    conf_thres: List[float],
) -> None:

    model_names = list(detections.keys())
    classes = list(detections[model_names[0]].keys())

    x = np.arange(len(classes))  # Class positions
    width = 0.7 / len(model_names)  # Adjust bar width based on number of models

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30, 10))
    fig.suptitle(
        f"Precision and Recall on 3 Training Runs", fontweight="bold", fontsize=24
    )
    for idx, model in enumerate(model_names):
        tps = [detections[model][cls]["tp"] for cls in classes]
        fps = np.array([detections[model][cls]["fp"] for cls in classes])
        total_gt = np.array([detections[model][cls]["total_gt"] for cls in classes])
        precisions = tps / (tps + fps)  # tps / total_dt
        recalls = tps / total_gt  # tps / (tps + fns)

        if len(tps[0]) > 1:
            precision_means = np.mean(precisions, axis=1)
            precision_stds = np.std(precisions, axis=1)
            recall_means = np.mean(recalls, axis=1)
            recall_stds = np.std(recalls, axis=1)

            # Precision bar plot
            axes[0].set_ylabel("Precision")
            axes[0].set_ylim([0, 1])
            axes[0].grid(ls="--", axis="y", alpha=0.5)
            axes[0].set_xticks(
                x + (width * (len(model_names) - 1)) / 2,
                classes,
                rotation=90,
                fontsize=16,
            )
            axes[0].bar(
                x + idx * width,
                precision_means,
                width,
                yerr=precision_stds,
                edgecolor="black",
                capsize=3,
                label=f"{model} {np.mean(precision_means) * 100:.2f}% ± {np.std(precision_means):.2f} IoU: {iou_thres[idx]} Conf: {conf_thres[idx]}",
            )

            axes[0].set_title(
                "Precision",
                size=20,
            )
            axes[0].legend(fontsize=16)

            # Recall bar plot
            axes[1].set_ylabel("Recall")
            axes[1].set_ylim([0, 1])
            axes[1].grid(ls="--", axis="y", alpha=0.5)
            axes[1].set_xticks(
                x + (width * (len(model_names) - 1)) / 2,
                classes,
                rotation=90,
                fontsize=16,
            )
            axes[1].bar(
                x + idx * width,
                recall_means,
                width,
                yerr=recall_stds,
                edgecolor="black",
                capsize=3,
                label=f"{model} {np.mean(recall_means) * 100:.2f}% ± {np.std(recall_means):.2f} IoU: {iou_thres[idx]} Conf: {conf_thres[idx]}",
            )

            axes[1].set_title(
                "Recall",
                size=20,
            )
            axes[1].legend(fontsize=16)

        elif len(tps[0]) == 1:
            print("TODO")

        else:
            print("Detections dictionary is empty!")

    plt.show()


def plot_ap(ap_scores: Dict[str, Dict[str, List]]) -> None:

    model_names = list(ap_scores.keys())
    classes = list(ap_scores[model_names[0]].keys())

    x = np.arange(len(classes))  # Class positions
    width = 0.7 / len(model_names)  # Adjust bar width based on number of models

    plt.figure(figsize=(10, 4))
    plt.ylabel("Average Precision (AP)")
    plt.ylim([0, 1])
    plt.grid(ls="--", axis="y", alpha=0.5)
    plt.xticks(rotation=90)

    for idx, model in enumerate(model_names):

        scores = list(ap_scores[model].values())
        ap_score_model = ap_scores[model]

        plt.title(
            f"Average Precisions on {len(scores[0])} Training Runs",
            size=10,
        )
        if len(scores[0]) > 1:
            ap_means = [np.mean(ap_score_model[cls]) for cls in classes]
            ap_stds = [np.std(ap_score_model[cls]) for cls in classes]

            plt.xticks(
                x + (width * (len(model_names) - 1)) / 2,
                classes,
                rotation=90,
            )
            plt.bar(
                x + idx * width,
                ap_means,
                width,
                yerr=ap_stds,
                edgecolor="black",
                capsize=3,
                label=f"{model} mAP@.5 = {np.mean(ap_means)*100:.2f}% ± {np.std(ap_means):.2f}",
            )

            plt.legend(loc="best")

        elif len(scores[0]) == 1:
            mAP = sum(ap_score_model.values()) / len(ap_score_model.values())
            plt.bar(classes, scores, color="skyblue")
            plt.title(f"{model} mAP@.5 = {mAP*100:.2f}%", size=10)

        else:
            print("AP scores dictionary is empty!")
    plt.show()


def plot_model_size(data):
    models = list(data["model"].keys())
    num_parameters = [data["model"][model]["num_parameter"] for model in models]
    memory_usage = [data["model"][model]["memory_usage"] for model in models]

    cmap = get_cmap()
    colors = [cmap(i / len(models)) for i in range(len(models))]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Model Size")

    for i, model in enumerate(models):
        axes[0].bar(model, num_parameters[i] / 1e6, color=colors[i])
        height = num_parameters[i] / 1e6
        axes[0].text(
            i, height + 0.001, f"{height:.3f} M", ha="center", va="bottom", fontsize=10
        )
    axes[0].set_title("Number of Parameters")
    axes[0].set_ylabel("Count (M)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)

    for i, model in enumerate(models):
        axes[1].bar(model, memory_usage[i], color=colors[i])
        height = memory_usage[i]
        axes[1].text(
            i, height + 0.001, f"{height:.3f} MB", ha="center", va="bottom", fontsize=10
        )

    axes[1].set_title("Memory Usage")
    axes[1].set_ylabel("Memory (MB)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)

    plt.show()


def plot_inference_time(data):

    models = list(data["model"].keys())
    inference_time_gpu = [
        data["model"][model]["inference_time_gpu"] for model in models
    ]
    inference_time_cpu = [
        data["model"][model]["inference_time_cpu"] for model in models
    ]
    cpu_name = data["hardware"]["cpu_name"]
    gpu_name = data["hardware"]["gpu_name"]

    cmap = get_cmap()
    colors = [cmap(i / len(models)) for i in range(len(models))]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Inference Time")

    for i, model in enumerate(models):
        axes[0].bar(model, inference_time_gpu[i], color=colors[i])
        height = inference_time_gpu[i]
        axes[0].text(
            i, height + 0.001, f"{height:.3f} s", ha="center", va="bottom", fontsize=10
        )

    high = max(inference_time_gpu)
    (
        axes[0].set_ylim([0, 0.5])
        if high < 0.5
        else axes[0].set_ylim([0, math.ceil(high + 0.5 * (high - 0))])
    )

    axes[0].set_title(f"GPU {gpu_name}")
    axes[0].set_ylabel("Time (s)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)

    for i, model in enumerate(models):
        axes[1].bar(model, inference_time_cpu[i], color=colors[i])
        height = inference_time_cpu[i]
        axes[1].text(
            i, height + 0.001, f"{height:.3f} s", ha="center", va="bottom", fontsize=10
        )

    high = max(inference_time_cpu)
    axes[1].set_ylim([0, math.ceil(high + 0.5 * (high - 0))])
    axes[1].set_title(f"CPU {cpu_name}")
    axes[1].set_ylabel("Time (s)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)

    plt.show()
