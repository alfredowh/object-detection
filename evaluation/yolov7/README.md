# YOLOv7 Evaluation

This folder implements the evaluation of object detection models on KITTI dataset and Gazebo simulation data. Evaluation metrics are (mean) average precisions, inference time and model size.

## Folder Structure

```bash
./
├── yolov7/                               # YOLOv7 evaluation
│   ├── detections/                       # Inference results on KITTI test data
│   ├── mAP_evaluation.ipynb              # Workflow of mAP evaluation for YOLOv7
│   ├── eval_metrics.ipynb                # Internal evaluation metrics with KITTI dataset
│   ├── mAP_evaluation_gazebo.ipynb       # Workflow of mAP evaluation for YOLOv7 on Gazebo data
│   └── eval_metrics_gazebo.ipynb         # Internal evaluation with Gazebo data
│   └── eval_pretrained_finetuned.ipynb   # Evaluation between pretrained and finetuned YOLOv7
│   └── other.ipynb                       # Other eval metrics
...
```

## Evaluation on KITTI dataset
After training the YOLOv7, we evaluate the model using a separate evaluation package. 
1. Run the inference script `yolov7/detect.py`with additional arguments `--save-txt --save-conf`to get inference detections in `*.txt`files
2. Copy the `*.txt` files in `detections/`
3. Follow the next steps on  [`mAP_evaluation.ipynb`](https://github.com/alfredowh/object-detection/blob/main/evaluation/yolov7/mAP_evaluation.ipynb) for preparing data to the mAP package
4. The evaluation results are located in `evaluation/output`
5. Visualize the evaluation metrics, including inference time and model size (see [`eval_metrics.ipynb`](https://github.com/alfredowh/object-detection/blob/main/evaluation/yolov7/eval_metrics.ipynb))


## Evaluation on Gazebo Simulation Data
The generated simulation data contains simple traffic scene with different weather conditions and time of day. The evaluation includes overall data and different **weather conditions** as well as **time of day**. 

The evaluation's workflow is similar to evaluation on KITTI dataset (for details see [`mAP_evaluation_gazebo.ipynb`](https://github.com/alfredowh/object-detection/blob/main/evaluation/yolov7/mAP_evaluation_gazebo.ipynb). The visualization of the evaluation results is located in [`eval_metrics_gazebo.ipynb`](https://github.com/alfredowh/object-detection/blob/main/evaluation/yolov7/eval_metrics_gazebo.ipynb).


## Evaluation between Pretrained and Finetuned YOLOv7
In this work, YOLOv7 was finetuned on KITTI dataset from YOLOv7 which is pretrained on MSCOCO dataset. The evaluation between the finetuned and pretrained model clarifies whether finetuning YOLOv7 on KITTI dataset brings any improvement compared to model pretrained on MSCOCO which rather contains general scenes (incl. traffic scene).

The evaluation's visualization is located in [`eval_pretrained_finetuned.ipynb`](https://github.com/alfredowh/object-detection/blob/main/evaluation/yolov7/eval_pretrained_finetuned.ipynb).
