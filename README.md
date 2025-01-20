# Object Detection
This repo aims to improve the environmental perception of mobile robots in public road traffic. YOLOv7 model is used and evaluated. The evaluation metrics include mean average precision, inference speed and model size.

Moreover, images of a simple road traffic scene are generated with the help of ROS2/Gazebo. This simulation data offers perspective from the angle of the mobile robot, cross domain testing and environment scenarios (time of day and weather conditions). So, the evaluation from this simulation gives more insights of the model performance. Implementation of the data generation is located in [Data Generation with ROS2/Gazebo](https://github.com/alfredowh/ros2-object-detection).

## Setup

1. Download the dataset using the `download_data.py` script in the `tools` folder.
2. Training / fine-tuning models using different training data splits.
3. Model evaluation [`YOLOv7`](https://github.com/alfredowh/object-detection/tree/main/evaluation/yolov7)
4. Evaluation on different time of day and weather conditions using Gazebo simulation data [`eval_metrics_gazebo.ipynb`](https://github.com/alfredowh/object-detection/blob/main/evaluation/yolov7/eval_metrics_gazebo.ipynb).


## Folder Structure

``` bash
.
├── input (symlink)                 # Symlink to the KITTI data (only for local development)
├── gazebo_data/                    # Simulation data from Gazebo 
├── evaluation/                     # Cross models evaluation
│   ├── mAP/                        # mAP calculation package
│   ├── yolov7/                     # YOLOv7 internal evaluation
│   └── results/                    # Evaluation results
├── yolov7/                         # YOLOv7 package
├── tools/                          # Utility functions
├── .gitignore          
└── README.md     
```

## Model Architecture

![image](https://github.com/user-attachments/assets/5419b2be-e0ce-4d36-947a-3eb1a5fa6da0)
source: [open-mmlab](https://github.com/open-mmlab/mmyolo/tree/main/configs/yolov7)

## Acknowledgements

This project utilizes significant portions of the implementation from the [YOLOv7 repository](https://github.com/WongKinYiu/yolov7) by [WongKinYiu](https://github.com/WongKinYiu) and contributors. YOLOv7 is a state-of-the-art object detection framework and has been instrumental in the development of this project.

### Integrated Features

The following components from YOLOv7 have been directly adapted or extended:

- **Model Architecture**: The core model structure for object detection.
- **Fine-tuning Pipeline**: Scripts for data preprocessing, training, and evaluation.
- **Inference Code**: Utilities for running predictions on new data.

### Citations

```bash
@inproceedings{wang2023yolov7,
  title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}
```

```bash
@article{wang2023designing,
  title={Designing Network Design Strategies Through Gradient Path Analysis},
  author={Wang, Chien-Yao and Liao, Hong-Yuan Mark and Yeh, I-Hau},
  journal={Journal of Information Science and Engineering},
  year={2023}
}
```



