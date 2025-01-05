# YOLOv7 on KITTI Dataset

Implementation of paper [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696)

## Folder Structure

```bash
IRO/
├── yolov7/  
│   ├── cfg/training        # config files
│   ├── data/               # Data and hyperparameters
│   ├── examples/           # Notebooks showing the usage of the yolov7 package
│   ├── models/             # Model architectures
│   ├── utils/              # Utility functions
...
```

## Evaluation
- Finetuned YOLOv7 vs. COCO pretrained YOLOv7 [`eval_pretrained_finetuned.ipynb`](https://github.com/BuildmodeOne/IRO/blob/main/evaluation/yolov7/eval_pretrained_finetuned.ipynb)
- Evaluation of the finetuned YOLOv7 on KITTI Dataset [`eval_metrics.ipynb`](https://github.com/BuildmodeOne/IRO/blob/main/evaluation/yolov7/eval_metrics.ipynb)
- Evaluation on Gazebo simulation data on different weather conditions and time of day [`eval_metrics_gazebo.ipynb`](https://github.com/BuildmodeOne/IRO/blob/main/evaluation/yolov7/eval_metrics_gazebo.ipynb)
- Cross model evaluation on KITTI dataset and Gazebo simulation data [`crossmodel_evaluation.ipynb`](https://github.com/BuildmodeOne/IRO/blob/main/evaluation/crossmodel_evaluation.ipynb)  


## Fine-Tuning

Workflow (see [`examples/finetuning_pipeline.ipynb`](https://github.com/BuildmodeOne/IRO/blob/main/yolov7/examples/finetuning_pipeline.ipynb)):

Pre-trained weights on COCO dataset [`yolov7-tiny.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt)

``` shell
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt

python3 train.py --workers 8 --device 0 --batch-size 32 --data data/kitti.yaml --img 640 640 --cfg cfg/training/yolov7-custom.yaml --weights 'yolov7-tiny.pt' --name split_1 --hyp data/hyp.scratch.custom.yaml
```

The trained weights `init.pt, best.pt, last.pt etc.` will be saved in `runs/train/split_1/weights/`.

## Inference

Workflow (see [`examples/inference_pipeline.ipynb`](https://github.com/BuildmodeOne/IRO/blob/main/yolov7/examples/inference_pipeline.ipynb)):

``` shell
mv yolov7-tiny-kitti/split_1/weights/best.pt ./yolov7-tiny-kitti.pt

python3 detect.py --weights yolov7-tiny-kitti.pt --conf 0.25 --img-size 640 --source ./test --name split_1 --save-txt --save--conf
```

Images with detections and the labels will be saved in `runs/detect/split_1`.