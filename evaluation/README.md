## Cross Model Evaluation

Object Detector is evaluated on KITTI dataset and Gazebo simulation data. Additionaly, the data exploration on the test data is conducted for better interpretation of the evaluation results.

- Model evaluation [`YOLOv7`](https://github.com/alfredowh/object-detection/tree/main/evaluation/yolov7)
- Evaluation on different time of day and weather conditions using Gazebo simulation data [`eval_metrics_gazebo.ipynb`](https://github.com/BuildmodeOne/IRO/blob/main/evaluation/yolov7/eval_metrics_gazebo.ipynb).
 

## Folder Structure

```bash
./
├── evaluation/                     
│   ├── mAP/                              # mAP calculation package
│   ├── results/                          # Evaluation results of each models
│   ├── yolov7/                           # Evaluation of YOLOv7
│   ├── kitti_data_exploration.ipynb      # Data exploration on KITTI dataset
│   └── gazebo_data_exploration.ipynb     # Data exploration on Gazebo simulation data
...
```

## Acknowledgements

This evaluation process utilizes significant portions of the implementation from the [mAP repository](https://github.com/Cartucho/mAP) by [Cartucho](https://github.com/Cartucho) and contributors. 
The repository includes mAP evaluation, scripts for label format conversion etc. which makes our cross model evaluations more efficient. For further details, please visit the [official repository](https://github.com/Cartucho/mAP).


### Citations

```bash
@INPROCEEDINGS{8594067,
  author={J. {Cartucho} and R. {Ventura} and M. {Veloso}},
  booktitle={2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Robust Object Recognition Through Symbiotic Deep Learning In Mobile Robots}, 
  year={2018},
  pages={2336-2341},
}
```
