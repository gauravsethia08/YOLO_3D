# YOLO 3D
ROS Package for Estimating 3D pose of an object using YOLOv3 Tiny from RGB + Depth Image. 

** The pakcage has been developed and tested on ROS Melodic. 

## Configuring
1. Install the required Libraries
```
$ pip install -r requirements.txt
```

2. Place the YOLO weights and config file in yolo folder and change the path accordingly in init function of Perception class.

3. Make the demo file executable
```
$ chmod +x demo.py
```

4. Execute the demo file
```
$ roslaunch YOLO_3D estimate_pose.launch
```

## Hardware Requirements

1. Depth Camera


## Training your custom YOLO Model

The current weights are trained on few objects from the YCB dataset. If you wish to train the model with custom dataset, follow this amazing [blog](https://medium.com/@today.rafi/train-your-own-tiny-yolo-v3-on-google-colaboratory-with-the-custom-dataset-2e35db02bf8f).

** The Jupyter Notebook provided by the blog author throw errors due to newer version of darknet. 
You can follow this notebook for training - [Notebook](https://colab.research.google.com/drive/1lJeAhFkzwXNxRljMiE0UfL6fPmwVwVi3?usp=sharing)


## Authors
[Gaurav Sethia](https://github.com/gauravsethia08), [Siddharth Ghodasara](https://github.com/SiddharthGhodasara), [Kaushik Balasundar](https://github.com/kaushikbalasundar)


## References
1. P. Adarsh, P. Rathi and M. Kumar, "YOLO v3-Tiny: Object Detection and Recognition using one stage improved model," 2020 6th International Conference on Advanced Computing and Communication Systems (ICACCS), 2020.

2. http://wiki.ros.org/image_geometry

3. https://medium.com/@today.rafi/train-your-own-tiny-yolo-v3-on-google-colaboratory-with-the-custom-dataset-2e35db02bf8f

 
