
- [About this project](#about)
- [Running the code](#running-the-code)

 <a id="about"></a>
# About this project

This repository contains the source code used in this thesis project: [Robust light source detection for AUV docking (diva)](https://www.diva-portal.org/smash/record.jsf?dswid=-2379&pid=diva2%3A1748034&c=1&searchType=SIMPLE&language=sv&query=Joar+Edlund&af=%5B%5D&aq=%5B%5B%5D%5D&aq2=%5B%5B%5D%5D&aqe=%5B%5D&noOfRows=50&sortOrder=author_sort_asc&sortOrder2=title_sort_asc&onlyFullText=false&sf=all) 

The project aims to develop a complete light source tracking pipeline for AUV docking. The project investigates the idea of using light sources in a known configuration as guidance for navigation of a smaller AUV, equiped with a camera, into a docking station. Specifically, this project aims to investigate a scenario where the light sources are placed in the front on a slowly submarine, acting as the docking station, and the camera is placed in the back of the AUV which is is approaching the docking station from the front.

The tracking pipeline developed for this project is summarized as: 
- detect potential light sources
- extract the correct light sources in the presence of disturbance (outliers) such as reflections from the water surface and other illuminaries
- estimate the pose from the extracted light sources

Here are a few aspects that might not have been considered by other publications on the topic:
- How to extract the correct light sources in the presence of outliers
- Configuration of the light sources for improved pose accuracy

A hybrid approach for the light source detection algorithm have been implemented:
- Local Peak algorithm: A novel iterative thresholding approach for finding light source candidates that finds a suitable threshold value for each peak (light source)
- Modified HATS: A tweaked version of [HATS](https://www.researchgate.net/publication/285042917_Reliable_pose_estimation_of_underwater_dock_using_single_camera_a_scene_invariant_approach)

 <a id="running-the-code"></a>
# Running the code

## tracker.launch

    roslaunch lolo_perception tracker.launch

Launches tracker_node.py + some extra helper nodes (if specified).
Launch file mainly for executing tracker_node.py (see below) with some extra functionality.

## tracker_node.py

    rosrun lolo_perception tracker_node.py

Runs the tracker (see tracker.py) in a continuous loop. 
The node subscribes to images which are fed into the tracker in tracker.py and publishes varius information such as the estimated pose from the tracker.

## image_analyze_node.py

Recorded images in a video/rosbag and want to analyze the performance of your tracker?
With image_anallyze_node.py you can create "datasets" from rosbags/videos which then can be played, acting as a source for image publishing.    

### Create a dataset

    # From a video (OpenCV supported format)
    rosrun lolo_perception image_analyze_node.py <dataset_name> -file <video_path>
    # From a rosbag
    rosrun lolo_perception image_analyze_node.py <dataset_name> -file <video_path> -topic <image_topic>
 
 A dataset created using image_analyze_node.py is a folder containing:
- the images from your video/rosbag in png format
- dataset_metadata.yaml: configuration of how the images should be played
- dataset_labelfile.txt: a human readable file containing the current labels for the dataset

Use the -h option to see all available options.

### Analyze images

    rosrun lolo_perception image_analyze_node.py <dataset_name>

While playing the dataset with image_analyze_node.py, the images will be published.

<img src="readme_images/image_analyze_node.png" title="Image analyze node" alt="quat" width="500"/>
