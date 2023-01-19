# https://stackoverflow.com/questions/25389095/python-get-path-of-root-project-structure

import os
import rospkg

def makeAbsolute(path, startPath):
    if not os.path.isabs(path):
        path = os.path.join(startPath, path)
    return path

PACKAGE_DIR = rospkg.RosPack().get_path("lolo_perception")
CONFIG_DIR = os.path.join(PACKAGE_DIR, "config")
CAMERA_CONFIG_DIR = os.path.join(CONFIG_DIR, "camera_calibration_data")
FEATURE_MODEL_CONFIG_DIR = os.path.join(CONFIG_DIR, "feature_models")
TRACKING_CONFIG_DIR = os.path.join(CONFIG_DIR, "tracking_config")
IMAGE_DATASET_DIR = os.path.join(PACKAGE_DIR, "image_datasets")
ROSBAG_DIR = os.path.join(PACKAGE_DIR, "rosbags")
VIDEO_DIR = os.path.join(PACKAGE_DIR, "videos")
LOGGING_DIR = os.path.join(PACKAGE_DIR, "logging")