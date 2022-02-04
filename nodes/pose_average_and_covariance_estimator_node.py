#!/usr/bin/env python
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
import roslib
import rospy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseWithCovarianceStamped
import os.path
import glob
import time
import numpy as np
import matplotlib.pyplot as plt

from lolo_perception.perception_utils import PoseAverageAndCovarianceEstimator
from lolo_perception.perception_ros_utils import vectorToPose, poseToVector


class AveragePosesAndImagePoints:
    def __init__(self, nSamples, poseTopic):
        self.estimator = PoseAverageAndCovarianceEstimator(nSamples=nSamples)
        self.frameID = None # will be the same as the subscribed poses

        self.poseSubscriber = rospy.Subscriber(poseTopic, 
                                               PoseWithCovarianceStamped, 
                                               self.poseCallback)
        self.posePublisher = rospy.Publisher(poseTopic + "/average", 
                                             PoseWithCovarianceStamped, 
                                             queue_size=1)

    def poseCallback(self, pose):
        self.frameID = pose.header.frame_id
        translationVector, rotationVector = poseToVector(pose)
        self.estimator.add(translationVector, rotationVector)

    def calcPose(self):
        poseAvg = self.estimator.calcAverage()
        poseCov = self.estimator.calcCovariance()
        
        return vectorToPose(self.frameID, poseAvg[:3], poseAvg[3:], poseCov)

    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            self.posePublisher.publish(self.calcPose())
            rate.sleep()


if __name__ == '__main__':
    pass