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

from lolo_perception.perception_ros_utils import vectorToPose, poseToVector

class PoseAverageAndCovarianceEstimator:
    """
    Utility class to use when evaluating uncertainty of pose and image points.
    Make sure all the samples are correct (watch out for outliers)
    """
    def __init__(self, nSamples=100):
        self.poseVecs = [] # list of pose vectors [tx, ty, tz, rx, ry, rz]
        self.nSamples = nSamples

    def add(self, translationVec, rotationVec):

        self.poseVecs.insert(0, list(translationVec) + list(rotationVec))
        self.poseVecs = self.poseVecs[:self.nSamples]

    def calcCovariance(self):
        if len(self.poseVecs) > 1:
            poseCov = np.cov(self.poseVecs, rowvar=False)
        else:
            poseCov = np.zeros((6,6), dtype=np.float32)

        return poseCov

    def calcAverage(self):
        # This only works for small differences in rotation and 
        # when rotations are not near +-pi
        if self.poseVecs:
            return np.mean(self.poseVecs, axis=0)

        return np.zeros((6,), dtype=np.float32)

class PoseAverageEstimatorNode:
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
    rospy.init_node("average_pose_estimator")
    poseTopic = rospy.get_param("~pose_topic")
    estimator = PoseAverageEstimatorNode(nSamples=100, poseTopic=poseTopic)
    estimator.run()
