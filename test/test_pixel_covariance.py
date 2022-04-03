#!/usr/bin/env python

import cv2 as cv
import numpy as np
import time
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from tf.transformations import quaternion_from_matrix

from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray, Pose, Quaternion, TransformStamped
import tf
import tf.msg
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

from lolo_perception.feature_extraction import LightSource
from lolo_perception.pose_estimation import DSPoseEstimator, calcPoseCovarianceFixedAxis
from lolo_perception.perception_utils import plotPosePoints, plotPoints, plotAxis, projectPoints, plotPosePointsWithReprojection
from lolo_perception.perception_ros_utils import vectorToPose, vectorToTransform, featurePointsToMsg


class PoseSimulation:
    def __init__(self, camera):
        self.camera = camera

    def plotErrorEllipse(self, img, center, pixelCovariance, color):
        lambdas, vs = np.linalg.eig(pixelCovariance)
        l1, l2 = lambdas
        v1, v2 = vs
        #angle = np.atan2()
        dir = -1
        if l1 > l2:
            #dir = -1
            angle = np.rad2deg(np.arctan2(v1[1], v1[0]))
            major = int(round(2*np.sqrt(5.991*l1))) # has to be int for some reason
            minor = int(round(2*np.sqrt(5.991*l2))) # has to be int for some reason
        else:
            #dir = 1
            angle = np.rad2deg(np.arctan2(v2[1], v2[0]))
            major = int(round(2*np.sqrt(5.991*l2))) # has to be int for some reason
            minor = int(round(2*np.sqrt(5.991*l1))) # has to be int for some reason

        center = int(round(center[0])), int(round(center[1]))
        for l, v in zip((l1, l2), (v1, v2)):
            v = v[0], -v[1]
            l = 2*np.sqrt(5.991*l)
            end = center[0]+v[0]*l, center[1]+v[1]*l
            end = int(round(end[0])), int(round(end[1]))
            cv.line(img, center, end, color=color)
        
        cv.ellipse(img, center, (major, minor), -angle, 0, 360, color=color)

    def test(self, sigma):
        
        self.camera.cameraMatrix[0, 0] = 300
        self.camera.cameraMatrix[1, 1] = 300

        img = np.zeros(self.camera.resolution, dtype=np.int8)
        img = np.stack((img,)*3, axis=-1)
        img = img.astype(np.uint8)
        cv.imshow("feature model modification", img)

        points3D = np.array([[0, 0, 0.5], [-0.5, -0.5, 1], [0.5, -0.5, 1], [-0.5, 0.5, 1], [0.5, 0.5, 1]])
        tempNoisedProjPoints = []
        
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():

            linInc = 0.1
            key = cv.waitKey(1)
            
            if key == ord('w'):
                points3D[:, 1] -= linInc
            elif key == ord("s"):
                points3D[:, 1] += linInc
            elif key == ord("d"):
                points3D[:, 0] += linInc
            elif key == ord("a"):
                points3D[:, 0] -= linInc
            elif key == ord("i"):
                points3D[:, 2] += linInc
            elif key == ord("k"):
                points3D[:, 2] -= linInc

            projPoints = []
            for p3D in points3D:
                p2D = projectPoints(p3D, np.zeros((1,3), dtype=np.float32), self.camera, np.array([[0., 0., 0.]]))
                projPoints.append(p2D[0])
            projPoints = np.array(projPoints)

            points3DNoised = points3D.copy()
            points3DNoised[:, 0] += np.random.normal(0, sigma, points3DNoised[:, 0].shape)
            points3DNoised[:, 1] += np.random.normal(0, sigma, points3DNoised[:, 1].shape)
            points3DNoised[:, 2] += np.random.normal(0, sigma, points3DNoised[:, 2].shape)

            projPointsNoised = []
            for p3D in points3DNoised:
                p2D = projectPoints(p3D, np.zeros((1,3), dtype=np.float32), self.camera, np.array([[0., 0., 0.]]))
                projPointsNoised.append(p2D[0])
            projPointsNoised = np.array(projPointsNoised)
            projPointsNoised = projPointsNoised.round().astype(np.int32)

            imgTemp = img.copy()

            plotPoints(imgTemp, projPoints, color=(0,255,0))
          
            fx = self.camera.cameraMatrix[0, 0]
            fy = self.camera.cameraMatrix[1, 1]
            for p3D, p2D in zip(points3D, projPoints):
                X, Y, Z = p3D
                sigmaU2 = sigma**2*fx**2*(X**2 + Z**2)/Z**4
                sigmaV2 = sigma**2*fy**2*(Y**2 + Z**2)/Z**4
                sigmaUV2 = sigma**2*fx*fy*X*Y/Z**4
                pixelCovariance = np.array([[sigmaU2, sigmaUV2], 
                                            [sigmaUV2, sigmaV2]])
                self.plotErrorEllipse(imgTemp, tuple(p2D), pixelCovariance, color=(0,255,0))


            tempNoisedProjPoints.append(projPointsNoised[0])
            tempNoisedProjPoints = tempNoisedProjPoints[-300:]
            plotPoints(imgTemp, tempNoisedProjPoints, color=(0,0,255))
            pixelCovarianceEst = np.cov(tempNoisedProjPoints, rowvar=False)
            if len(tempNoisedProjPoints) > 1:
                self.plotErrorEllipse(imgTemp, tuple(projPoints[0]), pixelCovarianceEst, color=(0,0,255))

            cv.imshow("feature model modification", imgTemp)
            
            rate.sleep()

if __name__ =="__main__":
    from lolo_perception.camera_model import Camera
    import os
    import rospkg

    rospy.init_node('pixel_covariance_node')

    cameraYaml = "usb_camera_720p_8.yaml"
    cameraYamlPath = os.path.join(rospkg.RosPack().get_path("lolo_perception"), "camera_calibration_data/{}".format(cameraYaml))
    camera = Camera.fromYaml(cameraYamlPath)
    camera.cameraMatrix = camera.projectionMatrix
    camera.distCoeffs *= 0

    sim = PoseSimulation(camera=camera)
    sim.test(0.01)
