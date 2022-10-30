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
from lolo_perception.pose_estimation import DSPoseEstimator, calcPoseCovarianceFixedAxis, ellipseToCovariance
from lolo_perception.perception_utils import plotPosePoints, plotPoints, plotAxis, projectPoints, plotPosePointsWithReprojection, plotErrorEllipse
from lolo_perception.perception_ros_utils import vectorToPose, vectorToTransform, featurePointsToMsg


class PoseSimulation:
    def __init__(self, camera):
        self.camera = camera


    def calcValidPoints(self, tempNoisedProjPoints, point3D, projPoint, sigma, mahaDistThresh=2.0):
        valid = []
        invalid = []

        fx = self.camera.cameraMatrix[0, 0]
        fy = self.camera.cameraMatrix[1, 1]

        X, Y, Z = point3D
        sigmaU2 = sigma**2*fx**2*(X**2 + Z**2)/Z**4
        sigmaV2 = sigma**2*fy**2*(Y**2 + Z**2)/Z**4
        sigmaUV2 = sigma**2*fx*fy*X*Y/Z**4
        pixelCovariance = np.array([[sigmaU2, sigmaUV2], 
                                    [sigmaUV2, sigmaV2]])


        pCovInv = np.linalg.inv(pixelCovariance)

        for p in tempNoisedProjPoints:
            err = np.array([p[0]-projPoint[0], p[1]-projPoint[1]])
            mahaDist = np.matmul(np.matmul(err.transpose(), pCovInv), err)
            mahaDist = np.sqrt(mahaDist)
            if mahaDist < mahaDistThresh:
                valid.append(p)
            else:
                invalid.append(p)

        return valid, invalid

    def __plotErrorEllipse(self, img, center, pixelCovariance, confidence=5.991, color=(0,0,255), displayAxis=True):
        lambdas, vs = np.linalg.eig(pixelCovariance)
        l1, l2 = lambdas
        v1, v2 = vs
        #angle = np.atan2()
        dir = -1
        axesScale = .5 # Correction for the 2
        if l1 > l2:
            #dir = -1
            angle = np.rad2deg(np.arctan2(v1[1], v1[0]))
            major = int(round(axesScale*2*np.sqrt(confidence*l1))) # has to be int for some reason
            minor = int(round(axesScale*2*np.sqrt(confidence*l2))) # has to be int for some reason
        else:
            #dir = 1
            angle = np.rad2deg(np.arctan2(v2[1], v2[0]))
            major = int(round(axesScale*2*np.sqrt(confidence*l2))) # has to be int for some reason
            minor = int(round(axesScale*2*np.sqrt(confidence*l1))) # has to be int for some reason

        center = int(round(center[0])), int(round(center[1]))
        if displayAxis:
            for l, v in zip((l1, l2), (v1, v2)):
                v = v[0], -v[1]
                l = axesScale*2*np.sqrt(confidence*l)
                end = center[0]+v[0]*l, center[1]+v[1]*l
                end = int(round(end[0])), int(round(end[1]))
                cv.line(img, center, end, color=color)
        
        cv.ellipse(img, center, (major, minor), -angle, 0, 360, color=color, thickness=2)

    def test(self, uncertainty, confidence):
        
        sigma = uncertainty/np.sqrt(confidence)

        self.camera.cameraMatrix[0, 0] = 300
        self.camera.cameraMatrix[1, 1] = 300
        self.camera.cameraMatrix[0, 2] = self.camera.resolution[1]/2 -1
        self.camera.cameraMatrix[1, 2] = self.camera.resolution[0]/2 -1

        img = 255*np.ones(self.camera.resolution, dtype=np.int8)
        img = np.stack((img,)*3, axis=-1)
        img = img.astype(np.uint8)*0
        cv.imshow("feature model modification", img)

        points3D = []
        for i in range(-10, 11):
            for j in range(-5, 6):
                points3D.append([1*(i+1), 1*(j+1), 2.])
        points3D = np.array(points3D)
        #points3D = np.array([[0, 0, 0.5], [-0.5, -0.5, 1], [0.5, -0.5, 1], [-0.5, 0.5, 1], [0.5, 0.5, 1]])
        #r = 0.06
        #points3D = np.array(list(points3D) + [[r,0,0.5], [-r,0,0.5], [0,r,0.5], [0,-r,0.5]])
        #print(points3D)

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

            #plotPoints(imgTemp, projPoints, color=(0,255,0))
     
            mahaDistThresh = np.sqrt(confidence)

            fx = self.camera.cameraMatrix[0, 0]
            fy = self.camera.cameraMatrix[1, 1]
            for p3D, p2D in zip(points3D, projPoints):
                X, Y, Z = p3D
                sigmaU2 = sigma**2*fx**2*(X**2 + Z**2)/Z**4
                sigmaV2 = sigma**2*fy**2*(Y**2 + Z**2)/Z**4
                sigmaUV2 = sigma**2*fx*fy*X*Y/Z**4
                pixelCovariance = np.array([[sigmaU2, sigmaUV2], 
                                            [sigmaUV2, sigmaV2]])
                major, minor, angle = plotErrorEllipse(imgTemp, tuple(p2D), pixelCovariance, confidence=confidence, color=(0,0,255), displayAxis=False)
                
            
            tempNoisedProjPoints.append(projPointsNoised[0])
            tempNoisedProjPoints = tempNoisedProjPoints[-100:]

            valid, invalid = self.calcValidPoints(tempNoisedProjPoints, points3D[0], projPoints[0], sigma, mahaDistThresh=mahaDistThresh)

            print("Ratio: {}".format(float(len(valid))/len(tempNoisedProjPoints)))
            plotPoints(imgTemp, valid, color=(255,0,0))
            plotPoints(imgTemp, invalid, color=(0,0,255))

            pixelCovarianceEst = np.cov(tempNoisedProjPoints, rowvar=False)
            if len(tempNoisedProjPoints) > 1:
                major, minor, angle = plotErrorEllipse(imgTemp, tuple(projPoints[0]), pixelCovarianceEst, confidence=confidence, color=(0,0,255))
                pixelCovarianceTest = ellipseToCovariance(major, minor, angle, confidence)
                plotErrorEllipse(imgTemp, tuple(projPoints[0]), pixelCovarianceTest, confidence=confidence, color=(255,0,255))
                
                #assert pixelCovariance[0,0] == pixelCovarianceTest[0,0]
                #assert pixelCovariance[0,1] == pixelCovarianceTest[0,1]
                #assert pixelCovariance[1,1] == pixelCovarianceTest[1,1]

            #pCovInv = np.linalg.inv(pCov)
            #mahaDist = np.matmul(np.matmul(err.transpose(), pCovInv), err)
            #mahaDist = np.sqrt(mahaDist)

            cv.imshow("feature model modification", imgTemp)
            
            rate.sleep()

if __name__ =="__main__":
    from lolo_perception.camera_model import Camera
    import os
    import rospkg

    rospy.init_node('pixel_covariance_node')

    cameraYaml = "usb_camera_720p_8.yaml"
    #cameraYaml = "contour.yaml"
    cameraYamlPath = os.path.join(rospkg.RosPack().get_path("lolo_perception"), "camera_calibration_data/{}".format(cameraYaml))
    camera = Camera.fromYaml(cameraYamlPath)
    camera.cameraMatrix = camera.projectionMatrix
    camera.distCoeffs *= 0

    sim = PoseSimulation(camera=camera)
    sim.test(uncertainty=0.06, 
             confidence=5.991
             )
