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

import sys
import os
dirPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dirPath, "../../simulation/scripts"))
from coordinate_system import CoordinateSystem, CoordinateSystemArtist
from feature import polygon, FeatureModel
from camera import usbCamera
from pose_estimation import DSPoseEstimator
from perception_utils import plotPosePoints, plotAxis, projectPoints
from perception_ros_utils import vectorToPose

def publishFeaturePoses(frameID, featurePoints, poseCovariance):
    timeStamp = rospy.Time.now()
    for i, point in enumerate(featurePoints):
        frameID = "{}/{}".format(frameID, i+1)
        covariance = np.zeros([0]*36) # TODO: use poseCovariance
        p = vectorToPose(frameID, point, np.zeros((1, 3)), covariance)

def vectorToTransform(frameID, childFrameID, translationVector, rotationVector):
    global posePublisher, transformPublisher

    t = TransformStamped()
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = frameID
    t.child_frame_id = childFrameID
    t.transform.translation.x = translationVector[0]
    t.transform.translation.y = translationVector[1]
    t.transform.translation.z = translationVector[2]

    rotMat = R.from_rotvec(rotationVector).as_dcm()
    rotMatHom = np.hstack((rotMat, np.zeros((3, 1))))
    rotMatHom = np.vstack((rotMatHom, np.array([0, 0, 0, 1])))
    q = quaternion_from_matrix(rotMatHom)
    t.transform.rotation = Quaternion(*q)
    return t

class PoseSimulation:
    def __init__(self):
        self.transformPublisher = rospy.Publisher("/tf", tf.msg.tfMessage, queue_size=1)
        self.cvBridge = CvBridge()
        self.imagePublisher = rospy.Publisher("pose/image", Image, queue_size=1)

        self.posePublisher = rospy.Publisher('light_true/pose', PoseWithCovarianceStamped, queue_size=1)
        self.featurePosesPublisher = rospy.Publisher('light_true/poses', PoseArray, queue_size=1)
        self.poseNoisedPublisher = rospy.Publisher('light_noised/pose', PoseWithCovarianceStamped, queue_size=1)
        self.featurePosesNoisedPublisher = rospy.Publisher('light_noised/poses', PoseArray, queue_size=1)

    def _getKey(self):
        pass

    def test2DNoiseError(self, camera, featureModel, sigmaX, sigmaY):
        """
        camera - Camera object
        featureModel - FeatureModel object
        pixelUncertainty - 2x2 matrix [[sigmaX, 0], [0, sigmaY]] where sigmaX and sigmaY are pixel std in x and y respectively
        """
        #img = cv.imread('../image_dataset/lights_in_scene.png')
        img = np.zeros(camera.resolution, dtype=np.int8)
        img = np.stack((img,)*3, axis=-1)
        cv.imshow("feature model modification", img)

        poseEstimator = DSPoseEstimator(camera, ignorePitch=False, ignoreRoll=False)
        featurePoints = featureModel.features

        trueTrans = np.array([0, 0, 1], dtype=np.float32)
        trueRotation = np.eye(3, dtype=np.float32)
        ax, ay, az = 0, 0, 0 # euler angles
        featureIdx = -1
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            
            linInc = 0.1
            angInc = 0.1
            key = cv.waitKey(1)
            if key == ord('w'):
                trueTrans[2] += linInc
            elif key == ord("s"):
                trueTrans[2] -= linInc
            elif key == ord("d"):
                trueTrans[0] += linInc
            elif key == ord("a"):
                trueTrans[0] -= linInc
            elif key == ord("i"):
                ax -= angInc
            elif key == ord("k"):
                ax += angInc
            elif key == ord("l"):
                ay += angInc
            elif key == ord("j"):
                ay -= angInc
            elif key == ord("n"):
                rotMat = R.from_euler("XYZ", (0, 0, -angInc)).as_dcm()
                if featureIdx == -1:
                    featurePoints = np.matmul(rotMat, featurePoints.transpose()).transpose()
                else:
                    featurePoints[featureIdx, :] = np.matmul(rotMat, featurePoints[featureIdx, :].transpose()).transpose()
            elif key == ord("m"):
                rotMat = R.from_euler("XYZ", (0, 0, angInc)).as_dcm()
                if featureIdx == -1:
                    featurePoints = np.matmul(rotMat, featurePoints.transpose()).transpose()
                else:
                    featurePoints[featureIdx, :] = np.matmul(rotMat, featurePoints[featureIdx, :].transpose()).transpose()
            elif key in [ord(str(i)) for i in range(len(featureModel.features))]:
                featureIdx = int(chr(key))
            else:
                if key != -1:
                    featureIdx = -1
                    print(chr(key))

            r = R.from_euler("YXZ", (ay, ax, 0))
            trueRotation = r.as_rotvec().transpose()
            projPoints = projectPoints(trueTrans, trueRotation, camera, featurePoints)

            # Introduce noise:
            # We systematically displace each feature point 2 standard deviations from its true value
            # 2 stds (defined by pixelUncertainty) to the left, top, right, and bottom
            #noise2D = np.random.normal(0, sigmaX, projPoints.shape)
            projPointsNoised = np.zeros(projPoints.shape)
            projPointsNoised[:, 0] = projPoints[:, 0] + np.random.normal(0, sigmaX, projPoints[:, 0].shape)
            projPointsNoised[:, 1] = projPoints[:, 1] + np.random.normal(0, sigmaY, projPoints[:, 1].shape)

            # estimate pose (translation and rotation in camera frame)
            estTranslationVec = trueTrans.copy()
            estRotationVec = trueRotation.copy()
            translationVector, rotationVector, covariance = poseEstimator.update(featurePoints, 
                                                                                projPointsNoised, 
                                                                                np.array([[4*sigmaX*sigmaX, 0.], # we set covariance as (2*std)^2 for 95% coverage
                                                                                        [0., 4*sigmaY*sigmaY]]),
                                                                                estTranslationVec,
                                                                                estRotationVec)

            _, _, covarianceUnNoised = poseEstimator.update(featurePoints, 
                                                            projPoints, 
                                                            np.array([[4*sigmaX*sigmaX, 0.],
                                                                    [0., 4*sigmaY*sigmaY]]),
                                                            estTranslationVec,
                                                            estRotationVec)

            covariance[3][3] = .1
            covariance[4][4] = .6

            pArray = PoseArray()
            #pArrayNoised = PoseArray()
            pArray.header.frame_id = "lights_noised"
            pArray.header.stamp = rospy.Time.now()
            for p in featurePoints:
                pose = Pose()
                pose.position.x = p[0]
                pose.position.y = p[1]
                pose.position.z = p[2]
                pose.orientation.w = 1
                pArray.poses.append(pose)
                #covariance = np.zeros([0]*36) # TODO: use poseCovariance
                #p = vectorToPose(frameID, point, np.zeros((1, 3)), covariance)

            self.featurePosesPublisher.publish(pArray)
            pArray.header.frame_id = "lights_true"
            self.featurePosesNoisedPublisher.publish(pArray)

            tTrue = vectorToTransform("camera1", "lights_true", trueTrans, trueRotation)
            tNoised = vectorToTransform("camera2", "lights_noised", translationVector, rotationVector)
            self.transformPublisher.publish(tf.msg.tfMessage([tTrue, tNoised]))

            self.posePublisher.publish( vectorToPose("camera1", trueTrans, trueRotation, covarianceUnNoised) )
            self.poseNoisedPublisher.publish( vectorToPose("camera2", translationVector, rotationVector, covariance) )

            imgTemp = img.copy()
            # true pose
            plotPosePoints(imgTemp, trueTrans, trueRotation, camera, featurePoints, color=(0,255,0))
            #plotAxis(imgTemp, rotation, translation, camera_matrix, dist_coeffs)

            # estimated pose
            plotPosePoints(imgTemp, translationVector, rotationVector, camera, featurePoints, color=(0,0,255))
            plotAxis(imgTemp, translationVector, rotationVector, camera, featurePoints, scale=0.043)

            self.imagePublisher.publish(self.cvBridge.cv2_to_imgmsg(imgTemp))

            rate.sleep()

if __name__ =="__main__":
    
    rospy.init_node('pose_estimation_simulation')

    camera = usbCamera
    #featureModel = FeatureModel([0, 0.06], [1, 4], [False, True], [0.043, 0])
    #featureModel = FeatureModel([0.06], [4], [True], [0])#, euler=(0, np.pi, 0))
    featureModel = FeatureModel([0, 0.3], [1, 4], [False, True], [0.3, 0])
    #featureModel = FeatureModel([0.3], [4], [True], [0])
    PoseSimulation().test2DNoiseError(camera, featureModel, sigmaX=1*camera.pixelWidth, sigmaY=1*camera.pixelHeight)