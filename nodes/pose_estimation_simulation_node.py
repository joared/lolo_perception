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
from lolo_perception.pose_estimation import DSPoseEstimator, calcPoseCovariance
from lolo_perception.perception_utils import plotPosePoints, plotPoints, plotAxis, projectPoints, plotPosePointsWithReprojection
from lolo_perception.perception_ros_utils import vectorToPose, vectorToTransform, featurePointsToMsg


class PoseSimulation:
    def __init__(self, camera, featureModel):
        self.transformPublisher = rospy.Publisher("/tf", tf.msg.tfMessage, queue_size=1)
        self.cvBridge = CvBridge()
        self.imagePublisher = rospy.Publisher("pose/image", Image, queue_size=1)

        self.truePoseTopic = 'docking_station_true/pose'
        self.trueLightsTopic = 'docking_station_true/lights'
        self.noisedPoseTopic = 'docking_station_noised/pose'
        self.noisedLightsTopic = 'docking_station_noised/lights'
        self.posePublisher = rospy.Publisher(self.truePoseTopic, PoseWithCovarianceStamped, queue_size=1)
        self.featurePosesPublisher = rospy.Publisher(self.trueLightsTopic, PoseArray, queue_size=1)
        self.poseNoisedPublisher = rospy.Publisher(self.noisedPoseTopic, PoseWithCovarianceStamped, queue_size=1)
        self.featurePosesNoisedPublisher = rospy.Publisher(self.noisedLightsTopic, PoseArray, queue_size=1)

        self.poseEstimator = DSPoseEstimator(camera, featureModel, ignorePitch=False, ignoreRoll=False)

    def _getKey(self):
        pass

    def test2DNoiseError(self, sigmaX, sigmaY):
        """
        camera - Camera object
        featureModel - FeatureModel object
        pixelUncertainty - 2x2 matrix [[sigmaX, 0], [0, sigmaY]] where sigmaX and sigmaY are pixel std in x and y respectively
        """
        #img = cv.imread('../image_dataset/lights_in_scene.png')
        img = np.zeros(camera.resolution, dtype=np.int8)
        img = np.stack((img,)*3, axis=-1)
        img = img.astype(np.uint8)
        cv.imshow("feature model modification", img)

        print("Feature model uncertainty:", featureModel.uncertainty)
        featureModel.uncertainty = 0.008545
        featureModel.uncertainty = 0.03
        print("Feature model uncertainty:", featureModel.uncertainty)

        featurePoints = featureModel.features
        biasedFeaturePoints = featurePoints.copy()
        biasedFeaturePoints[0][0] += 0.0
        biasedFeaturePoints[0][1] += 0.0
        biasedFeaturePoints[0][2] += 0.0

        trueTrans = np.array([0, 0, 6], dtype=np.float32)
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
                trueTrans[2] = max(linInc, trueTrans[2])
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

            #projPoints = projectPoints(trueTrans, trueRotation, camera, featurePoints)
            biasedProjPoints = projectPoints(trueTrans, trueRotation, camera, biasedFeaturePoints)

            # Introduce noise:
            # We systematically displace each feature point 2 standard deviations from its true value
            # 2 stds (defined by pixelUncertainty) to the left, top, right, and bottom
            #noise2D = np.random.normal(0, sigmaX, projPoints.shape)
            projPointsNoised = biasedProjPoints.copy()
            projPointsNoised[:, 0] += np.random.normal(0, sigmaX, biasedProjPoints[:, 0].shape)
            projPointsNoised[:, 1] += np.random.normal(0, sigmaY, biasedProjPoints[:, 1].shape)

            # estimate pose (translation and rotation in camera frame)
            estTranslationVec = trueTrans.copy()
            estRotationVec = trueRotation.copy()

            # TODO: calculate based on feature model uncertainty instead
            pixelCovariance = np.array([[sigmaX**2, 0], [0, sigmaY**2]])

            # TODO: use something else than max intensity (255)?
            lightSourcesNoised = [ LightSource( np.array([[[ p[0], p[1] ]]], dtype=np.int32), intensity=255 ) for p in projPointsNoised]
            dsPoseNoised = self.poseEstimator.estimatePose(lightSourcesNoised, 
                                                      estTranslationVec,
                                                      estRotationVec)
            
            # calculates covariance based on max reprojection rmse from feature uncertainty
            dsPoseNoised.calcCovariance()
            trueCovariance = calcPoseCovariance(camera, featureModel, trueTrans, trueRotation, pixelCovariance)
         
            timeStamp = rospy.Time.now()

            pArray = featurePointsToMsg(self.truePoseTopic, biasedFeaturePoints, timeStamp=timeStamp)
            self.featurePosesPublisher.publish(pArray)

            pArray = featurePointsToMsg(self.noisedPoseTopic, featurePoints, timeStamp=timeStamp)
            self.featurePosesNoisedPublisher.publish(pArray)

            tTrue = vectorToTransform("camera1", 
                                      self.truePoseTopic, 
                                      trueTrans, 
                                      trueRotation, 
                                      timeStamp=timeStamp)
            tNoised = vectorToTransform("camera2", 
                                        self.noisedPoseTopic, 
                                        dsPoseNoised.translationVector, 
                                        dsPoseNoised.rotationVector, 
                                        timeStamp=timeStamp)
            self.transformPublisher.publish(tf.msg.tfMessage([tTrue, tNoised]))

            self.posePublisher.publish( vectorToPose("camera1", trueTrans, trueRotation, trueCovariance, timeStamp=timeStamp) )
            self.poseNoisedPublisher.publish( vectorToPose("camera2", dsPoseNoised.translationVector, dsPoseNoised.rotationVector, dsPoseNoised.covariance, timeStamp=timeStamp) )

            imgTemp = img.copy()
            # true axis
            plotAxis(imgTemp, trueTrans, trueRotation, camera, featurePoints, scale=featureModel.maxX, color=(0,255,0), opacity=0.5)
            # estimated axis
            plotAxis(imgTemp, dsPoseNoised.translationVector, dsPoseNoised.rotationVector, camera, featurePoints, scale=featureModel.maxX, color=(0,0,255), opacity=0.5)

            # estimated featurepoints
            plotPosePointsWithReprojection(imgTemp, dsPoseNoised.translationVector, dsPoseNoised.rotationVector, camera, featurePoints, projPointsNoised, color=(0,0,255))

            # true featurepoints and biased featurepoints
            plotPosePoints(imgTemp, trueTrans, trueRotation, camera, biasedFeaturePoints, color=(0,255,0))
            plotPoints(imgTemp, projPointsNoised, color=(255,0,255))
            
            self.imagePublisher.publish(self.cvBridge.cv2_to_imgmsg(imgTemp))

            cv.imshow("feature model modification", imgTemp)
            rate.sleep()

if __name__ =="__main__":
    from lolo_perception.feature_model import FeatureModel
    from lolo_perception.camera_model import Camera
    import os
    import rospkg

    rospy.init_node('pose_estimation_simulation')

    featureModelYaml = rospy.get_param("~feature_model_yaml")
    featureModelYamlPath = os.path.join(rospkg.RosPack().get_path("lolo_perception"), "feature_models/{}".format(featureModelYaml))
    featureModel = FeatureModel.fromYaml(featureModelYamlPath)

    cameraYaml = rospy.get_param("~camera_yaml")
    cameraYamlPath = os.path.join(rospkg.RosPack().get_path("lolo_perception"), "camera_calibration_data/{}".format(cameraYaml))
    camera = Camera.fromYaml(cameraYamlPath)

    sim = PoseSimulation(camera, featureModel)
    sim.test2DNoiseError(sigmaX=2, sigmaY=2)