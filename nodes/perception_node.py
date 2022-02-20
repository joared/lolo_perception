#!/usr/bin/env python
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
#import roslib
import rospy
import tf.msg
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray
import time
import numpy as np
import itertools

from lolo_perception.feature_extraction import featureAssociation, AdaptiveThreshold2, AdaptiveThresholdPeak
from lolo_perception.pose_estimation import DSPoseEstimator
from lolo_perception.perception_utils import plotPoseImageInfo
from lolo_perception.perception_ros_utils import vectorToPose, vectorToTransform, lightSourcesToMsg, featurePointsToMsg
from lolo_perception.perception import Perception

class PerceptionNode:
    def __init__(self, featureModel):
        self.cameraTopic = "lolo_camera"
        self.cameraInfoSub = rospy.Subscriber("lolo_camera/camera_info", CameraInfo, self._getCameraCallback)
        self.camera = None
        while not rospy.is_shutdown() and self.camera is None:
            print("Waiting for camera info to be published")
            rospy.sleep(1)

        self.perception = Perception(self.camera, featureModel)
 
        self.imageMsg = None
        self.bridge = CvBridge()
        self.imgSubsciber = rospy.Subscriber('lolo_camera/image_rect_color', Image, self._imgCallback)

        # publish some images for visualization
        self.imgProcPublisher = rospy.Publisher('lolo_camera/image_processed', Image, queue_size=1)
        self.imgProcDrawPublisher = rospy.Publisher('lolo_camera/image_processed_draw', Image, queue_size=1)
        self.imgPosePublisher = rospy.Publisher('lolo_camera/image_processed_pose', Image, queue_size=1)

        # publish associated light source image points as a PoseArray
        self.associatedImagePointsPublisher = rospy.Publisher('lolo_camera/associated_image_points', PoseArray, queue_size=1)

        # publish estimated pose
        self.posePublisher = rospy.Publisher('docking_station/feature_model/estimated_pose', PoseWithCovarianceStamped, queue_size=1)

        # publish transform of estimated pose
        self.transformPublisher = rospy.Publisher("/tf", tf.msg.tfMessage, queue_size=1)

        # publish placement of the light sources as a PoseArray (published in the docking_station frame)
        self.featurePosesPublisher = rospy.Publisher('docking_station/feature_model/estimated_poses', PoseArray, queue_size=1)

    def _getCameraCallback(self, msg):
        """
        Use either K and D or just P
        https://answers.ros.org/question/119506/what-does-projection-matrix-provided-by-the-calibration-represent/
        https://github.com/dimatura/ros_vimdoc/blob/master/doc/ros-camera-info.txt
        """
        from lolo_perception.camera_model import Camera
        # Using only P (D=0), we should subscribe to the rectified image topic
        camera = Camera(cameraMatrix=np.array(msg.P, dtype=np.float32).reshape((3,4))[:, :3], 
                        distCoeffs=np.zeros((4,1), dtype=np.float32),
                        resolution=(msg.height, msg.width))
        # Using K and D, we should subscribe to the raw image topic
        #_camera = Camera(cameraMatrix=np.array(msg.K, dtype=np.float32).reshape((3,3)), 
        #                distCoeffs=np.array(msg.D, dtype=np.float32),
        #                resolution=(msg.height, msg.width))
        self.camera = camera

        # We only want one message
        self.cameraInfoSub.unregister()

    def _imgCallback(self, msg):
        self.imageMsg = msg

    def update(self, 
               imgColor, 
               estDSPose=None, 
               publishPose=True, 
               publishImages=True):

        (dsPose,
         poseAquired,
         candidates,
         processedImg,
         poseImg) = self.perception.estimatePose(imgColor, estDSPose)

        timeStamp = rospy.Time.now()
        # publish pose if pose has been aquired
        if publishPose and poseAquired:
            # publish transform
            dsTransform = vectorToTransform("lolo_camera_link", 
                                            "docking_station/feature_model_estimated_link", 
                                            dsPose.translationVector, 
                                            dsPose.rotationVector, 
                                            timeStamp=timeStamp)
            self.transformPublisher.publish(tf.msg.tfMessage([dsTransform]))

            # Publish placement of the light sources as a PoseArray (published in the docking_station frame)
            pArray = featurePointsToMsg("docking_station/feature_model_estimated_link", self.perception.featureModel.features, timeStamp=timeStamp)
            self.featurePosesPublisher.publish(pArray)
            
            # publish estimated pose
            self.posePublisher.publish(
                vectorToPose("lolo_camera_link", 
                dsPose.translationVector, 
                dsPose.rotationVector, 
                dsPose.covariance,
                timeStamp=timeStamp)
                )


        if publishImages:
            self.imgProcDrawPublisher.publish(self.bridge.cv2_to_imgmsg(processedImg))
            self.imgProcPublisher.publish(self.bridge.cv2_to_imgmsg(self.perception.featureExtractor.img))
            if dsPose:
                self.imgPosePublisher.publish(self.bridge.cv2_to_imgmsg(poseImg))

        if dsPose:
            # if the light source candidates have been associated, we pusblish the associated candidates
            self.associatedImagePointsPublisher.publish(lightSourcesToMsg(dsPose.associatedLightSources, timeStamp=timeStamp))
        else:
            # otherwise we publish all candidates
            self.associatedImagePointsPublisher.publish(lightSourcesToMsg(candidates, timeStamp=timeStamp))

        return dsPose, poseAquired

    def run(self, publishPose=True, publishImages=True):
        rate = rospy.Rate(30)

        # currently estimated docking station pose
        # send the pose as an argument in update
        # for the feature extraction to consider only a region of interest
        # near the estimated pose
        estDSPose = None
        while not rospy.is_shutdown():
            if self.imageMsg:
                tStart = time.time()
                try:
                    imgColor = self.bridge.imgmsg_to_cv2(self.imageMsg, 'bgr8')
                except CvBridgeError as e:
                    print(e)
                else:
                    self.imageMsg = None
                    (dsPose,
                     poseAquired) = self.update(imgColor, 
                                                estDSPose=estDSPose, 
                                                publishPose=publishPose, 
                                                publishImages=publishImages)

                    if not poseAquired:
                        estDSPose = None
                    else:
                        estDSPose = dsPose

                    tElapsed = time.time() - tStart
                    hz = 1/tElapsed
                    print("Frame rate: {}".format(hz))

            rate.sleep()


if __name__ == '__main__':
    from lolo_perception.feature_model import FeatureModel
    import os
    import rospkg
    import argparse
    rospy.init_node('perception_node')

    #parser = argparse.ArgumentParser(description='Perception node')
    #parser.add_argument('-feature_model_yaml', type=str, default="big_prototype_5.yaml",
    #                    help='')
    
    #args = parser.parse_args()

    featureModelYaml = rospy.get_param("~feature_model_yaml")
    #featureModelYaml = args.feature_model_yaml
    featureModelYamlPath = os.path.join(rospkg.RosPack().get_path("lolo_perception"), "feature_models/{}".format(featureModelYaml))
    featureModel = FeatureModel.fromYaml(featureModelYamlPath)

    perception = PerceptionNode(featureModel)
    perception.run(publishPose=True, publishImages=True)