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
from lolo_perception.camera_model import Camera

class Perception:
    def __init__(self, featureModel):
        self.cameraInfoSub = rospy.Subscriber("lolo_camera/camera_info", CameraInfo, self._getCameraCallback)
        self.camera = None
        while not rospy.is_shutdown() and self.camera is None:
            print("Waiting for camera info to be published")
            rospy.sleep(1)

        self.featureModel = featureModel
        
        # Use HATS when light sources are "large"
        # This feature extractor sorts candidates based on area
        self.hatsFeatureExtractor = AdaptiveThreshold2(len(self.featureModel.features), 
                                                       marginPercentage=0.01, 
                                                       minArea=10, 
                                                       minRatio=0.2,
                                                       thresholdType=cv.THRESH_BINARY)
        
        # Use local peak finding to initialize and when light sources are small
        # This feature extractor sorts candidates based on intensity and then area
        self.peakFeatureExtractor = AdaptiveThresholdPeak(len(self.featureModel.features), 
                                                          kernelSize=11, 
                                                          p=0.97)
        
        self.featureExtractor = self.peakFeatureExtractor

        # max additional light source candidates that will be considered by 
        # the feature extractor.
        # 6 gives a minimum frame rate of about 0.3 when all 
        # permutations of candidates are iterated over
        self.maxAdditionalCandidates = 6

        # margin of the region of interest when pose has been aquired
        self.roiMargin = 50

        # Pose estimator that calculates poses from detected light sources
        self.poseEstimator = DSPoseEstimator(self.camera, 
                                             self.featureModel,
                                             ignoreRoll=False, 
                                             ignorePitch=False, 
                                             flag=cv.SOLVEPNP_ITERATIVE)

        # valid orientation range [yawMinMax, pitchMinMax, rollMinMax]. Currently not used to disregard
        # invalid poses, but the axis and region of interest will be shown in red when a pose has 
        # an orientation within the valid range 
        self.validOrientationRange = [np.radians(35), np.radians(30), np.radians(15)]
 
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
        self.posePublisher = rospy.Publisher('docking_station/pose', PoseWithCovarianceStamped, queue_size=1)

        # publish transform of estimated pose
        self.transformPublisher = rospy.Publisher("/tf", tf.msg.tfMessage, queue_size=1)

        # publish placement of the light sources as a PoseArray (published in the docking_station frame)
        self.featurePosesPublisher = rospy.Publisher('lights/poses', PoseArray, queue_size=1)

    def _getCameraCallback(self, msg):
        """
        Use either K and D or just P
        https://answers.ros.org/question/119506/what-does-projection-matrix-provided-by-the-calibration-represent/
        https://github.com/dimatura/ros_vimdoc/blob/master/doc/ros-camera-info.txt
        """
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

        # information about contours extracted from the feature extractor is plotted in this image
        processedImg = imgColor.copy()
        
        # pose information and ROI is plotted in this image
        poseImg = imgColor.copy()

        # gray image to be processed
        gray = cv.cvtColor(imgColor, cv.COLOR_BGR2GRAY)

        # ROI contour
        roiCnt = None

        if not estDSPose:
            # if we are not given an estimated pose, we initialize
            self.featureExtractor = self.peakFeatureExtractor
        else:
            if all([ls.area > self.hatsFeatureExtractor.minArea for ls in estDSPose.associatedLightSources]):
                # change to hats
                self.featureExtractor = self.hatsFeatureExtractor
        
        # extract light source candidates from image
        _, candidates, roiCnt = self.featureExtractor(gray, 
                                                      maxAdditionalCandidates=self.maxAdditionalCandidates, 
                                                      estDSPose=estDSPose,
                                                      roiMargin=self.roiMargin, 
                                                      drawImg=processedImg)

        # return if pose has been aquired
        poseAquired = False
        # estimated pose
        dsPose = None
        # associated light sources (in the image) for the estimated pose
        associatedLightSources = None

        timeStamp = rospy.Time.now()
        if len(candidates) >= len(self.featureModel.features):

            lightCandidatePermutations = list(itertools.combinations(candidates, len(self.featureModel.features)))
            associatedPermutations = [featureAssociation(self.featureModel.features, candidates)[0] for candidates in lightCandidatePermutations]
                
            # findBestPose finds poses based on reprojection RMSE
            # the featureModel specifies the placementUncertainty and detectionTolerance
            # which determines the maximum allowed reprojection RMSE
            if estDSPose and self.featureExtractor == self.hatsFeatureExtractor:
                # if we use HATS, presumably we are close and we want to find the best pose
                dsPose = self.poseEstimator.findBestPose(associatedPermutations, firstValid=False)
            else:
                # if we use local peak, presumably we are far away, 
                # and the first valid pose is good enough in most cases 
                dsPose = self.poseEstimator.findBestPose(associatedPermutations, firstValid=True)

            if dsPose:
                if dsPose.rmse < dsPose.rmseMax:
                    associatedLightSources = dsPose.associatedLightSources
                    poseAquired = True

                rmseColor = (0,255,0) if poseAquired else (0,0,255)
                cv.putText(poseImg, 
                        "RMSE: {} < {}".format(round(dsPose.rmse, 2), round(dsPose.rmseMax, 2)), 
                        (20, 200), 
                        cv.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        color=rmseColor, 
                        thickness=2, 
                        lineType=cv.LINE_AA)
                cv.putText(poseImg, 
                        "RMSE certainty: {}".format(round(1-dsPose.rmse/dsPose.rmseMax, 2)), 
                        (20, 220), 
                        cv.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        color=rmseColor, 
                        thickness=2, 
                        lineType=cv.LINE_AA)
                cv.putText(poseImg, 
                        "HATS" if self.featureExtractor == self.hatsFeatureExtractor else "Peak", 
                        (poseImg.shape[1]/2, 40), 
                        cv.FONT_HERSHEY_SIMPLEX, 
                        2, 
                        color=(255,0,0), 
                        thickness=2, 
                        lineType=cv.LINE_AA)
            else:
                print("Pose estimation failed")
                
            # publish pose if pose has been aquired
            if publishPose and poseAquired:
                # publish transform
                dsTransform = vectorToTransform("lolo_camera", 
                                                "docking_station", 
                                                dsPose.translationVector, 
                                                dsPose.rotationVector, 
                                                timeStamp=timeStamp)
                self.transformPublisher.publish(tf.msg.tfMessage([dsTransform]))

                # Publish placement of the light sources as a PoseArray (published in the docking_station frame)
                pArray = featurePointsToMsg("docking_station", self.featureModel.features, timeStamp=timeStamp)
                self.featurePosesPublisher.publish(pArray)
                
                # publish estimated pose
                self.posePublisher.publish(
                    vectorToPose("lolo_camera", 
                    dsPose.translationVector, 
                    dsPose.rotationVector, 
                    dsPose.covariance,
                    timeStamp=timeStamp)
                    )


        if publishImages:
            self.imgProcDrawPublisher.publish(self.bridge.cv2_to_imgmsg(processedImg))
            self.imgProcPublisher.publish(self.bridge.cv2_to_imgmsg(self.featureExtractor.img))
            if dsPose:
                # plots pose axis, ROI, light sources etc.  
                plotPoseImageInfo(poseImg,
                                    dsPose,
                                    self.camera,
                                    self.featureModel,
                                    poseAquired,
                                    self.validOrientationRange,
                                    roiCnt)

                self.imgPosePublisher.publish(self.bridge.cv2_to_imgmsg(poseImg))

        if associatedLightSources:
            # if the light source candidates have been associated, we pusblish the associated candidates
            self.associatedImagePointsPublisher.publish(lightSourcesToMsg(associatedLightSources, timeStamp=timeStamp))
        else:
            # otherwise we publish all candidates
            self.associatedImagePointsPublisher.publish(lightSourcesToMsg(candidates, timeStamp=timeStamp))

        return (dsPose,
                poseAquired)

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
    from lolo_perception.feature_model import smallPrototype5, bigPrototype5

    rospy.init_node('perception_node')

    featureModels = {"small": smallPrototype5,
                     "big": bigPrototype5}

    featureModelStr = rospy.get_param("~feature_model")

    featureModel = featureModels[featureModelStr]

    print(featureModel.features)

    perception = Perception(featureModel)
    perception.run(publishPose=True, publishImages=True)