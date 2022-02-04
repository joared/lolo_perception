#!/usr/bin/env python
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
import roslib
import rospy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, PoseArray
import os.path
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
import itertools

from lolo_perception.feature_extraction import contourCentroid, LightSourceTracker, LightSource, featureAssociation, regionOfInterest, ThresholdFeatureExtractorWithInitialization, AdaptiveThreshold2, AdaptiveThresholdPeak
from lolo_perception.pose_estimation import DSPoseEstimator
from lolo_perception.reprojection_utils import calcPoseReprojectionRMSEThreshold
from lolo_perception.perception_utils import projectPoints, reprojectionError, plotPoseImageInfo, plotPosePoints, plotAxis, plotPoints, plotPoseInfo, plotCrosshair, PoseAndImageUncertaintyEstimator
from lolo_perception.perception_ros_utils import vectorToPose, lightSourcesToMsg
from lolo_perception.camera_model import Camera
from scipy.spatial.transform import Rotation as R, rotation


class Perception:
    def __init__(self, camera, featureModel):
        self.camera = camera
        rospy.Subscriber("lolo_camera/camera_info", CameraInfo, self._getCameraCallback)
        self.camera = None
        while not rospy.is_shutdown() and self.camera is None:
            print("Waiting for camera info to be published")
            rospy.sleep(1)

        self.featureModel = featureModel
        self.uncertaintyEst = PoseAndImageUncertaintyEstimator(len(featureModel.features), 
                                                               nSamples=1)

        self.featureExtractor = ThresholdFeatureExtractorWithInitialization(featureModel=self.featureModel, 
                                                                            camera=self.camera, 
                                                                            p=0.03,
                                                                            erosionKernelSize=5, 
                                                                            maxIter=4, 
                                                                            useKernel=False,
                                                                            nInitImages=10)
        
        self.hatsFeatureExtractor = AdaptiveThreshold2(len(self.featureModel.features), 
                                                       marginPercentage=0.01, 
                                                       minArea=10, 
                                                       minRatio=0.3,
                                                       thresholdType=cv.THRESH_BINARY)
        
        self.peakFeatureExtractor = AdaptiveThresholdPeak(len(self.featureModel.features), kernelSize=11, p=0.95)
        
        self.poseEstimator = DSPoseEstimator(self.camera, 
                                             self.featureModel,
                                             ignoreRoll=False, 
                                             ignorePitch=False, 
                                             flag=cv.SOLVEPNP_ITERATIVE)
 
        self.imageMsg = None
        self.bridge = CvBridge()
        self.imgSubsciber = rospy.Subscriber('lolo_camera/image_rect_color', Image, self._imgCallback)
        #self.imgSubsciber = rospy.Subscriber('lolo_camera/image_raw', Image, self.imgCallback)

        self.imgProcPublisher = rospy.Publisher('lolo_camera/image_processed', Image, queue_size=1)
        self.imgProcDrawPublisher = rospy.Publisher('lolo_camera/image_processed_draw', Image, queue_size=1)
        self.imgPosePublisher = rospy.Publisher('lolo_camera/image_processed_pose', Image, queue_size=1)

        self.associatedImagePointsPublisher = rospy.Publisher('lolo_camera/associated_image_points', PoseArray, queue_size=1)

        self.posePublisher = rospy.Publisher('docking_station/pose', PoseWithCovarianceStamped, queue_size=1)
        self.camPosePublisher = rospy.Publisher('lolo_camera/pose', PoseWithCovarianceStamped, queue_size=1)
        self.poseAvgPublisher = rospy.Publisher('docking_station/pose_average', PoseWithCovarianceStamped, queue_size=1)
        self.camPoseAvgPublisher = rospy.Publisher('lolo_camera/pose_average', PoseWithCovarianceStamped, queue_size=1)


        self.lsTrackers = []

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
        _camera = Camera(cameraMatrix=np.array(msg.K, dtype=np.float32).reshape((3,3)), 
                        distCoeffs=np.array(msg.D, dtype=np.float32),
                        resolution=(msg.height, msg.width))
        self.camera = camera

    def _imgCallback(self, msg):
        self.imageMsg = msg

    def _roiMask(self, gray, estDSPose, scaleMargin):
        featurePointsGuess = projectPoints(estDSPose.translationVector, 
                                           estDSPose.rotationVector, 
                                           self.camera, 
                                           self.featureModel.features)

        margin = max([ls.radius for ls in estDSPose.associatedLightSources])
        margin *= scaleMargin
        margin = max(int(10./720*gray.shape[0]), margin)
        roiCnt = regionOfInterest(featurePointsGuess, wMargin=margin, hMargin=margin)
        roiMask = np.zeros(gray.shape, np.int8)
        roiMask = cv.drawContours(roiMask, [roiCnt], -1, (255), -1)
        return roiMask, roiCnt

    def _sortCandidates(self, candidates, maxAdditionalCandidates):
        """
        Sort the candidates so that probably candidates are considered first
        maxAdditionalCandidates is to limit computation and might discard true lights
        """
        # true light are probably lower in the image (high y value)
        #candidates.sort(key=lambda p: p[1], reverse=True)
        # sort by areas
        candidates.sort(key=lambda p: p.area, reverse=True)
        nFeatures = len(self.featureModel.features)
        candidates = candidates[:nFeatures+maxAdditionalCandidates]
        
        return candidates

    def update(self, 
               imgColor, 
               estDSPose=None, 
               publishPose=True, 
               publishImages=False):

        processedImg = imgColor.copy()
        poseImg = imgColor.copy()

        gray = cv.cvtColor(imgColor, cv.COLOR_BGR2GRAY)

        roiCnt = None
        if estDSPose: # if translation exists, rotation should as well
            # create a region of interest
            roiMask, roiCnt = self._roiMask(gray, estDSPose, scaleMargin=2)
            roiImg = cv.bitwise_and(gray, gray, mask=roiMask)
            

            if not self.lsTrackers:
                for ls in estDSPose.associatedLightSources:
                    self.lsTrackers.append(LightSourceTracker(ls.center, radius=ls.radius, maxPatchRadius=50, minPatchRadius=15, p=.97))
            for lsTracker in self.lsTrackers:
                lsTracker.update(gray, drawImg=processedImg)
                if lsTracker.cnt is None:
                    estDSPose = None
                    
            gray = roiImg
        else:
            # initialize with peak extractor
            self.featureExtractor = self.peakFeatureExtractor


        ###################################################################################################
        """if estDSPose:
            maxAdditionalCandidates = 0
        else:
            maxAdditionalCandidates = 6
        res, candidates = self.featureExtractor(gray, maxAdditionalCandidates=maxAdditionalCandidates, drawImg=processedImg)
        candidates = self._sortCandidates(candidates, maxAdditionalCandidates=6)
        """

        ###################################################################################################
        if estDSPose:
            candidates = [LightSource(lsTracker.cnt) for lsTracker in self.lsTrackers]
        else:
            self.lsTrackers = []
            maxAdditionalCandidates = 6 
            res, candidates = self.featureExtractor(gray, maxAdditionalCandidates=maxAdditionalCandidates, drawImg=processedImg)
            candidates = self._sortCandidates(candidates, maxAdditionalCandidates=maxAdditionalCandidates) 

        ###################################################################################################
        associatedLightSources = candidates
        poseAquired = False
        dsPose = None
        if len(candidates) >= len(self.featureModel.features):
            dsPose = self.poseEstimator.findBestPose(candidates, firstValid=True)

            if dsPose:
                if dsPose.rmse < dsPose.rmseMax:
                    associatedLightSources = dsPose.associatedLightSources
                    # Show 2D representation, affects frame rate
                    #calcPoseReprojectionRMSEThreshold(dsPose.translationVector, 
                    #                              dsPose.rotationVector, 
                    #                              self.camera, 
                    #                              self.featureModel, 
                    #                              showImg=True)
                    poseAquired = True

                    for ls in associatedLightSources:
                        # change to hats if the lightsource is large enough
                        if False and ls.area > self.hatsFeatureExtractor.minArea:
                            self.featureExtractor = self.hatsFeatureExtractor


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
                        (20, 240), 
                        cv.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        color=(255,0,0), 
                        thickness=2, 
                        lineType=cv.LINE_AA)
            else:
                print("Pose estimation failed")
                

            if publishPose and poseAquired:
                self.posePublisher.publish(
                    vectorToPose("lolo_camera", 
                    dsPose.translationVector, 
                    dsPose.rotationVector, 
                    dsPose.covariance)
                    )
                self.camPosePublisher.publish(
                    vectorToPose("docking_station", 
                    dsPose.camTranslationVector, 
                    dsPose.camRotationVector, 
                    dsPose.camCovariance)
                    )
            if publishImages:
                plotPoseImageInfo(poseImg,
                                  dsPose,
                                  self.camera,
                                  self.featureModel,
                                  poseAquired,
                                  roiCnt)

                self.imgPosePublisher.publish(self.bridge.cv2_to_imgmsg(poseImg))
                self.imgProcDrawPublisher.publish(self.bridge.cv2_to_imgmsg(processedImg))
                self.imgProcPublisher.publish(self.bridge.cv2_to_imgmsg(self.featureExtractor.img))

        self.associatedImagePointsPublisher.publish(lightSourcesToMsg(associatedLightSources))

        return (dsPose,
                poseAquired)

    def run(self, publishPose=True, publishImages=True):
        avgFrameRate = 0
        i = 0
        rate = rospy.Rate(30)
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
                    i += 1
                    avgFrameRate = hz#(avgFrameRate*(i-1) + hz)/i
                    print("Frame rate: {}".format(avgFrameRate))

            rate.sleep()


if __name__ == '__main__':
    from lolo_perception.camera_model import usbCamera480p, usbCamera720p, contourCamera1080p
    from lolo_perception.feature_model import smallPrototype5, smallPrototypeSquare, smallPrototype9, bigPrototype5, bigPrototype52, idealModel

    rospy.init_node('perception_node')

    featureModel = bigPrototype5
    featureModel.features *= 1
    featureModel.maxRad *= 1
    #featureModel.features[0] = np.array([0.06, 0, 0])

    print(featureModel.features)

    perception = Perception(None, featureModel)
    perception.run()