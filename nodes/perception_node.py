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

from lolo_perception.feature_extraction import GradientFeatureExtractor, featureAssociation, drawInfo, ThresholdFeatureExtractor
from lolo_perception.pose_estimation import DSPoseEstimator
from lolo_perception.perception_utils import plotPosePoints, plotAxis, plotPoints, plotPoseInfo, plotPointsPixel
from lolo_perception.perception_ros_utils import vectorToPose

from scipy.spatial.transform import Rotation as R


class Perception:
    def __init__(self, camera, featureModel):
        self.camera = camera
        camInfoSub = rospy.Subscriber("lolo_camera/camera_info", CameraInfo, self.getCameraCallback)
        self.camera = None
        while not rospy.is_shutdown() and self.camera is None:
            print("Waiting for camera info to be published")
            rospy.sleep(1)

        self.featureModel = featureModel


        #featureExtractor = ThresholdFeatureExtractor(featureModel=self.featureModel, camera=self.camera, p=0.02, erosionKernelSize=7, maxIter=10, useKernel=False)
        self.featureExtractor = ThresholdFeatureExtractor(featureModel=self.featureModel, camera=self.camera, p=0.03, erosionKernelSize=5, maxIter=4, useKernel=False)
        self.gradFeatureExtractor = GradientFeatureExtractor(featureModel=self.featureModel, camera=self.camera)
        
        self.poseEstimator = DSPoseEstimator(self.camera, 
                                             ignoreRoll=False, 
                                             ignorePitch=False, 
                                             flag=cv.SOLVEPNP_ITERATIVE,
                                             calcCovariance=True)
 
        self.imageMsg = None
        self.bridge = CvBridge()
        self.imgSubsciber = rospy.Subscriber('lolo_camera/image_rect_color', Image, self.imgCallback)
        #self.imgSubsciber = rospy.Subscriber('lolo_camera/image_raw', Image, self.imgCallback)

        self.imgProcPublisher = rospy.Publisher('lolo_camera/image_processed', Image, queue_size=1)
        self.imgPosePublisher = rospy.Publisher('lolo_camera/image_processed_pose', Image, queue_size=1)
        
        self.imgThresholdPublisher = rospy.Publisher('lolo_camera/image_processed_thresholded', Image, queue_size=1)
        self.imgAdaOpenPublisher = rospy.Publisher('lolo_camera/image_processed_adaopen', Image, queue_size=1)

        self.imgGradPublisher = rospy.Publisher('lolo_camera/image_processed_grad', Image, queue_size=1)
        self.imgBinGradPublisher = rospy.Publisher('lolo_camera/image_processed_bin_grad', Image, queue_size=1)
            
        self.posePublisher = rospy.Publisher('docking_station/pose', PoseWithCovarianceStamped, queue_size=1)
        self.poseAvgPublisher = rospy.Publisher('docking_station/pose_average', PoseWithCovarianceStamped, queue_size=1)


    def getCameraCallback(self, msg):
        from lolo_perception.camera_model import Camera
        """
        Use either K and D or just P
        https://answers.ros.org/question/119506/what-does-projection-matrix-provided-by-the-calibration-represent/
        https://github.com/dimatura/ros_vimdoc/blob/master/doc/ros-camera-info.txt
        """
        # Using only P (D=0), we should subscribe to the rectified image topic
        camera = Camera(cameraMatrix=np.array(msg.P, dtype=np.float32).reshape((3,4))[:, :3], 
                        distCoeffs=np.zeros((4,1), dtype=np.float32),
                        resolution=(msg.height, msg.width), 
                        pixelWidth=2.8e-6, 
                        pixelHeight=2.8e-6, 
                        hz=15)
        # Using K and D, we should subscribe to the raw image topic
        _camera = Camera(cameraMatrix=np.array(msg.K, dtype=np.float32).reshape((3,3)), 
                        distCoeffs=np.array(msg.D, dtype=np.float32),
                        resolution=(msg.height, msg.width), 
                        pixelWidth=2.796875e-6, 
                        pixelHeight=2.8055555555e-6, 
                        hz=15)
        self.camera = camera

    def imgCallback(self, msg):
        self.imageMsg = msg

    def update(self, 
               imgColor, 
               estTranslationVector=None, 
               estRotationVector=None, 
               publishPose=True, 
               publishImages=False, 
               plot=False):

        processedImg = imgColor.copy()
        poseImg = imgColor.copy()
        #imgColor = imgColor.copy()

        gray = cv.cvtColor(imgColor, cv.COLOR_BGR2GRAY)
        waterSurfaceMask = np.zeros(gray.shape, dtype=np.uint8)
        surfaceLineRatio = 0#.45 # ratio 0-1, where 0 is the top of the image and 1 is the bottom
        waterSurfaceMask[int(waterSurfaceMask.shape[0]*surfaceLineRatio):, :] = 1

        gray = cv.bitwise_and(gray, gray, mask=waterSurfaceMask)

        #estTranslationVec = np.array([-0.3, 0.3, 1.5])
        #estRotationVec = np.array([0., np.pi, 0.])
        res, associatedPoints = self.featureExtractor(gray, 
                                                      processedImg, 
                                                      estTranslationVector, 
                                                      estRotationVector)

        #resGrad, _ = gradFeatureExtractor(gray, 
        #                                  processedImg, 
        #                                  estTranslationVec, 
        #                                  estRotationVec)

        if len(associatedPoints) == len(self.featureModel.features):
            sigmaX = .12 # std of pixel in x
            sigmaY = .22 # std of pixel in y
            (translationVector, 
                rotationVector, 
                covariance) = self.poseEstimator.update(
                                self.featureModel.features, 
                                associatedPoints, 
                                np.array([[self.camera.pixelWidth*self.camera.pixelWidth*sigmaX*sigmaX, 0], 
                                        [0, self.camera.pixelHeight*self.camera.pixelHeight*sigmaY*sigmaY]]),
                                estTranslationVector,
                                estRotationVector)

            # show lights pose wrt to camera
            rotMat = R.from_rotvec(rotationVector.transpose()).as_dcm()
            transl = translationVector

            # show camera pose wrt to lights
            transl = np.matmul(rotMat.transpose(), -translationVector)
            rotMat = rotMat.transpose()

            # convert to camera coordinates (x-rgiht, y-left, z-front)
            #translation = np.matmul(csRef.rotation, transl)
            #rotation = np.matmul(csRef.rotation, np.array(rotMat))

            translation = transl
            rotation = R.from_dcm(rotMat).as_rotvec()

            ############################
            global uncertaintyEst
            uncertaintyEst.add(self.poseEstimator.translationVector, 
                                self.poseEstimator.rotationVector, 
                                self.camera.metersToUV(associatedPoints))

            poseAvg, imageAvgs = uncertaintyEst.calcAverage()

            poseCov, imageCovs = uncertaintyEst.calcCovariance()
            print("mean image cov:", np.mean(imageCovs, axis=0))

            plotPointsPixel(poseImg, imageAvgs, (255, 0, 255), radius=3)
            pose = vectorToPose("lolo_camera",
                                poseAvg[:3],
                                poseAvg[3:],
                                poseCov)
            self.poseAvgPublisher.publish(pose)
            ############################

            if publishPose:
                pose = vectorToPose("lolo_camera", 
                                    self.poseEstimator.translationVector, 
                                    self.poseEstimator.rotationVector, 
                                    self.poseEstimator.poseCov)
                self.posePublisher.publish(pose)

        if False:
            hist = cv.calcHist([gray], [0], None, [256], [0, 256])
            histStart = 50
            maxH = max(hist[histStart:])
            hist = [v/maxH for v in hist[histStart:]]
            plt.cla()
            plt.xlim(0, 255)
            plt.ylim(0, 1)
            plt.plot(hist)
            plt.pause(0.000001)

        if publishImages or plot:
            plotAxis(poseImg, 
                    self.poseEstimator.translationVector, 
                    self.poseEstimator.rotationVector, 
                    self.camera, 
                    self.featureModel.features, 
                    self.featureModel.maxRad) # scaling for the axis shown
            plotPosePoints(poseImg, 
                        self.poseEstimator.translationVector, 
                        self.poseEstimator.rotationVector, 
                        self.camera, 
                        self.featureModel.features, 
                        color=(0, 0, 255))
            plotPoints(poseImg, self.camera, associatedPoints, (255, 0, 0))
            plotPoseInfo(poseImg, 
                         self.poseEstimator.translationVector, 
                         self.poseEstimator.rotationVector)

        if publishImages:
            self.imgThresholdPublisher.publish(self.bridge.cv2_to_imgmsg(self.featureExtractor.pHold.img))
            self.imgAdaOpenPublisher.publish(self.bridge.cv2_to_imgmsg(self.featureExtractor.adaOpen.img))
            self.imgPosePublisher.publish(self.bridge.cv2_to_imgmsg(poseImg))
            self.imgProcPublisher.publish(self.bridge.cv2_to_imgmsg(processedImg))
            #self.imgGradPublisher.publish(self.bridge.cv2_to_imgmsg(resGrad))

        if plot:
            cv.imshow("threshold", self.featureExtractor.pHold.img)
            cv.imshow("adaptive open", self.featureExtractor.adaOpen.img)
            cv.imshow("pose", poseImg)
            cv.imshow("processed image", processedImg)
            cv.waitKey(0)

        return (self.poseEstimator.translationVector, 
                self.poseEstimator.rotationVector, 
                self.poseEstimator.poseCov)

    def run(self, publishPose=True, publishImages=True, plot=False):
        avgFrameRate = 0
        i = 0
        rate = rospy.Rate(30)
        estTranslationVector = None
        estRotationVector = None
        while not rospy.is_shutdown():
            if self.imageMsg:
                tStart = time.time()
                try:
                    imgColor = self.bridge.imgmsg_to_cv2(self.imageMsg, 'bgr8')
                except CvBridgeError as e:
                    print(e)
                else:
                    

                    estTranslationVector = None # comment to use region of interest
                    estRotationVector = None    # comment to use region of interest

                    (estTranslationVector, 
                     estRotationVector, 
                     covariance) = self.update(
                                    imgColor, 
                                    estTranslationVector=estTranslationVector,
                                    estRotationVector=estRotationVector, 
                                    publishPose=publishPose, 
                                    publishImages=publishImages, 
                                    plot=plot)

                    tElapsed = time.time() - tStart
                    hz = 1/tElapsed
                    i += 1
                    avgFrameRate = (avgFrameRate*(i-1) + hz)/i
                    print("Average frame rate: {}".format(avgFrameRate))

            rate.sleep()


if __name__ == '__main__':
    from lolo_perception.camera_model import usbCamera480p, usbCamera720p, contourCamera1080p
    from lolo_perception.feature_model import smallPrototype5, smallPrototype9, bigPrototype5, bigPrototype52, idealModel
    from lolo_perception.perception_utils import PoseAndImageUncertaintyEstimator

    rospy.init_node('perception_node')

    featureModel = idealModel
    featureModel = smallPrototype5
    #featureModel = bigPrototype5
    featureModel.features *= 1
    featureModel.maxRad *= 1
    print(featureModel.features)
    featureModel.features[0] *= 1

    uncertaintyEst = PoseAndImageUncertaintyEstimator(len(featureModel.features), 
                                                      nSamples=1)

    perception = Perception(None, featureModel)
    perception.run()