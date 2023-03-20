#!/usr/bin/env python
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
import rospy
import tf.msg
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray
from std_msgs.msg import Float32
import time
from datetime import datetime
import numpy as np
import lolo_perception.py_logging as logging
from lolo_perception.msg import FeatureModel as FeatureModelMsg

import lolo_perception.definitions as loloDefs
from lolo_perception.utils import Timer
from lolo_perception.ros_utils import vectorToPose, vectorToTransform, poseToVector, lightSourcesToMsg, featurePointsToMsg
from lolo_perception.tracker import Tracker
from lolo_perception.image_processing import scaleImageToWidth
from lolo_perception.plotting_utils import plotFPS
from lolo_perception.camera_model import Camera

class PerceptionNode:
    def __init__(self):

        # Some custom logging, basically a rip off of pythons logging module. 
        # Logs are saved in the logging directory
        if rospy.get_param("~logging"):
            logging.basicConfig(filename=os.path.join(loloDefs.LOGGING_DIR, "{}.log".format(datetime.today())), 
                                level=logging.TRACE,
                                format="[{levelname:^8s}]:[{timestamp}]:[{messageindex:0>4}]:[{file:^20s}]:[{funcname: ^15s}]:[{lineno:^4}]: {message}",
                                printLevel=logging.INFO,
                                printFormat="[{levelname:^8s}]:[{messageindex:0>4}]: {message}",
                                )

        logging.info("------ Perception node setup ------")

        # Get the camera configuration
        logging.info("Waiting for camera info to be published...")
        msg = rospy.wait_for_message("lolo_camera/camera_info", CameraInfo, timeout=None)
        logging.info("Camera info received!")

        # Using only P (D=0), we should subscribe to the rectified image topic
        # Using K and D, we should subscribe to the raw image topic
        # Use either K and D or just P
        # https://answers.ros.org/question/119506/what-does-projection-matrix-provided-by-the-calibration-represent/
        # https://github.com/dimatura/ros_vimdoc/blob/master/doc/ros-camera-info.txt
        
        # Assume that subscribed images are rectified
        projectionMatrix = np.array(msg.P, dtype=np.float32).reshape((3,4))[:, :3]
        camera = Camera(cameraMatrix=projectionMatrix, 
                        distCoeffs=np.zeros((1,4), dtype=np.float32),
                        resolution=(msg.height, msg.width))

        # Get the docking station (feature model) configuration, either from yaml (if given) otherwise from message
        featureModelYaml = rospy.get_param("~feature_model_yaml", None)
        if featureModelYaml:
            # If yaml is provided, read it
            dockingStationYamlPath = os.path.join(os.path.join(loloDefs.FEATURE_MODEL_CONFIG_DIR, featureModelYaml))
            featureModel = FeatureModel.fromYaml(dockingStationYamlPath)
        else:
            # If no yaml is provided, wait for the feature model to be published
            logging.info("Waiting for feature model to be published...")
            msg = rospy.wait_for_message("feature_model", FeatureModelMsg, timeout=None)
            logging.info("Feature model received!")
            featureModel = FeatureModel(msg.name, 
                                        np.array([[p.x, p.y, p.z] for p in msg.features]), 
                                        msg.placementUncertainty, 
                                        msg.detectionTolerance)

        # Initialize the tracker
        configYaml = rospy.get_param("~tracker_yaml")
        trackingConfigPath = os.path.join(rospkg.RosPack().get_path("lolo_perception"), "config/tracking_config/{}".format(configYaml))
        self.tracker = Tracker.create(trackingConfigPath, camera, featureModel)
        
        # Operating frequency
        self.fps = rospy.get_param("~hz")
        self.fpsTimer = Timer("FPS timer", nAveragSamples=20)
        # If True, displayes the tracking images using opencv imshow
        self.cvShow = rospy.get_param("~cv_show")

        # How much information should be plotted in the images (disabled, simple or fancy)
        self.plotting = rospy.get_param("~plotting")

        # Setup subscribers/publishers
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
        self.camPosePublisher = rospy.Publisher('lolo_camera/estimated_pose', PoseWithCovarianceStamped, queue_size=10)
        self.mahalanobisDistPub = rospy.Publisher('docking_station/feature_model/estimated_pose/maha_dist', Float32, queue_size=1)

        # publish transform of estimated pose
        self.transformPublisher = rospy.Publisher("/tf", tf.msg.tfMessage, queue_size=1)

        # publish placement of the light sources as a PoseArray (published in the docking_station frame)
        self.featurePosesPublisher = rospy.Publisher('docking_station/feature_model/estimated_poses', PoseArray, queue_size=1)

        # set in tracker.py to update the estimated pose (estDSPose) for better prediction of the ROI
        self._cameraPoseMsg = None
        self.cameraPoseSub = rospy.Subscriber("lolo/camera/pose", PoseWithCovarianceStamped, self._cameraPoseSub)


    def _imgCallback(self, msg):
        self.imageMsg = msg


    def _cameraPoseSub(self, msg):
        self._cameraPoseMsg = msg


    def _publishPose(self, dsPose, timeStamp):
        # publish transform
        dsTransform = vectorToTransform("lolo_camera_link", 
                                        "docking_station/feature_model_estimated_link", 
                                        dsPose.translationVector, 
                                        dsPose.rotationVector, 
                                        timeStamp=timeStamp)
        self.transformPublisher.publish(tf.msg.tfMessage([dsTransform]))

        # Publish placement of the light sources as a PoseArray (published in the docking_station frame)
        pArray = featurePointsToMsg("docking_station/feature_model_estimated_link", self.tracker.featureModel.features, timeStamp=timeStamp)
        self.featurePosesPublisher.publish(pArray)
        
        # publish estimated pose
        self.posePublisher.publish(
            vectorToPose("lolo_camera_link", 
            dsPose.translationVector, 
            dsPose.rotationVector, 
            dsPose.covariance,
            timeStamp=timeStamp)
            )
        # publish mahalanobis distance
        if dsPose.mahaDist is not None:
            self.mahalanobisDistPub.publish(Float32(dsPose.mahaDist))


    def _publishCamPose(self, dsPose, timeStamp):
        if dsPose.camCovariance is None:
            logging.debug("Publishing cam pose with no covariance!")
        self.camPosePublisher.publish(
            vectorToPose("docking_station_link", 
            dsPose.camTranslationVector, 
            dsPose.camRotationVector, 
            dsPose.camCovariance if dsPose.camCovariance else np.eye(6)*0.00001, 
            #dsPose.calcCamPoseCovariance(),
            timeStamp=timeStamp)
            )
            

    def _publishImages(self, poseImg, processedImg, detectorImg):
        if poseImg is not None: self.imgPosePublisher.publish(self.bridge.cv2_to_imgmsg(poseImg))
        if processedImg is not None: self.imgProcDrawPublisher.publish(self.bridge.cv2_to_imgmsg(processedImg))
        if detectorImg is not None: self.imgProcPublisher.publish(self.bridge.cv2_to_imgmsg(detectorImg))


    def update(self, 
               imgColor, 
               estDSPose=None, 
               publishPose=True, 
               publishCamPose=False,
               publishImages=True):

        # TODO: move this to run()?
        cameraPoseVector = None
        if self._cameraPoseMsg:
            t, r = poseToVector(self._cameraPoseMsg)
            #t *= 0 # we disregard translation, ds has to be estimated 
            cameraPoseVector = np.array(list(t) + list(r))
            self._cameraPoseMsg = None
        
        self.fpsTimer.start()

        (dsPose,
         poseAquired,
         candidates,
         processedImg,
         poseImg,
         detectorImg) = self.tracker.estimatePose(imgColor, 
                                                  estDSPose, 
                                                  estCameraPoseVector=cameraPoseVector,
                                                  plotting=self.plotting)

        self.fpsTimer.stop()
        fpsVirtual = 1./self.fpsTimer.elapsed()
        fps = min(self.fps, fpsVirtual)
        plotFPS(poseImg, fps, fpsVirtual)

        logging.debug("Average FPS: {} (n = {})".format(1./self.fpsTimer.avg(), self.fpsTimer.nAveragSamples))

        timeStamp = rospy.Time.now()
        # publish pose if pose has been aquired
        if publishPose and poseAquired: 
            self._publishPose(dsPose, timeStamp)
        if publishCamPose and poseAquired: 
            self._publishCamPose(dsPose, timeStamp)
        if publishImages: 
            self._publishImages(poseImg, processedImg, detectorImg)

        if dsPose:
            # if the light source candidates have been associated, we pusblish the associated candidates
            self.associatedImagePointsPublisher.publish(lightSourcesToMsg(dsPose.associatedLightSources, timeStamp=timeStamp))
        else:
            # otherwise we publish all candidates
            self.associatedImagePointsPublisher.publish(lightSourcesToMsg(candidates, timeStamp=timeStamp))

        return dsPose, poseAquired, candidates


    def run(self, poseFeedback=True, publishPose=True, publishCamPose=False, publishImages=True):
        rate = rospy.Rate(self.fps)
        # currently estimated docking station pose
        # send the pose as an argument in update
        # for the feature extraction to consider only a region of interest
        # near the estimated pose
        estDSPose = None

        logging.info("------ Perception node started ------")

        while not rospy.is_shutdown():
            
            if self.imageMsg:
                try:
                    imgColor = self.bridge.imgmsg_to_cv2(self.imageMsg, 'bgr8')
                except CvBridgeError as e:
                    logging.error(e)
                else:
                    if not poseFeedback:
                        estDSPose = None

                    self.imageMsg = None
                    (estDSPose,
                     poseAquired,
                     candidates) = self.update(imgColor, 
                                                estDSPose=estDSPose, 
                                                publishPose=publishPose, 
                                                publishCamPose=publishCamPose,
                                                publishImages=publishImages)

            if self.cvShow:
                self.showImages(self.tracker.poseImg,
                                self.tracker.processedImg,
                                self.tracker.lightSourceDetector.img,
                                width=640)
               
            rate.sleep()


    def showImages(self, poseImg, processedImg, detectorImg, width=None):
        show = False
        
        for img, imgName in zip((poseImg, processedImg, detectorImg), ("Tracking", "Light source detection", "Image processing")):
            if img is not None:
                show = True
                if width:
                    img = scaleImageToWidth(img, width)
                cv.imshow(imgName, img)
        
        if show:
            cv.waitKey(1)


if __name__ == '__main__':
    from lolo_perception.feature_model import FeatureModel
    import os
    import rospkg
    
    rospy.init_node('perception_node')

    publishCamPose = rospy.get_param("~publish_cam_pose")
    poseFeedBack = rospy.get_param("~pose_feedback")

    perception = PerceptionNode()
    
    try: 
        perception.run(poseFeedback=poseFeedBack, publishPose=True, publishCamPose=publishCamPose, publishImages=True)
    except rospy.ROSInterruptException:
        pass