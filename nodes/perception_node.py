#!/usr/bin/env python
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
#import roslib
import rospy
import tf.msg
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray
from std_msgs.msg import Float32
import time
import numpy as np

from lolo_perception.perception_ros_utils import vectorToPose, vectorToTransform, poseToVector, lightSourcesToMsg, featurePointsToMsg
from lolo_perception.perception import Perception

class PerceptionNode:
    def __init__(self, featureModel, hz, cvShow=False, hatsMode="valley"):
        self.cameraTopic = "lolo_camera"
        self.cameraInfoSub = rospy.Subscriber("lolo_camera/camera_info", CameraInfo, self._getCameraCallback)
        self.camera = None
        while not rospy.is_shutdown() and self.camera is None:
            print("Waiting for camera info to be published")
            rospy.sleep(1)

        self.hz = hz
        self.cvShow = cvShow

        self.perception = Perception(self.camera, featureModel, hatsMode=hatsMode)

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

        # sed in perception.py to update the estimated pose (estDSPose) for better prediction of the ROI
        self._cameraPoseMsg = None
        self.cameraPoseSub = rospy.Subscriber("lolo/camera/pose", PoseWithCovarianceStamped, self._cameraPoseSub)

        self.hzs = []

    def _getCameraCallback(self, msg):
        """
        Use either K and D or just P
        https://answers.ros.org/question/119506/what-does-projection-matrix-provided-by-the-calibration-represent/
        https://github.com/dimatura/ros_vimdoc/blob/master/doc/ros-camera-info.txt
        """
        from lolo_perception.camera_model import Camera
        # Using only P (D=0), we should subscribe to the rectified image topic
        projectionMatrix = np.array(msg.P, dtype=np.float32).reshape((3,4))[:, :3]
        #scale = 1.36
        #projectionMatrix[0,0] *= scale
        #projectionMatrix[1,1] *= scale
        camera = Camera(cameraMatrix=projectionMatrix, 
                        distCoeffs=np.zeros((1,4), dtype=np.float32),
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

    def _cameraPoseSub(self, msg):
        self._cameraPoseMsg = msg

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
        

        start = time.time()

        (dsPose,
         poseAquired,
         candidates,
         processedImg,
         poseImg) = self.perception.estimatePose(imgColor, 
                                                 estDSPose, 
                                                 estCameraPoseVector=cameraPoseVector)

        if dsPose and dsPose.covariance is None:
            dsPose.calcCovariance()

        elapsed = time.time() - start
        virtualHZ = 1./elapsed
        hz = min(self.hz, virtualHZ)

        #if dsPose:
        self.hzs.append(virtualHZ)
    
        print("Average FPS:", sum(self.hzs)/float(len(self.hzs)))

        cv.putText(poseImg, 
                   "FPS {}".format(round(hz, 1)), 
                   (int(poseImg.shape[1]*4/5), 25), 
                   cv.FONT_HERSHEY_SIMPLEX, 
                   0.7, 
                   color=(0,255,0), 
                   thickness=2, 
                   lineType=cv.LINE_AA)

        cv.putText(poseImg, 
                   "Virtual FPS {}".format(round(virtualHZ, 1)), 
                   (int(poseImg.shape[1]*4/5), 45), 
                   cv.FONT_HERSHEY_SIMPLEX, 
                   0.7, 
                   color=(0,255,0), 
                   thickness=2, 
                   lineType=cv.LINE_AA)

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
            # publish mahalanobis distance
            if not dsPose.mahaDist and estDSPose:
                dsPose.calcMahalanobisDist(estDSPose)
                self.mahalanobisDistPub.publish(Float32(dsPose.mahaDist))

            if publishCamPose:
                print("!!!Publishing cam pose with no covariance!!!")
                self.camPosePublisher.publish(
                    vectorToPose("docking_station_link", 
                    dsPose.camTranslationVector, 
                    dsPose.camRotationVector, 
                    np.eye(6)*0.00001, 
                    #dsPose.calcCamPoseCovariance(),
                    timeStamp=timeStamp)
                    )

        if publishImages:
            self.imgProcDrawPublisher.publish(self.bridge.cv2_to_imgmsg(processedImg))
            self.imgProcPublisher.publish(self.bridge.cv2_to_imgmsg(self.perception.featureExtractor.img))
            #if dsPose:
            self.imgPosePublisher.publish(self.bridge.cv2_to_imgmsg(poseImg))

        if dsPose:
            # if the light source candidates have been associated, we pusblish the associated candidates
            self.associatedImagePointsPublisher.publish(lightSourcesToMsg(dsPose.associatedLightSources, timeStamp=timeStamp))
        else:
            # otherwise we publish all candidates
            self.associatedImagePointsPublisher.publish(lightSourcesToMsg(candidates, timeStamp=timeStamp))

        return dsPose, poseAquired, candidates

    def run(self, poseFeedback=True, publishPose=True, publishCamPose=False, publishImages=True):
        rate = rospy.Rate(self.hz)
        # currently estimated docking station pose
        # send the pose as an argument in update
        # for the feature extraction to consider only a region of interest
        # near the estimated pose
        estDSPose = None

        while not rospy.is_shutdown():
            
            if self.imageMsg:
                try:
                    imgColor = self.bridge.imgmsg_to_cv2(self.imageMsg, 'bgr8')
                except CvBridgeError as e:
                    print(e)
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
                show = False
                if self.perception.poseImg is not None:
                    show = True

                    img = self.perception.poseImg
                    #if img.shape[0] > 720:
                    img = cv.resize(img, (640,360))
                    cv.imshow("pose image", img)

                if self.perception.processedImg is not None:
                    show = True
                    
                    img = self.perception.processedImg
                    if img.shape[0] > 720:
                        img = cv.resize(img, (1280,720))
                    cv.imshow("processed image", img)

                if show:
                    cv.waitKey(1)

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
    hz = rospy.get_param("~hz")
    cvShow = rospy.get_param("~cv_show")
    publishCamPose = rospy.get_param("~publish_cam_pose")
    hatsMode = rospy.get_param("~hats_mode")
    poseFeedBack = rospy.get_param("~pose_feedback")
    #featureModelYaml = args.feature_model_yaml
    featureModelYamlPath = os.path.join(rospkg.RosPack().get_path("lolo_perception"), "feature_models/{}".format(featureModelYaml))
    featureModel = FeatureModel.fromYaml(featureModelYamlPath)

    perception = PerceptionNode(featureModel, hz, cvShow=cvShow, hatsMode=hatsMode)
    perception.run(poseFeedback=poseFeedBack, publishPose=True, publishCamPose=publishCamPose, publishImages=True)