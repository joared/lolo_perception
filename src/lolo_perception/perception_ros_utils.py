
import numpy as np
import rospy
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray, Pose, Point, Quaternion, TransformStamped
from sensor_msgs.msg import CameraInfo
from lolo_perception.msg import FeatureModel as FeatureModelMsg
from tf.transformations import quaternion_from_matrix
import yaml
import os
import rospkg

def vectorToPose(frameID, translationVector, rotationVector, covariance=None, timeStamp=None):
    rotMat = R.from_rotvec(rotationVector).as_dcm()
    rotMatHom = np.hstack((rotMat, np.zeros((3, 1))))
    rotMatHom = np.vstack((rotMatHom, np.array([0, 0, 0, 1])))
    q = quaternion_from_matrix(rotMatHom)

    p = PoseWithCovarianceStamped()
    p.header.frame_id = frameID
    p.header.stamp = timeStamp if timeStamp else rospy.Time.now()
    (p.pose.pose.position.x, 
     p.pose.pose.position.y, 
     p.pose.pose.position.z) = (translationVector[0], 
                                translationVector[1], 
                                translationVector[2])
    p.pose.pose.orientation = Quaternion(*q)

    if covariance is not None:
        p.pose.covariance = list(np.ravel(covariance))

    return p

def vectorToPoseStamped(frameID, translationVector, rotationVector, timeStamp=None):
    rotMat = R.from_rotvec(rotationVector).as_dcm()
    rotMatHom = np.hstack((rotMat, np.zeros((3, 1))))
    rotMatHom = np.vstack((rotMatHom, np.array([0, 0, 0, 1])))
    q = quaternion_from_matrix(rotMatHom)

    p = PoseStamped()
    p.header.frame_id = frameID
    p.header.stamp = timeStamp if timeStamp else rospy.Time.now()
    (p.pose.position.x, 
     p.pose.position.y, 
     p.pose.position.z) = (translationVector[0], 
                                translationVector[1], 
                                translationVector[2])
    p.pose.orientation = Quaternion(*q)

    return p

def vectorToTransform(frameID, childFrameID, translationVector, rotationVector, timeStamp=None):
    t = TransformStamped()
    t.header.stamp = timeStamp if timeStamp else rospy.Time.now()
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

def vectorQuatToTransform(frameID, childFrameID, translationVector, quaternion, timeStamp=None):
    t = TransformStamped()
    t.header.stamp = timeStamp if timeStamp else rospy.Time.now()
    t.header.frame_id = frameID
    t.child_frame_id = childFrameID
    t.transform.translation.x = translationVector[0]
    t.transform.translation.y = translationVector[1]
    t.transform.translation.z = translationVector[2]

    t.transform.rotation = Quaternion(*quaternion)
    return t

def poseToVector(pose):
    """
    PoseWithCovarianceStamped -> vector
    """
    translationVector = np.array([pose.pose.pose.position.x, 
                                  pose.pose.pose.position.y, 
                                  pose.pose.pose.position.z])

    quat = [pose.pose.pose.orientation.x,
            pose.pose.pose.orientation.y,
            pose.pose.pose.orientation.z,
            pose.pose.pose.orientation.w]

    rotationVector = R.from_quat(quat).as_rotvec()

    return translationVector, rotationVector

def lightSourcesToMsg(lightSources, timeStamp=None):
    poseArray = PoseArray()
    poseArray.header.stamp = timeStamp if timeStamp else rospy.Time.now()
    poseArray.header.frame_id = "lolo_camera/image_plane"
    for ls in lightSources:
        pose = Pose()
        pose.position.x = ls.center[0]
        pose.position.y = ls.center[1]
        pose.orientation.w = 1
        poseArray.poses.append(pose)

    return poseArray

def featurePointsToMsg(frameId, featurePoints, timeStamp=None):
    pArray = PoseArray()
    #pArrayNoised = PoseArray()
    pArray.header.frame_id = frameId
    pArray.header.stamp = timeStamp if timeStamp else rospy.Time.now()
    for p in featurePoints:
        pose = Pose()
        pose.position.x = p[0]
        pose.position.y = p[1]
        pose.position.z = p[2]
        pose.orientation.w = 1
        pArray.poses.append(pose)
    return pArray

def msgToImagePoints(msg):
    imgPoints = []
    for pose in msg.poses:
        imgPoints.append((int(round(pose.position.x)), 
                          int(round(pose.position.y))))

    return imgPoints

def yamlToFeatureModelMsg(featureModelYaml):
    from lolo_perception.feature_model import FeatureModel
    fm = FeatureModel.fromYaml(featureModelYaml)
    msg = FeatureModelMsg()
    msg.name = fm.name
    msg.detectionTolerance = fm.detectionTolerance
    msg.placementUncertainty = fm.placementUncertainty
    for f in fm.features:
        x, y, z = f
        p = Point(x=x, y=y, z=z)
        msg.features.append(p)

    return msg

def readCameraYaml(cameraYamlPath):
    with open(cameraYamlPath, "r") as file:
        calibData = yaml.load(file)
    
    msg = CameraInfo()
    msg.width = calibData["image_width"]
    msg.height = calibData["image_height"]
    msg.K = calibData["camera_matrix"]["data"]
    msg.D = calibData["distortion_coefficients"]["data"]
    msg.R = calibData["rectification_matrix"]["data"]
    msg.P = calibData["projection_matrix"]["data"]
    try:
        msg.distortion_model = calibData["distortion_model"]
    except KeyError:
        msg.distortion_model = calibData["camera_model"]

    return msg