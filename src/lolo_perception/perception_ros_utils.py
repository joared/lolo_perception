
import numpy as np
import rospy
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray, Pose, Quaternion, TransformStamped
from tf.transformations import quaternion_from_matrix

def vectorToPose(frameID, translationVector, rotationVector, covariance):
    rotMat = R.from_rotvec(rotationVector).as_dcm()

    # convert so that the pose is pointing in the z direction (as the light frame is defined)
    #lightRotation = R.from_euler("XYZ", (-np.pi/2, np.pi/2, 0)).as_dcm().transpose()
    #rotMat = np.matmul(rotMat, lightRotation)

    rotMatHom = np.hstack((rotMat, np.zeros((3, 1))))
    rotMatHom = np.vstack((rotMatHom, np.array([0, 0, 0, 1])))
    q = quaternion_from_matrix(rotMatHom)
    p = PoseWithCovarianceStamped()
    p.header.frame_id = frameID
    p.header.stamp = rospy.Time.now()
    (p.pose.pose.position.x, 
     p.pose.pose.position.y, 
     p.pose.pose.position.z) = (translationVector[0], 
                                translationVector[1], 
                                translationVector[2])
    p.pose.pose.orientation = Quaternion(*q)
    p.pose.covariance = list(np.ravel(covariance))

    return p