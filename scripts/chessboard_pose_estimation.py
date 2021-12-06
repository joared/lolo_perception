#!/usr/bin/env python
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
import roslib
import rospy
import rospkg
from sensor_msgs.msg import Image
import numpy as np

from perception_utils import plotPoseInfo

def callback(msg):
    global image_msg
    image_msg = msg
    return

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def readCameraMatrix(path):
    import yaml
    import os
    path = os.path.join(rospkg.RosPack().get_path("perception"), path)
    print(path)
    with open(path, "r") as file:
        d = yaml.load(file)

    print(d)

    P = np.array(d["projection_matrix"]["data"], 
                   dtype=np.float32).reshape((3,4))[:, :3]

    K = np.array(d["camera_matrix"]["data"], 
                   dtype=np.float32).reshape((3, 3))

    D = np.array(d["distortion_coefficients"]["data"], 
                   dtype=np.float32)

    return K, D
    #return P, np.zeros((4,1), dtype=np.float32)

def loop():
    """
    https://docs.opencv.org/4.x/d7/d53/tutorial_py_pose.html
    """
    global image_msg
    
    squareSize=0.04

    columns = 8
    rows = 6

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:columns,0:rows].T.reshape(-1,2)
    axis = np.float32([[1,0,0], [0,1,0], [0,0,-1]]).reshape(-1,3)

    print(objp)

    mtx, dist = readCameraMatrix("camera_calibration_data/usb_camera_720p_8.yaml")

    usePixelSize = True
    #objp *= squareSize
    #axis *= squareSize
    if usePixelSize:
        pw = 2.796875e-6
        ph = 2.8055555555e-6
        mtx[0, :] *= pw
        mtx[1, :] *= ph
        #mtx[2, :] *= squareSize
        objp *= squareSize
        axis *= squareSize

    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        if image_msg:
            try:
                img = bridge.imgmsg_to_cv2(image_msg, 'bgr8')
            except CvBridgeError as e:
                print(e)
            else:
                print("here!")
                gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
                ret, corners = cv.findChessboardCorners(gray, (columns,rows),None)
                if ret == True:
                    corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                    # Find the rotation and translation vectors.
                    featurePoints = corners2.copy()
                    if usePixelSize:
                        featurePoints = featurePoints.reshape(48, 2)
                        featurePoints[:, 0] *= pw
                        featurePoints[:, 1] *= ph

                    ret,rvecs, tvecs = cv.solvePnP(objp,
                                                   featurePoints, 
                                                   mtx, 
                                                   dist)
                    # project 3D points to image plane
                    imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
                    if usePixelSize:
                        print(imgpts.shape)
                        imgpts = imgpts.reshape(3, 2)

                        print(imgpts.shape)
                        imgpts[:, 0] /= pw
                        imgpts[:, 1] /= ph
                        imgpts.reshape(3, 1, 2)

                    img = draw(img, corners2, imgpts)

                    if False and not usePixelSize:
                        tvecs *= squareSize
                    print(tvecs.shape)
                    plotPoseInfo(img, tvecs[:, 0], rvecs[:, 0])
                    cv.imshow('img',img)
                    k = cv.waitKey(1) & 0xFF
                        

        rate.sleep()
    cv.destroyAllWindows()

if __name__ == "__main__":
    rospy.init_node('chessboard')

    image_msg = None
    bridge = CvBridge()
    sub_pose = rospy.Subscriber('lolo_camera/image_raw', Image, callback)

    loop()