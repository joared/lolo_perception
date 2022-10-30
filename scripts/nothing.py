#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import cv2 as cv

from scipy.spatial.transform import Rotation as R

from lolo_perception.feature_model import FeatureModel

def calcPoseDiffVec(pose1, pose2):
    translDiff = pose1[:3] - pose2[:3]

    ax1, ay1, az1 = pose1[3:]
    r1Vec = R.from_euler("YXZ", (ay1, ax1, az1)).as_rotvec()

    ax2, ay2, az2 = pose2[3:]
    r2Vec = R.from_euler("YXZ", (ay2, ax2, az2)).as_rotvec()

    rDiff = r1Vec-r2Vec

    ayDiff, axDiff, azDiff = R.from_rotvec(rDiff).as_euler("YXZ")

    poseDiff = np.array(list(translDiff) + [axDiff, ayDiff, azDiff])
    return poseDiff

def calcPoseDiff(pose1, pose2):
    """
    pose - [x,y,z,ax,ay,az] where ax,ay,az is with euler order YXZ
    """
    translDiff = pose1[:3] - pose2[:3]

    ax1, ay1, az1 = pose1[3:]
    r1 = R.from_euler("YXZ", (ay1, ax1, az1))

    ax2, ay2, az2 = pose2[3:]
    r2 = R.from_euler("YXZ", (ay2, ax2, az2))
 
    #e = r1*r2.inv() # error in world (not this one, because it is fixed frame rotation)
    #e = r1.inv()*r2 # 2 wrt to 1 (should not be this one)
    e = r2.inv()*r1 # 1 wrt to 2
    print("Err rotvec: ", e.as_rotvec())
    print(logStuff(r1.as_dcm(), r2.as_dcm()))
    ayDiff, axDiff, azDiff = e.as_euler("YXZ")

    poseDiff = np.array(list(translDiff) + [axDiff, ayDiff, azDiff])

    return poseDiff

def skew(m):
    return np.array([[   0, -m[2],  m[1]], 
                    [ m[2],    0, -m[0]], 
                     [-m[1], m[0],     0]])

def exponentialMap(rotVec):
    K = skew(rotVec)
    #rotMat = np.eye(3) + np.sin(theta)*K + (1- np.cos(theta))*np.linalg.matrix_power(K, 2)
    rotMat = expm(K)
    return rotMat

def logMap(rotMat):
    thetaTimesK = logm(rotMat)
    rotVec = np.array([thetaTimesK[2, 1], thetaTimesK[0, 2], thetaTimesK[1, 0]])
    return rotVec

def _motionExpMap(transl, rotVec):
    theta = np.linalg.norm(rotVec)
    K = skew(rotVec)
    mat = np.zeros((4,4), dtype=np.float32)
    mat[0:3, 0:3] = K
    mat[:3, 3] = transl
    T = expm(mat)
    return T

def motionExpMap(transl, rotVec):
    theta = np.linalg.norm(rotVec)
    mat = np.zeros((4,4), dtype=np.float32)
    mat[0:3, 0:3] = exponentialMap(rotVec) # rotation matrix

    K = skew(rotVec/theta)
    if theta == 0:
        K = np.zeros((3, 3))

    t = theta*np.eye(3) + (1- np.cos(theta))*K + (theta - np.sin(theta))*np.linalg.matrix_power(K, 2)
    print(t)
    t = np.matmul(t, transl)
    mat[:3, 3] = t
    print(t)
    T = mat
    return T

def logStuff(rotMat1, rotMat2):
    return logMap(np.matmul(rotMat2.transpose(), rotMat1))

if __name__ == "__main__":
    import rospkg
    import os
    from scipy.linalg import expm, logm
    import rospy

    p1 = np.array([1., 0, 1, -0.1, 0.3, -0.2])
    p2 = np.array([1., 0, 0, 0.2, 0.1, 0.1])

    featureModelYaml = "big_prototype_5.yaml"
    featureModelYamlPath = os.path.join(rospkg.RosPack().get_path("lolo_perception"), "feature_models/{}".format(featureModelYaml))
    featureModel = FeatureModel.fromYaml(featureModelYamlPath)
    r = R.from_rotvec([0., np.pi, 0])
    

    #cv.imshow("img", gray)
    #cv.waitKey(0)

    #print(logMap(mat))
    #print(mat)
    #print(exponentialMap(rotVec))
    #diff = calcPoseDiff(p1, p2)
    #print(diff)

    def constraint(r, x, z):
        return z > r and r <= x+z and r <= z-x

    gridSizeZ = 1000
    gridSizeX = 500
    img = np.zeros((gridSizeZ, gridSizeX, 3), dtype=np.uint8)

    r = 100
    for z in np.arange(0, gridSizeZ, 20):
        for x in np.arange(-gridSizeX/2, gridSizeX/2, 20):
            
            #valid = constraint(r, x, z) # THis is assuming zero relative orientation
            valid = True
            if not valid:
                cv.circle(img, (x+gridSizeX/2, z), 1, (0,0,255), -1)
            else:
                cv.circle(img, (x+gridSizeX/2, z), 1, (0,255,0), -1)
            
                validOrientations = []
                for beta in np.arange(-np.pi/2, np.pi/2, 0.01):

                    validLeft = r < np.cos(beta)*(x+z) + np.sin(beta)*(z-x) # constraint on angle for left light
                    validRight = r < np.cos(beta)*(z-x) - np.sin(beta)*(x+z) # constraint on angle for right light

                    valid = validLeft and validRight

                    if valid:
                        validOrientations.append(beta)

                if validOrientations:
                    maxBeta = np.rad2deg(max(validOrientations))
                    minBeta = np.rad2deg(min(validOrientations))

                    maxRelativeAngle = 10
                    color = (0,255,0)
                    if maxBeta < maxRelativeAngle or minBeta > -maxRelativeAngle:
                        color = (0,0,255)
                    cv.ellipse(img, (x+gridSizeX/2, z), (10, 10), -90, minBeta, maxBeta, color=color, thickness=-1)

    #img = cv.flip(img, 0)
    #img = cv.flip(img, 1)
    cv.imshow("img", cv.transpose(img))
    cv.waitKey(0)