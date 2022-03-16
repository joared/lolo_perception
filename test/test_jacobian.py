import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as R

def skew(m):
    return np.array([[   0, -m[2],  m[1]], 
            [ m[2],    0, -m[0]], 
            [-m[1], m[0],     0]])

def imageJacobian(tVec, rVec, cameraMatrix, objectPoint):
    rot = R.from_rotvec(-rVec)

    imgJacobian = np.zeros((2, 6), dtype=np.float32)

    imgPoints = projectPoints(tVec, rVec, cameraMatrix, np.array([objectPoint]))
    imgPoint = imgPoints[0]

    for i in range(6):
        tVecTemp = tVec.copy()
        if i < 3:
            tVecTemp[i] += 1
        else:
            w = np.zeros(3, dtype=np.float32)
            w[i-3] = 1
            w = R.from_rotvec(w)
            rotTemp = R.from_rotvec(rVec)
            rVecTemp = ""
        imgPointsTemp = projectPoints(tVecTemp, rVecTemp, cameraMatrix, np.array([objectPoint]))
        imgPointTemp = imgPointsTemp[0]
        du = (imgPointTemp[0] - imgPoint[0])
        dv = (imgPointTemp[1] - imgPoint[1])
        imgJacobian[:, i] = du, dv

    return imgJacobian

def numericalJacobian(tVec, rVec, cameraMatrix, objectPoints):
    deltaT = 0.00001
    imgJacobian = np.zeros((2, 6), dtype=np.float32)

    imgPoints = projectPoints(tVec, rVec, cameraMatrix, objectPoints)

    for i in range(6):
        tVecTemp = tVec.copy()
        rVecTemp = rVec.copy()
        if i < 3:
            tVecTemp[i] += deltaT
        else:
            rVecTemp[i-3] += deltaT
        imgPointsTemp = projectPoints(tVecTemp, rVecTemp, cameraMatrix, objectPoints)
        du = (imgPointsTemp[:, 0] - imgPoints[:, 0])/deltaT
        dv = (imgPointsTemp[:, 1] - imgPoints[:, 1])/deltaT
        imgJacobian[:, i] = du, dv

    return imgJacobian
    

def projectPoints(tVec, rVec, cameraMatrix, objectPoints):
    rotMat = R.from_rotvec(rVec).as_dcm()
    points3D = np.matmul(rotMat, objectPoints.transpose()).transpose() + tVec
    imgPoints = np.matmul(cameraMatrix, points3D.transpose()).transpose()
    imgPoints[:, :] /= imgPoints[:, [-1]]
    imgPoints = imgPoints[:, :2]
    return imgPoints

if __name__ == "__main__":
    centerPoint = np.array([0., 1., 1.])
    #objectPoints = np.array([centerPoint, [-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]])
    objectPoints = np.array([[0., 0., -1.]])
    tVec = np.array([0., 0., 4.5])
    rVec = np.array([0., 0., 0.])
    cameraMatrix = np.array([[1400., 0., 639.],
                             [0., 1400., 359.],
                             [0., 0., 1]])
    
    imgPoints, jacobian = cv.projectPoints(objectPoints, 
                                           rVec, 
                                           tVec, 
                                           cameraMatrix, 
                                           distCoeffs=np.zeros((1,4), dtype=np.float32))
                                           #distCoeffs=np.array([0.1, 0.002, -0.1, -0.005]))
    #jacobian2 = numericalJacobian(tVec, rVec, cameraMatrix, objectPoints[0])
    #print(jacobian2)

    imgPoints2 = projectPoints(tVec, rVec, cameraMatrix, objectPoints)
    print(imgPoints2)
    imgPoints = np.array([p[0] for p in imgPoints])
    img = np.zeros((720, 1280, 3), dtype=np.uint8)

    for p in imgPoints:
        cv.circle(img, (int(round(p[0])), int(round(p[1]))), 5, (255,0,0), -1)

    jacobian = jacobian.round(5)
    for i in range(jacobian.shape[0]/2):
        print("du: {}".format(list(jacobian[i*2, 3:6]) + list(jacobian[i*2, :3])))
        print("dv: {}".format(list(jacobian[i*2+1, 3:6]) + list(jacobian[i*2+1, :3])))
    cv.imshow("img", img)
    cv.waitKey(0)
    cv.destroyAllWindows()