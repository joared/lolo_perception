import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as R

from lolo_perception.perception_utils import plotAxis, plotVector

from lolo_perception.pose_estimation_utils import objectJacobianLie, objectJacobianOpenCV, objectJacobianOpenCV2, numericalJacobian, rotvecFromMat, matFromRotvec

class CameraDummy:
    def __init__(self, cameraMatrix, distCoeffs):
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs

if __name__ == "__main__":
    centerPoint = np.array([0., 0., -1.])
    objectPoints = np.array([centerPoint, [-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]])*0.5
    #objectPoints = np.array([[-1., -1., 0.], [1., -1., 0.], [1., 1., 0.], [-1., 1., 0.]])
    #objectPoints = np.array([[1., 2., -1.]])
    tVec = np.array([1., -1., 5.])
    rVec = np.array([0.1, np.pi/3, -0.3])

    #tVec = np.array([0., 0., 10.])*0.5
    #rVec = np.array([0, 0, 0.])

    #rVec = np.array([0.1, 0, -0.3])
    #rVec = np.array([0, np.pi, 0])
    f = 500. # J ~f, H~1/f^2
    cameraMatrix = np.array([[f, 0., 639.],
                             [0., f, 359.],
                             [0., 0., 1]])
    

    imgPoints, jacobian = cv.projectPoints(objectPoints, 
                                           rVec, 
                                           tVec, 
                                           cameraMatrix, 
                                           distCoeffs=np.zeros((1,4), dtype=np.float32))

    imgPoints, jacobianApplied = cv.projectPoints(R.from_rotvec(rVec).apply(objectPoints), 
                                                  rVec*0, 
                                                  tVec, 
                                                  cameraMatrix, 
                                                  distCoeffs=np.zeros((1,4), dtype=np.float32))

    #jacobian2 = numericalJacobian(tVec, rVec, cameraMatrix, objectPoints[0])
    #print(jacobian2)

    jacobianLocal = objectJacobianLie(cameraMatrix, tVec, rVec, objectPoints, method="local")
    jacobianGlobal = objectJacobianLie(cameraMatrix, tVec, rVec, objectPoints, method="global")
    #jacobianGlobal = objectJacobianPose(cameraMatrix, tVec, rVec, objectPoints)
    jacobianNumericalCross = numericalJacobian(cameraMatrix, tVec, rVec, objectPoints, "rvec", jacType="cross")
    jacobianNumericalGlobal = numericalJacobian(cameraMatrix, tVec, rVec, objectPoints, "rvec", jacType="global")
    jacobianNumericalLocal = numericalJacobian(cameraMatrix, tVec, rVec, objectPoints, "rvec", jacType="local")
    jacobianNumericalFixedAxis = numericalJacobian(cameraMatrix, tVec, rVec, objectPoints, "xyz", jacType="cross")
    jacobianOpenCV = objectJacobianOpenCV(cameraMatrix, tVec, rVec, objectPoints)
    jacobianOpenCV2Left = objectJacobianOpenCV2(cameraMatrix, tVec, rVec, objectPoints, mode="left")
    jacobianOpenCV2Right = objectJacobianOpenCV2(cameraMatrix, tVec, rVec, objectPoints, mode="right")
    #imgPoints2 = projectPoints(tVec, rVec, cameraMatrix, objectPoints)
    #print(imgPoints2)

    jacobian = np.hstack((jacobian[:, 3:6], jacobian[:, :3]))
    jacobianApplied = np.hstack((jacobianApplied[:, 3:6], jacobianApplied[:, :3]))
    print("-------OpenCV------")
    print(jacobian.round(2))
    print("---OpenCV remake---")
    print(jacobianOpenCV.round(2))
    print("OpenCV remake 2 (left/right)")
    print(jacobianOpenCV2Left.round(2))
    print(jacobianOpenCV2Right.round(2))
    print("--Numerical Cross-")
    print(jacobianNumericalCross.round(2))
    print("---OpenCV applied--")
    print(jacobianApplied.round(2))
    print("--Numerical Global-")
    print(jacobianNumericalGlobal.round(2))
    print("---Numerical Fixed axis--")
    print(jacobianNumericalFixedAxis.round(2))
    print("-----Global Lie----")
    print(jacobianGlobal.round(2))
    print("-----Local Lie-----")
    print(jacobianLocal.round(2))
    print("--Numerical Local--")
    print(jacobianNumericalLocal.round(2))



    imgPoints = np.array([p[0] for p in imgPoints])
    img = np.zeros((720, 1280, 3), dtype=np.uint8)

    for p in imgPoints:
        cv.circle(img, (int(round(p[0])), int(round(p[1]))), 5, (255,0,0), -1)

    camera = CameraDummy(cameraMatrix, np.zeros((1,4), dtype=np.float32))

    plotAxis(img, tVec, rVec, camera, objectPoints, scale=1, color=None, thickness=2, opacity=1)

    print(np.linalg.inv(np.matmul(jacobian.transpose(), jacobian).round(2)))
    print(np.matmul(jacobian.transpose(), jacobian).round(2))
    print(np.linalg.matrix_rank(jacobian))


    cv.imshow("img", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    exit()
    ########################################################################
    delta = np.array([0.1, -0.7, 0.8])*0
    #print(rVec + delta)
    if np.linalg.norm(rVec) != 0:
        rVec *= (np.pi-0.0001)/np.linalg.norm(rVec)
    #rVec *= 0
    theta = np.linalg.norm(rVec)
    
    rotMat = R.from_rotvec(rVec).as_dcm()

    rVecNew = rotvecFromMat(rotMat)
    thetaNew = np.linalg.norm(rVecNew)

    print("Theta:", theta)
    print("Theta new:", thetaNew)

    print(rVec)
    print(rVecNew)

    plotVector(img, rVec/np.pi, tVec, rVec, camera, color=(255,0,255))
    plotVector(img, rVecNew/np.pi, tVec, rVec, camera, color=(0,255,255))

    print(rotMat)
    print(matFromRotvec(rVecNew))

    np.testing.assert_almost_equal(rVec, 
                                   rVecNew, 
                                   decimal=7, 
                                   err_msg='Rotation vectors not equal', 
                                   verbose=True)
