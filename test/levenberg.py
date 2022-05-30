import numpy as np
import cv2 as cv
import time
import random

from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from scipy.spatial.transform import Rotation as R

from lolo_simulation.coordinate_system import CoordinateSystemArtist, CoordinateSystem
from lolo_perception.perception_utils import plotAxis
from lolo_perception.pose_estimation_utils import lmSolve, projectPoints2


def lmSolveRoutine(objectPoints, detectedPoints, camera, tVec, rVec, tVecGuess, rVecGuess, jacType, ax, ax2, camCS, color="r", scale=1., sigmaX=0, sigmaY=0):
    cameraMatrix = camera.cameraMatrix
    if sigmaX != 0 or sigmaX != 0:
        detectedPoints[:, 0] += np.random.normal(0, sigmaX, detectedPoints[:, 0].shape)
        detectedPoints[:, 1] += np.random.normal(0, sigmaY, detectedPoints[:, 1].shape)


    estCS = CoordinateSystemArtist()

    img = np.zeros((720, 1280, 3))
    for pose, lambdas, grads, errors in lmSolve(cameraMatrix, 
                                                detectedPoints, 
                                                objectPoints, 
                                                tVecGuess, 
                                                rVecGuess, 
                                                jacType,
                                                maxIter=50,
                                                mode="lm",
                                                generate=True,
                                                verbose=1):


        projPoints = projectPoints2(pose[:3], pose[3:], cameraMatrix, objectPoints)
        for p in projPoints:
            p = p.round().astype(np.int32)
            cv.circle(img, tuple(p), 3, (0,0,255), -1)
        plotAxis(img, pose[:3], pose[3:], camera, objectPoints, color=(0,0,255), scale=scale, opacity=1)

        for imgP in detectedPoints:
            imgP = imgP.round().astype(np.int32)
            cv.circle(img, tuple(imgP), 3, (0,255,0), -1)
        plotAxis(img, tVec, rVec, camera, objectPoints, color=(0,255,0), scale=scale)

        ax2.cla()
        ax2.plot(errors)
        ax2.plot(lambdas)
        ax2.plot(grads)
        ax2.legend(["F(x)", "mu", "||g||"])

        estCS.cs.translation = pose[:3]
        estCS.cs.rotation = R.from_rotvec(pose[3:]).as_dcm()
        estCS.drawRelative(ax, camCS.cs, colors=("gray",)*3, scale=1, alpha=0.5)

        print(pose.round(2))
        cv.imshow("img", img)
        cv.waitKey(10)
        plt.pause(.1)
        img *= 0
    
    estCS.drawRelative(ax, camCS.cs, colors=(color,)*3, scale=1, alpha=1)

def randomVec(rx, ry, rz):

    return np.array([(random.random()-0.5)*rx,
                    (random.random()-0.5)*ry,
                    (random.random()-0.5)*rz])

def lmIllustration(cameraMatrix, objectPoints, tVec, rVec, tVecGuess, rVecGuess, jacType="opencv", sigmaX=0, sigmaY=0, scale=1):
        detectedPoints = projectPoints2(tVec, rVec, cameraMatrix, objectPoints)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        fig2 = plt.figure()
        ax2 = fig2.gca()

        maxZ = tVec[2]*1.2
        ax.set_xlim(0, maxZ)
        ax.set_ylim(0, maxZ)
        ax.set_zlim(0, maxZ)

        camCS = CoordinateSystemArtist(CoordinateSystem(translation=(maxZ/2, 0, maxZ/2), euler=(-np.pi/2, 0, 0)))
        trueCS = CoordinateSystemArtist(CoordinateSystem(translation=tVec, euler=R.from_rotvec(rVec).as_euler("XYZ")))
        camCS.draw(ax, colors=("b",)*3)
        trueCS.drawRelative(ax, camCS.cs, colors=("g",)*3)

        print(tVec, rVec)
        lmSolveRoutine(objectPoints, 
                    detectedPoints, 
                    CameraDummy(cameraMatrix), 
                    tVec, 
                    rVec, 
                    tVecGuess, 
                    rVecGuess, 
                    jacType, 
                    ax, 
                    ax2, 
                    camCS,
                    color="r", 
                    sigmaX=sigmaX, 
                    sigmaY=sigmaY,
                    scale=scale)

        while True:
            plt.pause(0.01)
            key = cv.waitKey(10)
            if key == ord("q"):
                break
        cv.destroyAllWindows()


if __name__ =="__main__":
    from lolo_perception.feature_model import polygon
    random.seed(12345)
    
    class CameraDummy:
        def __init__(self, cameraMatrix):
            self.cameraMatrix = cameraMatrix
            self.distCoeffs = np.zeros((1,4), dtype=np.float32)

    # Square
    #objectPoints = np.array([[1., 1., 0], [-1., 1., 0], [1., -1., 0], [-1., -1., 0]])*0.3
    # Assymteric planar
    #objectPoints = np.array([[1., 1., 0], [-1., 1., 0], [1., -1., 0], [-.0, -.0, 0.9]])
    # symmetric planar polygon
    #objectPoints = np.array(polygon(1, 8, shift=False, zShift=0)[:, :-1])
    # 5 Lights
    objectPoints = np.array([[1., 1., 0], [-1., 1., 0], [1., -1., 0], [-1., -1., 0], [0., -.0, -.5]])*.33
    
    

    # Difficult pose
    tVec = np.array([-5., -5., 25.])
    rVec = R.from_euler("YXZ", (np.deg2rad(-110), np.deg2rad(90), np.deg2rad(90))).as_rotvec()
    
    print(np.concatenate((tVec, rVec)).round(2))
    #rVec = np.array([0, np.pi, 0])
    f = 1000.0
    cameraMatrix = np.array([[f, 0., 639.],
                             [0., f, 359.],
                             [0., 0., 1]])

    np.random.seed(123456)
    lmIllustration(cameraMatrix, 
                   objectPoints, 
                   np.array([0., 0., 12]), 
                   np.array([0., np.deg2rad(30), 0.]),
                   np.array([0., 0., 1.]),
                   np.array([0., np.deg2rad(-160), 0.]),
                   sigmaX=2,
                   sigmaY=2,
                   scale=0.3)
    exit()

    success = {"opencvTrue":0, "opencv":0, "local":0, "global":0}
    maxZ = 25.0

    failedPoses = []
    for i in range(300):
        tVec = randomVec(5, 5, maxZ)
        tVec[2] += maxZ/2 + np.max(np.abs(objectPoints))
        angle = 10
        rVec = R.from_euler("YXZ", randomVec(np.deg2rad(angle), np.deg2rad(angle), np.deg2rad(angle))).as_rotvec()

        tVecGuess = np.array([tVec[0]/tVec[2], tVec[1]/tVec[2], np.max(np.abs(objectPoints))+1])
        rVecGuess = np.array([0.0, 0., 0.])

        tVecGuess = np.array([0, 0, 5.])
        rVecGuess = np.array([0.0, 0., 0.])

        detectedPoints = projectPoints2(tVec, rVec, cameraMatrix, objectPoints)
        for jacType, color in zip(["opencvTrue", "opencv", "local", "global"], ["r", "b", "m", "y"]):
            if jacType == "opencvTrue":
                succ, rotationVector, translationVector = cv.solvePnP(objectPoints,
                                                                    detectedPoints,
                                                                    cameraMatrix,
                                                                    np.zeros((1,4), dtype=np.float32),
                                                                    useExtrinsicGuess=True,
                                                                    tvec=tVecGuess.copy().reshape((3,1)),
                                                                    rvec=rVecGuess.copy().reshape((3,1)),
                                                                    flags=cv.SOLVEPNP_ITERATIVE)
                pose = np.array(list(translationVector.reshape((3,))) + list(rotationVector.reshape((3,))))
            else:
                pose = lmSolve(cameraMatrix, 
                                detectedPoints, 
                                objectPoints, 
                                tVec=tVecGuess, 
                                rVec=rVecGuess, 
                                jacobianCalc=jacType,
                                maxIter=200,
                                mode="lm",
                                generate=False,
                                verbose=0).next()[0]

            rtol = 10e-7
            if np.allclose(pose[:3], tVec, rtol=rtol) and np.allclose(R.from_rotvec(pose[3:]).as_dcm(), R.from_rotvec(rVec).as_dcm(), rtol=10-4):
                success[jacType] += 1
            else:
                if jacType == "opencvTrue":
                    print(succ)
                failedPoses.append([tVec, rVec])
                print(jacType)
                print(np.array([list(tVec) + list(rVec)]))
                print(np.array(pose))

    print(success)

    for tVec, rVec in failedPoses:
        
        #tVec, rVec = np.array([1.83979545, 2.27883223, 1.00268141]), np.array([0., 0., 0.])
        detectedPoints = projectPoints2(tVec, rVec, cameraMatrix, objectPoints)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        fig2 = plt.figure()
        ax2 = fig2.gca()

        ax.set_xlim(0, maxZ)
        ax.set_ylim(0, maxZ)
        ax.set_zlim(0, maxZ)

        camCS = CoordinateSystemArtist(CoordinateSystem(translation=(maxZ/2, 0, maxZ/2), euler=(-np.pi/2, 0, 0)))
        trueCS = CoordinateSystemArtist(CoordinateSystem(translation=tVec, euler=R.from_rotvec(rVec).as_euler("XYZ")))
        camCS.draw(ax, colors=("b",)*3)
        trueCS.drawRelative(ax, camCS.cs, colors=("g",)*3)

        print(tVec, rVec)
        lmSolveRoutine(objectPoints, 
                    detectedPoints, 
                    CameraDummy(cameraMatrix), 
                    tVec, 
                    rVec, 
                    tVecGuess, 
                    rVecGuess, 
                    jacType, 
                    ax, 
                    ax2, 
                    camCS,
                    color="r", 
                    sigmaX=0, 
                    sigmaY=0)

        while True:
            plt.pause(0.01)
            key = cv.waitKey(10)
            if key == ord("q"):
                break
        cv.destroyAllWindows()












    N = 100
    for mode in ("opencvTrue", "opencv", "global", "local"):
        start = time.time()
        for j in range(N):
            if mode == "opencvTrue":
                success, rotationVector, translationVector = cv.solvePnP(objectPoints,
                                                                    detectedPoints,
                                                                    cameraMatrix,
                                                                    np.zeros((1,4), dtype=np.float32),
                                                                    useExtrinsicGuess=True,
                                                                    tvec=np.array([0., 0., 2.]).reshape((3,1)),
                                                                    rvec=np.array([0.0, 0., 0.]).reshape((3,1)),
                                                                    flags=cv.SOLVEPNP_ITERATIVE)
                pose = np.array(list(translationVector) + list(rotationVector))
            else:
                (pose, lambdas, grads, errors) in lmSolve(cameraMatrix, 
                                                        detectedPoints, 
                                                        objectPoints, 
                                                        np.array([0., 0., 2.]), 
                                                        np.array([0.0, 0., 0.]), 
                                                        mode,
                                                        maxIter=200,
                                                        mode="lm",
                                                        generate=False,
                                                        verbose=0)

        elapsed = time.time()-start
        avg = elapsed/N
        print("{}: {}".format(mode, round(avg, 5)))
        print(pose)

