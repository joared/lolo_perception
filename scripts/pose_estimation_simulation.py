import cv2 as cv
import numpy as np
import time
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from scipy.spatial.transform import Rotation as R, rotation
from scipy.spatial.transform import Slerp
from tf.transformations import quaternion_multiply

import sys
sys.path.append("../../simulation/scripts")
from coordinate_system import CoordinateSystem, CoordinateSystemArtist

def projectPoints(points3D, cameraMatrix):
    # 3D points projected and converted to image coordinates (x,y in top left corner)
    points2D = np.matmul(cameraMatrix, points3D.transpose())
    points2D = points2D.transpose()
    for p in points2D:
        p[0] /= p[2]
        p[1] /= p[2]
        p[2] /= p[2]
    return points2D

def measurePose(points_3D, points_2D, camera_matrix, dist_coeffs, transEst, rotEst, guess, flag):
    
    """
    https://docs.opencv.org/4.5.3/d9/d0c/group__calib3d.html
    cv::SOLVEPNP_ITERATIVE = 0,
    cv::SOLVEPNP_EPNP = 1,
    cv::SOLVEPNP_P3P = 2,
    cv::SOLVEPNP_DLS = 3,
    cv::SOLVEPNP_UPNP = 4,
    cv::SOLVEPNP_AP3P = 5,
    cv::SOLVEPNP_IPPE = 6,
    cv::SOLVEPNP_IPPE_SQUARE = 7,
    cv::SOLVEPNP_SQPNP = 8
    """
    if flag == cv.SOLVEPNP_EPNP:
        points_2D = points_2D.reshape((points_2D.shape[0], 1, points_2D.shape[1]))
    success, rotation_vector, translation_vector = cv.solvePnP(np.array(list(points_3D)), 
                                                               np.array(list(points_2D)), 
                                                               camera_matrix, 
                                                               dist_coeffs, 
                                                               useExtrinsicGuess=guess,
                                                               tvec=transEst,
                                                               rvec=rotEst,
                                                               flags=flag)
    return success, rotation_vector, translation_vector

def plotPose(img, rotation_vector, translation_vector, camera_matrix, dist_coeffs, projectedPoints, color=(0,255,0)):

    zDir, jacobian = cv.projectPoints(np.array([(0.0, 0.0, 100.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    xDir, jacobian = cv.projectPoints(np.array([(100.0, 0.0, 0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    yDir, jacobian = cv.projectPoints(np.array([(0.0, 100.0, 0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    #for d, c in zip((xDir, yDir, zDir), ((0,0,255), (0,255,0), (255,0,0))):
    for d, c in zip((zDir,), (color,)):
        center = projectPoints(translation_vector.transpose(), camera_matrix)
        cx = center[0][0]
        cy = center[0][1]
        point1 = (int(round(cx)), int(round(cy)))
        point2 = (int(round(d[0][0][0])), int(round(d[0][0][1])))
        cv.line(img, point1, point2, c, 2)

    ss = 5 # square size
    for f in projectedPoints:
        img = cv.circle(img, tuple(np.rint((f[:2])).astype(int)), 3, color, 2)
        #img = cv.circle(img, tuple(np.rint((pf[:2])).astype(int)), 3, (0, 0, 255), 2)
        #img = cv.rectangle(img, 
        #                   (int(round(t[0]-ss)), int(round(t[1]+ss))), 
        #                   (int(round(t[0]+ss)), int(round(t[1]-ss))), 
        #                   (255, 255, 0), 
        #                   1)
    return img

def plotAxis(img, rotation_vector, translation_vector, camera_matrix, dist_coeffs):

    zDir, jacobian = cv.projectPoints(np.array([(0.0, 0.0, 100.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    xDir, jacobian = cv.projectPoints(np.array([(100.0, 0.0, 0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    yDir, jacobian = cv.projectPoints(np.array([(0.0, 100.0, 0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    print(zDir)
    
    for d, c in zip((xDir, yDir, zDir), ((0,0,255), (0,255,0), (255,0,0))):
        center = projectPoints(translation_vector.transpose(), camera_matrix)
        cx = center[0][0]
        cy = center[0][1]
        point1 = (int(round(cx)), int(round(cy)))
        point2 = (int(round(d[0][0][0])), int(round(d[0][0][1])))
        cv.line(img, point1, point2, c, 2)

    return img

def displayRotation(img, objectPpoints, euler, translation, camera_matrix, dist_coeffs):
    r = R.from_euler("XYZ", euler)
    rotation = r.as_rotvec().transpose()
    nIter = 100
    
    for i in range(nIter):
        imgTemp = img.copy()
        plotAxis(imgTemp, i*rotation/nIter, translation, camera_matrix, dist_coeffs)

        # Transform and project estimated measured points
        r = R.from_rotvec((i*rotation/nIter).transpose())
        T = np.hstack((r.as_dcm(), np.array([[0], [0], [800]])))
        points3D = np.matmul(T, objectPpoints.transpose()).transpose()
        points2D = projectPoints(points3D, camera_matrix)

        plotPose(imgTemp, i*rotation/nIter, translation, camera_matrix, dist_coeffs, points2D, (255, 0, 0))
        cv.imshow("Image", imgTemp)
        cv.waitKey(10)

    """
    # Quaternion slerp
    quatTo = r.as_quat()
    r = R.from_quat([[0,0,0,1], quatTo])
    slerp = Slerp([0, nIter], r)
    r = slerp(list(range(nIter)))
    print(r)
    """

def polygon(rad, n, shift=False, zShift=0):
    theta = 2*np.pi/n
    if shift is True:
        #points = np.array([[0, 0, 0, 1]] + [ [rad*np.sin(theta*(i + 0.5)), rad*np.cos(theta*(i + 0.5)), 0, 1] for i in range(n)], dtype=np.float32)
        points = np.array([ [rad*np.sin(theta*(i + 0.5)), rad*np.cos(theta*(i + 0.5)), zShift, 1] for i in range(n)] , dtype=np.float32)
    else:
        points = np.array([ [rad*np.sin(theta*i), rad*np.cos(theta*i), zShift, 1] for i in range(n)], dtype=np.float32)

    return points

def pose2():
    import sys
    sys.path.append("../../simulation/scripts")
    from camera import usbCamera
    # Tutorial: https://www.pythonpool.com/opencv-solvepnp/
    img = cv.imread('../image_dataset/lights_in_scene.png')
    img = np.zeros((480, 640))
    img = np.stack((img,)*3, axis=-1)

    """
    px = 320
    py = 240
    camera_matrix = np.array([[812.2540283203125,   0,    		        px],
                              [   0,               814.7816162109375, 	py], 
	                          [   0,     		     0,   		       	1]], dtype=np.float32)
    
    camera_matrix[0,0] = 800
    camera_matrix[1,1] = 800
    dist_coeffs = np.zeros((4,1), dtype=np.float32)
    """
    camera = usbCamera
    camera_matrix = camera.cameraMatrixPixel
    px = camera_matrix[0, 2]#/camera.pixelWidth
    py = camera_matrix[1, 1]#/camera.pixelHeight
    dist_coeffs = camera.distCoeffs
    # if the features was on the image plane
    # first point has to be in center!!! Otherwise plot doesn't work
    halfHyp = 100*np.cos(np.pi/4)
    square = np.array([[0., 0., 0., 1.],
                         [-halfHyp, -halfHyp, 0, 1],
                         [halfHyp, -halfHyp, 0, 1],
                         [halfHyp + 50, halfHyp, 0, 1],
                         [-halfHyp, halfHyp, 0, 1]], dtype=np.float32)

    squareTwist = np.array([[0, 0, 0, 1],
                        #[0, -100, 0, 1],
                        [-100, -100, 0, 1],
                        [100, -100, 0, 1],
                        [150, 150, 0, 1],
                        [-100, 100, 0, 1]], dtype=np.float32)

    points_3D = polygon(rad=100, n=4, shift=True, zShift=0)
    #points_3D = np.append(points_3D, polygon(rad=100, n=4, shift=False, zShift=20), axis=0)
    
    #points_3D = np.append(points_3D, polygon(rad=20, n=3, shift=True, zShift=-100), axis=0)
    #points_3D = np.append(points_3D, [[0, 0, 0, 1.]], axis=0)
    points_3D = np.append(points_3D, [[0, 0, 70.7, 1.]], axis=0)
    #points_3D = np.append(points_3D, [[0,-30,-200,1]], axis=0)
    #points_3D = np.append(points_3D, [[-30,30,-200,1]], axis=0)
    #points_3D = np.append(points_3D, [[30,30,-200,1]], axis=0)
    #points_3D = np.append(points_3D, [[-30,0,0,1]], axis=0)
    #points_3D = np.append(points_3D, [[30,0,0,1]], axis=0)
    #points_3D = square
    #points_3D[:, :3] *= camera.pixelWidth
    print(points_3D.shape)

    pz = 1200.#*camera.pixelWidth
    pzGuess = 800.#*camera.pixelWidth
    ax = 0#np.pi/4
    ay = np.pi
    ayGuess = np.pi
    az = 0#.6
    theta = 0

    displayRotation(img.copy(), points_3D, (ax, ay, az), np.array([[0.], [0.], [pz]], dtype=np.float32), camera_matrix, dist_coeffs)

    transEst = np.array([[0.], [0.], [pzGuess]], dtype=np.float32)
    rotTemp = R.from_euler("XYZ", (ax, ayGuess, az)).as_rotvec()
    rotEst = np.array([[rotTemp[0]], [rotTemp[1]], [rotTemp[2]]], dtype=np.float32)
    velEst = np.array([[0., 0., 0.]]).transpose()

    fig = plt.figure()
    axes = fig.gca(projection='3d')
    cs = CoordinateSystem()
    csArt = CoordinateSystemArtist(cs)
    cs2 = CoordinateSystem()
    csArt2 = CoordinateSystemArtist(cs2)

    errors = []
    nIterations = 500
    start = time.time()
    for i in range(nIterations):
        
        #translation from the camera origin?
        #translation = np.array([[np.cos(theta)*50], [np.sin(theta)*50], [pz]])
        translation = np.array([[0.0], [0.0], [pz]])

        #theta += 0.05
        #pz += 15
        #ax += 0.03
        ay += 0.06*np.sin(i*0.07-np.pi/2)
        #az += 0.01

        r = R.from_euler("XYZ", (ax, ay, az))
        rotation = r.as_rotvec().transpose()
        objectT = np.hstack((r.as_dcm(), translation))

        # 3D points in camera frame
        transformed_3D = np.matmul(objectT, points_3D.transpose()).transpose()

        # 2D projected points
        points_2D = projectPoints(transformed_3D, camera_matrix)


        # 1) introduce noise in 3D
        #noise3D = np.random.normal(0, 1, transformed_3D.shape) # Maybe introduce noise in image plane instead??
        #points_2D_noised = projectPoints(transformed_3D  + noise3D, camera_matrix)
        # 2) introduce noise in 2D image plane
        noise2D = np.random.normal(0, 3, points_2D.shape) # Maybe introduce noise in image plane instead??
        noise2D[:, 2] = 0
        points_2D_noised = points_2D + noise2D

        # estimate pose (translation and rotation in camera frame)
        # we take away the center point
        # Guessing may result in can mess up the first measurement (maybe average the first X measurements?)
        # Remove slices, even if it seems to work? Documentation says slices not allowed
        guess = True
        #if i == 0:
        #    guess = True
        success, rotation_vector, translation_vector = measurePose(points_3D[:, :3], 
                                                                   points_2D_noised[:, :2], 
                                                                   camera_matrix, 
                                                                   dist_coeffs, 
                                                                   #transEst.copy(), 
                                                                   #rotEst.copy(), 
                                                                   np.array([[0.], [0.], [pzGuess]]),
                                                                   np.array([[0.], [ayGuess], [0.]]),
                                                                   guess=guess,
                                                                   flag=cv.SOLVEPNP_ITERATIVE)
                                                                   #flag=cv.SOLVEPNP_EPNP)

        #print("NSOLUTIONS:", len(rotation_vector))

        if not success:
            print("FAILED PnP")

        alpha = 0.6
        beta = 0.3
        velEst += beta*(translation_vector - transEst)
        transEst = transEst + alpha*(translation_vector-transEst) + velEst
        
        citrus = 0.0
        angleVelMeas = 0
        delta = 0.5
        
        # Quaternion slerp
        quatEst, = R.from_rotvec(rotEst.transpose()).as_quat()
        quatMeas, = R.from_rotvec(rotation_vector.transpose()).as_quat()
        r = R.from_quat([quatEst, quatMeas])
        slerp = Slerp([0, 1], r)
        r = slerp([delta])
        rotEst = r.as_rotvec().transpose()
        
        """
        # Quaternion slerp
        quatEst, = R.from_rotvec(rotEst.transpose()).as_quat()
        quatMeas, = R.from_rotvec(rotation_vector.transpose()).as_quat()
        r = R.from_quat([quatEst, quatMeas])
        slerp = Slerp([0, 1], r)
        r = slerp([delta])
        rotEst = r.as_rotvec().transpose()

        # Angle axis with error as matrix multiplication (works!!)
        rEst = R.from_rotvec(rotEst.transpose())
        rMeas = R.from_rotvec(rotation_vector.transpose())
        # from rMeas = rErr*rEst
        rErr = rMeas*rEst.inv()
        rotErr = rErr.as_rotvec().transpose()
        rotErr *= delta
        print("Rot err:", rotErr)
        rEst = R.from_rotvec(rotErr.transpose())*rEst
        rotEst = rEst.as_rotvec().transpose()

        # Quaternion thing (I don't know...)
        quatEst, = R.from_rotvec(rotEst.transpose()).as_quat()
        quatMeas, = R.from_rotvec(rotation_vector.transpose()).as_quat()
        quatEstInv = np.append(quatEst[:3], -quatEst[3])
        quatErr = quaternion_multiply(quatMeas, quatEstInv)
        # DO something with the error
        quatErr = ...
        quatEst = quaternion_multiply(quatErr, quatEst)
        rotEst = R.from_quat(quatEst).as_rotvec()
        rotEst = np.array([rotEst]).transpose()
        
        # Angle axis (kind of works, but jump at pi somewhere still...)
        #rotEst = (1-delta)*rotEst + delta*rotation_vector + citrus*angVelMeas # this averaging is wrong!!!
        diff = rotation_vector-rotEst
        diff = (diff + np.pi) % (2*np.pi) - np.pi
        rotEst = rotEst + delta*(diff) # citrus*angVelMeas
        rotEst = (rotEst + np.pi) % (2*np.pi) - np.pi
        """

        quatTrue = R.from_rotvec(rotation).as_quat()
        quatMeas, = R.from_rotvec(rotation_vector.transpose()).as_quat()
        quatMeasInv = np.append(quatMeas[:3], -quatMeas[3])
        quatErr = quaternion_multiply(quatTrue, quatMeasInv)

        #errors.append(quatErr[:3])
        errors.append(translation - translation_vector)
        print("Mean trans err: ", np.mean(errors, axis=0).transpose())
        print("Mean trans std: ", np.std(errors, axis=0).transpose())
        #ax.cla()
        #ax.scatter(*zip(*errors))
        #ax.scatter(*zip(*enumerate(errors)))

        # Transform and project estimated measured points
        r = R.from_rotvec(rotation_vector.transpose())
        T = np.hstack((r.as_dcm()[0], translation_vector))
        pointsSolved3D = np.matmul(T, points_3D.transpose()).transpose()
        pointsSolved2D = projectPoints(pointsSolved3D, camera_matrix)
        
        # Transform and project estimated points
        r = R.from_rotvec(rotEst.transpose())
        T = np.hstack((r.as_dcm()[0], transEst))
        pointsEst3D = np.matmul(T, points_3D.transpose()).transpose()
        pointsEst2D = projectPoints(pointsEst3D, camera_matrix)

        temp3D = transformed_3D.copy()
        for p in temp3D: 
            p[0] += px
            p[1] += py

        #plotPose(img.copy(), rotation_vector, translation_vector, camera_matrix, dist_coeffs, temp3D, points_2D, transformedProj)
        imgTemp = img.copy()

        # ground truth
        #plotPose(imgTemp, rotation, translation, camera_matrix, dist_coeffs, points_2D, color=(0,255,0))
        #plotAxis(imgTemp, rotation, translation, camera_matrix, dist_coeffs)

        # estimated measure points_2D and points_2D_noised should be used?: no because thats just what the PnP algotihm based the pose estimation on
        #plotAxis(imgTemp, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        plotPose(imgTemp, rotation_vector, translation_vector, camera_matrix, dist_coeffs, pointsSolved2D, color=(0,0,255))
        
        # estimated
        plotPose(imgTemp, rotEst, transEst, camera_matrix, dist_coeffs, pointsEst2D, color=(255,255,0))
        #plotAxis(imgTemp, rotEst, transEst, camera_matrix, dist_coeffs)

        cv.imshow("Image", imgTemp)
        cv.waitKey(10)
        rotMat = R.from_rotvec(rotation_vector[:,0].transpose()).as_dcm()
        cs.setTransform([1, 0, 0], np.array(rotMat))
        
        rotMat = R.from_rotvec(rotation).as_dcm()
        cs2.setTransform([-1, 0, 0], np.array(rotMat))

        axes.cla()
        size = 2
        axes.set_xlim(-size, size)
        axes.set_ylim(-size, size)
        axes.set_zlim(-size, size)
        csArt.draw(axes)
        csArt2.draw(axes)
        plt.pause(0.01)
    #plt.show()
    elapsed = time.time() - start
    print("Computation time avg:", elapsed/nIterations)

if __name__ =="__main__":
    pose2()