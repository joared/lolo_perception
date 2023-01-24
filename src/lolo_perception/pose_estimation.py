import cv2 as cv
import numpy as np
import lolo_perception.py_logging as logging
from scipy.spatial.transform import Rotation as R
import scipy

def interactionMatrix(x, y, Z):
    """
    IBVS interaction matrix
    """
    return [[-1/Z, 0, x/Z, x*y, -(1+x*x), y],
                [0, -1/Z, y/Z, 1+y*y, -x*y, -x]]


def projectPoints(translationVector, rotationVector, camera, objPoints):
    projPoints, _ = cv.projectPoints(objPoints, 
                                     rotationVector, 
                                     translationVector, 
                                     camera.cameraMatrix, 
                                     camera.distCoeffs)
    projPoints = np.array([p[0] for p in projPoints])
    return projPoints


def reprojectionError(translationVector, rotationVector, camera, objPoints, imagePoints):
    """

    https://github.com/strawlab/image_pipeline/blob/master/camera_calibration/nodes/cameracheck.py
    """
    projPoints = projectPoints(translationVector, rotationVector, camera, objPoints)

    reprojErrs = imagePoints - projPoints
    rms = np.sqrt(np.sum(reprojErrs ** 2)/np.product(reprojErrs.shape))

    return reprojErrs, rms


def calcImageCovariance(translationVector, rotationVector, camera, featureModel, confidence):
    sigma3D = featureModel.uncertainty/np.sqrt(confidence)

    fx = camera.cameraMatrix[0, 0]
    fy = camera.cameraMatrix[1, 1]

    fx2 = fx**2
    fy2 = fy**2
    sigma23D = sigma3D**2

    covariances = []
    r = R.from_rotvec(rotationVector)    
    for pLocal in featureModel.features:
        X,Y,Z = r.apply(pLocal) + translationVector

        sigma2U = sigma23D*fx2*(X**2+Z**2)/Z**4
        sigma2V = sigma23D*fy2*(Y**2+Z**2)/Z**4
        sigma2UV = sigma23D*fx*fy*X*Y/Z**4

        pixelCov = [[sigma2U, sigma2UV],
                    [sigma2UV, sigma2V]]

        covariances.append(pixelCov)
    
    return np.array(covariances)


def calcMahalanobisDist(estDSPose, dsPose, SInv=None):
    if SInv is None:
        logging.error("This should not happen")
        #SInv = cv.invert(estDSPose.covariance, flags=cv.DECOMP_CHOLESKY)
        SInv = np.linalg.inv(estDSPose.covariance)

    translErr = estDSPose.translationVector - dsPose.translationVector
    #translErr[2] = 0 # ignore z translation
    
    r1 = R.from_rotvec(estDSPose.rotationVector)
    r2 = R.from_rotvec(dsPose.rotationVector)
 
    #e = r1*r2.inv() # error in world (not this one, because it is fixed frame rotation)
    #e = r1.inv()*r2 # 2 wrt to 1 (should not be this one)
    rotationErr = r2.inv()*r1 # 1 wrt to 2
    rotationErr = rotationErr.as_euler("xyz") # fixed axis "xyz"

    err = np.array(list(translErr) + list(rotationErr))
    #print(np.matmul(err, np.linalg.inv(S)))
    mahaDist = np.matmul(np.matmul(err.transpose(), SInv), err)
    mahaDist = np.sqrt(mahaDist)
    
    return mahaDist


def calcCamPoseCovariance(camera, featureModel, translationVector, rotationVector, pixelCovariance):
    """
    projPoints = projectPoints(translationVector, 
                               rotationVector, 
                               camera, 
                               featureModel.features)
    """
    J = np.zeros((2*len(featureModel.features), 6))

    fx = camera.cameraMatrix[0, 0]
    fy = camera.cameraMatrix[1, 1]

    for i, p in enumerate(featureModel.features):
        X, Y, Z = R.from_rotvec(rotationVector).apply(p) + translationVector

        intMat = np.array(interactionMatrix(X/Z, Y/Z, Z))
        intMat[0, :]*=fx
        intMat[1, :]*=fy

        J[i*2:i*2+2, :] = intMat

    sigma = scipy.linalg.block_diag(*[pixelCovariance]*len(featureModel.features))

    sigmaInv = np.linalg.inv(sigma)
    try:
        mult = np.matmul(np.matmul(J.transpose(), sigmaInv), J)
        covariance = np.linalg.inv(mult)
    except np.linalg.LinAlgError as e:
        print("Singular matrix")

    return covariance


def calcPoseCovarianceFixedAxis(camera, featureModel, translationVector, rotationVector, pixelCovariance):
    # pixelCovariance - 2x2 or Nx2x2 where N is the number of features

    allCovariancesGiven = False
    if len(pixelCovariance.shape) == 3:
        assert pixelCovariance.shape == (len(featureModel.features),2,2), "Incorrect shape of pixelCovariance '{}', should be '{}'".format(pixelCovariance.shape, (len(featureModel.features),2,2))
        allCovariancesGiven = True

    # some magic to convert to fixed axis
    r = R.from_rotvec(rotationVector)
    featuresTemp = r.apply(featureModel.features)
    rotationVector = np.array([0., 0., 0.])

    _, jacobian = cv.projectPoints(featuresTemp, 
                                   rotationVector.reshape((3, 1)), 
                                   translationVector.reshape((3, 1)), 
                                   camera.cameraMatrix, 
                                   camera.distCoeffs)

    rotJ = jacobian[:, :3]
    transJ = jacobian[:, 3:6]
    J = np.hstack((transJ, rotJ)) # reorder covariance as used in PoseWithCovarianceStamped

    if allCovariancesGiven:
        sigma = scipy.linalg.block_diag(*pixelCovariance)
    else:
        sigma = scipy.linalg.block_diag(*[pixelCovariance]*len(featureModel.features))

    retval, sigmaInv = cv.invert(sigma, flags=cv.DECOMP_CHOLESKY)
    
    if not retval:
        logging.error("Inversion of sigma failed with return value: {}".format(retval))
    try:
        # C = (J^T * S^-1 * J)^-1
        mult = np.matmul(np.matmul(J.transpose(), sigmaInv), J)
        retval, covariance = cv.invert(mult, flags=cv.DECOMP_CHOLESKY)
        
        if not retval:
            logging.error("Inversion of the hessian failed with return value: {}".format(retval))

    except np.linalg.LinAlgError as e:
        logging.error("Singular matrix")

    return covariance


def calcPoseCovariance(camera, featureModel, translationVector, rotationVector, pixelCovariance):
    _, jacobian = cv.projectPoints(featureModel.features, 
                                   rotationVector.reshape((3, 1)), 
                                   translationVector.reshape((3, 1)), 
                                   camera.cameraMatrix, 
                                   camera.distCoeffs)

    rotJ = jacobian[:, :3]
    transJ = jacobian[:, 3:6]
    J = np.hstack((transJ, rotJ)) # reorder covariance as used in PoseWithCovarianceStamped

    # How to rotate covariance: https://robotics.stackexchange.com/questions/2556/how-to-rotate-covariance

    sigma = scipy.linalg.block_diag(*[pixelCovariance]*len(featureModel.features))

    sigmaInv = np.linalg.inv(sigma)
    try:
        mult = np.matmul(np.matmul(J.transpose(), sigmaInv), J)
        covariance = np.linalg.inv(mult)
    except np.linalg.LinAlgError as e:
        print("Singular matrix")

    return covariance