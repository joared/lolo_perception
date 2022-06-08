
import cv2 as cv
import numpy as np
import time
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import scipy
from lolo_perception.perception_utils import projectPoints, reprojectionError
from lolo_perception.feature_extraction import featureAssociation


from lolo_perception.reprojection_utils import calcPoseReprojectionRMSEThreshold, calcPoseReprojectionThresholds
from lolo_perception.pose_estimation_utils import lmSolve, interactionMatrix

from numpy.linalg import lapack_lite
#lapack_routine = lapack_lite.dgesv

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
        # From test_pixel_covariance
        #sigmaU2 = sigma**2*fx**2*(X**2 + Z**2)/Z**4
        #sigmaV2 = sigma**2*fy**2*(Y**2 + Z**2)/Z**4
        #sigmaUV2 = sigma**2*fx*fy*X*Y/Z**4
    
    return np.array(covariances)


def calcMahalanobisDist(estDSPose, dsPose, SInv=None):
    if SInv is None:
        SInv = np.linalg.inv(estDSPose.covariance)

    translErr = estDSPose.translationVector - dsPose.translationVector
    translErr *= 0 # ignore translation for now
    
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

# Looking one step deeper, we see that solve performs many sanity checks.  
# Stripping these, we have:
def fastInverse(A):
    """
    https://stackoverflow.com/questions/11972102/is-there-a-way-to-efficiently-invert-an-array-of-matrices-with-numpy
    """
    b = np.identity(A.shape[2], dtype=A.dtype)

    n_eq = A.shape[1]
    n_rhs = A.shape[2]
    pivots = np.zeros(n_eq, np.intc)
    identity  = np.eye(n_eq)
    def lapack_inverse(a):
        b = np.copy(identity)
        pivots = np.zeros(n_eq, np.intc)
        results = lapack_lite.dgelsd(n_eq, n_rhs, a, n_eq, pivots, b, n_eq, 0)
        if results['info'] > 0:
            raise np.LinAlgError('Singular matrix')
        return b

    return np.array([lapack_inverse(a) for a in A])

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
    #sigmaInv = fastInverse(np.reshape(sigma, (sigma.shape[0], sigma.shape[1], 1))).reshape(sigma.shape)
    try:
        mult = np.matmul(np.matmul(J.transpose(), sigmaInv), J)
        covariance = np.linalg.inv(mult)
        #covariance = fastInverse(np.reshape(mult, (mult.shape[0], mult.shape[1], 1))).reshape(mult.shape)
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

    # How to rotate covariance: https://robotics.stackexchange.com/questions/2556/how-to-rotate-covariance

    if allCovariancesGiven:
        sigma = scipy.linalg.block_diag(*pixelCovariance)
    else:
        sigma = scipy.linalg.block_diag(*[pixelCovariance]*len(featureModel.features))

    sigmaInv = np.linalg.inv(sigma)
    #sigmaInv = fastInverse(np.reshape(sigma, (sigma.shape[0], sigma.shape[1], 1))).reshape(sigma.shape)
    try:
        mult = np.matmul(np.matmul(J.transpose(), sigmaInv), J)
        covariance = np.linalg.inv(mult)
        #covariance = fastInverse(np.reshape(mult, (mult.shape[0], mult.shape[1], 1))).reshape(mult.shape)
    except np.linalg.LinAlgError as e:
        print("Singular matrix")

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
    #sigmaInv = fastInverse(np.reshape(sigma, (sigma.shape[0], sigma.shape[1], 1))).reshape(sigma.shape)
    try:
        mult = np.matmul(np.matmul(J.transpose(), sigmaInv), J)
        covariance = np.linalg.inv(mult)
        #covariance = fastInverse(np.reshape(mult, (mult.shape[0], mult.shape[1], 1))).reshape(mult.shape)
    except np.linalg.LinAlgError as e:
        print("Singular matrix")

    return covariance

class DSPose:
    def __init__(self, 
                 translationVector, 
                 rotationVector, 
                 camTranslationVector, 
                 camRotationVector,
                 associatedLightSources,
                 camera,
                 featureModel):

        self.translationVector = translationVector
        self.rotationVector = rotationVector
        self._covariance = None
        self.yaw, self.pitch, self.roll = R.from_rotvec(self.rotationVector).as_euler("YXZ")

        self._validOrientation = () # yaw, pitch, roll

        self.camTranslationVector = camTranslationVector
        self.camRotationVector = camRotationVector
        self._camCovariance = None # not used at the moment
        #self.camYaw, self.camPitch, self.camRoll = R.from_rotvec(self.camRotationVector).as_euler("YXZ") # not used

        self.associatedLightSources = associatedLightSources
        self.camera = camera
        self.featureModel = featureModel

        self.pixelCovariances = None
        self.reprErrors = None
        self.reprErrorsMax = None
        self._rmse = None
        self._rmseMax = None

        self.chiSquaredConfidence = 5.991
        self.mahaDist = None
        self.mahaDistThresh = None
        
        # increase this to keep track of how many valid poses have 
        # been detected in sequence
        self.detectionCount = 1

        # how many attempts the pose estimation algorithm did until it found the pose
        self.attempts = 0
        # number of total combinations that the pose algorithm considered 
        self.combinations = 0 

    @property
    def rmse(self):
        if self._rmse:
            return self._rmse
        else:
            return self.calcRMSE()[0]

    @property
    def rmseMax(self):
        if self._rmseMax:
            return self._rmseMax
        else:
            return self.calcRMSE()[1]

    @property
    def covariance(self):
        if self._covariance is not None:
            return self._covariance
        else:
            return self.calcCovariance()

    @property
    def camCovariance(self):
        if self._camCovariance is not None:
            return self._camCovariance
        else:
            return np.zeros((6, 6))

    def calcMahalanobisDist(self, estDSPose, SInv=None):
        mahaDist = calcMahalanobisDist(estDSPose, self, SInv)
        self.mahaDist = mahaDist
        return self.mahaDist

    def reProject(self):
        return projectPoints(self.translationVector, 
                             self.rotationVector, 
                             self.camera, 
                             self.featureModel.features)

    def reprErrMinCertainty(self):
        if self.reprErrors is None or self.reprErrorsMax is None:
            self.calcRMSE()

        diffs = self.reprErrorsMax - abs(self.reprErrors)
        certainties = diffs / self.reprErrorsMax
        certainty = np.min(certainties)
        return certainty

    def validReprError_old_reprojection(self):
        if self.reprErrors is None or self.reprErrorsMax is None:
            self.calcRMSE()

        for err, errMax in zip(self.reprErrors, self.reprErrorsMax):
            if abs(err[0]) > errMax[0] or abs(err[1]) > errMax[1]:
                return False
        return True

    def validReprError(self, minThreshold=0.7071):
        if self.reprErrors is None or self.pixelCovariances is None:
            self.calcRMSE()
        
        #return True # TODO: remove
        #return self.validReprError_old_reprojection() # old version
        for err, pCov in zip(self.reprErrors, self.pixelCovariances):
            try:
                pCovInv = np.linalg.inv(pCov)
            except np.linalg.LinAlgError as e:
                print("Singular image covariance matrix")
                return False

            if np.linalg.norm(err) < minThreshold:
                continue

            mahaDistSquare = np.matmul(np.matmul(err.transpose(), pCovInv), err)
            if mahaDistSquare > self.chiSquaredConfidence:
                return False

        return True

    def calcRMSE_old_reprojection(self, showImg=False):
        if self._rmse:
            return self._rmse
            #raise Exception("Already calculated")

        errs, rmse = reprojectionError(self.translationVector, 
                                       self.rotationVector, 
                                       self.camera, 
                                       self.featureModel.features, 
                                       np.array([ls.center for ls in self.associatedLightSources], dtype=np.float32))

        reprThresholds = calcPoseReprojectionThresholds(self.translationVector, 
                                                        self.rotationVector, 
                                                        self.camera, 
                                                        self.featureModel)
        # Minimum absolute reprojection error is 1
        reprThresholds[reprThresholds < 1] = 1

        rmseMaxModel = np.sqrt(np.sum(reprThresholds**2)/np.product(reprThresholds.shape))


        self.reprErrorsMax = reprThresholds
        self.reprErrors = errs

        self._rmse = rmse
        self._rmseMax = rmseMaxModel
        #self.rmseMax += rmseMaxLightsource # light source rejection is unrealiable since large areas (noise) may not be rejected 
        self._rmseMax += 1 # add 1 pixel for far distance detections
        return self._rmse, self._rmseMax

    def calcRMSE(self, showImg=False):
        if self._rmse:
            return self._rmse
            #raise Exception("Already calculated")

        errs, rmse = reprojectionError(self.translationVector, 
                                       self.rotationVector, 
                                       self.camera, 
                                       self.featureModel.features, 
                                       np.array([ls.center for ls in self.associatedLightSources], dtype=np.float32))

        reprThresholds = calcPoseReprojectionThresholds(self.translationVector, 
                                                        self.rotationVector, 
                                                        self.camera, 
                                                        self.featureModel)
        # Minimum absolute reprojection error is 1
        reprThresholds[reprThresholds < 1] = 1

        rmseMaxModel = np.sqrt(np.sum(reprThresholds**2)/np.product(reprThresholds.shape))

        pixelCovariances = calcImageCovariance(self.translationVector, 
                                               self.rotationVector, 
                                               self.camera, 
                                               self.featureModel,
                                               confidence=self.chiSquaredConfidence)
        self.pixelCovariances = pixelCovariances

        self.reprErrorsMax = reprThresholds
        self.reprErrors = errs

        self._rmse = rmse
        self._rmseMax = rmseMaxModel
        #self.rmseMax += rmseMaxLightsource # light source rejection is unrealiable since large areas (noise) may not be rejected 
        self._rmseMax += 1 # add 1 pixel for far distance detections
        return self._rmse, self._rmseMax

    def calcCamPoseCovariance(self, pixelCovariance=None, sigmaScale=4.0):
        if pixelCovariance is None:
            # https://www.thoughtco.com/range-rule-for-standard-deviation-3126231
            # 2 - 95 %, 4 - 99 %
            sigmaX = self.rmseMax/sigmaScale
            sigmaY = self.rmseMax/sigmaScale
            pixelCovariance = np.array([[sigmaX**2, 0], [0, sigmaY**2]])

        covariance = calcCamPoseCovariance(self.camera, 
                                           self.featureModel, 
                                           self.translationVector, 
                                           self.rotationVector, 
                                           pixelCovariance)

        return covariance

    def calcCovariance(self, pixelCovariance=None, sigmaScale=4.0):
        # AUV homing and docking for remote operations
        # About covariance: https://manialabs.wordpress.com/2012/08/06/covariance-matrices-with-a-practical-example/
        # Article: https://www.sciencedirect.com/science/article/pii/S0029801818301367
        # Stack overflow: https://stackoverflow.com/questions/36618269/uncertainty-on-pose-estimate-when-minimizing-measurement-errors
        # Standard deviation = reprojection rmse/4: https://www.thoughtco.com/range-rule-for-standard-deviation-3126231
        #jacobian - 2*nPoints * 14
        # jacobian - [rotation, translation, focal lengths, principal point, dist coeffs]

        # TODO: is RMSE a godd approximation of the standard deviation? (same formula?)
        if pixelCovariance is None:
            # https://www.thoughtco.com/range-rule-for-standard-deviation-3126231
            # 2 - 95 %, 4 - 99 %
            #sigmaX = self.rmseMax/sigmaScale
            #sigmaY = self.rmseMax/sigmaScale
            #pixelCovariance = np.array([[sigmaX**2, 0], [0, sigmaY**2]])
            if self.pixelCovariances is None:
                self.calcRMSE()
            pixelCovariance = self.pixelCovariances

        covariance = calcPoseCovarianceFixedAxis(self.camera, 
                                                 self.featureModel, 
                                                 self.translationVector, 
                                                 self.rotationVector, 
                                                 pixelCovariance)

        self._covariance = covariance

        return self._covariance

    def validOrientation(self, yawRange, pitchRange, rollRange):
        if self._validOrientation:
            return self._validOrientation

        validYaw = -yawRange <= self.yaw <= yawRange
        validPitch = -pitchRange <= self.pitch <= pitchRange
        validRoll = -rollRange <= self.roll <= rollRange

        self._validOrientation = (validYaw, validPitch, validRoll)
        
        return self._validOrientation
        
class DSPoseEstimator:
    #def __init__(self, auv, dockingStation, camera, featureModel):
    def __init__(self, 
                 camera, 
                 featureModel,
                 ignoreRoll=False, 
                 ignorePitch=False, 
                 initFlag=cv.SOLVEPNP_EPNP,
                 flag=cv.SOLVEPNP_ITERATIVE,
                 refine=False):

        self.camera = camera
        self.featureModel = featureModel

        # from camera frame: yaw-pitch-roll (y, x, z)
        self.ignoreRoll = ignoreRoll
        self.ignorePitch = ignorePitch
        
        self.initFlag = initFlag
        self.flag = flag
        self.refine = refine

    def _estimatePoseEPnP(self, featurePoints, associatedPoints):
        associatedPoints = associatedPoints.reshape((len(associatedPoints), 1, 2))
            
        # On axis-angle: https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation#Relationship_to_other_representations
        success, rotationVector, translationVector = cv.solvePnP(featurePoints,
                                                                associatedPoints,
                                                                self.camera.cameraMatrix,
                                                                self.camera.distCoeffs,
                                                                useExtrinsicGuess=False,
                                                                tvec=None,
                                                                rvec=None,
                                                                flags=cv.SOLVEPNP_EPNP)
        return success, translationVector, rotationVector

    def _estimatePoseLM(self, featurePoints, associatedPoints, guessTrans, guessRot):
        guessTrans = guessTrans.copy().reshape((3, 1))
        guessRot = guessRot.copy().reshape((3, 1))
        success, rotationVector, translationVector = cv.solvePnP(featurePoints,
                                                                 associatedPoints,
                                                                 self.camera.cameraMatrix,
                                                                 self.camera.distCoeffs,
                                                                 useExtrinsicGuess=True,
                                                                 tvec=guessTrans,
                                                                 rvec=guessRot,
                                                                 flags=cv.SOLVEPNP_ITERATIVE)
        return success, translationVector, rotationVector

    def _estimatePose(self, 
                      associatedPoints,
                      estTranslationVec=None,
                      estRotationVec=None):

        featurePoints = np.array(list(self.featureModel.features[:, :3]))

        flag = self.flag
        if estTranslationVec is None:
            flag = self.initFlag


        if flag in ("opencv", "local", "global"):
            success = True
            pose = lmSolve(self.camera.cameraMatrix, 
                            associatedPoints, 
                            featurePoints, 
                            tVec=estTranslationVec, 
                            rVec=estRotationVec, 
                            jacobianCalc=self.flag,
                            maxIter=50,
                            mode="lm",
                            generate=False,
                            verbose=0).next()[0]
            translationVector, rotationVector = np.reshape(pose[:3], (3,1)), np.reshape(pose[3:], (3,1))
        
        elif flag == cv.SOLVEPNP_EPNP:
            success, translationVector, rotationVector = self._estimatePoseEPnP(featurePoints, associatedPoints)
            if self.refine:
                success, translationVector, rotationVector = self._estimatePoseLM(featurePoints, associatedPoints, translationVector, rotationVector)

        elif flag == cv.SOLVEPNP_ITERATIVE:
            if estRotationVec is None:
                estTranslationVec = np.array([0., 0., 1.])
                estRotationVec = np.array([0., 0., 0.])
            success, translationVector, rotationVector = self._estimatePoseLM(featurePoints, associatedPoints, estTranslationVec, estRotationVec)
            
        else:
            raise Exception("Invalid pose estimation flag {}".format(flag))

        return success, translationVector, rotationVector

    def estimatePose(self, 
                     associatedLightSources, 
                     estTranslationVec=None, 
                     estRotationVec=None):
        """
        featurePoints - points of the feature model
        associatedPoints - detected and associated points in the image
        pointCovariance - uncertainty of the detected points
        """
        associatedPoints = np.array([ls.center for ls in associatedLightSources], dtype=np.float32)
        success, translationVector, rotationVector = self._estimatePose(associatedPoints, estTranslationVec, estRotationVec)
                                                                 
        if not success:
            print("Pose estimation failed, no solution")
            return 

        translationVector = translationVector[:, 0]
        rotationVector = rotationVector[:, 0]
        
        ay, ax, az = R.from_rotvec(rotationVector).as_euler("YXZ")
        if self.ignorePitch:
            ax = 0
        if self.ignoreRoll:
            az = 0
        self.rotationVector = R.from_euler("YXZ", (ay, ax, az)).as_rotvec()

        rotMat = R.from_rotvec(self.rotationVector).as_dcm()
        camTranslationVector = np.matmul(rotMat.transpose(), -translationVector)
        camRotationVector = R.from_dcm(rotMat.transpose()).as_rotvec()

        return DSPose(translationVector, 
                      rotationVector, 
                      camTranslationVector,
                      camRotationVector,
                      associatedLightSources,
                      self.camera,
                      self.featureModel)

    def _old_estimatePose(self, 
                     associatedLightSources, 
                     estTranslationVec=None, 
                     estRotationVec=None):
        """
        featurePoints - points of the feature model
        associatedPoints - detected and associated points in the image
        pointCovariance - uncertainty of the detected points
        """
        associatedPoints = np.array([ls.center for ls in associatedLightSources], dtype=np.float32)

        featurePoints = np.array(list(self.featureModel.features[:, :3]))


        if estTranslationVec is not None:
            if self.flag in ("opencv", "local", "global"):
                success = True
                pose = lmSolve(self.camera.cameraMatrix, 
                                associatedPoints, 
                                featurePoints, 
                                tVec=estTranslationVec, 
                                rVec=estRotationVec, 
                                jacobianCalc=self.flag,
                                maxIter=50,
                                mode="lm",
                                generate=False,
                                verbose=0).next()[0]
                translationVector, rotationVector = np.reshape(pose[:3], (3,1)), np.reshape(pose[3:], (3,1))
            
            else:
                if self.flag == cv.SOLVEPNP_EPNP:
                    associatedPoints = associatedPoints.reshape((len(associatedPoints), 1, 2))

                guessTrans = estTranslationVec.copy().reshape((3, 1))
                guessRot = estRotationVec.copy().reshape((3, 1))
                # On axis-angle: https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation#Relationship_to_other_representations
                success, rotationVector, translationVector = cv.solvePnP(featurePoints,
                                                                        associatedPoints,
                                                                        self.camera.cameraMatrix,
                                                                        self.camera.distCoeffs,
                                                                        useExtrinsicGuess=True,
                                                                        tvec=guessTrans,
                                                                        rvec=guessRot,
                                                                        flags=self.flag)
        else:
            guessTrans = np.array([0., 0., 1.])
            guessRot = np.array([0., 0., 0.])

            if self.initFlag in ("opencv", "local", "global"):
                success = True
                pose = lmSolve(self.camera.cameraMatrix, 
                                associatedPoints, 
                                featurePoints, 
                                tVec=guessTrans, 
                                rVec=guessRot, 
                                jacobianCalc=self.initFlag,
                                maxIter=20,
                                mode="lm",
                                generate=False,
                                verbose=0).next()[0]

                translationVector, rotationVector = np.reshape(pose[:3], (3,1)), np.reshape(pose[3:], (3,1))
            else:
                if self.initFlag == cv.SOLVEPNP_EPNP:
                    associatedPoints = associatedPoints.reshape((len(associatedPoints), 1, 2))

                # On axis-angle: https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation#Relationship_to_other_representations
                success, rotationVector, translationVector = cv.solvePnP(featurePoints,
                                                                        associatedPoints,
                                                                        self.camera.cameraMatrix,
                                                                        self.camera.distCoeffs,
                                                                        useExtrinsicGuess=True,
                                                                        tvec=guessTrans.reshape((3,1)),
                                                                        rvec=guessRot.reshape((3,1)),
                                                                        flags=self.initFlag)

                                                                 
        if not success:
            print("Pose estimation failed, no solution")
            return 

        translationVector = translationVector[:, 0]
        rotationVector = rotationVector[:, 0]
        
        ay, ax, az = R.from_rotvec(rotationVector).as_euler("YXZ")
        if self.ignorePitch:
            ax = 0
        if self.ignoreRoll:
            az = 0
        self.rotationVector = R.from_euler("YXZ", (ay, ax, az)).as_rotvec()

        rotMat = R.from_rotvec(self.rotationVector).as_dcm()
        camTranslationVector = np.matmul(rotMat.transpose(), -translationVector)
        camRotationVector = R.from_dcm(rotMat.transpose()).as_rotvec()

        return DSPose(translationVector, 
                      rotationVector, 
                      camTranslationVector,
                      camRotationVector,
                      associatedLightSources,
                      self.camera,
                      self.featureModel)

    def findBestPose(self, associatedLightSourcePermutations, estDSPose=None, firstValid=False, mahaDistThresh=None):
        """
        Assume that if firstValid == False, we don't care about 
        mahalanobis distance and just want to find the best pose based on RMSE ratio RMSE/RMSE_MAX
        """
        estTranslationVec = None
        estRotationVec = None
        SInv = None
        if estDSPose:
            estTranslationVec = estDSPose.translationVector
            estRotationVec = estDSPose.rotationVector
            SInv = np.linalg.inv(estDSPose.covariance)

        poses = []
        rmseRatios = []
        attempts = 0
        N = len(associatedLightSourcePermutations)
        for associtatedLights in associatedLightSourcePermutations:
            attempts += 1
            dsPose = self.estimatePose(associtatedLights, 
                                    estTranslationVec=estTranslationVec, 
                                    estRotationVec=estRotationVec)
            dsPose.mahaDistThresh = mahaDistThresh
            if estDSPose:
                # If estDSPose is given, mahaDistThresh has to be given
                if mahaDistThresh is None:
                    raise Exception("Mahalanobis distance has to be given")

                mahaDist = dsPose.calcMahalanobisDist(estDSPose, SInv)
                validMahaDist = mahaDist < mahaDistThresh
            else:
                validMahaDist = True

            validReprError = dsPose.validReprError()
            valid = validReprError and validMahaDist

            if firstValid:
                # TODO: Temporary, remove
                if validReprError and not validMahaDist:
                    print("Maha dist outlier")

                if valid:
                    dsPose.attempts = attempts
                    dsPose.combinations = N
                    return dsPose
            else:
                if valid:
                    rmseRatios.append(float(dsPose.rmse)/dsPose.rmseMax)
                    poses.append(dsPose)

        if rmseRatios:
            # Find the best pose in terms of RMSE ratio
            bestIdx = np.argmin(rmseRatios)
            bestPose = poses[bestIdx]
            bestPose.attempts = attempts
            dsPose.combinations = N
            return bestPose

if __name__ =="__main__":
    from lolo_simulation.coordinate_system import CoordinateSystemArtist, CoordinateSystem
    from lolo_perception.perception_utils import plotAxis
    class CameraDummy:
        def __init__(self, cameraMatrix):
            self.cameraMatrix = cameraMatrix
            self.distCoeffs = np.zeros((1,4), dtype=np.float32)

    # Square
    objectPoints = np.array([[1., 1., 0], [-1., 1., 0], [1., -1., 0], [-1., -1., 0]])
    # 5 Lights
    #objectPoints = np.array([[1., 1., 0], [-1., 1., 0], [1., -1., 0], [-1., -1., 0], [0., 0., -1.0]])
    
    # Difficult pose
    tVec = np.array([-5., -5., 25.])
    rVec = R.from_euler("YXZ", (np.deg2rad(-110), np.deg2rad(90), np.deg2rad(90))).as_rotvec()
    
    # Random pose
    tVec = np.array([-5., -5., 25.])
    rVec = R.from_euler("YXZ", (np.deg2rad(-90), np.deg2rad(45), np.deg2rad(45))).as_rotvec()
    
    print(np.concatenate((tVec, rVec)).round(2))
    #rVec = np.array([0, np.pi, 0])
    cameraMatrix = np.array([[1400., 0., 639.],
                             [0., 1400., 359.],
                             [0., 0., 1]])

    detectedPoints = projectPoints2(tVec, rVec, cameraMatrix, objectPoints)
    sigmaX = 0
    sigmaY = 0
    detectedPoints[:, 0] += np.random.normal(0, sigmaX, detectedPoints[:, 0].shape)
    detectedPoints[:, 1] += np.random.normal(0, sigmaY, detectedPoints[:, 1].shape)

    camCS = CoordinateSystemArtist(CoordinateSystem(translation=(tVec[2]/2, 0, tVec[2]/2), euler=(-np.pi/2, 0, 0)))
    trueCS = CoordinateSystemArtist(CoordinateSystem(translation=tVec, euler=R.from_rotvec(rVec).as_euler("XYZ")))
    estCS = CoordinateSystemArtist()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlim(0, tVec[2])
    ax.set_ylim(0, tVec[2])
    ax.set_zlim(0, tVec[2])
    camCS.draw(ax, colors=("b",)*3)
    trueCS.drawRelative(ax, camCS.cs, colors=("g",)*3)
    
    fig2 = plt.figure()
    ax2 = fig2.gca()

    img = np.zeros((720, 1280, 3))
    for pose, lambdas, grads, errors in lmSolve(cameraMatrix, 
                                                detectedPoints, 
                                                objectPoints, 
                                                np.array([0., 0., 2.]), 
                                                np.array([0.0, 0., 0.]), 
                                                "opencv",
                                                maxIter=200,
                                                mode="lm",
                                                generate=True,
                                                verbose=1):

        for imgP in detectedPoints:
            imgP = imgP.round().astype(np.int32)
            cv.circle(img, tuple(imgP), 3, (255,0,0), -1)
        plotAxis(img, tVec, rVec, CameraDummy(cameraMatrix), objectPoints, scale=1)

        projPoints = projectPoints2(pose[:3], pose[3:], cameraMatrix, objectPoints)
        for p in projPoints:
            p = p.round().astype(np.int32)
            cv.circle(img, tuple(p), 3, (255,0,255), -1)
        plotAxis(img, pose[:3], pose[3:], CameraDummy(cameraMatrix), objectPoints, scale=1, opacity=0.002)

        ax2.cla()
        ax2.plot(errors)
        ax2.plot(lambdas)
        ax2.plot(grads)
        ax2.legend(["F(x)", "mu", "||g||"])

        estCS.cs.translation = pose[:3]
        estCS.cs.rotation = R.from_rotvec(pose[3:]).as_dcm()
        estCS.drawRelative(ax, camCS.cs, colors=("r",)*3, scale=0.5, alpha=0.5)

        print(pose.round(2))
        cv.imshow("img", img)
        cv.waitKey(10)
        plt.pause(0.01)
        img *= 0

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
                pose, lambdas, grads, errors in lmSolve(cameraMatrix, 
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

    while True:
        plt.pause(0.01)
        key = cv.waitKey(10)
        if key == ord("q"):
            break
    cv.destroyAllWindows()