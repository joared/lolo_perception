
import cv2 as cv
import numpy as np
import lolo_perception.py_logging as logging
from scipy.spatial.transform import Rotation as R
import scipy
from lolo_perception.perception_utils import projectPoints, reprojectionError


from lolo_perception.reprojection_utils import calcPoseReprojectionThresholds
from lolo_perception.pose_estimation_utils import interactionMatrix


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
        self.covariance = None
        self.yaw, self.pitch, self.roll = R.from_rotvec(self.rotationVector).as_euler("YXZ")

        self.camTranslationVector = camTranslationVector
        self.camRotationVector = camRotationVector
        self.camCovariance = None # not used at the moment
        #self.camYaw, self.camPitch, self.camRoll = R.from_rotvec(self.camRotationVector).as_euler("YXZ") # not used

        self.associatedLightSources = associatedLightSources
        self.camera = camera
        self.featureModel = featureModel

        # image MD threshold
        self.chiSquaredConfidence = 5.991
        self.mahaDist = None
        self.mahaDistThresh = None
        
        # increase this to keep track of how many valid poses have 
        # been detected in sequence
        self.detectionCount = 0
        # how many attempts the pose estimation algorithm did until it found the pose
        self.attempts = 0
        # number of total combinations that the pose algorithm considered
        self.combinations = 0 

        self.reprErrors, self.rmse = reprojectionError(self.translationVector, 
                                                       self.rotationVector, 
                                                       self.camera, 
                                                       self.featureModel.features, 
                                                       np.array([ls.center for ls in self.associatedLightSources], dtype=np.float32))

        self.pixelCovariances = calcImageCovariance(self.translationVector, 
                                                    self.rotationVector, 
                                                    self.camera, 
                                                    self.featureModel,
                                                    confidence=self.chiSquaredConfidence)        

        self.reprErrorsMax = calcPoseReprojectionThresholds(self.translationVector, 
                                                            self.rotationVector, 
                                                            self.camera, 
                                                            self.featureModel)

        self.rmseMax = np.sqrt(np.sum(self.reprErrorsMax**2)/np.product(self.reprErrorsMax.shape))


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
        diffs = self.reprErrorsMax - abs(self.reprErrors)
        certainties = diffs / self.reprErrorsMax
        certainty = np.min(certainties)
        return certainty


    def validReprError(self, 
                       minThreshold=2.0
                       #minThreshold=0.7071
                       ):

        for err, pCov in zip(self.reprErrors, self.pixelCovariances):
            try:
                pCovInv = np.linalg.inv(pCov)
            except np.linalg.LinAlgError as e:
                logging.warn("Singular image covariance matrix")
                return False

            # Any reprojection below imThreshold is OK
            #if np.linalg.norm(err) < minThreshold:
            #    continue
            #else:
            #    return False

            mahaDistSquare = np.matmul(np.matmul(err.transpose(), pCovInv), err)
            if mahaDistSquare > self.chiSquaredConfidence:
                return False

        return True


    def calcCovariance(self, pixelCovariance=None):
        if pixelCovariance is None:
            pixelCovariance = self.pixelCovariances

        # TODO: this is not the convention of the covariance that ROS Pose messages uses
        # ROS says that fixed axis (euler) should be used. However this covariance is calculated 
        # using the rotation vector.
        covariance = calcPoseCovarianceFixedAxis(self.camera, 
                                                 self.featureModel, 
                                                 self.translationVector, 
                                                 self.rotationVector, 
                                                 pixelCovariance)

        self.covariance = covariance

        logging.debug("Calculated covariance for pose")
        return self.covariance

    def validOrientation(self, yawRange, pitchRange, rollRange):

        validYaw = -yawRange <= self.yaw <= yawRange
        validPitch = -pitchRange <= self.pitch <= pitchRange
        validRoll = -rollRange <= self.roll <= rollRange

        return (validYaw, validPitch, validRoll)
    

class DSPoseEstimator:

    FLAG_LM = "LM"
    FLAG_EPNP = "EPNP"

    def __init__(self, 
                 camera, 
                 featureModel,
                 initFlag=cv.SOLVEPNP_EPNP,
                 flag=cv.SOLVEPNP_ITERATIVE,
                 refine=False):

        self.camera = camera
        self.featureModel = featureModel
        
        assert initFlag in (self.FLAG_LM, self.FLAG_EPNP), "Invalid pose estimation flag {}".format(initFlag)
        self.initFlag = initFlag
        assert flag in (self.FLAG_LM, self.FLAG_EPNP), "Invalid pose estimation flag {}".format(flag)
        self.flag = flag
        self.currentFlag = self.initFlag
        self.refine = refine


    def __repr__(self):
        s = self.currentFlag
        if self.refine and self.currentFlag == self.FLAG_EPNP:
            s += "+" + self.FLAG_LM
        return s

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
        
        self.currentFlag = flag

        if flag == self.FLAG_EPNP:
            success, translationVector, rotationVector = self._estimatePoseEPnP(featurePoints, associatedPoints)
            
            if self.refine:
                success, translationVector, rotationVector = self._estimatePoseLM(featurePoints, associatedPoints, translationVector, rotationVector)

        elif flag == self.FLAG_LM:
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

        associatedPoints = np.array([ls.center for ls in associatedLightSources], dtype=np.float32)
        success, translationVector, rotationVector = self._estimatePose(associatedPoints, estTranslationVec, estRotationVec)
                                                                 
        if not success:
            logging.info("Pose estimation failed, no solution")
            return 

        translationVector = translationVector[:, 0]
        rotationVector = rotationVector[:, 0]

        rotMat = R.from_rotvec(rotationVector).as_dcm()
        camTranslationVector = np.matmul(rotMat.transpose(), -translationVector)
        camRotationVector = R.from_dcm(rotMat.transpose()).as_rotvec()

        return DSPose(translationVector, 
                      rotationVector, 
                      camTranslationVector,
                      camRotationVector,
                      associatedLightSources,
                      self.camera,
                      self.featureModel)

    def findBestPose(self, associatedLightSourceCombinations, N, estDSPose=None, firstValid=False, mahaDistThresh=None):
        estTranslationVec = None
        estRotationVec = None
        SInv = None 
        if estDSPose:
            estTranslationVec = estDSPose.translationVector
            estRotationVec = estDSPose.rotationVector
            
            if mahaDistThresh:
                _, SInv = cv.invert(estDSPose.covariance, flags=cv.DECOMP_CHOLESKY)
           
        poses = []
        rmseRatios = []
        attempts = 0

        for associtatedLights in associatedLightSourceCombinations:
            
            attempts += 1
            dsPose = self.estimatePose(associtatedLights, 
                                       estTranslationVec=estTranslationVec, 
                                       estRotationVec=estRotationVec)

            dsPose.mahaDistThresh = mahaDistThresh
            if estDSPose and mahaDistThresh:
                mahaDist = dsPose.calcMahalanobisDist(estDSPose, SInv)
                dsPose.mahaDist = mahaDist
                validMahaDist = mahaDist < mahaDistThresh
            else:
                validMahaDist = True

            validReprError = dsPose.validReprError()
            valid = validReprError and validMahaDist

            if firstValid:
                # TODO: Temporary, remove
                if validReprError and not validMahaDist:
                    logging.info("Mahalanobis outlier, rejecting pose")

                if valid:
                    dsPose.attempts = attempts
                    dsPose.combinations = N
                    dsPose.calcCovariance()
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
    pass