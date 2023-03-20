import cv2 as cv
import numpy as np
import lolo_perception.py_logging as logging
from scipy.spatial.transform import Rotation as R
from lolo_perception.pose_estimation import projectPoints, reprojectionError, calcImageCovariance, calcMahalanobisDist, calcPoseCovarianceEuler, calcPoseCovariance


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

        # TODO: might be better to use the detected contour as image covariance
        # instead of using the spherical projection with predetermined radius (detectionTolerance and displacement uncertainty)
        self.pixelCovariances = calcImageCovariance(self.translationVector, 
                                                    self.rotationVector, 
                                                    self.camera, 
                                                    self.featureModel,
                                                    confidence=self.chiSquaredConfidence)


    def calcMahalanobisDist(self, estDSPose, SInv=None):
        mahaDist = calcMahalanobisDist(estDSPose, self, SInv)
        self.mahaDist = mahaDist
        return self.mahaDist


    def reProject(self):
        return projectPoints(self.translationVector, 
                             self.rotationVector, 
                             self.camera, 
                             self.featureModel.features)


    def validReprError(self):

        for err, pCov in zip(self.reprErrors, self.pixelCovariances):
            try:
                pCovInv = np.linalg.inv(pCov)
            except np.linalg.LinAlgError as e:
                logging.warn("Singular image covariance matrix")
                return False

            # TODO: Any reprojection below imThreshold is OK
            #minThreshold=0.7071
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

        # TODO: Not sure if the euler covariance calculation is correct
        #covariance = calcPoseCovarianceEuler(self.camera, 
        #                                     self.featureModel, 
        #                                     self.translationVector, 
        #                                     self.rotationVector, 
        #                                     pixelCovariance)

        # Calculates the covariance using the rotation vector
        covariance = calcPoseCovariance(self.camera, 
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