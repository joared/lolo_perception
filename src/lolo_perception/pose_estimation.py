
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


from lolo_perception.reprojection_utils import calcPoseReprojectionRMSEThreshold

from numpy.linalg import lapack_lite
#lapack_routine = lapack_lite.dgesv

def calcMahalanobisDist(estDSPose, dsPose, SInv=None):
    if SInv is None:
        SInv = np.linalg.inv(estDSPose.covariance)

    translErr = estDSPose.translationVector - dsPose.translationVector
    translErr *= 0
    
    r1 = R.from_rotvec(estDSPose.rotationVector)
    r2 = R.from_rotvec(dsPose.rotationVector)
 
    #e = r1*r2.inv() # error in world (not this one, because it is fixed frame rotation)
    #e = r1.inv()*r2 # 2 wrt to 1 (should not be this one)
    rotationErr = r2.inv()*r1 # 1 wrt to 2
    rotationErr = rotationErr.as_euler("xyz") # fixed axis "xyz"

    err = np.array(list(translErr) + list(rotationErr))
    #print(np.matmul(err, np.linalg.inv(S)))
    mahaDist = np.matmul(np.matmul(err, SInv), err.transpose())
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

def calcPoseCovarianceFixedAxis(camera, featureModel, translationVector, rotationVector, pixelCovariance):

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

        self._rmse = None
        self._rmseMax = None

        self.mahaDist = None

        # increase this to keep track of how many valid poses have 
        # been detected in sequence
        self.detectionCount = 1

        # how many attempts the pose estimation algorithm did until it found the pose
        self.attempts = 0

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

    def calcRMSE(self):
        if self._rmse or self._rmseMax:
            raise Exception("NONONO")

        errs, rmse = reprojectionError(self.translationVector, 
                                       self.rotationVector, 
                                       self.camera, 
                                       self.featureModel.features, 
                                       np.array([ls.center for ls in self.associatedLightSources], dtype=np.float32))
        rmseMaxModel = calcPoseReprojectionRMSEThreshold(self.translationVector, 
                                                         self.rotationVector, 
                                                         self.camera, 
                                                         self.featureModel,
                                                         showImg=False)

        lightSourceReprojections = np.array([ls.radius for ls in self.associatedLightSources], dtype=np.float32)
        #rmseMaxLightsource = np.sqrt(np.sum(lightSourceReprojections**2)/len(self.associatedLightSources))

        self._rmse = rmse
        self._rmseMax = rmseMaxModel
        #self.rmseMax += rmseMaxLightsource # light source rejection is unrealiable since large areas (noise) may not be rejected 
        self._rmseMax += 1 # add 1 pixel for far distance detections
        return self._rmse, self._rmseMax

    def calcCovariance(self, pixelCovariance=None):
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
            sigmaX = self.rmseMax/4
            sigmaY = self.rmseMax/4
            pixelCovariance = np.array([[sigmaX**2, 0], [0, sigmaY**2]])

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
                 flag=cv.SOLVEPNP_ITERATIVE):

        self.camera = camera
        self.featureModel = featureModel

        # from camera frame: yaw-pitch-roll (y, x, z)
        self.ignoreRoll = ignoreRoll
        self.ignorePitch = ignorePitch
        
        self.flag = flag

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

        if self.flag == cv.SOLVEPNP_EPNP:
            associatedPoints = associatedPoints.reshape((len(associatedPoints), 1, 2))

        featurePoints = np.array(list(self.featureModel.features[:, :3]))
        if estTranslationVec is not None:
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
                                                                    flags=cv.SOLVEPNP_ITERATIVE)
        else:
            guessTrans = np.array([[0.], [0.], [5]])
            #print("Guess trans", guessTrans)
            guessRot = np.array([[0.], [0.], [0.]])
            # On axis-angle: https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation#Relationship_to_other_representations
            success, rotationVector, translationVector = cv.solvePnP(featurePoints,
                                                                    associatedPoints,
                                                                    self.camera.cameraMatrix,
                                                                    self.camera.distCoeffs,
                                                                    useExtrinsicGuess=True,
                                                                    tvec=guessTrans,
                                                                    rvec=guessRot,
                                                                    flags=self.flag)
                                                                 
        if not success:
            print("Pose estimation failed")
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
        for associtatedLights in associatedLightSourcePermutations:
            attempts += 1
            dsPose = self.estimatePose(associtatedLights, 
                                    estTranslationVec=estTranslationVec, 
                                    estRotationVec=estRotationVec)
            
            if estDSPose:
                # If estDSPose is given, mahaDistThresh has to be given
                if mahaDistThresh is None:
                    raise Exception("Mahalanobis distance has to be given")

                mahaDist = dsPose.calcMahalanobisDist(estDSPose, SInv)
                validMahaDist = mahaDist < mahaDistThresh
            else:
                validMahaDist = True

            validRMSE = dsPose.rmse < dsPose.rmseMax
            valid = validRMSE and validMahaDist

            if firstValid:
                # TODO: Temporary, remove
                if validRMSE and not validMahaDist:
                    print("Maha dist outlier")

                if valid:
                    dsPose.attempts = attempts
                    return dsPose
            else:
                if validRMSE:
                    rmseRatios.append(float(dsPose.rmse)/dsPose.rmseMax)
                    poses.append(dsPose)

        if rmseRatios:
            # Find the best pose in terms of RMSE ratio
            bestIdx = np.argmin(rmseRatios)
            bestPose = poses[bestIdx]
            bestPose.attempts = attempts
            return bestPose

if __name__ =="__main__":
    pass
