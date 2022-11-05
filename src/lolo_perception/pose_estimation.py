
import cv2 as cv
import numpy as np
import time
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import scipy
from lolo_perception.perception_utils import projectPoints, reprojectionError
from lolo_perception.image_processing import featureAssociation, withinContour


from lolo_perception.reprojection_utils import calcPoseReprojectionRMSEThreshold, calcPoseReprojectionThresholds
from lolo_perception.pose_estimation_utils import lmSolve, interactionMatrix

from numpy.linalg import lapack_lite

from scipy import sparse
#lapack_routine = lapack_lite.dgesv

def ellipseToCovariance(majorAxis, minorAxis, angle, confidence):
    angle = np.deg2rad(angle)
    majorAxis = majorAxis/np.sqrt(confidence)
    minorAxis = minorAxis/np.sqrt(confidence)
    varX = majorAxis**2 * np.cos(angle)**2 + minorAxis**2 * np.sin(angle)**2
    varY = majorAxis**2 * np.sin(angle)**2 + minorAxis**2 * np.cos(angle)**2
    cov = (majorAxis**2 - minorAxis**2) * np.sin(angle) * np.cos(angle)
    return np.array([[varX, cov], [cov, varY]])

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
    #translErr[2] = 0 # ignore z translation for now
    
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

    #sigmaInv = np.linalg.inv(sigma)
    sigma = sparse.csr_matrix(sigma)
    sigmaInv = sparse.linalg.inv(sigma)
    #sigmaInv = fastInverse(np.reshape(sigma, (sigma.shape[0], sigma.shape[1], 1))).reshape(sigma.shape)
    try:
        #mult = np.matmul(np.matmul(J.transpose(), sigmaInv), J)
        #covariance = np.linalg.inv(mult)
        
        # new sparse calculation
        J = sparse.csr_matrix(J)
        JTranspose = sparse.csr_matrix(J.transpose())
        mult =  sparse.csr_matrix.dot(sparse.csr_matrix.dot(JTranspose, sigmaInv), J)
        covariance = sparse.linalg.inv(mult).toarray()
        
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

        # These are calculated in init() at the moment
        self.pixelCovariances = None
        self.reprErrors = None
        self.reprErrorsMax = None
        self.rmse = None                # TODO: Shuld these be
        self.rmseMax = None             # use in findBestPose()?

        # image MD threshold
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

        # TODO: this might not be the cleanest way
        # Calculate reprojection error and pixel covariances
        self.init()

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

    def init(self):
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
                       minThreshold=2.0 # 3.0
                       #minThreshold=0.7071
                       ):

        for err, pCov in zip(self.reprErrors, self.pixelCovariances):
            try:
                pCovInv = np.linalg.inv(pCov)
            except np.linalg.LinAlgError as e:
                print("Singular image covariance matrix")
                return False # TODO: why was this True before???

            #if np.linalg.norm(err) < 6.0:
            #    continue
            #else:
            #    return False

            mahaDistSquare = np.matmul(np.matmul(err.transpose(), pCovInv), err)
            if mahaDistSquare > self.chiSquaredConfidence:
                return False

        return True

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
            
            pixelCovariance = self.pixelCovariances

            #pixelCovariance = []
            # New stuff
            #for ls, pCov in zip(self.associatedLightSources, self.pixelCovariances):
            #pixelCovariance = np.array(pixelCovariance)
            

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
        rotationVector = R.from_euler("YXZ", (ay, ax, az)).as_rotvec()

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

    def findBestPose(self, associatedLightSourceCombinations, estDSPose=None, firstValid=False, mahaDistThresh=None):
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

            if mahaDistThresh:
                SInv = np.linalg.inv(estDSPose.covariance)
           
        poses = []
        rmseRatios = []
        attempts = 0
        N = len(associatedLightSourceCombinations)
        for associtatedLights in associatedLightSourceCombinations:
            attempts += 1
            dsPose = self.estimatePose(associtatedLights, 
                                    estTranslationVec=estTranslationVec, 
                                    estRotationVec=estRotationVec)
            dsPose.mahaDistThresh = mahaDistThresh
            if estDSPose and mahaDistThresh:
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