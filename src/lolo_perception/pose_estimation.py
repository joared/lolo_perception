
import cv2 as cv
import numpy as np
import time
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from scipy.spatial.transform import Rotation as R, rotation
from scipy.spatial.transform import Slerp
import scipy

class DSPoseEstimator:
    #def __init__(self, auv, dockingStation, camera, featureModel):
    def __init__(self, 
                 camera, 
                 featureModel=None,
                 ignoreRoll=False, 
                 ignorePitch=False, 
                 flag=cv.SOLVEPNP_ITERATIVE, 
                 calcCovariance=True):

        self.camera = camera
        self.featureModel = featureModel

        # from camera frame: yaw-pitch-roll (y, x, z)
        self.ignoreRoll = ignoreRoll
        self.ignorePitch = ignorePitch
        
        self.flag = flag
        self._calcCovariance = calcCovariance

        # pose of docking station wrt camera
        self.translationVector = np.zeros(3)
        self.rotationVector = np.zeros(3)
        self.poseCov = np.zeros((6, 6))

        # pose of camera wrt to docking station
        self.camTranslationVector = np.zeros(3)
        self.camRotationVector = np.zeros(3)
        self.camPoseCov = np.zeros((6, 6)) # rotation the same but translation is nonlinear

    def calcCovariance(self, 
                       featurePoints, 
                       translationVector, 
                       rotationVector, 
                       pixelCovariance):
        # AUV homing and docking for remote operations
        # About covariance: https://manialabs.wordpress.com/2012/08/06/covariance-matrices-with-a-practical-example/
        # Article: https://www.sciencedirect.com/science/article/pii/S0029801818301367
        # Stack overflow: https://stackoverflow.com/questions/36618269/uncertainty-on-pose-estimate-when-minimizing-measurement-errors
        #jacobian - 2*nPoints * 14
        # jacobian - [rotation, translation, focal lengths, principal point, dist coeffs]

        _, jacobian = cv.projectPoints(featurePoints, 
                                        rotationVector.reshape((3, 1)), 
                                        translationVector.reshape((3, 1)), 
                                        self.camera.cameraMatrix, 
                                        self.camera.distCoeffs)

        rotJ = jacobian[:, :3]
        transJ = jacobian[:, 3:6]
        J = np.hstack((transJ, rotJ)) # reorder covariance as used in PoseWithCovarianceStamped

        # How to rotate covariance: https://robotics.stackexchange.com/questions/2556/how-to-rotate-covariance

        sigma = scipy.linalg.block_diag(*[pixelCovariance]*len(featurePoints))
        sigmaInv = np.linalg.inv(sigma)
        try:
            poseCov = np.linalg.inv(np.matmul(np.matmul(J.transpose(), sigmaInv), J))
        except np.linalg.LinAlgError as e:
            print("Singular matrix")

        return poseCov

    def update(self, 
               featurePoints, 
               associatedPoints, 
               pixelCovariance, 
               estTranslationVec=None, 
               estRotationVec=None):
        """
        Assume that predict have been called just before this
        featurePoints - points of the feature model [m]
        associatedPoints - detected and associated points in the image [m]
        pointCovariance - uncertainty of the detected points
        """

        if self.flag == cv.SOLVEPNP_EPNP:
            associatedPoints = associatedPoints.reshape((len(associatedPoints), 1, 2))

        featurePoints = np.array(list(featurePoints[:, :3]))
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
            guessTrans = np.array([[0.], [0.], [1]])
            print("Guess trans", guessTrans)
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

        self.translationVector = translationVector[:, 0]
        self.rotationVector = rotationVector[:, 0]
        
        ay, ax, az = R.from_rotvec(self.rotationVector).as_euler("YXZ")
        if self.ignorePitch:
            ax = 0
        if self.ignoreRoll:
            az = 0
        self.rotationVector = R.from_euler("YXZ", (ay, ax, az)).as_rotvec()

        rotMat = R.from_rotvec(self.rotationVector).as_dcm()
        self.camTranslationVector = np.matmul(rotMat.transpose(), -self.translationVector)
        self.camRotationVector = R.from_dcm(rotMat.transpose()).as_rotvec()

        self.poseCov = np.zeros((6,6))
        if self._calcCovariance:
            self.poseCov = self.calcCovariance(featurePoints, 
                                               self.translationVector, 
                                               self.rotationVector, 
                                               pixelCovariance)

        return (self.translationVector, 
                self.rotationVector, 
                self.poseCov,
                self.camTranslationVector,
                self.camRotationVector)


if __name__ =="__main__":
    pass