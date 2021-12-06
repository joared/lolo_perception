
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
        self.calcCovariance = calcCovariance

        self.imageCovariance = np.array([[self.camera.pixelWidth*2, 0], 
                                         [0, self.camera.pixelHeight*2]])

        self.poseAcquired = False

        # relative pose of the docking station w.r.t the AUV
        # updated when update is called
        self.translationVector = np.zeros(3)
        self.rotationVector = np.zeros(3)
        self.poseCov = np.zeros((6, 6))

    def calcCovariance(self, imagePointCovariance):
        # AUV homing and docking for remote operations
        # About covariance: https://manialabs.wordpress.com/2012/08/06/covariance-matrices-with-a-practical-example/
        # Article: https://www.sciencedirect.com/science/article/pii/S0029801818301367
        # Stack overflow: https://stackoverflow.com/questions/36618269/uncertainty-on-pose-estimate-when-minimizing-measurement-errors
        #jacobian - 2*nPoints * 14
        # jacobian - [rotation, translation, focal lengths, principal point, dist coeffs]
        rotJ = jacobian[:, :3]
        transJ = jacobian[:, 3:6]
        J = np.hstack((transJ, rotJ)) # reorder covariance as used in PoseWithCovarianceStamped

        # How to rotate covariance: https://robotics.stackexchange.com/questions/2556/how-to-rotate-covariance

        sigma = scipy.linalg.block_diag(*[pointCovariance]*len(featurePoints))
        sigmaInv = np.linalg.inv(sigma)
        try:
            poseCov = np.linalg.inv(np.matmul(np.matmul(J.transpose(), sigmaInv), J))
        except np.linalg.LinAlgError as e:
            print("Singular matrix")

    def update(self, featurePoints, associatedPoints, pointCovariance, estTranslationVec=None, estRotationVec=None):
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
            guessTrans = np.array([[0.], [0.], [1/2.8*1e6*self.camera.pixelWidth]])
            print("Guess trans", guessTrans)
            guessRot = np.array([[0.], [np.pi], [0.]])
            # On axis-angle: https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation#Relationship_to_other_representations
            success, rotationVector, translationVector = cv.solvePnP(featurePoints,
                                                                    associatedPoints,
                                                                    self.camera.cameraMatrix,
                                                                    self.camera.distCoeffs,
                                                                    useExtrinsicGuess=True,
                                                                    tvec=guessTrans,
                                                                    rvec=guessRot,
                                                                    flags=self.flag)
                                                                 

        
        poseCov = np.zeros((6,6))

        if self.calcCovariance:
            # AUV homing and docking for remote operations
            # About covariance: https://manialabs.wordpress.com/2012/08/06/covariance-matrices-with-a-practical-example/
            # Article: https://www.sciencedirect.com/science/article/pii/S0029801818301367
            # Stack overflow: https://stackoverflow.com/questions/36618269/uncertainty-on-pose-estimate-when-minimizing-measurement-errors
            #jacobian - 2*nPoints * 14
            # jacobian - [rotation, translation, focal lengths, principal point, dist coeffs]
            _, jacobian = cv.projectPoints(featurePoints, 
                                           rotationVector, 
                                           translationVector, 
                                           self.camera.cameraMatrix, 
                                           self.camera.distCoeffs)
            rotJ = jacobian[:, :3]
            transJ = jacobian[:, 3:6]
            J = np.hstack((transJ, rotJ)) # reorder covariance as used in PoseWithCovarianceStamped

            # How to rotate covariance: https://robotics.stackexchange.com/questions/2556/how-to-rotate-covariance

            sigma = scipy.linalg.block_diag(*[pointCovariance]*len(featurePoints))
            sigmaInv = np.linalg.inv(sigma)
            try:
                poseCov = np.linalg.inv(np.matmul(np.matmul(J.transpose(), sigmaInv), J))
            except np.linalg.LinAlgError as e:
                print("Singular matrix")

        if not success:
            input("DIDNT SUCCEED")

        self.translationVector = translationVector[:, 0]
        self.rotationVector = rotationVector[:, 0]
        self.poseCov = poseCov

        ay, ax, az = R.from_rotvec(self.rotationVector).as_euler("YXZ")
        if self.ignorePitch:
            ax = 0
        if self.ignoreRoll:
            az = 0
        self.dsRotation = R.from_euler("YXZ", (ay, ax, az)).as_dcm()

        return self.translationVector, self.rotationVector, poseCov


if __name__ =="__main__":
    pass