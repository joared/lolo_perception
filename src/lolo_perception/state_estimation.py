
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
                 calcCovariance=False):
        self.camera = camera
        self.featureModel = featureModel

        # both of these needs to be clearly defined what they mean
        self.ignoreRoll = ignoreRoll
        self.ignorePitch = ignorePitch
        
        self.flag = flag
        self._calcCovariance = calcCovariance

        self.imageCovariance = np.array([[self.camera.pixelWidth*2, 0], 
                                         [0, self.camera.pixelHeight*2]])

        self.poseAcquired = False
        
        # pose relative to the pose at last update
        # updated when predict is called
        self.auvTranslation = np.zeros(3)
        self.auvRotation = np.zeros((3,3))
        self.auvTranslationCov = np.zeros(3)
        self.auvRotationCov = np.zeros((3,3))

        # relative pose of the docking station w.r.t the AUV
        # updated when update is called
        self.dsTranslation = np.zeros(3)
        self.dsTranslationCov = np.zeros(3)
        self.dsRotation = np.zeros((3,3))
        self.dsRotationCov = np.zeros((3,3))
        
        # estimated velocity of the docking station in the body frame
        # updated when update is called
        self.dsVelocity = np.zeros(6)
        self.dsVelocityCov = np.zeros(6)
        

    def _propagateAuv(self, translation, rotation, vel, dt, translationCov=np.zeros(3), rotationCov=np.zeros((3,3))):
        v = np.matmul(rotation, np.array(vel[:3])*dt)
        w = np.array(vel[3:])*dt
        
        translation[0] += v[0]
        translation[1] += v[1]
        translation[2] += v[2]

        r = R.from_matrix(rotation)
        rw = R.from_rotvec(w)
        r = r*rw # multiplying from the right makes the rotation in the local frame

        rotation = np.array(list(r.as_matrix()))

        return translation, translationCov, rotation, rotationCov

    def predict(self, auvVel, dt):
        """
        predict the relative pose between the AUV and the docking station
        The AUV and docking station pose is relative the AUV pose from the last update
        """
        self.self.dsVelocity
        pass

    def _transformFeaturePoseToDSPose(self):
        """
        Measurements are the pose of the feature model (the lights) w.r.t the camera.
        We need to transform the pose using the model of the docking station
        to estimate the pose/velocity of the docking station.
        """
        pass

    def _filterPose(self, translation, rotation):
        self._transformFeaturePoseToDSPose()
        return translation, rotation

    def _estimateDsState(self, translation, rotation, dt):
        lastVel = self.dsVelocity[:3]
        vel = (translation - self.dsTranslation)/dt
        alpha = 1
        vel = vel*alpha + lastVel*(1-alpha)
        # vel = (translation - self.dsTranslation)/dt
        vel = np.concatenate([vel, np.zeros(3)])

        return translation, rotation, vel

    def update(self, featurePoints, associatedPoints, pointCovariance, dt, estTranslationVec=None, estRotationVec=None):
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
            guessTrans = np.array([[0.], [0.], [1.]])
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
                                                                 
        projectedPoints, jacobian = cv.projectPoints(featurePoints, 
                                                     rotationVector, 
                                                     translationVector, 
                                                     self.camera.cameraMatrix, 
                                                     self.camera.distCoeffs)
        
        poseCov = np.zeros((6,6))

        if self._calcCovariance:
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

        if not success:
            input("DIDNT SUCCEED")
        rotationMatrix = R.from_rotvec(rotationVector[:, 0]).as_dcm()
        translation, rotation = self._filterPose(translationVector[:, 0], rotationMatrix)
        translation, rotation, vel = self._estimateDsState(translation, rotation, dt)
        
        self.dsTranslation = translation
        self.dsRotation = rotation
        self.dsVelocity = vel
        #return translationVector[:, 0], rotationVector[:, 0]

        ay, ax, az = R.from_dcm(self.dsRotation).as_euler("YXZ")
        if self.ignorePitch:
            ax = 0
        if self.ignoreRoll:
            az = 0
        self.dsRotation = R.from_euler("YXZ", (ay, ax, az)).as_dcm()
        #if self.ignoreRoll and self.ignorePitch:
        #    ay, ax, az = R.from_dcm(self.dsRotation).as_euler("YXZ")
        #    self.dsRotation = R.from_euler("YXZ", (ay, 0, 0)).as_dcm()
        #if self.ignoreRoll and not self.ignorePitch:
        #    ay, az, ax = R.from_dcm(self.dsRotation).as_euler("YZX")
        #    self.dsRotation = R.from_euler("YZX", (ay, 0, ax)).as_dcm()
         # remove roll info

        return self.dsTranslation, R.from_dcm(self.dsRotation).as_rotvec(), poseCov


if __name__ =="__main__":
    pass