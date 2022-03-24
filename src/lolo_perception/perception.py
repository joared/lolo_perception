from concurrent.futures import process
import cv2 as cv
import time
import numpy as np
import itertools
from scipy.spatial.transform import Rotation as R

from lolo_perception.feature_extraction import featureAssociation, regionOfInterest, LightSourceTrackInitializer, AdaptiveThreshold2, AdaptiveThresholdPeak
from lolo_perception.pose_estimation import DSPoseEstimator, calcMahalanobisDist
from lolo_perception.perception_utils import plotPoseImageInfo

class Perception:
    def __init__(self, camera, featureModel):
        self.camera = camera
        self.featureModel = featureModel
        
        # This scaling might not be accurate, better to adjust manually
        #minPatchRadius = int(self.camera.cameraMatrix[0, 0]*self.camera.resolution[1]/69120.0)
        #radius = int(minPatchRadius * 1.2)
        #maxPatchRadius = int(minPatchRadius * 7.2)
        #maxMovement = int(minPatchRadius * 1.5)

        minPatchRadius = 7
        radius = 7
        maxPatchRadius = 20
        maxMovement = 20


        # Initialize light source trackers
        self.lightSourceTracker = LightSourceTrackInitializer(radius=radius, 
                                                              maxPatchRadius=maxPatchRadius, 
                                                              minPatchRadius=minPatchRadius,
                                                              p=0.97,
                                                              maxIntensityChange=0.7,
                                                              maxMovement=maxMovement)
        # Number of trackers in the initialization phase (stage 1 and 2)
        self.nLightSourceTrackers = 20

        # Use HATS when light sources are "large"
        # This feature extractor sorts candidates based on area
        self.hatsFeatureExtractor = AdaptiveThreshold2(len(self.featureModel.features), 
                                                       marginPercentage=0.01, 
                                                       minArea=10, 
                                                       minRatio=0.2,
                                                       thresholdType=cv.THRESH_BINARY)

        # Use local peak finding to initialize and when light sources are small
        # This feature extractor sorts candidates based on intensity and then area
        self.peakFeatureExtractor = AdaptiveThresholdPeak(len(self.featureModel.features), 
                                                          kernelSize=11, 
                                                          p=0.97,
                                                          maxIntensityChange=0.7)
        
        # start with peak
        self.featureExtractor = self.peakFeatureExtractor

        # max additional light source candidates that will be considered by 
        # the feature extractor.
        # 6 gives a minimum frame rate of about 0.3 when all 
        # permutations of candidates are iterated over
        self.maxAdditionalCandidates = 6

        # margin of the region of interest when pose has been aquired
        self.roiMargin = int(self.camera.cameraMatrix[0, 0]*self.camera.resolution[1]/25700.)

        # Pose estimator that calculates poses from detected light sources
        self.poseEstimator = DSPoseEstimator(self.camera, 
                                             self.featureModel,
                                             ignoreRoll=False, 
                                             ignorePitch=False, 
                                             flag=cv.SOLVEPNP_ITERATIVE)

        # valid orientation range [yawMinMax, pitchMinMax, rollMinMax]. Currently not used to disregard
        # invalid poses, but the axis and region of interest will be shown in red when a pose has 
        # an orientation within the valid range 
        self.validOrientationRange = [np.radians(35), np.radians(30), np.radians(15)]

        # mahalanobis distance threshold
        self.mahalanobisDistThresh = 12.592

        # This should be sent in as an argument (deducted from some state estimator)
        # and is used to update the estimated pose (estDSPose) for better prediction of the ROI
        self.estCameraPoseVector = None # [x, y, z, ax, ay, az]

        # stages of the perception module:
        # 1 - initialize light source trackers
        # 2 - track light sources
        # 3 - initialize pose
        # 4 - acquire pose
        # 5 - track pose
        self.startStage = 1
        self.stage = self.startStage
        self.stage2Iterations = 15 # Tracking light sources for this amount of frames
        self.stage4Iterations = 10 # Acquiring pose for this amount of frames

    def updateDSPoseFromNewCameraPose(self, estDSPose, estCameraPoseVector):
        if self.estCameraPoseVector is not None and estCameraPoseVector is not None:
            print("Updating estDSPose")
            c1ToGTransl = self.estCameraPoseVector[:3]
            c1ToGRot = R.from_rotvec(self.estCameraPoseVector[3:])
            c2ToGTransl = estCameraPoseVector[:3]
            c2ToGRot = R.from_rotvec(estCameraPoseVector[3:])
            dsToC1Transl = estDSPose.translationVector
            dsToC1Rot = R.from_rotvec(estDSPose.rotationVector)

            gToC1Rot = c1ToGRot.inv()
            gToC2Rot = c2ToGRot.inv()
            c2ToC1Transl = gToC1Rot.apply(c2ToGTransl-c1ToGTransl)
            #c2ToC1Transl[2] = 0 # disregard displacement in z
            c1ToC2Rot = gToC2Rot*c1ToGRot

            dsToC2Transl = c1ToC2Rot.apply(dsToC1Transl-c2ToC1Transl)
            dsToC2Rot = gToC2Rot*c1ToGRot*dsToC1Rot

            estDSPose.translationVector = dsToC2Transl
            estDSPose.rotationVector = dsToC2Rot.as_rotvec()

        # increase covariance
        rotK = 5e-5
        covNew = estDSPose.covariance + np.eye(6)*rotK
        estDSPose._covariance = covNew

        return estDSPose

    def _updateStage(self, estDSPose):
        if self.stage in (1, 2):
            # if estDSPose is given from something else than the perception module
            #if estDSPose:
            #    self.stage = 3

            nTrackers = len(self.lightSourceTracker.trackers)

            if self.stage == 1:
                if nTrackers >= len(self.featureModel.features):
                    self.stage = 2

            elif self.stage == 2:
                if nTrackers < len(self.featureModel.features):
                    self.stage = self.startStage

                elif self.lightSourceTracker.iterations >= self.stage2Iterations or nTrackers == len(self.featureModel.features):
                    self.stage = 3

        elif self.stage in (3, 4, 5):
            if estDSPose:
                if self.stage == 3:
                    self.stage = 4

                elif self.stage == 4:
                    if estDSPose.detectionCount >= self.stage4Iterations:
                        self.stage = 5

            else:
                 self.stage = self.startStage
        else:
            raise Exception("Invalid stage '{}'".format(self.stage))

    def estimatePose(self, 
                     imgColor, 
                     estDSPose=None,
                     estCameraPoseVector=None):

        #estCameraPoseVector = None ###

        # Keeping track of FPS
        start = time.time()

        # information about contours extracted from the feature extractor is plotted in this image
        processedImg = imgColor.copy()
        
        # pose information and ROI is plotted in this image
        poseImg = imgColor.copy()

        # gray image to be processed
        gray = cv.cvtColor(imgColor, cv.COLOR_BGR2GRAY)

        # ROI contour
        roiCnt = None
        roiCntUpdated = None

        self._updateStage(estDSPose)


        # TODO: move this to perception_node?
        if estDSPose:
            featurePointsGuess = estDSPose.reProject()
            roiMargin = self.roiMargin
            roiMargin += max([ls.radius for ls in estDSPose.associatedLightSources])
            _, roiCnt = regionOfInterest(featurePointsGuess, wMargin=roiMargin, hMargin=roiMargin)

            estDSPose = self.updateDSPoseFromNewCameraPose(estDSPose, estCameraPoseVector)
        self.estCameraPoseVector = estCameraPoseVector

        # return if pose has been aquired or not
        poseAquired = False
        # new estimated pose
        dsPose = None

        if self.stage == 1:
            self.lightSourceTracker.reset()
            self.featureExtractor = self.peakFeatureExtractor
            _, candidates, roiCntUpdated = self.peakFeatureExtractor(gray, 
                                                                     maxAdditionalCandidates=self.nLightSourceTrackers, 
                                                                     estDSPose=estDSPose,
                                                                     roiMargin=self.roiMargin, 
                                                                     drawImg=processedImg)
            self.lightSourceTracker.update(gray, [ls.center for ls in candidates], drawImg=processedImg)
            candidates = self.lightSourceTracker.getCandidates()

        elif self.stage == 2:
            self.lightSourceTracker.update(gray, drawImg=processedImg)
            candidates = self.lightSourceTracker.getCandidates()

        elif self.stage in (3, 4, 5):
            if self.stage == 3:
                candidates = self.lightSourceTracker.getCandidates(len(self.featureModel.features) + self.maxAdditionalCandidates)
                roiCntUpdated = roiCnt
            elif self.stage in (4,5):
                # choose feature extractor
                self.featureExtractor = self.peakFeatureExtractor
                if estDSPose:
                    if all([ls.area > self.hatsFeatureExtractor.minArea for ls in estDSPose.associatedLightSources]):
                        # change to hats
                        self.featureExtractor = self.hatsFeatureExtractor

                # extract light source candidates from image
                _, candidates, roiCntUpdated = self.featureExtractor(gray, 
                                                                    maxAdditionalCandidates=self.maxAdditionalCandidates, 
                                                                    estDSPose=estDSPose,
                                                                    roiMargin=self.roiMargin, 
                                                                    drawImg=processedImg)

            if len(candidates) >= len(self.featureModel.features):

                lightCandidatePermutations = list(itertools.combinations(candidates, len(self.featureModel.features)))
                associatedPermutations = [featureAssociation(self.featureModel.features, candidates)[0] for candidates in lightCandidatePermutations]
                    
                # findBestPose finds poses based on reprojection RMSE
                # the featureModel specifies the placementUncertainty and detectionTolerance
                # which determines the maximum allowed reprojection RMSE
                if estDSPose and self.featureExtractor == self.hatsFeatureExtractor:
                    # if we use HATS, presumably we are close and we want to find the best pose
                    dsPose = self.poseEstimator.findBestPose(associatedPermutations, 
                                                             estDSPose=estDSPose, 
                                                             firstValid=False, 
                                                             mahaDistThresh=self.mahalanobisDistThresh)
                else:
                    # if we use local peak, presumably we are far away, 
                    # and the first valid pose is good enough in most cases 
                    dsPose = self.poseEstimator.findBestPose(associatedPermutations, 
                                                             estDSPose=estDSPose, 
                                                             firstValid=True,
                                                             mahaDistThresh=self.mahalanobisDistThresh)

                if dsPose:
                    # TODO: remove this, it's done in pose_estimation
                    if dsPose.rmse < dsPose.rmseMax:
                        
                        # TODO: Do this is pose_estimation.py?
                        if estDSPose:
                            # mahanalobis distance check
                            # mahaDist = dsPose.calcMahalanobisDist(estDSPose)
                            #if mahaDist <= self.mahalanobisDistThresh:
                                # disregard poses with large mahalaobis distance
                                #pass
                            dsPose.detectionCount += estDSPose.detectionCount
                        poseAquired = True

                    rmseColor = (0,255,0) if poseAquired else (0,0,255)
                    cv.putText(poseImg, 
                            "RMSE: {} < {}".format(round(dsPose.rmse, 2), round(dsPose.rmseMax, 2)), 
                            (20, 200), 
                            cv.FONT_HERSHEY_SIMPLEX, 
                            1, 
                            color=rmseColor, 
                            thickness=2, 
                            lineType=cv.LINE_AA)
                    cv.putText(poseImg, 
                            "RMSE certainty: {}".format(round(1-dsPose.rmse/dsPose.rmseMax, 2)), 
                            (20, 220), 
                            cv.FONT_HERSHEY_SIMPLEX, 
                            1, 
                            color=rmseColor, 
                            thickness=2, 
                            lineType=cv.LINE_AA)

                else:
                    print("Pose estimation failed")

        if self.stage in (1, 2):
            # In stage 1 and two, it could be useful to see the 
            # lightsource trackers in the poseImg
            poseImg = processedImg.copy()

        cv.putText(poseImg, 
                "{} S{}".format("HATS" if self.featureExtractor == self.hatsFeatureExtractor else "Peak", self.stage), 
                (int(poseImg.shape[1]/2.5), 45), 
                cv.FONT_HERSHEY_SIMPLEX, 
                2, 
                color=(255,0,255), 
                thickness=2, 
                lineType=cv.LINE_AA)

        progress = 0
        if self.stage == 2:
            progress = self.lightSourceTracker.iterations/float(self.stage2Iterations)
        elif self.stage == 4:
            if dsPose:
                progress = dsPose.detectionCount/float(self.stage4Iterations)
        elif self.stage == 5:
            progress = 1

        # plots pose axis, ROI, light sources etc.  
        plotPoseImageInfo(poseImg,
                          dsPose,
                          self.camera,
                          self.featureModel,
                          poseAquired,
                          self.validOrientationRange,
                          roiCnt,
                          roiCntUpdated,
                          progress=progress)

        elapsed = time.time()-start
        cv.putText(poseImg, 
                   "FPS {}".format(round(1./elapsed, 1)), 
                   (int(poseImg.shape[1]*4/5), 25), 
                   cv.FONT_HERSHEY_SIMPLEX, 
                   1, 
                   color=(0,255,0), 
                   thickness=2, 
                   lineType=cv.LINE_AA)

        # TODO: return candidates?
        return dsPose, poseAquired, candidates, processedImg, poseImg


if __name__ == '__main__':
    pass
