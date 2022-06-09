import cv2 as cv
import time
import numpy as np
import itertools
from scipy.spatial.transform import Rotation as R

from lolo_perception.feature_extraction import featureAssociation, featureAssociationSquare, featureAssociationSquareImproved, localMax, refineCentroidGradient, LightSourceTrackInitializer, AdaptiveThreshold2, AdaptiveThresholdPeak, ModifiedHATS, LocalMaxHATS
from lolo_perception.pose_estimation import DSPoseEstimator, calcMahalanobisDist
from lolo_perception.perception_utils import plotPoseImageInfo, regionOfInterest

class AssociateCombinationGenerator:
    def __init__(self, lsCombs, associateFunc):
        self._comb = lsCombs
        self._assFunc = associateFunc

    def __iter__(self):
        for lsComb in self._comb:
            yield self._assFunc(lsComb)[0]

    def __len__(self):
        return len(self._comb)

class Perception:
    def __init__(self, camera, featureModel, hatsMode=ModifiedHATS.MODE_VALLEY):
        self.camera = camera
        self.featureModel = featureModel

        # Choose association method
        def assFunc(comb):
            return featureAssociation(self.featureModel.features, comb)
            #return featureAssociationSquare(self.featureModel.features, comb, self.camera.resolution)
            #return featureAssociationSquareImproved(self.featureModel.features, comb)

        self.associationFunc = assFunc

        res1080p = True
        if res1080p:
            minArea = 40
            blurKernelSize = 11 # 11
            localMaxKernelSize = 11 # 25
        else:
            minArea = 20
            blurKernelSize = 5
            localMaxKernelSize = 5 # 11

        minCircleExtent = 0.1 # 0.2
        maxIntensityChange = 0.7
        self.areaScale = 3 # change to HATS when all areas > minArea*areaScale 
        self.hatsFeatureExtractor = ModifiedHATS(len(self.featureModel.features), 
                                                 peakMargin=0, # this should be zero if using MODE_VALLEY or MODE_PEAK
                                                 minArea=minArea, 
                                                 minRatio=minCircleExtent, # might not be good for outlier detection, convex hull instead?
                                                 maxIntensityChange=maxIntensityChange,
                                                 blurKernelSize=blurKernelSize,
                                                 thresholdType=cv.THRESH_BINARY,
                                                 mode=hatsMode,
                                                 ignorePeakAtMax=True,
                                                 showHistogram=False)

        # Use local peak finding to initialize and when light sources are small
        # This feature extractor sorts candidates based on intensity and then area
        pMin = .8
        pMax = .975
        maxIter = 100
        self.peakFeatureExtractor = AdaptiveThresholdPeak(len(self.featureModel.features), 
                                                          kernelSize=localMaxKernelSize, # 11 for 720p, 25 for 1080p
                                                          pMin=pMin, #0.93 set pMin = pMax for fixed p
                                                          pMax=pMax, # 0.975
                                                          maxIntensityChange=maxIntensityChange,
                                                          minArea=minArea,
                                                          minCircleExtent=minCircleExtent,
                                                          blurKernelSize=blurKernelSize,  # 5 for 720p, 11 for 1080p
                                                          ignorePAtMax=True,
                                                          maxIter=maxIter)
        
        # start with peak
        self.featureExtractor = self.peakFeatureExtractor

        # max additional light source candidates that will be considered by 
        # the feature extractor.
        # 6 gives a minimum frame rate of about 1 when all 
        # combinations of candidates are iterated over
        self.maxAdditionalCandidates = 6 # 6

        # margin of the region of interest when pose has been aquired
        self.roiMargin = int(round(0.0626*self.camera.cameraMatrix[0, 0]))

        # This scaling might not be accurate, better to adjust manually
        #minPatchRadius = int(self.camera.cameraMatrix[0, 0]*self.camera.resolution[1]/69120.0)
        #radius = int(minPatchRadius * 1.2)
        #maxPatchRadius = int(minPatchRadius * 7.2)
        #maxMovement = int(minPatchRadius * 1.5)
        minPatchRadius = self.roiMargin
        radius = 20
        maxPatchRadius = 100
        maxMovement = 20

        # Initialize light source trackers
        self.lightSourceTracker = LightSourceTrackInitializer(radius=radius, 
                                                              maxPatchRadius=maxPatchRadius, 
                                                              minPatchRadius=minPatchRadius,
                                                              p=0.975,
                                                              maxIntensityChange=0.7,
                                                              maxMovement=maxMovement)
        # Number of trackers in the initialization phase (stage 1 and 2)
        self.nLightSourceTrackers = 20

        # Pose estimator that calculates poses from detected light sources
        initPoseEstimationFlag = cv.SOLVEPNP_ITERATIVE # cv.SOLVEPNP_ITERATIVE or cv.SOLVEPNP_EPNP 
        poseEstimationFlag = cv.SOLVEPNP_ITERATIVE
        self.poseEstimator = DSPoseEstimator(self.camera, 
                                             self.featureModel,
                                             ignoreRoll=False, 
                                             ignorePitch=False,
                                             initFlag=initPoseEstimationFlag, 
                                             flag=poseEstimationFlag,
                                             refine=False)

        # valid orientation range [yawMinMax, pitchMinMax, rollMinMax]. Currently not used to disregard
        # invalid poses, but the axis and region of interest will be shown in red when a pose has 
        # an orientation within the valid range 
        self.validOrientationRange = [np.radians(80), np.radians(30), np.radians(30)]

        # mahalanobis distance threshold
        self.mahalanobisDistThresh = 12.592

        # This should be sent in as an argument (deducted from some state estimator)
        # and is used to update the estimated pose (estDSPose) for better prediction of the ROI
        self.estCameraPoseVector = None # [x, y, z, ax, ay, az]

        # stages of the perception module:
        # 1 - initialize light source trackers
        # 2 - track light sources
        # 3 - initialize pose from light source trackers
        # ----------------------
        # 4 - acquire pose
        # 5 - track pose
        self.startStage = 4 # 1 or 4
        self.stage = self.startStage
        self.stage2Iterations = 15#15 # Tracking light sources for this amount of frames
        self.stage4Iterations = 10 # Acquiring pose for this amount of frames

        # Access images from perception_node
        self.processedImg = None
        self.poseImg = None

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
        #rotK = 5e-5
        rotK = 5e-4
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
                     estCameraPoseVector=None,
                     colorCoeffs=None,
                     calcCovariance=False):

        #estCameraPoseVector = None ###

        # information about contours extracted from the feature extractor is plotted in this image
        processedImg = imgColor.copy()
        
        # pose information and ROI is plotted in this image
        poseImg = imgColor.copy()

        # gray image to be processed
        if colorCoeffs is not None:
            m = np.array(colorCoeffs).reshape((1,3))
            gray = cv.transform(imgColor, m)
        else:
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
                self.lightSourceTracker.update(gray, drawImg=processedImg)
                candidates = self.lightSourceTracker.getCandidates(len(self.featureModel.features) + self.maxAdditionalCandidates)
                roiCntUpdated = roiCnt
            elif self.stage in (4,5):
                # choose feature extractor
                
                if estDSPose:
                    if self.featureExtractor == self.peakFeatureExtractor:
                        changeToHATS = all([ls.area > self.areaScale*self.hatsFeatureExtractor.minArea for ls in estDSPose.associatedLightSources])
                    else:
                        changeToHATS = all([ls.area > self.areaScale/2.0*self.hatsFeatureExtractor.minArea for ls in estDSPose.associatedLightSources])
                        
                    if changeToHATS:
                        self.featureExtractor = self.hatsFeatureExtractor
                    else:
                        self.featureExtractor = self.peakFeatureExtractor
                else:
                    self.featureExtractor = self.peakFeatureExtractor

                #self.featureExtractor = self.localMaxHATS
                #self.featureExtractor = self.hatsFeatureExtractor # TODO: remove
                #self.featureExtractor = self.peakFeatureExtractor #TODO remove

                # extract light source candidates from image
                _, candidates, roiCntUpdated = self.featureExtractor(gray, 
                                                                    maxAdditionalCandidates=self.maxAdditionalCandidates, 
                                                                    estDSPose=estDSPose,
                                                                    roiMargin=self.roiMargin, 
                                                                    drawImg=processedImg)

            if len(candidates) >= len(self.featureModel.features):


                # N_C! / (N_F! * (N_C-N_F)!)
                lightCandidateCombinations = list(itertools.combinations(candidates, len(self.featureModel.features)))
                     
                # TODO: does this take a lot of time?
                # sort combinations
                if self.featureExtractor == self.hatsFeatureExtractor:
                    # sort by summed circle extent
                    lightCandidateCombinations.sort(key=lambda comb: sum([ls.circleExtent() for ls in comb]), reverse=True)
                elif self.featureExtractor == self.peakFeatureExtractor:
                    # sort by summed intensity
                    # TODO: not sure how much this improves
                    #lightCandidateCombinations.sort(key=lambda comb: (sum([ls.intensity for ls in comb]), sum([ls.area for ls in comb])), reverse=True)
                    lightCandidateCombinations.sort(key=lambda comb: (sum([ls.intensity for ls in comb]), sum([ls.circleExtent() for ls in comb])), reverse=True)
                else:
                    # using some other 
                    print("sorting by intensity, then area")
                    lightCandidateCombinations.sort(key=lambda comb: (sum([ls.intensity for ls in comb]), sum([ls.area for ls in comb])), reverse=True)

                associatedCombinations = AssociateCombinationGenerator(lightCandidateCombinations, 
                                                                       self.associationFunc)

                # findBestPose finds poses based on reprojection RMSE
                # the featureModel specifies the placementUncertainty and detectionTolerance
                # which determines the maximum allowed reprojection RMSE
                if estDSPose and self.featureExtractor == self.hatsFeatureExtractor:
                    # if we use HATS, presumably we are close and we want to find the best pose
                    dsPose = self.poseEstimator.findBestPose(associatedCombinations, 
                                                             estDSPose=estDSPose, 
                                                             firstValid=True, 
                                                             mahaDistThresh=self.mahalanobisDistThresh)
                else:
                    # if we use local peak, presumably we are far away, 
                    # and the first valid pose is good enough in most cases 
                    dsPose = self.poseEstimator.findBestPose(associatedCombinations, 
                                                             estDSPose=estDSPose, 
                                                             firstValid=True,
                                                             mahaDistThresh=self.mahalanobisDistThresh)

                if dsPose:

                    # TODO: Do this is pose_estimation.py?
                    if estDSPose:
                        # mahanalobis distance check
                        # mahaDist = dsPose.calcMahalanobisDist(estDSPose)
                        #if mahaDist <= self.mahalanobisDistThresh:
                            # disregard poses with large mahalaobis distance
                            #pass
                        dsPose.detectionCount += estDSPose.detectionCount
                    if calcCovariance:
                        dsPose.calcCovariance()
                    poseAquired = True

                    # TODO: Use refined centers?
                    #centroids = refineCentroidGradient(gray, [ls.cnt for ls in dsPose.associatedLightSources], ksize=7)
                    #for ls, centroid in zip(candidates, centroids):
                    #    #ls.center = (ls.center[0] + centroid[0])/2, (ls.center[1] + centroid[1])/2 
                    #    cv.circle(poseImg, centroid, 1, (255,0,255), 1)

                else:
                    print("Pose estimation failed")

        if self.stage in (1, 2):
            # In stage 1 and two, it could be useful to see the 
            # lightsource trackers in the poseImg
            poseImg = processedImg.copy()

        progress = 0
        if self.stage == 2:
            progress = self.lightSourceTracker.iterations/float(self.stage2Iterations)
        elif self.stage == 4:
            if dsPose:
                progress = dsPose.detectionCount/float(self.stage4Iterations)
        elif self.stage == 5:
            progress = 1

        poseEstFlag = self.poseEstimator.flag if estDSPose else self.poseEstimator.initFlag

        if poseEstFlag == cv.SOLVEPNP_ITERATIVE:
            poseEstMethod = "L-M"
        elif poseEstFlag == cv.SOLVEPNP_EPNP:
            poseEstMethod = "EPnP"
            if self.poseEstimator.refine:
                poseEstMethod += "+L-M"
        else:
            poseEstMethod = poseEstFlag
            
        # plots pose axis, ROI, light sources etc. 
        plotPoseImageInfo(poseImg,
                          "{} S{}".format("HATS" if self.featureExtractor == self.hatsFeatureExtractor else "Peak", self.stage),
                          dsPose,
                          self.camera,
                          self.featureModel,
                          poseAquired,
                          self.validOrientationRange,
                          poseEstMethod,
                          roiCnt,
                          roiCntUpdated,
                          progress=progress,
                          fixedAxis=False)

        if roiCntUpdated is not None:# TODO: remove
            #processedImg = imgColor.copy()
            cv.drawContours(processedImg, [roiCntUpdated], -1, (0,255,0), 3) 
        #processedImg = self.hatsFeatureExtractor.img

        self.processedImg = processedImg
        self.poseImg = poseImg
        return dsPose, poseAquired, candidates, processedImg, poseImg


if __name__ == '__main__':
    pass
