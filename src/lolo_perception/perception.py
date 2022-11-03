import cv2 as cv
import yaml
import numpy as np
import itertools
from scipy.spatial.transform import Rotation as R

from lolo_perception.image_processing import featureAssociation, featureAssociationSquare, featureAssociationSquareImproved, featureAssociationSquareImprovedWithFilter
from lolo_perception.pose_estimation import DSPoseEstimator
from lolo_perception.light_source_deteector import LocalPeak, ModifiedHATS
from lolo_perception.perception_utils import plotPoseImageInfo, regionOfInterest


class AssociateCombinationGenerator:
    def __init__(self, lsCombs, associateFunc):
        self._comb = lsCombs
        self._assFunc = associateFunc

    def __iter__(self):
        for lsComb in self._comb:
            lsCombAss = self._assFunc(lsComb)[0]
            if lsCombAss:
                yield lsCombAss

    def __len__(self):
        return len(self._comb)

class Perception:
    def __init__(self, camera, featureModel, hatsMode=ModifiedHATS.MODE_VALLEY):
        self.camera = camera
        self.featureModel = featureModel

        # Choose association method
        def assFunc(comb):
            #return featureAssociation(self.featureModel.features, comb)
            #return featureAssociationSquare(self.featureModel.features, comb, self.camera.resolution)
            #return featureAssociationSquareImproved(self.featureModel.features, comb)
            return featureAssociationSquareImprovedWithFilter(self.featureModel.features, comb, p=0.07)

        self.associationFunc = assFunc
        
        if self.camera.resolution[1] > 1280:
            print("Using large kernels")
            minArea = 80 #40
            blurKernelSize = 11 # 11
            localMaxKernelSize = 21 # 11
        elif self.camera.resolution[1] == 1280:
            print("Using small kernels")
            minArea = 80 #20
            blurKernelSize = 11 #9# 9/11
            localMaxKernelSize = 11 # 11
        else:
            print("Using tiny kernels")
            minArea = 80
            blurKernelSize = 5 #5
            localMaxKernelSize = 11 # 11

        minCircleExtent = 0.2 # 0.55 for underwater, 0 when above ground (set low during heavy motion blur)
        self.maxIntensityChange = 0.7
        self.toHATSScale = 3 #3 # change to HATS when min area > minArea*areaScale
        self.toPeakScale = 1.5 # change back to peak when min area < minArea*areaScale
        ignoreMax = True
        self.hatsFeatureExtractor = ModifiedHATS(len(self.featureModel.features), 
                                                 peakMargin=0, # this should be zero if using MODE_VALLEY or MODE_PEAK
                                                 minArea=minArea, 
                                                 minRatio=minCircleExtent, # might not be good for outlier detection, convex hull instead?
                                                 maxIntensityChange=self.maxIntensityChange,
                                                 blurKernelSize=blurKernelSize,
                                                 mode=hatsMode,
                                                 ignorePeakAtMax=ignoreMax,
                                                 showHistogram=False)

        # Use local peak finding to initialize and when light sources are small
        # This feature extractor sorts candidates based on intensity and then area
        pMin = .975 # .8
        pMax = .975 #.975
        maxIter = 100
        self.peakFeatureExtractor = LocalPeak(len(self.featureModel.features), 
                                              kernelSize=localMaxKernelSize, # 11 for 720p, 25 for 1080p
                                              pMin=pMin, #0.93 set pMin = pMax for fixed p
                                              pMax=pMax, # 0.975
                                              maxIntensityChange=self.maxIntensityChange,
                                              minArea=minArea,
                                              minCircleExtent=minCircleExtent,
                                              blurKernelSize=blurKernelSize,  # 5 for 720p, 11 for 1080p
                                              ignorePAtMax=ignoreMax,
                                              maxIter=maxIter)
        
        # start with peak
        self.featureExtractor = self.peakFeatureExtractor

        # max additional light source candidates that will be considered by 
        # the feature extractor.
        # 6 gives a minimum frame rate of about 1 when all 
        # combinations of candidates are iterated over
        self.maxAdditionalCandidates = 6 # 6

        # margin of the region of interest when pose has been aquired
        #self.roiMargin = int(round(0.0626*self.camera.cameraMatrix[0, 0]))
        self.roiMargin = 90

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
        # an orientation outside the valid range 
        self.validOrientationRange = [np.radians(90), np.radians(30), np.radians(30)]

        # mahalanobis distance threshold
        self.mahalanobisDistThresh = 12.592

        # This should be sent in as an argument (deducted from some state estimator)
        # and is used to update the estimated pose (estDSPose) for better prediction of the ROI
        self.estCameraPoseVector = None # [x, y, z, ax, ay, az]

        self.detectionCountThresh = 10

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

        # choose feature extractor (should be part of image processing module)
        if estDSPose:
            if self.featureExtractor == self.peakFeatureExtractor:
                changeToHATS = all([ls.area > self.toHATSScale*self.hatsFeatureExtractor.minArea for ls in estDSPose.associatedLightSources])
            else:
                changeToHATS = all([ls.area > self.toPeakScale*self.hatsFeatureExtractor.minArea for ls in estDSPose.associatedLightSources])
        
            if changeToHATS:
                self.featureExtractor = self.hatsFeatureExtractor
            else:
                self.featureExtractor = self.peakFeatureExtractor
        else:
            self.featureExtractor = self.peakFeatureExtractor

        # extract light source candidates from image
        _, candidates, roiCntUpdated = self.featureExtractor(gray, 
                                                            maxAdditionalCandidates=self.maxAdditionalCandidates, 
                                                            estDSPose=estDSPose,
                                                            roiMargin=self.roiMargin, 
                                                            drawImg=processedImg)

        # should be part of image processing module
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
            # The first valid pose is good enough in most cases 
            dsPose = self.poseEstimator.findBestPose(associatedCombinations, 
                                                     estDSPose=estDSPose, 
                                                     firstValid=True,
                                                     mahaDistThresh=self.mahalanobisDistThresh)

            if dsPose:
                # TODO: Do this is pose_estimation.py?
                if estDSPose:
                    dsPose.detectionCount += estDSPose.detectionCount
                    if dsPose.detectionCount >= self.detectionCountThresh:
                        poseAquired = True
                if calcCovariance:
                    dsPose.calcCovariance()
            else:
                print("Pose estimation failed")

        progress = 0
        if dsPose:
            progress = dsPose.detectionCount/float(self.detectionCountThresh)
            progress = min(1, progress)

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
                          self,
                          "{}".format("HATS" if self.featureExtractor == self.hatsFeatureExtractor else "Peak"),
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

        self.processedImg = processedImg
        self.poseImg = poseImg
        return dsPose, poseAquired, candidates, processedImg, poseImg

    @staticmethod
    def fromYaml(yamlPath):
        raise Exception("Not implemented")
        with open(yamlPath, "r") as file:
            perceptionData = yaml.safe_load(file)
        

if __name__ == '__main__':
    pass
