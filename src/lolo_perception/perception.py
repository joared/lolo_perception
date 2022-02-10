import cv2 as cv
import time
import numpy as np
import itertools

from lolo_perception.feature_extraction import featureAssociation, AdaptiveThreshold2, AdaptiveThresholdPeak
from lolo_perception.pose_estimation import DSPoseEstimator
from lolo_perception.perception_utils import plotPoseImageInfo

class Perception:
    def __init__(self, camera, featureModel):
        self.camera = camera
        self.featureModel = featureModel
        
        # Use HATS when light sources are "large"
        # This feature extractor sorts candidates based on area
        self.hatsFeatureExtractor = AdaptiveThreshold2(len(self.featureModel.features), 
                                                       marginPercentage=0.004, # For 255, this wil correspond to 254 
                                                       minArea=10, 
                                                       minRatio=0.2,
                                                       thresholdType=cv.THRESH_BINARY)
        
        # Use local peak finding to initialize and when light sources are small
        # This feature extractor sorts candidates based on intensity and then area
        self.peakFeatureExtractor = AdaptiveThresholdPeak(len(self.featureModel.features), 
                                                          kernelSize=11, 
                                                          p=0.97)
        
        self.featureExtractor = self.peakFeatureExtractor

        # max additional light source candidates that will be considered by 
        # the feature extractor.
        # 6 gives a minimum frame rate of about 0.3 when all 
        # permutations of candidates are iterated over
        self.maxAdditionalCandidates = 6

        # margin of the region of interest when pose has been aquired
        self.roiMargin = 50

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

    def estimatePose(self, 
                     imgColor, 
                     estDSPose=None):

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

        if not estDSPose:
            # if we are not given an estimated pose, we initialize
            self.featureExtractor = self.peakFeatureExtractor
        else:
            if all([ls.area > self.hatsFeatureExtractor.minArea for ls in estDSPose.associatedLightSources]):
                # change to hats
                self.featureExtractor = self.hatsFeatureExtractor
        
        # extract light source candidates from image
        _, candidates, roiCnt = self.featureExtractor(gray, 
                                                      maxAdditionalCandidates=self.maxAdditionalCandidates, 
                                                      estDSPose=estDSPose,
                                                      roiMargin=self.roiMargin, 
                                                      drawImg=processedImg)

        # return if pose has been aquired
        poseAquired = False
        # estimated pose
        dsPose = None

        if len(candidates) >= len(self.featureModel.features):

            lightCandidatePermutations = list(itertools.combinations(candidates, len(self.featureModel.features)))
            associatedPermutations = [featureAssociation(self.featureModel.features, candidates)[0] for candidates in lightCandidatePermutations]
                
            # findBestPose finds poses based on reprojection RMSE
            # the featureModel specifies the placementUncertainty and detectionTolerance
            # which determines the maximum allowed reprojection RMSE
            if estDSPose and self.featureExtractor == self.hatsFeatureExtractor:
                # if we use HATS, presumably we are close and we want to find the best pose
                dsPose = self.poseEstimator.findBestPose(associatedPermutations, firstValid=False)
            else:
                # if we use local peak, presumably we are far away, 
                # and the first valid pose is good enough in most cases 
                dsPose = self.poseEstimator.findBestPose(associatedPermutations, firstValid=True)

            if dsPose:
                if dsPose.rmse < dsPose.rmseMax:
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
                cv.putText(poseImg, 
                        "HATS" if self.featureExtractor == self.hatsFeatureExtractor else "Peak", 
                        (poseImg.shape[1]/2, 45), 
                        cv.FONT_HERSHEY_SIMPLEX, 
                        2, 
                        color=(255,0,255), 
                        thickness=2, 
                        lineType=cv.LINE_AA)
            else:
                print("Pose estimation failed")

        
        if dsPose:
            # plots pose axis, ROI, light sources etc.  
            plotPoseImageInfo(poseImg,
                              dsPose,
                              self.camera,
                              self.featureModel,
                              poseAquired,
                              self.validOrientationRange,
                              roiCnt)

        elapsed = time.time()-start
        cv.putText(poseImg, 
                   "FPS {}".format(round(1./elapsed, 1)), 
                   (poseImg.shape[1]*2/3, 45), 
                   cv.FONT_HERSHEY_SIMPLEX, 
                   2, 
                   color=(0,255,0), 
                   thickness=2, 
                   lineType=cv.LINE_AA)

        # TODO: return candidates?
        return dsPose, poseAquired, candidates, processedImg, poseImg


if __name__ == '__main__':
    pass