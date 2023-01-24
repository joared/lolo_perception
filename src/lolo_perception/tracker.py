import cv2 as cv
import time
import yaml
import os
import numpy as np
import itertools
import py_logging as logging
import math
from scipy.spatial.transform import Rotation as R

from lolo_perception.image_processing import featureAssociation, featureAssociationSquareImprovedWithFilter, regionOfInterest
from lolo_perception.pose_estimator import DSPoseEstimator
from lolo_perception.light_source_detector import getLightSourceDetectorFromName
from lolo_perception.plotting_utils import plotPoseImageInfoSimple, plotPoseImageInfo
from lolo_perception.utils import Timer

class LightSourceCombinationGenerator:

    ASSOCIATE_SIMPLE = "simple"
    ASSOCIATE_SQUARE_W_FILTER = "square_filter"

    def __init__(self, featureModel, associationFlag, sort=True):

        self.featureModel = featureModel

        assert associationFlag in (self.ASSOCIATE_SIMPLE, self.ASSOCIATE_SQUARE_W_FILTER), "Invalid association flag"
        
        # Choose association method
        def assFunc(comb):
            if associationFlag == self.ASSOCIATE_SIMPLE:
                return featureAssociation(self.featureModel.features, comb)
            elif associationFlag == self.ASSOCIATE_SQUARE_W_FILTER:
                return featureAssociationSquareImprovedWithFilter(self.featureModel.features, comb, p=0.33) # p = 0.07, 0.33
            else:
                raise Exception("Invalid association flag")
            
        self.assFunc = assFunc
        self.sort = sort

    def preFilter(self, candidates):
        filteredCandidates = candidates
        # filteredCandidates = []
        # for ls in candidates:
        #     if ls.intensity < 180 and ls.radius > 10:
        #         # invalid
        #         pass
        #     else:
        #         # valid
        #         filteredCandidates.append(ls)

        return filteredCandidates

    def group(self, candidates):
        # TODO: Grouping the candidates (based on features such as area, size and intensity)
        # can speed up the iteration.

        # At least one group
        groups = [candidates]

        return groups

    def preAssociationFilter(self, candidates):
        # TODO: Filter out candidates before association
        return True

    def postAssociationFilter(self, associated):
        # TODO: Filter out candidates after association
        return True

    def _generate(self, candidates):
        # TODO: group candidates
        for groupedCandidates in self.group(candidates):
            # Note: if len(groupedCandidates) < number of features, there will be no combinations
            for combination in itertools.combinations(groupedCandidates, len(self.featureModel.features)):
                yield combination

    def generate(self, candidates):
        
        # TODO: This should be done by the light source detector
        candidates = self.preFilter(candidates)

        lightCandidateCombinations = self._generate(candidates)

        if self.sort:
            lightCandidateCombinations = sorted(lightCandidateCombinations,
                                                key=lambda comb: (sum([ls.intensity for ls in comb]), sum([ls.circleExtent() for ls in comb])), 
                                                reverse=True)           

        for comb in lightCandidateCombinations:
            
            # TODO: 
            if not self.preAssociationFilter(comb):
               # If the pre filter does not accept the combination of light sources, continue to the next
               continue
            
            # Associate the light sources
            # If the association failed for some reason, continue to the next
            associated = self.assFunc(comb)
            if associated is None:
                continue

            
            if not self.postAssociationFilter(associated):
               # If the post filter does not accept the combination of light sources, continue to the next
               continue

            yield associated

class Tracker:

    def __init__(self, 
                 camera, 
                 featureModel,
                 lightSourceDetector,
                 poseEstimator,
                 associationFlag,
                 sortCombinations,
                 additionalCandidates,
                 roiMargin,
                 mahalanobisDistThresh,
                 detectionCountThresh,
                 detectionDeCount,
                 validOrientationRange,
                 covTransInc,
                 covRotInc):

        self.camera = camera
        self.featureModel = featureModel
        
        self.combinationGenerator = LightSourceCombinationGenerator(self.featureModel, associationFlag, sort=sortCombinations)

        # Pose estimator that calculates poses from detected light sources
        self.poseEstimator = poseEstimator
        self.lightSourceDetector = lightSourceDetector

        # max additional light source candidates that will be considered by 
        # the feature extractor.
        # 6 gives a minimum frame rate of about 1 when all 
        # combinations of candidates are iterated over
        self.additionalCandidates = additionalCandidates # 6

        # margin of the region of interest when pose has been aquired
        #self.roiMargin = int(round(0.0626*self.camera.cameraMatrix[0, 0]))
        self.roiMargin = roiMargin # 90

        # valid orientation range [yawMinMax, pitchMinMax, rollMinMax]. Currently not used to reject
        # invalid poses, but the axis and region of interest will be shown in red when a pose has 
        # an orientation outside the valid range 
        self.validOrientationRange = [np.radians(a) for a in validOrientationRange]

        # mahalanobis distance threshold
        self.mahalanobisDistThresh = mahalanobisDistThresh #12.592

        # Increase covariance of the pose each iteration
        # TODO: ideally this is done by a state estimator
        self.covTransInc = covTransInc
        self.covRotInc = covRotInc

        self.detectionCountThresh = detectionCountThresh
        self.detectionDeCount = detectionDeCount

        # This should be sent in as an argument (from some state estimator)
        # and is used to update the estimated pose (estDSPose) for better prediction of the ROI
        self.estCameraPoseVector = None # [x, y, z, ax, ay, az]

        # Access images from perception_node
        self.processedImg = None
        self.poseImg = None

    @staticmethod
    def create(trackingYaml, camera, dockingStation):
        from lolo_perception.camera_model import Camera
        from lolo_perception.feature_model import FeatureModel

        # Create the camera (camera is ieither a camera or a camera yaml)
        if isinstance(camera, str) and os.path.splitext(camera)[1] == ".yaml":
            camera = Camera.fromYaml(camera)
        elif isinstance(camera, Camera):
            pass
        else:
            raise Exception("'{}' is not a {} instance or a yaml path.".format(camera, Camera))

        # Create the docking station model (dockingStation is either a feature model or a feature model yaml)
        if isinstance(dockingStation, str) and os.path.splitext(dockingStation)[1] == ".yaml":
            dockingStation = FeatureModel.fromYaml(dockingStation)
        elif isinstance(dockingStation, FeatureModel):
            pass
        else:
            raise Exception("'{}' is not a {} instance or a yaml path.".format(dockingStation, FeatureModel))

        with open(trackingYaml, "r") as file:
            params = yaml.safe_load(file)

        # Create light source detector
        detectorParams = params["light_source_detector"]
        keys = detectorParams.keys()
        if len(keys) > 1:
            raise Exception("More than 1 class entries for light_source_detector.")
        
        className = keys[0]
        print("CLALLALLSLS", className)
        detectorClass = getLightSourceDetectorFromName(className)
        detector = detectorClass(**detectorParams[className])
        
        # Create pose estimator
        poseEstParams = params["pose_estimator"]
        poseEstimator = DSPoseEstimator(camera, dockingStation, **poseEstParams)

        # Create tracker
        trackingParams = params["tracker"]
        tracker = Tracker(camera, dockingStation, detector, poseEstimator, **trackingParams)

        return tracker

    def updateDSPoseFromNewCameraPose(self, estDSPose, estCameraPoseVector):
        if self.estCameraPoseVector is not None and estCameraPoseVector is not None:
            logging.debug("Updating the estimated pose from new camera pose")
            c1ToGTransl = self.estCameraPoseVector[:3]
            c1ToGRot = R.from_rotvec(self.estCameraPoseVector[3:])
            c2ToGTransl = estCameraPoseVector[:3]
            c2ToGRot = R.from_rotvec(estCameraPoseVector[3:])
            dsToC1Transl = estDSPose.translationVector
            dsToC1Rot = R.from_rotvec(estDSPose.rotationVector)

            gToC1Rot = c1ToGRot.inv()
            gToC2Rot = c2ToGRot.inv()
            c2ToC1Transl = gToC1Rot.apply(c2ToGTransl-c1ToGTransl)
            #c2ToC1Transl[2] = 0 # to ignore displacement in z
            c1ToC2Rot = gToC2Rot*c1ToGRot

            dsToC2Transl = c1ToC2Rot.apply(dsToC1Transl-c2ToC1Transl)
            dsToC2Rot = gToC2Rot*c1ToGRot*dsToC1Rot

            estDSPose.translationVector = dsToC2Transl
            estDSPose.rotationVector = dsToC2Rot.as_rotvec()
        
        if estDSPose.covariance is not None:
            covInc = np.eye(6)
            covInc[:3, :3] *= self.covTransInc
            covInc[3:,3:] *= self.covRotInc
            covNew = estDSPose.covariance + covInc
            estDSPose.covariance = covNew
        else:
            logging.error("Estimated pose has no covariance, could not increase it")

        return estDSPose

    def estimatePose(self, 
                     imgColor, 
                     estDSPose=None,
                     estCameraPoseVector=None,
                     colorCoeffs=None):

        # Initialize timers for recording the processing time
        imgProcTimer = Timer("Image Proc")
        poseEstTimer = Timer("Pose Est")
        totTimer = Timer("Total")
        totTimer.start()

        # Customize the color->gray conversion (currently not used)
        # Could be beneficial if the light sources are of known color
        if colorCoeffs is not None:
            m = np.array(colorCoeffs).reshape((1,3))
            gray = cv.transform(imgColor, m)
        else:
            gray = cv.cvtColor(imgColor, cv.COLOR_BGR2GRAY)

        # information about contours extracted from the feature extractor is plotted in this image
        # WARNING: Some overhead and slows down the execution time a bit
        processedImg = cv.cvtColor(gray, cv.COLOR_GRAY2BGR) # Information about the light source detection is plotted here
        poseImg = imgColor.copy()                           # Information about pose, fps etc. is plotted on this image
        #processedImg = None
        #poseImg = None

        roiCnt = None           # ROI of the passed estDSPose
        roiCntUpdated = None    # ROI of the new dsPose
        poseAquired = False     # Determined from detectionCount
        dsPose = None           # New estimated pose

        if estDSPose:
            roiMargin = self.roiMargin
            roiMargin += max([ls.radius for ls in estDSPose.associatedLightSources])
            _, roiCnt = regionOfInterest(estDSPose.reProject(), wMargin=roiMargin, hMargin=roiMargin, shape=gray.shape)

            # Update the estDSPose from the given estCameraPoseVector and increase covariance
            estDSPose = self.updateDSPoseFromNewCameraPose(estDSPose, estCameraPoseVector)
        self.estCameraPoseVector = estCameraPoseVector

        # Extract light source candidates from image
        imgProcTimer.start()
        _, candidates, roiCntUpdated = self.lightSourceDetector.detect(gray, 
                                                                       nFeatures=len(self.featureModel.features),
                                                                       additionalCandidates=self.additionalCandidates, 
                                                                       estDSPose=estDSPose,
                                                                       roiMargin=self.roiMargin, 
                                                                       drawImg=processedImg)
        imgProcTimer.stop()
        
        if len(candidates) >= len(self.featureModel.features):

            # Generate light source combinations that are fed into the pose estimation module
            # sort=True for the combination generator will result in some extra overhead
            associatedCombinations = self.combinationGenerator.generate(candidates)
            
            NC = len(candidates)
            NF = len(self.featureModel.features)
            N = math.factorial(NC) / (math.factorial(NF) * math.factorial(NC-NF))
            
            # Find a valid pose by iteration through the combinations
            poseEstTimer.start()
            dsPose = self.poseEstimator.findBestPose(associatedCombinations, 
                                                     N=N,
                                                     estDSPose=estDSPose, 
                                                     firstValid=True,
                                                     mahaDistThresh=self.mahalanobisDistThresh)
            poseEstTimer.stop()

            if dsPose and estDSPose:
                # Increment the detection count.
                dsPose.detectionCount = estDSPose.detectionCount + 1
                # If the detection count is above the threshold, the pose is considered acquired
                # and is an indication that it is "safe" to use
                if dsPose.detectionCount >= self.detectionCountThresh:
                    poseAquired = True
            elif dsPose and not estDSPose:
                logging.info("New pose found")
            elif not dsPose and estDSPose:
                logging.info("No valid pose found")
        else:
            logging.info("Did not find enough light source candidates {} < {}".format(len(candidates), len(self.featureModel.features)))

        # If a pose have been seen before but we loose it temporarily,
        # we keep the previously estimate pose for self.detectionDeCount iterations
        if not dsPose and estDSPose:
            if estDSPose.detectionCount > 1:
                dsPose = estDSPose
                dsPose.detectionCount = min(self.detectionDeCount, dsPose.detectionCount)
                dsPose.detectionCount -= 1

        totTimer.stop()
        totRestElapsed = totTimer.elapsed() - imgProcTimer.elapsed() - poseEstTimer.elapsed()

        # plots pose axis, ROI, light sources etc.
        if poseImg is not None:
            progress = 0
            if dsPose:
                progress = min(1, dsPose.detectionCount/float(self.detectionCountThresh))

            #poseImg = plotPoseImageInfoSimple(poseImg, dsPose, poseAquired, roiCntUpdated, progress)

            piChartArgs = {"Image Proc": imgProcTimer.elapsed(),
                           "Pose Est": poseEstTimer.elapsed(),
                           "Rest": totRestElapsed}

            poseImg = plotPoseImageInfo(poseImg,
                                        str(self.lightSourceDetector),
                                        dsPose,
                                        self.camera,
                                        self.featureModel,
                                        poseAquired,
                                        self.validOrientationRange,
                                        str(self.poseEstimator),
                                        piChartArgs,
                                        roiCnt,
                                        roiCntUpdated,
                                        progress=progress,
                                        fixedAxis=False)

        if poseImg is None:
            poseImg = np.zeros((10,10,3), dtype=np.uint8)
        if processedImg is None:
            processedImg = np.zeros((10,10,3), dtype=np.uint8)

        self.poseImg = poseImg
        self.processedImg = processedImg

        return dsPose, poseAquired, candidates, processedImg, poseImg
        

if __name__ == '__main__':
    pass
