import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as R, rotation

def plotAxis(img, translationVector, rotationVector, camera, points, scale):
    points = points[:, :3].copy()
    #print(points)
    rad = scale

    zDir, jacobian = cv.projectPoints(np.array([(0.0, 0.0, rad)]), 
                                      rotationVector, 
                                      translationVector, 
                                      camera.cameraMatrix, 
                                      camera.distCoeffs)

    yDir, jacobian = cv.projectPoints(np.array([(0.0, rad, 0.0)]), 
                                      rotationVector, 
                                      translationVector, 
                                      camera.cameraMatrix, 
                                      camera.distCoeffs)

    xDir, jacobian = cv.projectPoints(np.array([(rad, 0.0, 0.0)]), 
                                      rotationVector, 
                                      translationVector, 
                                      camera.cameraMatrix, 
                                      camera.distCoeffs)

    center, jacobian = cv.projectPoints(np.array([(0.0, 0.0, 0.0)]), 
                                        rotationVector, 
                                        translationVector, 
                                        camera.cameraMatrix, 
                                        camera.distCoeffs)

    center = center[0][0][0], center[0][0][1]   
    for d, c in zip((xDir, yDir, zDir), ((0,0,255), (0,255,0), (255,0,0))):
        cx = center[0]
        cy = center[1]
        point1 = (int(round(cx)), int(round(cy)))
        point2 = (int(round(d[0][0][0])), int(round(d[0][0][1])))
        cv.line(img, point1, point2, c, 5)

def plotPosePoints(img, translationVector, rotationVector, camera, points, color):
    projPoints = projectPoints(translationVector, rotationVector, camera, points)
    plotPoints(img, projPoints, color)

def plotPoints(img, points, color, radius=1):
    for p in points:
        x = int( round(p[0]) )
        y = int( round(p[1]) )
        cv.circle(img, (x,y), radius, color, 3)

def projectPoints(translationVector, rotationVector, camera, points):
    projPoints, jacobian = cv.projectPoints(points, 
                                        rotationVector, 
                                        translationVector, 
                                        camera.cameraMatrix, 
                                        camera.distCoeffs)
    projPoints = np.array([p[0] for p in projPoints])
    return projPoints

def plotPoseInfo(img, translationVector, rotationVector):
    distance = np.linalg.norm(translationVector)
    yaw, pitch, roll = R.from_rotvec(rotationVector).as_euler("YXZ")
    yaw, pitch, roll = np.rad2deg(yaw), np.rad2deg(pitch), np.rad2deg(roll)
    org = (20, 25)
    
    if distance < 2:
        unit = "cm"
        distance = round(distance*100, 2)
        translationVector = translationVector.copy()
        translationVector *= 100
    else:
        unit = "m"
        distance = round(distance, 2)

    inc = 25
    cv.putText(img, "Range: {} {}".format(distance, unit), org, cv.FONT_HERSHEY_SIMPLEX, 1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
    org = (org[0], org[1]+inc)
    cv.putText(img, "X: {} {}".format(round(translationVector[0], 2), unit), org, cv.FONT_HERSHEY_SIMPLEX, 1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
    org = (org[0], org[1]+inc)
    cv.putText(img, "Y: {} {}".format(round(translationVector[1], 2), unit), org, cv.FONT_HERSHEY_SIMPLEX, 1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
    org = (org[0], org[1]+inc)
    cv.putText(img, "Z: {} {}".format(round(translationVector[2], 2), unit), org, cv.FONT_HERSHEY_SIMPLEX, 1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
    org = (org[0], org[1]+inc)
    cv.putText(img, "Yaw: {} deg".format(round(yaw, 2)), org, cv.FONT_HERSHEY_SIMPLEX, 1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
    org = (org[0], org[1]+inc)
    cv.putText(img, "Pitch: {} deg".format(round(pitch, 2)), org, cv.FONT_HERSHEY_SIMPLEX, 1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
    org = (org[0], org[1]+inc)
    cv.putText(img, "Roll: {} deg".format(round(roll, 2)), org, cv.FONT_HERSHEY_SIMPLEX, 1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)

class PoseAndImageUncertaintyEstimator:
    """
    Utility class to use when evaluating uncertainty of pose and image points.
    Make sure all the samples are correct (watch out for outliers)
    """
    def __init__(self, nImagePoints, nSamples=100):
        self.associatedImagePoints = [[] for _ in range(nImagePoints)]
        self.poseVecs = [] # list of pose vectors [tx, ty, tz, rx, ry, rz]
        self.nSamples = nSamples

    def add(self, translationVec, rotationVec, assImgPoints):

        vec = [_ for _ in translationVec] + [_ for _ in rotationVec]
        self.poseVecs.insert(0, vec)
        self.poseVecs = self.poseVecs[:self.nSamples]

        for i, p in enumerate(assImgPoints):
            self.associatedImagePoints[i].insert(0, p)
            self.associatedImagePoints[i] = self.associatedImagePoints[i][:self.nSamples]

    def calcCovariance(self):
        if len(self.poseVecs) > 1:
            poseCov = np.cov(self.poseVecs, rowvar=False)
        else:
            poseCov = np.zeros((6,6))
        imageCovs = []
        for points in self.associatedImagePoints:
            imageCovs.append(np.cov(points, rowvar=False))

        return poseCov, imageCovs

    def calcAverage(self):
        # This only works for small differences in rotation and 
        # when rotations are not near +-pi
        poseAvg = np.mean(self.poseVecs, axis=0)
        imageAvgs = []
        for points in self.associatedImagePoints:
            imageAvgs.append(np.mean(points, axis=0))

        return poseAvg, imageAvgs