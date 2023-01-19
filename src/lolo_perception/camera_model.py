from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2 as cv
import yaml

class Camera:

    def __init__(self, cameraMatrix, distCoeffs, projectionMatrix=None, resolution=None):
        self.cameraMatrix = cameraMatrix
        
        self.distCoeffs = distCoeffs
        self.resolution = resolution # (h, w), maybe change resolution to be (width, height) instead of (height, width)

        self.projectionMatrix = projectionMatrix
        self.roi = None
        if projectionMatrix is None:
            if resolution is not None:
                h, w = self.resolution
                P, roi = cv.getOptimalNewCameraMatrix(self.cameraMatrix, 
                                                    self.distCoeffs, 
                                                    (w,h), 
                                                    0, 
                                                    (w,h))

                self.projectionMatrix = P
                self.roi = roi
            else:
                self.projectionMatrix = self.cameraMatrix

        # Initialize distortion maps (used in undistortImage() and distortPoints())
        # We want the undistorted image to be of the same size so we pass the resolution (height and width)
        # in cv.initUndistortRectifyMap().
        # Note: these are actually mapping from undistorted coordinates to distorted coordinates
        # (see https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga7dfb72c9cf9780a347fbe3d1c47e5d5a) 
        self.mapx, self.mapy = cv.initUndistortRectifyMap(self.cameraMatrix, 
                                                          self.distCoeffs, 
                                                          None, #np.eye(3), 
                                                          self.projectionMatrix, 
                                                          (self.resolution[1], self.resolution[0]), 
                                                          m1type=cv.CV_32FC1)


    def undistortImage(self, img):
        """
        Undistorts the image with self.mapx and self.mapy.
        """
        imgRect = cv.remap(img, self.mapx, self.mapy, interpolation=cv.INTER_LINEAR)
        #if self.roi is not None:
        #    x, y, w, h = self.roi
        #    imgRect = imgRect[y:y+h, x:x+w]
        return imgRect


    # def undistortImage(self, img):
    #     # How to undistort image: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    #     h, w = img.shape[:2]
    #     newcameramtx, roi = cv.getOptimalNewCameraMatrix(self.cameraMatrix, 
    #                                                     self.distCoeffs, 
    #                                                     (w,h), 
    #                                                     0, 
    #                                                     (w,h))

    #     imgRect = cv.undistort(img, self.cameraMatrix, self.distCoeffs, None, newcameramtx)
    #     x, y, w, h = roi
    #     imgRect = imgRect[y:y+h, x:x+w]
    #     return imgRect, newcameramtx


    def undistortPoints(self, points):
        """
        Undistorts the points using cv.undistortPoints().
        The new points are rounded to the nearest integer.
        """
        points = np.array(points)
        points = points.reshape((len(points), 1, 2))

        pointsUndist = cv.undistortPoints(points, self.cameraMatrix, self.distCoeffs, P=self.projectionMatrix)
        pointsUndist = pointsUndist.reshape((len(pointsUndist), 2))

        pointsUndistInt = []
        for p in pointsUndist:
            pointsUndistInt.append((int(round(p[0])), 
                                    int(round(p[1]))))

        return pointsUndistInt

    def distortPoints(self, points):
        """
        Distorts the points using self.mapx and self.mapy distortion maps.
        The input points have to be valid pixel coordinates (integer values).
        """
        distortedPoints = []

        for p in points:
            distortedPoints.append((self.mapx(*p), 
                                    self.mapy(*p)))

        return distortedPoints


    @staticmethod
    def fromYaml(cameraYamlPath):
        with open(cameraYamlPath, "r") as file:
            calibData = yaml.load(file)
  
        projectionMatrix = np.array(calibData["projection_matrix"]["data"], dtype=np.float32).reshape((3,4))[:,:3]

        cameraMatrix = np.array(calibData["camera_matrix"]["data"], dtype=np.float32).reshape((3,3))
        distCoeffs = np.array(calibData["distortion_coefficients"]["data"], dtype=np.float32)
        resolution = calibData["image_height"], calibData["image_width"]

        return Camera(cameraMatrix=cameraMatrix, 
                      distCoeffs=distCoeffs,
                      projectionMatrix=projectionMatrix,
                      resolution=resolution)


if __name__ == "__main__":
    pass
    


