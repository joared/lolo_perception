from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2 as cv
import yaml

class Camera:
    # https://stackoverflow.com/questions/11140163/plotting-a-3d-cube-a-sphere-and-a-vector-in-matplotlib
    def __init__(self, cameraMatrix, distCoeffs, projectionMatrix=None, resolution=None):
        self.cameraMatrix = cameraMatrix
        
        self.distCoeffs = distCoeffs
        self.resolution = resolution # maybe change resolution to be (width, height) instead of (height, width)

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

        # image_proc is using this to undistort image

        self.mapx, self.mapy = cv.initUndistortRectifyMap(self.cameraMatrix, 
                                                          self.distCoeffs, 
                                                          None, #np.eye(3), 
                                                          self.projectionMatrix, 
                                                          (self.resolution[1], self.resolution[0]), 
                                                          m1type=cv.CV_32FC1)


    def undistortImage(self, img):
        # currently only used in image_analyze_node.py
        #imgRect = cv.undistort(img, self.cameraMatrix, self.distCoeffs, None, self.projectionMatrix)

        imgRect = cv.remap(img, self.mapx, self.mapy, interpolation=cv.INTER_LINEAR)
        #if self.roi is not None:
        #    x, y, w, h = self.roi
        #    imgRect = imgRect[y:y+h, x:x+w]
        return imgRect

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

        """
        return Camera(cameraMatrix=projectionMatrix, 
                      distCoeffs=np.zeros((1,4), dtype=np.float32),
                      projectionMatrix=None,
                      resolution=resolution)
        """

if __name__ == "__main__":
    pass
    


