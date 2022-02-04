from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2 as cv

class Camera:
    # https://stackoverflow.com/questions/11140163/plotting-a-3d-cube-a-sphere-and-a-vector-in-matplotlib
    def __init__(self, cameraMatrix, distCoeffs, resolution=None):
        self.cameraMatrixPixel = cameraMatrix
        self._cameraMatrix = cameraMatrix.copy()
        
        self.distCoeffs = distCoeffs
        self.resolution = resolution # maybe change resolution to be (width, height) instead of (height, width)

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
            self.projectionMatrix = np.eye(3)
            self.roi = None

    @property
    def cameraMatrix(self):
        return self._cameraMatrix.copy()

    def undistortImage(self, img):
        imgRect = cv.undistort(img, self.cameraMatrix, self.distCoeffs, None, self.projectionMatrix)
        if self.roi is not None:
            x, y, w, h = self.roi
            imgRect = imgRect[y:y+h, x:x+w]
        return imgRect


# this camera matrix is wrong, should use the projection matrix instead since images are already rectified
usbCamera480p = Camera(cameraMatrix=np.array([[812.2540283203125,   0,    		    329.864062734141 ],
                                         [   0,               814.7816162109375, 239.0201541966089], 
                                         [   0,     		     0,   		       1             ]], dtype=np.float32), 
                   distCoeffs=np.zeros((4,1), dtype=np.float32),
                   resolution=(480, 640))

contourCamera1080p = Camera(cameraMatrix=np.array([[ 884.36572,    0.     ,  994.04928],
                                             [    0.     , 1096.93066,  567.01791],
                                             [    0.     ,    0.     ,    1.     ]], dtype=np.float32), 
                        distCoeffs=np.zeros((4,1), dtype=np.float32),
                        resolution=(1080, 1920))

usbCamera720p = Camera(cameraMatrix=np.array([[1607.87793,    0.     ,  649.18249],
                                              [   0.     , 1609.64954,  293.20127],
                                              [   0.     ,    0.     ,    1.     ]], dtype=np.float32), 
                        distCoeffs=np.zeros((4,1), dtype=np.float32),
                        resolution=(720, 1280))

if __name__ == "__main__":
    pass
    


