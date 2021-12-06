from scipy.spatial.transform import Rotation as R
import numpy as np

class Camera:
    # https://stackoverflow.com/questions/11140163/plotting-a-3d-cube-a-sphere-and-a-vector-in-matplotlib
    def __init__(self, cameraMatrix, distCoeffs, resolution):
        self.cameraMatrixPixel = cameraMatrix
        self._cameraMatrix = cameraMatrix.copy()
        
        self.distCoeffs = distCoeffs
        self.resolution = resolution # maybe change resolution to be (width, height) instead of (height, width)

    @property
    def cameraMatrix(self):
        return self._cameraMatrix.copy()


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
    


