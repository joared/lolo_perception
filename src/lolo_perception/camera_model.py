from scipy.spatial.transform import Rotation as R
import numpy as np

class Camera:
    # https://stackoverflow.com/questions/11140163/plotting-a-3d-cube-a-sphere-and-a-vector-in-matplotlib
    def __init__(self, cameraMatrix, distCoeffs, resolution, pixelWidth, pixelHeight, hz, *args, **kwargs):
        self.cameraMatrixPixel = cameraMatrix
        self._cameraMatrix = cameraMatrix.copy()
        self._cameraMatrix[0, :] *= pixelWidth
        self._cameraMatrix[1, :] *= pixelHeight
        
        self.distCoeffs = distCoeffs
        self.resolution = resolution # maybe change resolution to be (width, height) instead of (height, width)
        self.pixelWidth = pixelWidth
        self.pixelHeight = pixelHeight
        self.imWidth = self.resolution[1]*self.pixelWidth
        self.imHeight = self.resolution[0]*self.pixelHeight
        self.cx = self.cameraMatrix[0][2]
        self.cy = self.cameraMatrix[1][2]
        self.hz = hz # assume that perception algorithm can perform at this rate

        # Points
        f = self.cameraMatrix[1][1] # focal length (take average between fx and fy?)
        self.f = f
        self.imSize = 1

    @property
    def cameraMatrix(self):
        return self._cameraMatrix.copy()

    def project(self, point):
        """
        TODO: Handle singular matrix
        """
        l0 = np.array(point) # position vector for line
        vl = l0 - np.array(self.translation) # direction vector for line
        p0 = np.array(self.translation) + np.array(self.rotation)[:, 2]*self.f # center of plane
        mat = np.column_stack((np.array(self.rotation)[:, 0], np.array(self.rotation)[:, 1], -vl))
        x = np.linalg.solve(mat, l0-p0)

        # Check if points are within image plane
        if not self.pointWithinImagePlane(x[0], x[1]):
            print("Projection: Point not within image plane")
            return False

        # Check if the projection is from the "front" of the camera/plane
        if np.dot(np.array(self.rotation)[:, 2], vl) < self.f:
            print("Projection: Point behind image plane")
            return False

        return l0 + x[2]*vl

    def projectLocal3D(self, point):
        """
        TODO: Handle singular matrix
        Use open cv project instead?
        """
        l0 = np.array(point) # position vector for line
        vl = l0
        p0 = np.array([0, 0, self.f]) # center of plane
        mat = np.column_stack((np.array([1, 0, 0]).transpose(), np.array([0, 1, 0]).transpose(), -vl))
        x = np.linalg.solve(mat, l0-p0)

        if not self.pointWithinImagePlane(x[0], x[1]):
            return False

        # Check if the projection is from the "front" of the camera/plane
        #if np.dot(np.array([0, 0, 1]), vl) < self.f:

        return l0 + x[2]*vl

    def pointWithinImagePlane(self, x, y):
        x = x + self.cx
        y = y + self.cy
        return x > 0 and x < self.imWidth and y > 0 and y < self.imHeight

    def uvToMeters(self, points):
        points[:, 0] *= self.pixelWidth
        points[:, 1] *= self.pixelHeight
        return points

    def metersToUV(self, points):
        points = points.copy()
        points[:, 0] /= self.pixelWidth
        points[:, 1] /= self.pixelHeight
        return points


# this camera matrix is wrong, should use the projection matrix instead since images are already rectified
usbCamera480p = Camera(cameraMatrix=np.array([[812.2540283203125,   0,    		    329.864062734141 ],
                                         [   0,               814.7816162109375, 239.0201541966089], 
                                         [   0,     		     0,   		       1             ]], dtype=np.float32), 
                   distCoeffs=np.zeros((4,1), dtype=np.float32),
                   resolution=(480, 640), 
                   pixelWidth=2.796875e-6, 
                   pixelHeight=2.8055555555e-6, 
                   hz=15)

contourCamera1080p = Camera(cameraMatrix=np.array([[ 884.36572,    0.     ,  994.04928],
                                             [    0.     , 1096.93066,  567.01791],
                                             [    0.     ,    0.     ,    1.     ]], dtype=np.float32), 
                        distCoeffs=np.zeros((4,1), dtype=np.float32),
                        resolution=(1080, 1920), 
                        pixelWidth=2.8e-6,  # not known
                        pixelHeight=2.8e-6, # not known
                        hz=15)

usbCamera720p = Camera(cameraMatrix=np.array([[1607.87793,    0.     ,  649.18249],
                                              [   0.     , 1609.64954,  293.20127],
                                              [   0.     ,    0.     ,    1.     ]], dtype=np.float32), 
                        distCoeffs=np.zeros((4,1), dtype=np.float32),
                        resolution=(720, 1280), 
                        pixelWidth=2.796875e-6, 
                        pixelHeight=2.8055555555e-6, 
                        hz=15)

if __name__ == "__main__":
    pass
    


