from matplotlib.colors import LightSource
import numpy as np
from scipy.spatial.transform import Rotation as R

def polygon(rad, n, shift=False, zShift=0):
    """
    Creates points in the xy-plane
    """
    theta = 2*np.pi/n
    if shift is True:
        points = np.array([ [rad*np.sin(theta*(i + 0.5)), rad*np.cos(theta*(i + 0.5)), zShift, 1] for i in range(n)] , dtype=np.float32)
    else:
        points = np.array([ [rad*np.sin(theta*i), rad*np.cos(theta*i), zShift, 1] for i in range(n)], dtype=np.float32)

    return points

def polygons(rads, ns, shifts, zShifts):
    assert len(rads) == len(ns) == len(shifts) == len(zShifts), "All args need to be the same length"
    points = None
    for r, n, s, z in zip(rads, ns, shifts, zShifts):
        if points is None:
            points = polygon(r, n, s, z)
        else:
            points = np.append(points, polygon(r, n, s, z), axis=0)
    return points


class FeatureModel:

    # default light source placement uncertainty percentage (percentage of max radius)
    DEFAULT_PLACEMENT_UNCERTAINTY_P = 0.01

    # default light source detection tolerance percentage (percentage of max radius)
    DEFAULT_DETECTION_TOLERANCE_P = 0.02

    def __init__(self, features, euler=(0, 0, 0), placementUncertainty=None, detectionTolerance=None):
        self.features = features
        rotMat = R.from_euler("XYZ", euler).as_dcm()
        self.features = np.matmul(rotMat, self.features[:, :3].transpose()).transpose()
        self.features = self.features[:, :3].copy() # Don't need homogenious

        self.nFeatures = len(self.features)
        self.maxRad = max([np.linalg.norm(f) for f in self.features])
        self.maxX = max([abs(f[0]) for f in self.features])
        self.maxY = max([abs(f[1]) for f in self.features])

        self.placementUncertainty = placementUncertainty
        if placementUncertainty is None:
            print("FeatureModel WARNING: placement uncertainty not specified, using default")
            self.placementUncertainty = self.maxRad*self.DEFAULT_PLACEMENT_UNCERTAINTY_P

        self.detectionTolerance = detectionTolerance
        if detectionTolerance is None:
            print("FeatureModel WARNING: detection tolerance not specified, using default")
            self.detectionTolerance = self.maxRad*self.DEFAULT_DETECTION_TOLERANCE_P

        # This uncertainty is used to calculate the maximum allowed reprojection error RMSE
        # when estimating a pose from detected light sources
        self.uncertainty = self.placementUncertainty + self.detectionTolerance

smallPrototype5 = FeatureModel(polygons([0, 0.06], 
                                        [1, 4], 
                                        [False, True], 
                                        [-0.043, 0]),
                                        placementUncertainty=.0012,
                                        detectionTolerance=0.0006)

smallPrototypeSquare = FeatureModel(polygons([0.06], 
                                        [4], 
                                        [True], 
                                        [0]),
                                        placementUncertainty=.0012,
                                        detectionTolerance=0.0006)

smallPrototype9 = FeatureModel(polygons([0, 0.06], 
                                        [1, 8], 
                                        [False, True], 
                                        [-0.043, 0]))

bigPrototype5 = FeatureModel(np.array([[0, 0, -0.465], 
                                       [-0.33, -0.2575, 0], 
                                       [0.33, -0.2575, 0], 
                                       [0.33, 0.2575, 0], 
                                       [-0.33, 0.2575, 0]]))

bigPrototype52 = FeatureModel(np.array([[-0.04, -0.2575-0.05, 0], 
                                       [-0.33, -0.2575, 0], 
                                       [0.33, -0.2575, 0], 
                                       [0.33, 0.2575, 0], 
                                       [-0.33, 0.2575, 0]]))
        
idealModel = FeatureModel(polygons([0, 1], 
                                   [1, 4], 
                                   [False, True], 
                                   [-0.7167, 0]))

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fm = FeatureModel([0.06, 0], [4, 1], [False, False], [0, 0.043])
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(*zip(*fm.features))
    size = 0.1
    ax.set_xlim(-size, size)
    ax.set_ylim(-size, size)
    ax.set_zlim(-size, size)
    plt.show()