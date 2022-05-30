import numpy as np
from numpy import place
from scipy.spatial.transform import Rotation as R
import yaml

def polygon(rad, n, shift=False, zShift=0):
    """
    Creates points in the xy-plane
    """
    theta = 2*np.pi/n
    if shift is True:
        points = np.array([ [rad*np.sin(theta*(i + 0.5)), -rad*np.cos(theta*(i + 0.5)), zShift, 1] for i in range(n)] , dtype=np.float32)
    else:
        points = np.array([ [rad*np.sin(theta*i), -rad*np.cos(theta*i), zShift, 1] for i in range(n)], dtype=np.float32)

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
    DEFAULT_DETECTION_TOLERANCE_P = 0.05
    
    # Previous value
    # DEFAULT_DETECTION_TOLERANCE_P = 0.02

    def __init__(self, name, features, placementUncertainty=0, detectionTolerance=0, euler=(0, 0, 0)):
        self.name = name
        
        self.features = features
        rotMat = R.from_euler("XYZ", euler).as_dcm()
        self.features = np.matmul(rotMat, self.features[:, :3].transpose()).transpose()
        self.features = self.features[:, :3].copy() # Don't need homogenious

        self.nFeatures = len(self.features)
        self.maxRad = max([np.linalg.norm(f) for f in self.features])
        self.maxX = max([abs(f[0]) for f in self.features])
        self.maxY = max([abs(f[1]) for f in self.features])

        self.placementUncertainty = placementUncertainty
        if placementUncertainty == 0:
            print("FeatureModel WARNING: placement uncertainty not specified, using default")
            self.placementUncertainty = self.maxRad*self.DEFAULT_PLACEMENT_UNCERTAINTY_P
            print("Placement: {}".format(self.placementUncertainty))

        self.detectionTolerance = detectionTolerance
        if detectionTolerance == 0:
            print("FeatureModel WARNING: detection tolerance not specified, using default")
            self.detectionTolerance = self.maxRad*self.DEFAULT_DETECTION_TOLERANCE_P
            print("Tolerance: {}".format(self.detectionTolerance))

        # This uncertainty is used to calculate the maximum allowed reprojection error RMSE
        # when estimating a pose from detected light sources
        self.uncertainty = self.placementUncertainty + self.detectionTolerance

    @staticmethod
    def fromYaml(yamlPath):
        with open(yamlPath, "r") as file:
            featureModelData = yaml.load(file)

        features = featureModelData["features"]
        if features[0] == "polygon":
            rad, n, shift = features[1:]
            rad = float(rad)
            n = int(n)
            shift = bool(shift)
            features = polygon(rad, n, shift)
        else:
            features = np.array(featureModelData["features"], dtype=np.float32)


        return FeatureModel(featureModelData["model_name"],
                            features,
                            featureModelData["placement_uncertainty"],
                            featureModelData["detection_tolerance"])


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    import rospy
    import rospkg
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("feature_model_yaml", help="feature model yaml file")
    args = parser.parse_args()

    featureModelYaml = args.feature_model_yaml
    yamlPath = os.path.join(rospkg.RosPack().get_path("lolo_perception"), "feature_models/{}".format(featureModelYaml))
    fm = FeatureModel.fromYaml(yamlPath)

    r = R.from_euler("XYZ", (-np.pi/2,0,0))
    featurePoints = r.apply(fm.features)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(*zip(*featurePoints))

    l = fm.maxRad
    xAxis = r.as_dcm()[:, 0]*l
    yAxis = r.as_dcm()[:, 1]*l
    zAxis = r.as_dcm()[:, 2]*l

    ax.plot(*zip(xAxis, [0]*3), color="r")
    ax.plot(*zip(yAxis, [0]*3), color="g")
    ax.plot(*zip(zAxis, [0]*3), color="b")

    for i, fp in enumerate(featurePoints):
        ax.text(fp[0]+l*0.03, fp[1]+l*0.03, fp[2]+l*0.03, str(i), [0, 0, 0])

    size = l
    ax.set_xlim(-size, size)
    ax.set_ylim(-size, size)
    ax.set_zlim(-size, size)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    plt.show()