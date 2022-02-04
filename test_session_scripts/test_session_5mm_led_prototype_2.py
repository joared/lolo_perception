import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # for python 2.7
from coordinate_system import CoordinateSystemArtist

import rosbag
from lolo_perception.perception_ros_utils import poseToVector

class TestSession:
    def __init__(self):
        pass

    def cameraRelativeLightTransform(self, pose):
        euler = [np.deg2rad(v) for v in pose[3:]]
        lightRotRelativeCamera = R.from_euler("YXZ", euler).as_dcm()

        cameraTransRelativeLight = -np.matmul(lightRotRelativeCamera.transpose(), np.array(pose[:3]))
        cameraRotRelativeLight = lightRotRelativeCamera.transpose()
        return cameraTransRelativeLight, cameraRotRelativeLight

    def calculateErrors(self, poses, posesTrue):
        for i, (pose, poseTrue) in enumerate(zip(poses, posesTrue)):
            #print("Reference trans: {}".format(transLightToCameraRef))
            rangeErr = np.linalg.norm(poseTrue[:3]) - np.linalg.norm(pose[:3])
            transErr = np.array(poseTrue[:3]) - np.array(pose[:3])
            eulerErr = np.array(poseTrue[3:]) - np.array(pose[3:])
            print("Light pose {} err: range {} trans {} rot {}".format(i+1, rangeErr, transErr, eulerErr))

        for i, (pose, poseTrue) in enumerate(zip(poses, posesTrue)):
            camToLightTrans, camToLightRot = self.cameraRelativeLightTransform(pose)
            camToLightTransTrue, camToLightRotTrue = self.cameraRelativeLightTransform(poseTrue)
            camToLightRot = np.rad2deg(R.from_dcm(camToLightRot).as_euler("YXZ"))
            camToLightRotTrue = np.rad2deg(R.from_dcm(camToLightRotTrue).as_euler("YXZ"))

            rangeErr = np.linalg.norm(camToLightTransTrue) - np.linalg.norm(camToLightTrans)
            transErr = np.array(camToLightTransTrue) - np.array(camToLightTrans)
            eulerErr = np.array(camToLightRotTrue) - np.array(camToLightRot)
            print("Cam pose {} err: range {} trans {} rot {}".format(i+1, rangeErr, transErr, eulerErr))

    def getCoordinateSystems(self, poses, posesTrue):
        # light prototype pose in world coordinates
        lightPrototypeTrans = [0, 0, 0]
        lightPrototypeRot = R.from_euler("XYZ", (-np.pi/2, np.pi/2, 0)).as_dcm()
        csLightPrototype = CoordinateSystemArtist(translation=lightPrototypeTrans, 
                                                  euler=(-np.pi/2, np.pi/2, 0))

        coordSystems = []      # camera relative to world
        coordSystemsTrue = []  # camera relative to world
        coordSystemsLight = [] # light relative world (assuming measured 
                            # camera poses are at the true pose)
        for pose, poseTrue in zip(poses, posesTrue):

            # measured poses in world coordinates
            cameraTransRelativeLight, cameraRotRelativeLight = self.cameraRelativeLightTransform(pose)
            cameraTransRelativeWorld = np.matmul(lightPrototypeRot, cameraTransRelativeLight)
            cameraRotRelativeWorld = np.matmul(lightPrototypeRot, cameraRotRelativeLight)

            coordSystems.append(CoordinateSystemArtist(translation=cameraTransRelativeWorld, 
                                                    euler=R.from_dcm(cameraRotRelativeWorld).as_euler("YXZ"),
                                                    seq="YXZ"))

            # true poses in world coordinates
            cameraTransRelativeLight, cameraRotRelativeLight = self.cameraRelativeLightTransform(poseTrue)
            cameraTransRelativeWorld = np.matmul(lightPrototypeRot, cameraTransRelativeLight)
            cameraRotRelativeWorld = np.matmul(lightPrototypeRot, cameraRotRelativeLight)
    
            coordSystemsTrue.append(CoordinateSystemArtist(translation=cameraTransRelativeWorld, 
                                                        euler=R.from_dcm(cameraRotRelativeWorld).as_euler("YXZ"),
                                                        seq="YXZ"))

            # light relative world (assuming measured camera poses are at the true pose)
            lightTransRelativeWorld = cameraTransRelativeWorld + np.matmul(cameraRotRelativeWorld, np.array(pose[:3]))
            lightRotRelativeWorld = np.matmul(
                cameraRotRelativeWorld, 
                R.from_euler("YXZ", tuple(np.deg2rad(pose[3:]))).as_dcm()
                )
            coordSystemsLight.append(CoordinateSystemArtist(translation=lightTransRelativeWorld, 
                                                            euler=R.from_dcm(lightRotRelativeWorld).as_euler("YXZ"),
                                                            seq="YXZ"))

        return csLightPrototype, coordSystems, coordSystemsTrue, coordSystemsLight

    def displayPoses(self, poses, posesTrue):
        assert len(poses) == len(posesTrue), "Must have the same number of poses as ground truths"

        csLightPrototype, coordSystems, coordSystemsTrue, coordSystemsLight = self.getCoordinateSystems(poses, posesTrue)

        fig = plt.figure()
        fig.add_subplot(projection="3d")
        ax = Axes3D(fig)

        s = 120
        ax.set_xlim(-s, 0)
        ax.set_ylim(-s/2, s/2)
        ax.set_zlim(-s/2, s/2)

        axisScale = 6
        poseIdx = 1
        for cs, csTrue, csLight in zip(coordSystems, coordSystemsTrue, coordSystemsLight):
            cs.draw(ax, scale=axisScale, color=("r", "r", "r"))
            csLight.draw(ax, scale=axisScale, color=("salmon", "lightgreen", "lightblue"))
            csTrue.draw(ax, scale=axisScale, color=("b", "b", "b"), text="Cam {}".format(poseIdx))
            poseIdx += 1
        csLightPrototype.draw(ax, scale=axisScale, text="Light")
        plt.show()

        fig = plt.figure()
        fig.add_subplot(projection="3d")
        ax = Axes3D(fig)

        while True:
            for csLightRef in coordSystemsLight:
                ax.clear()
                ax.set_xlim(-s, 0)
                ax.set_ylim(-s/2, s/2)
                ax.set_zlim(-s/2, s/2)

                csLightPrototype.draw(ax, scale=axisScale)
                for cs, csTrue, csLight in zip(coordSystems, coordSystemsTrue, coordSystemsLight):
                    cs.draw(ax, scale=axisScale, color=("r", "r", "r"))
                    if csLightRef == csLight:
                        csTrue.draw(ax, scale=axisScale, color=("g", "g", "g"))
                        csLight.draw(ax, scale=axisScale, color=("salmon", "lightgreen", "lightblue"))
                    else:
                        csTrue.draw(ax, scale=axisScale, color=("b", "b", "b"))
                plt.pause(1)
        plt.show()

class RosbagToPoseVecs:
	def __init__(self, poseTopic):
		self.poseTopic = poseTopic

	def getPoses(self, filePath):
		bag = rosbag.Bag(filePath)

		for topic, msg, t in bag.read_messages():
			t = t.to_sec()
			if topic == self.poseTopic:
				translationVector, rotationVector = poseToVector(msg)
                print("YAh")
        

def getPosesFromRosbags(rosbags):
    
    poses = [[]*len(rosbags)]
    poseAvgs = []
    poseCovs = []
    for rb, poseList in zip(rosbags, poses):
        # get all poses
        for pose in rb:
            poseList.append(pose)
        # calc average and covariance
        poseAvgs.append(poseAvg)
        poseCovs.append(poseCov)

if  __name__ == "__main__":
    # pose: x, y, z, yaw, pitch, roll
    poses = [
             # poses from long range (120 cm)
             (-3.58, -.12, 120.03, -4.25, 1.85, 4.45),
             (18.57, 2.16, 120.31, -2.75, 1.42, 4.77),
             (-25.42, 4.45, 120.02, -4.44, -1.21, 4.13),
             (39.75, 7.79, 120.41, -2.7, -0.61, 4.23),
             (-47.95, 2.98, 119.51, -4.45, -1.04, 4.08),
             # poses from mid range (60 cm)
             (-2.17, 2.45, 59.9, -4.31, -0.54, 3.88),
             (19.94, 3.39, 60.99, -2.71, -0.39, 4.44),
             (-22.9, 1.52, 60.86, -3.46, -0.52, 4.55),
             # pose from short range (20 cm)
             (-0.8, 0.16, 19.74, -3.08, 1.45, 1.47)
             ]
    groundTruths = [
                    # true poses from long range (120 cm)
                    (0, 0, 120, 0, 0, 0),
                    (21.2, 0, 120, 0, 0, 0),
                    (-21.2, 0, 120, 0, 0, 0),
                    (43.7, 0, 120, 0, 0, 0),
                    (-43.7, 0, 120, 0, 0, 0),
                    # true poses from mid range (60 cm)
                    (0, 0, 60, 0, 0, 0),
                    (21.8, 0, 60, 0, 0, 0),
                    (-21.8, 0, 60, 0, 0, 0),
                    # true pose from short range (20 cm)
                    (0, 0, 20, 0, 0, 0)
                    ]

    # test 2
    poses = [
             # poses from long range (120 cm)
             (-3.58, -.12, 120.03, -4.25, 1.85, 4.45),
             (18.57, 2.16, 120.31, -2.75, 1.42, 4.77),
             (-25.42, 4.45, 120.02, -4.44, -1.21, 4.13),
             (39.75, 7.79, 120.41, -2.7, -0.61, 4.23),
             (-47.95, 2.98, 119.51, -4.45, -1.04, 4.08),
             # poses from mid range (60 cm)
             (-2.17, 2.45, 59.9, -4.31, -0.54, 3.88),
             (19.94, 3.39, 60.99, -2.71, -0.39, 4.44),
             (-22.9, 1.52, 60.86, -3.46, -0.52, 4.55),
             # pose from short range (20 cm)
             (-0.8, 0.16, 19.74, -3.08, 1.45, 1.47)
             ]
    camPoses =
            [
             # poses from long range (120 cm)
             (-0.97, -2.21, -120.07, .5, -1., 0.28),
             (18.57, 2.16, 120.31, -2.75, 1.42, 4.77),
             (-25.42, 4.45, 120.02, -4.44, -1.21, 4.13),
             (39.75, 7.79, 120.41, -2.7, -0.61, 4.23),
             (-47.95, 2.98, 119.51, -4.45, -1.04, 4.08),
             # poses from mid range (60 cm)
             (-2.17, 2.45, 59.9, -4.31, -0.54, 3.88),
             (19.94, 3.39, 60.99, -2.71, -0.39, 4.44),
             (-22.9, 1.52, 60.86, -3.46, -0.52, 4.55),
             # pose from short range (20 cm)
             (-0.8, 0.16, 19.74, -3.08, 1.45, 1.47)
             ] 
    yDisp = -0.9
    groundTruths = [
                    # true poses from long range (120 cm)
                    (0, yDisp, 120, 0, 0, 0),
                    (21.2, yDisp, 120, 0, 0, 0),
                    (-21.2, yDisp, 120, 0, 0, 0),
                    (43.7, yDisp, 120, 0, 0, 0),
                    (-43.7, yDisp, 120, 0, 0, 0),
                    # true poses from mid range (60 cm)
                    (0, yDisp, 60, 0, 0, 0),
                    (21.8, yDisp, 60, 0, 0, 0),
                    (-21.8, yDisp, 60, 0, 0, 0),
                    # true pose from short range (20 cm)
                    (0, yDisp, 20, 0, 0, 0)
                    ]

    ts = TestSession()
    #ts.calculateErrors(poses, groundTruths)
    #ts.displayPoses(poses, groundTruths)

    import argparse
    parser = argparse.ArgumentParser(description='Test session 5mm light prototype.')
    parser.add_argument('rosbagdir', type=str, default="../rosbags/test_session_5mm_led_prototype",
                        help='')
    
    args = parser.parse_args()

    RosbagToPoseVecs("docking_station/pose").getPoses(args.rosbagdir + "/pos_1.bag")