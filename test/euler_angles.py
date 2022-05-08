import numpy as np
from scipy.spatial.transform import Rotation as R

if __name__ == "__main__":
    thetaX = 0.4
    thetaY = -0.1
    thetaZ = 0.7
    Rx = R.from_euler("XYZ", (thetaX, 0, 0)).as_dcm()
    Ry = R.from_euler("xyz", (0, thetaY, 0)).as_dcm()
    Rz = R.from_euler("XYZ", (0, 0, thetaZ)).as_dcm()

    

    print("------ Euler angles -----")
    # multiply to the right
    rotMat = np.matmul(np.matmul(Rx, Ry), Rz)
    print("Actual")
    print(rotMat)
    print("True")
    print(R.from_euler("XYZ", (thetaX, thetaY, thetaZ)).as_dcm())

    print("-------- Fixed ----------")
    # multiply to the left
    rotMat = np.matmul(np.matmul(Rz, Ry), Rx)
    print("Actual")
    print(rotMat)
    print("True")
    print(R.from_euler("xyz", (thetaX, thetaY, thetaZ)).as_dcm())

    # Gimbal lock
    # second rotation = pi/2
    print("----- Gimbal lock---------")
    rotMat = R.from_euler("XYZ", (0, np.pi/2, 0)).as_dcm()
    R.from_dcm(rotMat).as_euler("XYZ")
