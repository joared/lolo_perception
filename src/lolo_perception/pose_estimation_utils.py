from cmath import exp
import numpy as np
import cv2 as cv
from scipy.spatial.transform import Rotation as R

def interactionMatrix(x, y, Z):
    """
    IBVS interaction matrix
    """
    return [[-1/Z, 0, x/Z, x*y, -(1+x*x), y],
                [0, -1/Z, y/Z, 1+y*y, -x*y, -x]]

if __name__ == "__main__":
    pass