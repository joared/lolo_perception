import numpy as np
import cv2 as cv
import time

#from lolo_perception.pose_estimation_utils import 

def test_cvinvert_vs_npinv():
    c = np.array([[1.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                  [0.2, 2.1, 1.1, 1.1, 1.1, 1.1],
                  [0.3, 1.1, 3.1, 1.1, 1.1, 1.1],
                  [0.4, 1.1, 1.1, 4.1, 1.1, 1.1],
                  [0.5, 1.1, 1.1, 1.1, 5.1, 1.1],
                  [0.6, 1.1, 1.1, 1.1, 1.1, 1.1]])

    times = []
    for _ in range(1000):
        start = time.time()
        np.linalg.inv(c)
        elapsed = time.time() - start
        times.append(elapsed)
    
    m = np.mean(times)
    print("Avg numpy.linalg.inv(): {} ({} Hz)".format(m, 1./m))

    times = []
    for _ in range(1000):
        start = time.time()
        cv.invert(c, flags=cv.DECOMP_CHOLESKY)
        elapsed = time.time() - start
        times.append(elapsed)
    
    m = np.mean(times)
    print("Avg cv2.invert(cv.DECOMP_CHOLESKY): {} ({} Hz)".format(m, 1./m))


if __name__ == "__main__":
    test_cvinvert_vs_npinv()