from lolo_perception.feature_extraction import pDecay
import numpy as np
from matplotlib import pyplot as plt



if __name__ == "__main__":
    b = .17501
    pMin = .8
    pMax = .975
    I = np.array(range(1,256))*1.0

    p = pDecay(b, pMin, pMax, I, IMax=255.)

    print(p)
    plt.plot(p)
    plt.show()