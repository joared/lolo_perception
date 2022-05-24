import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import signal
from scipy import ndimage

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def gaussianBlur(y, kernelSize):
    sigma = 0.3*((kernelSize-1)*0.5 - 1) + 0.8
    yBlur = ndimage.gaussian_filter(y, sigma, order=0)
    return yBlur

def meanBlur(y, kernelSize):
    yBlur = []
    i = 0
    while True:
        if i >= len(y):
            break
        mean = sum(y[i:i+kernelSize])/kernelSize
        yBlur.append(mean)
        i += 1

    return np.array(yBlur)

def medianBlur(y, kernelSize):
    yBlur = []
    i = 0
    while True:
        if i >= len(y):
            break
        median = np.median(y[i:i+kernelSize])
        yBlur.append(median)
        i += 1

    return np.array(yBlur)

def localPeak(y, blurSize, locMaxSize, p):
    # local max on non-blured image
    locMaxIdxs = list(signal.find_peaks(y, width=locMaxSize)[0])
    if blurSize > 0:
        y = gaussianBlur(y, blurSize)

    # local max on blured image
    #locMaxIdxs = list(signal.find_peaks(y, width=locMaxSize)[0])

    locMax = [y[i] if i in locMaxIdxs else 0 for i in range(len(y))]
    locMaxTemp = [y[i] if i in locMaxIdxs else 0 for i in range(len(y))]


    peaks = [] # [(peak, left, right), ...]
    removedPeaks = []
    edgePeaks = []
    iterations = 0
    while True:
        iterations += 1
        #print(locMaxTemp)
        maxIdx = np.argmax(locMaxTemp)
        maxI = locMaxTemp[maxIdx]
        
        
        if maxI <= 0:
            break
        threshold = maxI*p

        # find left side of peak
        lefti = 0
        while True:
            lefti += 1
            idx = maxIdx -lefti
            if idx > 0:
                if y[idx] < threshold:
                    break
            else:
                break
        righti = 0
        while True:
            righti += 1
            idx = maxIdx + righti
            if idx < len(locMaxTemp)-1:
                if y[idx] < threshold:
                    break
            else:
                break

        # Connected to edge?
        if maxIdx-lefti == 0 or maxIdx+righti+1 == len(y):
            print("edge")
            edgePeaks.append((maxIdx, lefti, righti))
            locMaxTemp[maxIdx-lefti:maxIdx+righti+1] = [0]*len(locMaxTemp[maxIdx-lefti:maxIdx+righti+1])
            continue

        for peak, l, r in peaks:
            # other peaks within this peak?
            if peak > maxIdx-lefti and peak < maxIdx+righti:
                # Don't add, within other peak
                print("other peak within")
                removedPeaks.append((maxIdx, lefti, righti))
                break
        else:
            peaks.append((maxIdx, lefti, righti))

        locMaxTemp[maxIdx-lefti:maxIdx+righti+1] = [0]*len(locMaxTemp[maxIdx-lefti:maxIdx+righti+1])
    print(iterations)
    return peaks, removedPeaks, edgePeaks, y, np.array(locMax)

if __name__ == "__main__":
    np.random.seed(12345)

    # peaks we want to find
    loc1 = 35
    loc2 = 65
    loc3 = 50
    y1 = lambda x: gaussian(x, loc1, 3)*100
    y2 = lambda x: gaussian(x, loc2, 3)*70
    y3 = lambda x: gaussian(x, loc3, 5)*255*2/3.

    n1 = lambda x: gaussian(x, 29, 1)*50*0.44
    n2 = lambda x: gaussian(x, 59, 1)*40*0.44

    energy = lambda x: gaussian(x, 50, 100)*400
    ambient = lambda x: gaussian(x, 212, 60)*400


    x = np.arange(0, 200, 0.1)
    
    noiseFuncs = []
    for i in range(100):
        mean = np.random.randint(1, 200)
        sigma = np.random.normal(.1, 1)
        intensity = np.random.randint(1, 5)
        n = gaussian(x, mean, sigma)*intensity
        noiseFuncs.append(n)

    noise = sum([n for n in noiseFuncs])
    noise += n1(x) + n2(x)
    gaussNoise = np.random.normal(0, .005*255, len(x))#.005

    yTrue = y1(x) + y2(x) + y3(x) + energy(x) + ambient(x)
    yTrue = 255./max(yTrue)*yTrue

    yNoised = (yTrue + noise + gaussNoise)
    
    locMaxSize = 7
    blurSize = 3 # 3
    p = .955

    peaks, removedPeaks, edgePeaks, yPeak, locMax = localPeak(yNoised, blurSize, locMaxSize, p=p)
    plt.scatter(x[np.nonzero(locMax)], locMax[np.nonzero(locMax)], color="g")
    plt.fill_between(x, yPeak, alpha=0.4, interpolate=True, color="b")

    plt.vlines(loc1, ymin=0, ymax=yNoised[np.where(x == loc1)], colors="black")
    plt.vlines(loc2, ymin=0, ymax=yNoised[np.where(x == loc2)], colors="black")
    plt.vlines(loc3, ymin=0, ymax=yNoised[np.where(x == loc3)], colors="black")
    plt.xlabel("1D pixel")
    plt.ylabel("Intensity")
    plt.legend(["local max"])
    
    #ax = plt.gca()
    #ax.add_patch(Rectangle((61,200), 7.5, 30, fill=False, edgecolor="black"))

    plt.figure()
    plt.scatter(x[np.nonzero(locMax)], locMax[np.nonzero(locMax)], color="g", label="local max")


    if True:
        legend = {"b": "found", "r":"overlapping", "y":"edge"}
        for color, peaks in zip(("b", "r", "y"), (peaks, removedPeaks, edgePeaks)):
            for i, peak in enumerate(peaks):
                idx = peak[0]
                lidx = peak[1]
                ridx = peak[2]

                label = legend[color] if i == 0 else None
                plt.scatter(x[idx], yPeak[idx], color=color, label=label)
                
                threshold = yPeak[idx]*p
                plt.hlines(threshold, xmin=x[idx-lidx], xmax=x[idx+ridx], colors=color)
                
                plt.vlines(x[idx], ymin=threshold, ymax=yPeak[idx], colors=color)

                # adjust center with centroid
                if color == "b":
                    center = int((idx-lidx + (ridx+lidx)/2.0))
                    label = "adjusted center" if i == 0 else None
                    plt.scatter(x[center], yPeak[idx], color="purple", label=label)

    plt.fill_between(x, yPeak, alpha=0.4, interpolate=True, color="b")
    plt.vlines(loc1, ymin=0, ymax=yNoised[np.where(x == loc1)], colors="black")
    plt.vlines(loc2, ymin=0, ymax=yNoised[np.where(x == loc2)], colors="black")
    plt.vlines(loc3, ymin=0, ymax=yNoised[np.where(x == loc3)], colors="black")
    plt.xlabel("1D pixel")
    plt.ylabel("Intensity")
    
    plt.show()