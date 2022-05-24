import os
import cv2 as cv
import numpy as np

def readLabelFile(filepath):
    landmarks = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                col, row = line.split()
                col, row = int(col), int(row)
                landmarks.append([col, row])

    assert len(landmarks) == 8, "Wrong number of landmarks in '{}'".format(os.path.basename(filepath))
    return landmarks

def tuneLandmarks(path, labels, radius):
    tunedLabels = {}
    for basename in labels:
        tunedLandmarks = []
        img = cv.imread(os.path.join(path, basename) + ".jpg")
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        for lm in labels[basename]:
            x,y = lm
            patch = gray[y-radius:y+radius+1, x-radius:x+radius+1]
            maxIndx = np.unravel_index(np.argmax(patch), patch.shape)

            x += maxIndx[1] - radius
            y += maxIndx[0] - radius

            tunedLandmarks.append([x,y])

        tunedLabels[basename] = tunedLandmarks

    return tunedLabels

if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-tune', action='store_true', default=False, help="Tune labels such that they are located at the local maxima")
    args = parser.parse_args()

    labels = {}
    path = "../image_dataset/dataset_recovery"

    for f in os.listdir(path):
        filepath = os.path.join(path, f)
        filename, file_extension = os.path.splitext(filepath)
        filename = os.path.basename(filename)
        if "_bad" in filename:
            print("ignoring bad file '{}'".format(filename))
            continue

        try:
            int(filename)
        except:
            print("ignoring file '{}'".format(filename))
            continue
        else:
            if file_extension == ".txt":
                landmarks = readLabelFile(filepath)
                labels[filename] = landmarks

    if args.tune:
        print("Tuning labels")
        labels = tuneLandmarks(path, labels, radius=7)

    # Redundancy check
    for i in range(1, 2307):
        assert i == int(sorted(labels.keys(), key=lambda name: int(name))[i-1])
        
    videoName = "donn"
    savefile = videoName + ".txt"
    savePath = os.path.join(path, savefile)
    with open(savePath, "w") as f:
        for l in sorted(labels.keys(), key=lambda name: int(name)):
            s = videoName + "_" + l
            s += ":["
            for i, lMark in enumerate(labels[l]):
                s += "'({},{},{})'".format(lMark[0], lMark[1], 10) # arbitrary radius of 10
                if i < len(labels[l])-1:
                    s += ", "
            s += "]\n"

            f.write(s)