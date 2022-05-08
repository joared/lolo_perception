import os

def readLabelFile(filepath):
    landmarks = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                print(line)
                col, row = line.split()
                col, row = int(col), int(row)
                landmarks.append([col, row])

    assert len(landmarks) == 8, "Wrong number of landmarks in '{}'".format(os.path.basename(filepath))
    return landmarks

if __name__ == "__main__":
    
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

    # Redundancy check
    for i in range(1, 2307):
        assert i == int(sorted(labels.keys(), key=lambda name: int(name))[i-1])

    videoName = "donn"
    savefile = videoName + ".txt"
    savePath = os.path.join(path, savefile)
    with open(savePath, "w") as f:
        for l in sorted(labels.keys(), key=lambda name: int(name)):
            print(l)
            s = videoName + "_" + l
            s += ":["
            for i, lMark in enumerate(labels[l]):
                s += "'({},{},{})'".format(lMark[0], lMark[1], 10) # arbitrary radius of 10
                if i < len(labels[l])-1:
                    s += ", "
            s += "]\n"

            f.write(s)