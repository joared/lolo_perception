import os
import yaml
import threading
import multiprocessing
import time
import cv2 as cv
import numpy as np
import sys
import select


class ImageDataset:

    LABEL_FILE_NAME = "dataset_labels.txt"
    METADATA_FILE_NAME = "dataset_metadata.yaml"
    IMAGE_NAME_FORMAT = "image_{index}{extension}"
    IMAGE_FORMAT = ".png"

    def __init__(self, datasetDir, loadBatchSize=500, maxNLoadedImages=2000):
        if not os.path.isdir(datasetDir):
            raise Exception("Dataset '{}' does not exist, use create() to create a new dataset.".format(datasetDir))

        self._datasetDir = datasetDir # The directory which contains the dataset. Use create() to create a new dataset.

        self.labeledImages = self.readLabels()  # {image_X: labels}
        self.metadata = self.readMetadata()     # {data: value}
        self._size = self.readSize()            # Counts all the images (with correct name format)
        
        # Number of threads = maxLoadedImages/batchSize
        self.loader = ImageLoader(self._datasetDir, 
                                  datasetSize=self._size,
                                  idxToImageNameFunc=self.idxToImageName,
                                  batchSize=loadBatchSize, # 50/500
                                  maxNLoadedImages=maxNLoadedImages) # 500/2000


    def __iter__(self):
        for imgIdx in range(self._size):
            yield self.loadImage(imgIdx)


    def __len__(self):
        return self._size


    def isReady(self):
        return self.loader.isReady()


    def waitUntilReady(self, waitSleep=1, printInfo=True):
        # Initialize loading
        self.loadImage(0)

        while True:
            if not self.isReady():
                if printInfo:
                    os.system('cls' if os.name == 'nt' else 'clear')
                    self.loader.printBatchStatus(0)
                    print("Loading dataset...")
            else:
                break
            time.sleep(waitSleep)

        if printInfo:
            os.system('cls' if os.name == 'nt' else 'clear')
            self.loader.printBatchStatus(0)
            print("Dataset loaded!")


    def readSize(self):
        l = 0
        i = 0
        while True:
            filePath = os.path.join(self._datasetDir, self.idxToImageName(i))
            if os.path.isfile(filePath):
                l += 1
            else:
                break
            i += 1

        return l


    def getImage(self, imgIdx):
        return self.loader.getImage(imgIdx)


    def loadImage(self, imgIdx):
        return self.loader.loadImage(imgIdx)


    @classmethod
    def idxToImageName(cls, i):
        return cls.IMAGE_NAME_FORMAT.format(index=i, extension=cls.IMAGE_FORMAT)


    @classmethod
    def create(cls, datasetDir, imageGenerator, metadata, startIdx=0, endIdx=np.inf, overwriteExisting=False):
        if not overwriteExisting and os.path.isdir(datasetDir):
            raise Exception("Dataset {} already exists. You can overwrite an existing dataset by using overwriteExisting=True.".format(datasetDir))

        # Create the directory
        os.mkdir(datasetDir)
        print("Created dataset {}".format(datasetDir))

        # Create metadata file
        with open(os.path.join(datasetDir, cls.METADATA_FILE_NAME), "w"):
            print("Created '{}'".format(cls.METADATA_FILE_NAME))

        # Create label file
        with open(os.path.join(datasetDir, cls.LABEL_FILE_NAME), "w"):
            print("Created '{}'".format(cls.LABEL_FILE_NAME))            

        dataset = cls(datasetDir)
        dataset.metadata = metadata
        dataset.save()

        print("Saving images in {} format...".format(cls.IMAGE_FORMAT))
        saver = ImageSaver(datasetDir, 
                           nActiveBatches=8, 
                           maxNImages=200*8, 
                           imgIdxToNameFunc=cls.idxToImageName)
        saver.saveImages(imageGenerator, startIdx, endIdx)

        return cls(datasetDir)


    def zipImages(self):
        import zipfile
        print("Zipping images")
        with zipfile.ZipFile(os.path.join(self._datasetDir, "images.zip"), mode="w", allowZip64=True) as archive:
            for imgIdx in range(self._size):
                print("Image {}".format(imgIdx))
                imageFile = os.path.join(self._datasetDir, self.idxToImageName(imgIdx))
                archive.write(imageFile)
        print("Done zipping images")


    def addLabels(self, labels, imgIdx):
        self.labeledImages[imgIdx] = labels


    def getLabels(self, imgIdx):
        if imgIdx in self.labeledImages:
            return self.labeledImages[imgIdx]


    def saveLabels(self):
        with open(os.path.join(self._datasetDir, self.LABEL_FILE_NAME), "w") as f:
            for imgIdx in self.labeledImages:
                if self.labeledImages[imgIdx]:
                    labelsText = ["({},{},{})".format(x,y,r) for x,y,r in self.labeledImages[imgIdx]]
                    f.write("{}:{}\n".format(imgIdx, labelsText))


    def readLabels(self):
        labeledImgs = {}
        with open(os.path.join(self._datasetDir, self.LABEL_FILE_NAME), "r") as f:
            for line in f:
                imgIdx, labels = line.split(":")
                imgIdx = int(imgIdx)

                labels = labels.strip()[1:-1] # remove []
                labels = labels.split(", ")
                labels = [tuple(map(int, s[2:-2].split(","))) for s in labels]
                
                labeledImgs[imgIdx] = labels

        return labeledImgs


    def save(self):
        self.saveMetadata()
        self.saveLabels()


    def saveMetadata(self):
        with open(os.path.join(self._datasetDir, self.METADATA_FILE_NAME), "w") as f:
            yaml.dump(self.metadata, f, default_flow_style=False)


    def readMetadata(self):
        with open(os.path.join(self._datasetDir, self.METADATA_FILE_NAME), "r") as f:
            metadata = yaml.safe_load(f)

        return metadata


    def saveVideo(self, fps):
        # TODO: not used atm
        height,width, _ = self.loadImage(0).shape

        # Compressed video
        #video = cv.VideoWriter(os.path.join(self._datasetDir, "video.avi"), 0, self.fps, (width,height))

        # Lossless video
        video = cv.VideoWriter(os.path.join(self._datasetDir, "video.avi"), cv.VideoWriter.fourcc("F", "F", "V", "1"), fps, (width,height))
        print("Saving video")
        for i in range(self._size):
            video.write(self.loadImage(i))
        video.release()


    def printInfo(self, printLoadingStatus=False):
        nameStr = "-------------- {} --------------".format(os.path.basename(self._datasetDir))
        print(nameStr)
        for k in self.metadata:
            print("{}: {}".format(k, self.metadata[k]))

        print("Labeled images: {}/{}".format(len(self.labeledImages), self._size))

        if printLoadingStatus:
            print("Loaded: {}".format(self.isReady()))


class ImageLoader:

    class BatchLoader:
        def __init__(self, datasetDir, batchSize, size, batchIdx, idxToImageNameFunc, debug=False):
            self.datasetDir = datasetDir
            self.batchIdx = batchIdx
            self.size = size # Actual size of the batch (last usually smaller)
            self.batchSize = batchSize
            self.idxToImageNameFunc = idxToImageNameFunc
            self.images = []

            self._reset()
            self._readError  = False # Only used for status atm
            self.debug = debug


        def __len__(self):
            return len(self.images)


        def _reset(self):
            self._isActive = False
            self._isDone = False
            self._cancel = False
            self.images = []       


        def isReady(self):
            return not self._isActive


        def load(self):
            if self._isActive or self._isDone:
                return

            self._reset()
            self._isActive = True
            
            x = threading.Thread(target=self._load, args=())
            x.setDaemon(True)
            x.start()

            
        def _load(self):
            if self.debug:
                print("Loading batch {}".format(self.batchIdx))

            for i in range(self.size):
                if self._cancel:
                    self.images = [] # TODO: necessary? Done in reset
                    self._reset()
                    if self.debug:
                        print("Canceled batch {}".format(self.batchIdx))
                    return
                imgIdx = i + self.batchIdx*self.batchSize
                img = self.readImage(imgIdx)
                if img is not None:
                    self.images.append(img)
                else:
                    self._readError = True
                    break
            
            self._isActive = False
            self._isDone = True
            if self.debug:
                print("Batch {} done".format(self.batchIdx))


        def cancelLoad(self):
            if self._isDone and self.debug:
                
                print("Resetting batch {}".format(self.batchIdx))
            self._reset()
            self._cancel = True


        def getImage(self, imgIdx):
            i = imgIdx-self.batchIdx*self.batchSize
            if i < 0 or i >= self.batchSize:
                raise Exception("Wrong index")

            if i < len(self.images):
                return self.images[i]
            else:
                if self.debug:
                    print("Image not loaded yet")


        def readImage(self, imgIdx):
            fileName = os.path.join(self.datasetDir, self.idxToImageNameFunc(imgIdx))
            img = cv.imread(fileName)
            if img is None:
                raise Exception("Read error: image '{}' could not be read.".format(fileName))

            return img


    def __init__(self, datasetDir, datasetSize, idxToImageNameFunc, batchSize, maxNLoadedImages):
        self.datasetDir = datasetDir
        self.datasetSize = datasetSize
        self.batchSize = batchSize
        self.nBatches = 1 if datasetSize <= self.batchSize else datasetSize/self.batchSize+min(datasetSize%self.batchSize, 1)
        self.batches = []
        for i in range(self.nBatches):
            b = self.BatchLoader(datasetDir, 
                                 self.batchSize, 
                                 self.batchSize if datasetSize/((i+1)*self.batchSize) >= 1 else datasetSize%self.batchSize,
                                 i, 
                                 idxToImageNameFunc)
            self.batches.append(b)

        self.maxNLoadedImages = maxNLoadedImages
        self.nActiveBatches = min(self.nBatches, maxNLoadedImages/self.batchSize)
        self.centerBatch = max(0, self.nActiveBatches/2-1)
        self._currentBatchIdx = -1


    def __len__(self):
        return self.datasetSize


    def isReady(self):
        for b in self.batches:
            if not b.isReady():
                return False
        return True


    def getImage(self, imgIdx):
        theBatchIdx = imgIdx / self.batchSize
        theBatch = self.batches[theBatchIdx]
        theImage = theBatch.getImage(imgIdx)
        if theImage is None:
            print("Image not loaded reading from disk...")
            theImage = theBatch.readImage(imgIdx)
            if theImage is None:
                print("Image does not exist (yet)")
        return theImage


    def loadImage(self, imgIdx):
        theBatchIdx = imgIdx / self.batchSize
        if theBatchIdx == self._currentBatchIdx:
            theBatch = self.batches[theBatchIdx]
            theImage = theBatch.getImage(imgIdx)
            if theImage is None:
                #if self.debug:
                #print("Image not loaded reading from disk...")
                theImage = theBatch.readImage(imgIdx)
            return theImage
        
        self._currentBatchIdx = theBatchIdx

        firstActiveBatch = theBatchIdx-self.centerBatch
        lastActiveBatch = theBatchIdx + self.nActiveBatches-1 -self.centerBatch

        if firstActiveBatch < 0:
            lastActiveBatch += abs(firstActiveBatch)
            firstActiveBatch = 0

            if lastActiveBatch >= self.nBatches:
                raise Exception("Last active batch idx calc failed")

        elif lastActiveBatch >= self.nBatches:
            firstActiveBatch -= lastActiveBatch - self.nBatches + 1
            
            lastActiveBatch = self.nBatches -1

            if firstActiveBatch < 0:
                raise Exception("First active batch idx calc failed")

        theImage = None
        for b in self.batches:
            if b.batchIdx < firstActiveBatch:
                b.cancelLoad()
            elif b.batchIdx == theBatchIdx:
                #if not b._isDone and not b._isActive:
                b.load()
                theImage = b.getImage(imgIdx)
                if theImage is None:
                    #print("Image not loaded reading from disk...")
                    theImage = b.readImage(imgIdx)
            elif b.batchIdx <= lastActiveBatch:
                b.load()
            else:
                b.cancelLoad()

        return theImage

    def loadPercentage(self):
        toBeLoaded = 0
        nLoaded = 0
        for b in self.batches:
            if b._isActive or b._isDone:
                toBeLoaded += b.size
                nLoaded += len(b.images)

        return int(round(nLoaded/float(toBeLoaded)*100))

    def printBatchStatus(self, imgIdx):
        theBatchIdx = imgIdx / self.batchSize
        
        for b in self.batches:
            if b._readError:
                status = "Read error"
            elif b._isDone:
                status = "Done"
            elif b._isActive:
                status = "Loading"
            else:
                status = "Inactive"

            status += " {}/{}".format(len(b), b.size)

            if b.batchIdx == theBatchIdx:
                status += " <-- {}".format(imgIdx)

            print("Batch {} ({}-{}): {}".format(b.batchIdx, b.batchIdx*b.batchSize, b.batchIdx*b.batchSize+b.size-1, status))


    def drawBatchStatus(self, img, imgIdx):
        width = 150
        height = 15
        centerX = int(img.shape[1]/1.1)
        start = (centerX-int(width/2), img.shape[0]-20)
        end = (centerX+int(width/2), start[1]-height)

        # Draw a rectangle that represents the loading bar
        thickness = 1
        startLoadWindow = (start[0]-thickness, start[1]+thickness)
        endLoadWindow = (end[0]+thickness, end[1]-thickness)
        cv.rectangle(img, startLoadWindow, endLoadWindow, (0,255,0), thickness)
        cv.rectangle(img, start, end, (50,50,50), -1)

        statusColors = {"Loading": (0,255,255), 
                        "Done":(0,255,0),
                        "Inactive":(50,50,50),
                        "Read error":(0,0,255)}

        wRatio = width/float(self.datasetSize)

        # Draw the rectangles representing the amount of images loaded by each batch
        for i, b in enumerate(self.batches):
            if b._readError: status = "Read error"
            elif b._isDone: status = "Done"
            elif b._isActive: status = "Loading"
            else: status = "Inactive"
            
            bStart = (int(start[0] + wRatio*b.batchSize*i), start[1])
            bEnd = (int(bStart[0]+wRatio*len(b)), end[1])

            # Just to prevent the last load indicator to not stop at one pixel too short
            if i == len(self.batches)-1 and b._isDone:
                bEnd = (end[0], end[1])
            
            if len(b) == 0:
                continue
            color = statusColors[status]
            cv.rectangle(img, bStart, bEnd, color, -1)

            bStart = (bStart[0]+wRatio*b.size, bStart[1])

        # Draw a blue line indicating the location of the current image index
        x = start[0] + int(imgIdx*wRatio)
        cv.line(img, (x, start[1]), (x, end[1]), (255,0,0), thickness)

        # Draw an info text indicating if the load is complete
        ready = self.isReady()
        infoText = "Images ready" if ready else "Loading images {}%".format(self.loadPercentage())
        color = statusColors["Done"] if ready else statusColors["Loading"]
        cv.putText(img, infoText, (start[0], end[1]-height), cv.FONT_HERSHEY_SIMPLEX, .3, color, 1, cv.LINE_AA)

        return img
        

class ImageSaver:

    class BatchSaver:
        def __init__(self, datasetDir, batchSize, size, batchIdx, imgIdxToNameFunc):
            self.datasetDir = datasetDir
            self.batchSize = batchSize
            self.batchIdx = batchIdx
            self.size = size
            self.imgIdxToName = imgIdxToNameFunc

            self._isActive = False
            self.n = 0 # Updated in _save()
            self.x = None # The thread


        def isActive(self):
            return self._isActive


        def save(self, images):
            if self._isActive:
                raise Exception("Batch saver is already active.")

            self._isActive = True
            self.n = 0
            
            self.x = threading.Thread(target=self._save, args=(images,))
            self.x.daemon = True
            self.x.start()


        def _save(self, images):
            if not images:
                raise Exception("'images' is empty")

            self.n = 0
            for i, img in enumerate(images):
                imgIdx = i + self.batchIdx*self.batchSize
                retval = cv.imwrite(os.path.join(self.datasetDir, self.imgIdxToName(imgIdx)), img)
                if not retval:
                    raise Exception("Failed to write {}".format(self.imgIdxToName(imgIdx)))
                self.n += 1

            self._isActive = False


    def __init__(self, datasetDir, nActiveBatches, maxNImages, imgIdxToNameFunc):
        self.datasetDir = datasetDir
        self.nActiveBatches = nActiveBatches
        self.maxNImages = maxNImages
        self.imgIdxToName = imgIdxToNameFunc
        self.batchSize = self.maxNImages/self.nActiveBatches

        self.batches = []

        self.imgIdx = 0 # updated in _getImageBatch()


    def saveImages(self, imageGenerator, startIdx=0, endIdx=None, verbose=1, waitUntilDone=True):
        batchIdx = 0
        if endIdx is None:
            endIdx = np.inf
        while True:
            if verbose:
                self.printBatchStatus()
            
            while [b.isActive() for b in self.batches].count(True) >= self.nActiveBatches: pass

            imgBatch = self._getImageBatch(imageGenerator, self.batchSize, startIdx, endIdx)
            if not imgBatch:
                break
            b = self.BatchSaver(self.datasetDir, self.batchSize, len(imgBatch), batchIdx, self.imgIdxToName)
            self.batches.append(b)
            batchIdx += 1
    
            b.save(imgBatch)
            
        if waitUntilDone:
            self.waitUntilDone(verbose)

    
    def _getImageBatch(self, imageGenerator, batchSize, startIdx, endIdx):
        images = []

        while len(images) < batchSize:
            
            if self.imgIdx > endIdx:
                #print("breaking", self.imgIdx, endIdx)
                break
            
            try:
                img = imageGenerator.next()
            except StopIteration:
                break
            
            if self.imgIdx >= startIdx:
                #print("continuing", self.imgIdx, startIdx)
                #pass
                images.append(img)
            self.imgIdx += 1

        return images


    def waitUntilDone(self, verbose=1):
        while any([b.isActive() for b in self.batches]):
            if verbose:
                os.system('cls' if os.name == 'nt' else 'clear')
                print("Waiting for batches to finnish...")
                self.printBatchStatus()
            time.sleep(1)

        if verbose:
            os.system('cls' if os.name == 'nt' else 'clear')
            self.printBatchStatus()
            print("Images saved")


    def printBatchStatus(self):
        for b in self.batches:
            if b.isActive():
                status = "Saving"
            else:
                status = "Done"
            status += " {}/{}".format(b.n, b.size)

            print("Batch {} ({}-{}): {}".format(b.batchIdx, b.batchIdx*b.batchSize, b.batchIdx*b.batchSize+b.size-1, status))
