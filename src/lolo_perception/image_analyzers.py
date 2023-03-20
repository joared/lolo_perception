import cv2 as cv
import numpy as np

class AbstractImageAnalyzer:
    def __init__(self):
        self.currentPoint = None

    def click(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.currentPoint = (x, y)

        # check to see if the left mouse button was released
        elif event == cv.EVENT_LBUTTONUP:
            pass
        
    def _analyze(self, img):
        # Not implemented
        pass

    def analyze(self, imgName, img):
        cv.setMouseCallback(imgName, self.click)
        img = self._analyze(img)

        cv.imshow(imgName, img)
        #key = cv.waitKey(1) & 0xFF
        key = 0

        return key, img

class FloodFillAnalyzer(AbstractImageAnalyzer):
    def __init__(self, debug=False):
        AbstractImageAnalyzer.__init__(self)

    def _floodFill(self, center, img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        floodfillMask = gray.copy()
        mask = np.zeros((gray.shape[0] + 2, gray.shape[1] + 2), dtype=np.uint8)
        p = 0.97
        loDiff = int(gray[center[1], center[0]]*(1-p))
        retval, image, mask, rect = cv.floodFill(floodfillMask, mask, center, 255, loDiff=loDiff, upDiff=255, flags=cv.FLOODFILL_FIXED_RANGE)
        mask *= 255
        cv.imshow("floodfill", mask)
        mask = mask[1:mask.shape[0]-1, 1:mask.shape[1]-1]
        # Find contours of unique color
        _, contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        print("CONTOURS", len(contours))
        cv.drawContours(img, contours, -1, (255,0,0), 1)
        return img

    def _analyze(self, img):
        # import win32api
        # import win32con
        # win32api.SetCursor(win32api.LoadCursor(0, win32con.IDC_SIZEALL))
        if self.currentPoint:
            img = self._floodFill(self.currentPoint, img)

        return img