import cv2 as cv
import argparse

if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()
    parser.add_argument('file', help="File to be converted to ROI")
    parser.add_argument('roi', help="Image ROI x,y,w,h")
    
    args = parser.parse_args()
    
    img = cv.imread(args.file)
    if img is None:
        raise Exception("Image '{}' doesn't seem to exist")

    try:
        x, y, w, h = map(int, [s.strip() for s in args.roi.split(",")])
    except Exception as e:
        print(e)
        raise Exception("ROI could not be parsed '{}'".format(args.roi))

    cv.imshow("image", img[y:y+h, x:x+w])
    cv.waitKey(0)
    cv.destroyAllWindows()