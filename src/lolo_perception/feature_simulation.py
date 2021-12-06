import cv2 as cv
import numpy as np

def psf(I0, r):
    """
    theta - angle
    b - scattering coefficient
    tau = optical length
    theta0 - mean scatttering angle
    K - functional constant dependent on scattering angle theta0
    w0 - scattering albedo
    """

    def K(theta0):
        #return 1*(np.pi-theta0)
        return 1

    theta = np.pi/2
    theta0 = np.pi/2
    w0 = .87
    b = .1
    L = 1
    tau = 1.39807 * r/20
    m = 1/(w0 - 2*r*theta0)
    #r = 7

    #return K(theta0)*np.random.poisson(lam=b*r)*np.exp(-tau) / (2*np.pi*pow(theta, m))
    
    #return I0*np.exp(-b*r)/10
    den = (2*np.pi*pow(theta, m))
    #print(den)
    return I0*np.exp(-b*r)/ den

def scatterPoint(img, point, r, size):
    r = r*2.2 # hardcoded to make simulation a bit nicer
    I0 = img[point[0], point[1]]
    I0 = psf(I0, r)
    img[point[0], point[1]] = I0

    for dx in range(-size, size):
        for dy in range(-size, size):
            x = point[0] + dx
            y = point[1] + dy
            if dx == 0 and dy == 0:
                continue
            
            if x > 0 and x < img.shape[0] and y > 0 and y < img.shape[1]:
                xyr = np.linalg.norm([dx, dy])*2
                img[x, y] += psf(I0, xyr)
                img[x, y] = min(img[x, y], 255)

if __name__ == "__main__":
    import time
    size = (480, 640)
    img = np.zeros(size, dtype=np.float32)
    
    points = [(np.random.randint(480), np.random.randint(640)) for i in range(10)]
        
    for point in points:
        img[point[0], point[1]] = 255

    for point in points:
        scatterPoint(img, point, r=3, size=50)

    size = 20
    #for point in points:
    #    cv.rectangle(img, (point[1]-size, point[0]-size), (point[1]+size, point[0]+size), color=(1,0,0))
    cv.imshow("Image", img)
    cv.waitKey(0)