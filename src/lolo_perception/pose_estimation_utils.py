from cmath import exp
import numpy as np
import cv2 as cv
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

def interactionMatrix(x, y, Z):
    """
    IBVS interaction matrix
    """
    return [[-1/Z, 0, x/Z, x*y, -(1+x*x), y],
                [0, -1/Z, y/Z, 1+y*y, -x*y, -x]]

def skew(m):
    return np.array([[   0, -m[2],  m[1]], 
            [ m[2],    0, -m[0]], 
            [-m[1], m[0],     0]])

def unSkew(mat):
    w = np.array([mat[2, 1], mat[0, 2], mat[1, 0]])
    return w

def matFromRotvec(w):
    theta = np.linalg.norm(w)
    if theta == 0:
        return np.eye(3)*1.0
    wNorm = w/theta
    skewW = skew(wNorm)
    skewW2 = np.matmul(skewW, skewW)
    return np.eye(3) + np.sin(theta)*skewW + (1-np.cos(theta))*skewW2

def rotvecFromMat(rotMat):
    thetaNew = np.arccos((np.trace(rotMat) -1)/2.0)
    rVecCross = thetaNew/2/np.sin(thetaNew)*(rotMat-rotMat.transpose())
    rVecNew = unSkew(rVecCross)

    # if theta ~ pi ?
    if abs(thetaNew - np.pi) < 10e-8:
        print("pi detected")
        B = 0.5 * (rotMat + np.eye(3))
        ax = np.sqrt(B[0,0])
        ay = np.sqrt(B[1,1])
        az = np.sqrt(B[2,2])

        # TODO: which off diagonal elements
        signs = np.sign(unSkew(B))
        #signs = np.sign(B[0, 1]), np.sign(B[0, 2]), np.sign(B[1, 0])
        
        rVecNewNorm = np.array([signs[0]*ax, signs[1]*ay, signs[2]*az])
        rVecNew = thetaNew*rVecNewNorm

    elif thetaNew == 0:
        rVecNew = np.array([0., 0., 0.])

    return rVecNew

def projectPoints2(tVec, rVec, cameraMatrix, objectPoints):
    rotMat = R.from_rotvec(rVec).as_dcm()
    points3D = np.matmul(rotMat, objectPoints.transpose()).transpose() + tVec
    imgPoints = np.matmul(cameraMatrix, points3D.transpose()).transpose()
    imgPoints[:, :] /= imgPoints[:, [-1]]
    imgPoints = imgPoints[:, :2]
    return imgPoints

def imageJacobian(tVec, rVec, cameraMatrix, objectPoint):
    rot = R.from_rotvec(-rVec)

    imgJacobian = np.zeros((2, 6), dtype=np.float32)

    imgPoints = projectPoints2(tVec, rVec, cameraMatrix, np.array([objectPoint]))
    imgPoint = imgPoints[0]

    for i in range(6):
        tVecTemp = tVec.copy()
        if i < 3:
            tVecTemp[i] += 1
        else:
            w = np.zeros(3, dtype=np.float32)
            w[i-3] = 1
            w = R.from_rotvec(w)
            rotTemp = R.from_rotvec(rVec)
            rVecTemp = ""
        imgPointsTemp = projectPoints2(tVecTemp, rVecTemp, cameraMatrix, np.array([objectPoint]))
        imgPointTemp = imgPointsTemp[0]
        du = (imgPointTemp[0] - imgPoint[0])
        dv = (imgPointTemp[1] - imgPoint[1])
        imgJacobian[:, i] = du, dv

    return imgJacobian

def objectJacobianCamera(cameraMatrix, tVec, rVec, objectPoints):
    # Projection of a point: - e + A + p
    # http://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf
    # Uses "left" jacobian
    J = np.zeros((2*len(objectPoints), 6))

    fx = cameraMatrix[0, 0]
    fy = cameraMatrix[1, 1]

    for i, p in enumerate(objectPoints):
        X, Y, Z = R.from_rotvec(rVec).apply(p) + tVec
    
        Jt = [[fx/Z, 0, -fx*X/Z**2],
              [0, fy/Z, -fy*Y/Z**2]]

        right = np.array([[],
                          [], 
                          []])

        Jp = np.hstack((-rotMat.transpose(), ))

        intMat = np.array(interactionMatrix(X/Z, Y/Z, Z))
        intMat[0, :]*=fx
        intMat[1, :]*=fy
        print(-intMat.round(2))



        J[i*2:i*2+2, :] = Jp

    return J

def objectJacobianPose(cameraMatrix, tVec, rVec, objectPoints):
    # Projection of a point: - e + A + p
    # http://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf
    # This is just the interaction matrix????
    # Uses "left" jacobian
    J = np.zeros((2*len(objectPoints), 6))

    fx = cameraMatrix[0, 0]
    fy = cameraMatrix[1, 1]

    for i, p in enumerate(objectPoints):
        X, Y, Z = R.from_rotvec(rVec).apply(p) + tVec

        Jp = [[fx/Z, 0, -fx*X/Z**2, -fx*X*Y/Z**2, fx*(1+X**2/Z**2), -fx*Y/Z],
              [0, fy/Z, -fy*Y/Z**2, -fy*(1+Y**2/Z**2), fy*X*Y/Z**2, fy*X/Z]]

        J[i*2:i*2+2, :] = Jp

    return J


def objectJacobianLie(cameraMatrix, tVec, rVec, objectPoints, method="global"):

    # method - gloabl (left multiplication) or local (right multiplication)

    J = np.zeros((2*len(objectPoints), 6))

    fx = cameraMatrix[0, 0]
    fy = cameraMatrix[1, 1]

    for i, p in enumerate(objectPoints):

        Xc, Yc, Zc = R.from_rotvec(rVec).apply(p) + tVec

        J1 = [[fx/Zc,    0, -fx*Xc/Zc**2],
              [   0, fy/Zc, -fy*Yc/Zc**2]]

        if method == "local":
            # This is dR/dR
            # https://www.youtube.com/watch?v=gy8U7S4LWzs&t=4067s

            # rotation: -R[p]_x
            X, Y, Z = p
            rotMat = R.from_rotvec(rVec).as_dcm()
            skewX = np.array([[0, -Z, Y],
                              [Z, 0, -X],
                              [-Y, X, 0]])
            mat = np.hstack((np.eye(3), np.matmul(rotMat, -skewX)))
            
            Jp = np.matmul(J1, mat)
            J[i*2:i*2+2, :] = Jp

        elif method == "global":
            # This is dR^T/dR and same as OpenCV applied
            # https://www.youtube.com/watch?v=gy8U7S4LWzs&t=4067s

            # rotation: -[Rp]_x
            X, Y, Z = R.from_rotvec(rVec).apply(p)
            skewX = np.array([[0, -Z, Y],
                            [Z, 0, -X],
                            [-Y, X, 0]])
            mat = np.hstack((np.eye(3), -skewX))

            Jp = np.matmul(J1, mat)
            J[i*2:i*2+2, :] = Jp
        else:
            raise Exception("Invalid method '{}'".format(method))

    return J

def objectJacobianLieGlobal(cameraMatrix, tVec, rVec, objectPoints):
    return objectJacobianLie(cameraMatrix, tVec, rVec, objectPoints, method="global")

def objectJacobianLieLocal(cameraMatrix, tVec, rVec, objectPoints):
    return objectJacobianLie(cameraMatrix, tVec, rVec, objectPoints, method="local")


def objectJacobianOpenCV(cameraMatrix, tVec, rVec, objectPoints, returnOpenCV=False):

    if returnOpenCV:
        _, jacobian = cv.projectPoints(objectPoints, 
                                    rVec.reshape((3, 1)), 
                                    tVec.reshape((3, 1)), 
                                    cameraMatrix, 
                                    np.zeros((1,4), dtype=np.float32))
        rotJ = jacobian[:, :3]
        transJ = jacobian[:, 3:6]
        J = np.hstack((transJ, rotJ))
        return J

    # https://arxiv.org/pdf/1312.0788.pdf
    J = np.zeros((2*len(objectPoints), 6))

    fx = cameraMatrix[0, 0]
    fy = cameraMatrix[1, 1]

    for i, p in enumerate(objectPoints):
        X, Y, Z = R.from_rotvec(rVec).apply(p) + tVec

        Jt = [[fx/Z, 0, -fx*X/Z**2],
              [0, fy/Z, -fy*Y/Z**2]]

        rotMat = R.from_rotvec(rVec).as_dcm()
        theta = np.linalg.norm(rVec)

        # this is the correct one (similar to right multiply)
        leftMat = np.matmul(-rotMat, skew(p))
        # this is the other one (similar to left multiply)
        #leftMat = skew(np.matmul(-rotMat, p))
        if theta != 0:
            mat = np.matmul(rVec.reshape(3,1), rVec.reshape(1,3))
            mat = (mat + np.matmul(rotMat.transpose()-np.eye(3), skew(rVec)))/theta**2
            mat = np.matmul(leftMat, mat)
        else:
            mat = leftMat

        Jr = np.matmul(Jt, mat)

        Jp = np.hstack((Jt, Jr))

        J[i*2:i*2+2, :] = Jp

    return J

def objectJacobianOpenCV2(cameraMatrix, tVec, rVec, objectPoints, mode="right"):
    # https://arxiv.org/pdf/1312.0788.pdf
    J = np.zeros((2*len(objectPoints), 6))

    fx = cameraMatrix[0, 0]
    fy = cameraMatrix[1, 1]

    for i, p in enumerate(objectPoints):
        X, Y, Z = R.from_rotvec(rVec).apply(p) + tVec

        Jt = [[fx/Z, 0, -fx*X/Z**2],
              [0, fy/Z, -fy*Y/Z**2]]

        rotMat = R.from_rotvec(rVec).as_dcm()
        theta = np.linalg.norm(rVec)

        # Eq (143) in micro theory
        thetaSkew = skew(rVec)

        # Doesn't matter if we use left or right here
        # Both models perturbations of the rotvec parameters

        if mode == "right":
            # Right
            mat = -np.matmul(rotMat, skew(p))
        elif mode == "left":
            # Left
            mat = -skew(np.matmul(rotMat, p))
        else:
            raise Exception("Invalid mode '{}'".format(mode))

        if theta != 0:
            JRight = np.eye(3) - (1-np.cos(theta))/theta**2*thetaSkew + (theta-np.sin(theta))/theta**3*np.matmul(thetaSkew, thetaSkew) 
            JLeft = JRight.transpose()
            
            if mode == "right":
                mat = np.matmul(mat, JRight)
            elif mode == "left":
                mat = np.matmul(mat, JLeft)


        Jr = np.matmul(Jt, mat)

        Jp = np.hstack((Jt, Jr))

        J[i*2:i*2+2, :] = Jp

    return J

def numericalJacobian(cameraMatrix, tVec, rVec, objectPoints, method="rvec", jacType=True):
    """
    method - rvec or euler order
    jacType - global, local or cross
    """
    deltaT = 0.00001
    J = np.zeros((2*len(objectPoints), 6), dtype=np.float32)

    imgPoints = projectPoints2(tVec, rVec, cameraMatrix, objectPoints)

    for i in range(6):
        tVecTemp = tVec.copy()
        rVecTemp = rVec.copy()
        if i < 3:
            tVecTemp[i] += deltaT
        else:
            if jacType in ("global", "local"):
                rVecTemp = np.array([0., 0., 0.])
            if method == "rvec":
                rVecTemp[i-3] += deltaT
            else:
                # euler
                euler = np.array(R.from_rotvec(rVecTemp).as_euler(method))
                euler[i-3] += deltaT
                rVecTemp = R.from_euler(method, tuple(euler)).as_rotvec()

            if jacType == "local":
                # local changes
                rTemp = R.from_rotvec(rVec)*R.from_rotvec(rVecTemp)
                rVecTemp = rTemp.as_rotvec()

            elif jacType == "global":
                # global changes
                rTemp = R.from_rotvec(rVecTemp)*R.from_rotvec(rVec)
                rVecTemp = rTemp.as_rotvec()

        imgPointsTemp = projectPoints2(tVecTemp, rVecTemp, cameraMatrix, objectPoints)
        du = (imgPointsTemp[:, 0] - imgPoints[:, 0])/deltaT
        dv = (imgPointsTemp[:, 1] - imgPoints[:, 1])/deltaT
        J[:, i] = np.stack((du,dv)).transpose().ravel()

    return J

def plot3DEllipse(ax, A, center, confidence=7.815):
    # https://localcoder.org/plotting-ellipsoid-with-matplotlib

    

    # find the rotation matrix and radii of the axes
    U, s, rotation = linalg.svd(A)
    radii = 1.0/np.sqrt(s)*np.sqrt(confidence)

    print(radii)
    #radii = (1, 1, 1)

    # now carry on with EOL's answer
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
   
    center = np.reshape(center, (3,1)) 
    ellipsoid = (np.matmul(A,  np.stack((x, y, z), 0).reshape(3, -1)) + center).reshape(3, *x.shape)

    #ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color='b', alpha=0.2)
    ax.plot_surface(*ellipsoid,  rstride=4, cstride=4, color='b', alpha=0.2)


def reprojectionError2(imgPoints, reProjectedPoints):
    err = []
    for imgP, reprP in zip(imgPoints, reProjectedPoints):
        err.append(imgP[0]-reprP[0])
        err.append(imgP[1]-reprP[1])

    return np.array(err)

def lmSolve(cameraMatrix, detectedPoints, objectPoints, tVec, rVec, jacobianCalc="opencv", maxIter=100, mode="lm", generate=False, verbose=0):
    if jacobianCalc == "opencvTrue":
        jacobianFunc = lambda *args: objectJacobianOpenCV(*args, returnOpenCV=False)
    elif jacobianCalc == "opencv":
        jacobianFunc = objectJacobianOpenCV
    elif jacobianCalc == "global":
        jacobianFunc = objectJacobianLieGlobal
    elif jacobianCalc == "local":
        jacobianFunc = objectJacobianLieLocal
    else:
        raise Exception("Unknown jacobian calc '{}'".format(jacobianCalc))
    

    Rhi = 0 # 0.75
    Rlo = 0 # 0.25
    lc = 0.75
    k = 0
    v = 2
    x = np.concatenate((tVec, rVec)).astype(np.float64)
    J = -jacobianFunc(cameraMatrix, x[:3], x[3:], objectPoints)
    A = np.matmul(J.transpose(), J)
    D = np.diag(A)
    projPoints = projectPoints2(x[:3], x[3:], cameraMatrix, objectPoints)
    err = reprojectionError2(detectedPoints, projPoints)
    g = np.matmul(J.transpose(), err)

    lamb = 1.0
    if mode == "gd":
        lamb = 1000000.0 #max(np.diag(A))
    elif mode == "gn":
        lamb = 0.0

    lambdas = []
    grads = []
    errors = []

    while True:
        if generate:
            yield x, lambdas, grads, errors
        k += 1

        if mode == "lm":
            Ap = A.copy()
            Ap += np.diag(D)*lamb
        elif mode == "gn":
            Ap = A.copy()
        elif mode == "gd":
            Ap = A.copy()
            #Ap = np.diag(D)*lamb
            Ap = np.eye(6)*lamb
        else:
            raise Exception("Invalid mode '{}'".format(mode))

        h = np.linalg.solve(Ap, -g)
        #h = np.matmul(np.linalg.inv(Ap), -g) 

        found = False

        epsilon1 = 10e-12
        epsilon3 = 10e-12
        if np.linalg.norm(h) < epsilon1*(np.linalg.norm(x) + epsilon3):
            if verbose == 1:
                print("change really small, exiting")
            found = True
        else:
            if jacobianCalc in ("opencvTrue", "opencv"):
                xNew = x + h
            elif jacobianCalc == "global":
                tNew = x[:3]+h[:3]
                rNew = R.from_rotvec(h[3:])*R.from_rotvec(x[3:])
                xNew = np.concatenate((tNew, rNew.as_rotvec()))
            elif jacobianCalc == "local":
                tNew = x[:3]+h[:3]
                rNew = R.from_rotvec(x[3:])*R.from_rotvec(h[3:])
                xNew = np.concatenate((tNew, rNew.as_rotvec()))
            else:
                raise Exception("Unknown jacobian function '{}'".format(jacobianFunc))

            Fx = sum(err**2)

            lambdas.append(lamb)
            grads.append(np.linalg.norm(g, ord=np.inf))
            errors.append(Fx)

            projPointsNew = projectPoints2(xNew[:3], xNew[3:], cameraMatrix, objectPoints)
            errNew = reprojectionError2(detectedPoints, projPointsNew)
            
            FxNew = sum(errNew**2)

            if mode == "gd":
                Ldiff = 1.0
            else:
                Ldiff = 1/2*np.matmul(h.transpose(), (lamb*h - g))
        
            if Ldiff == 0:
                Ldiff = 10e-20

            godness = (Fx - FxNew)/Ldiff # TODO: abs(Ldiff) here?

            if Ldiff < 0:
                raise Exception("Ldiff lower than zero, shouldn't happen")


            if godness > 0 or mode == "gn":

                x = xNew

                J = -jacobianFunc(cameraMatrix, x[:3], x[3:], objectPoints)
                A = np.matmul(J.transpose(), J)
                D = np.diag(A)
                projPoints = projPointsNew
                err = errNew.copy()
                g = np.matmul(J.transpose(), err)

                epsilon2 = 10e-12
                found = np.linalg.norm(g, ord=np.inf) < epsilon2
                if found and verbose == 1:
                    print("gradient small, exiting")
                
                
                #if mode == "lm":
                #if godness > Rhi:
                #lamb = lamb*max(0.33, 1-(2*godness-1)**3)
                lamb *= 0.5
                v = 2
                #if lamb < lc:
                    #    lamb = 0
                
                
            else:
                lamb *= v
                #if mode == "lm":
                    #v = min(max(v, 2.0), 10.0)
                v = 2*v
                #v = min(v, 10.0)

        if found:
            break

        if k == maxIter:
            if verbose == 1:
                print("Maxiter reached")
            break
    if not generate:
        yield x, lambdas, grads, errors


def RPnP(cameraMatrix, detectedPoints, objectPoints, tVec, rVec):
    pointsNorm = detectedPoints[:, :2]/detectedPoints[:, 2]

if __name__ == "__main__":
    A = np.array([[2.020e-09, -3.756e-09, -4.912e-08 ],
                [-3.756e-09,  9.147e-09,  1.300e-07],
                [-4.912e-08,  1.300e-07,  2.226e-06]])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plot3DEllipse(ax, A, center = [0,0,0])
    plt.show()
    plt.close(fig)
    del fig