import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# https://localcoder.org/plotting-ellipsoid-with-matplotlib

# your ellispsoid and center in matrix form
A = np.array([[1,0.5,0],[0.5,2,0],[0,0,2]])

A = np.array([[2.020e-09, -3.756e-09, -4.912e-08 ],
              [-3.756e-09,  9.147e-09,  1.300e-07],
              [-4.912e-08,  1.300e-07,  2.226e-06]])


center = [0,0,0]

# find the rotation matrix and radii of the axes
U, s, rotation = linalg.svd(A)
radii = 1.0/np.sqrt(s)

# now carry on with EOL's answer
u = np.linspace(0.0, 2.0 * np.pi, 100)
v = np.linspace(0.0, np.pi, 100)
x = radii[0] * np.outer(np.cos(u), np.sin(v))
y = radii[1] * np.outer(np.sin(u), np.sin(v))
z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
for i in range(len(x)):
    for j in range(len(x)):
        [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color='b', alpha=0.2)
ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b', alpha=0.2)
plt.show()
plt.close(fig)
del fig