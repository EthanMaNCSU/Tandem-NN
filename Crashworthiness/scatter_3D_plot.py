import numpy
import random
numpy.random.seed(7)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def calculate_color_from_R2(R2):
    if( R2<0 ): R2 = 0
    R2_new = R2 * 510
    if R2_new <= 255:
        red = 255
        green = R2_new
        blue = 0
    elif R2_new>255 and R2_new<=510:
        red = 510 - R2_new
        green = 255
        blue = 0
    return [(red/255, green/255, blue/255)]

fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlim3d(0,1)
ax.set_ylim3d(0,1)
ax.set_zlim3d(0,1)

subdivision_R2_scores = numpy.load("subdivision_R2_6.npy")
for point in subdivision_R2_scores:
    ax.scatter(point[0], point[1], point[2], c=calculate_color_from_R2(point[3]))

# data = numpy.load("data_Y_space.npy")
#
# f1 = data[:,5]
# f2 = data[:,6]
# f3 = data[:,7]
#
# ax.set_xlabel('f1')
# ax.set_ylabel('f2')
# ax.set_zlabel('f3')
#
# ax.scatter(f1, f2, f3, c="red", linewidths= 0, alpha=0.5)
# plt.scatter(f1, f2, c="red", linewidths= 0, alpha=0.5)
# plt.ylim(0, 1)
# plt.xlim(0, 1)
# plt.xlabel("f1")
# plt.ylabel("f2")
plt.show()