'''
Created on Feb 27, 2014

@author: AllBodyScan3D
'''
import logging

from pylab import *
# from mpl_toolkits.axes_grid.axislines import SubplotZero
from scipy.cluster.vq import vq, kmeans, kmeans2, whiten

import matplotlib.pyplot as plt
import numpy as np


logging.basicConfig(level=logging.DEBUG)

mu, sigma = 0, 0.1

m1 = np.random.normal(mu, sigma, (100,2)) + 1
m2 = np.random.normal(mu, sigma, (100,2)) - 1
logging.debug(m1)
m = np.concatenate((m1, m2), axis=0)
 
logging.debug(m)
 
fig = plt.figure()
# ax = SubplotZero(fig, 111)
ax = fig.add_subplot(1, 1, 1)
fig.add_subplot(ax)
xlabel('x')
ylabel('y')

# for direction in ["xzero", "yzero"]:
#     ax.axis[direction].set_axisline_style("-|>")
#     ax.axis[direction].set_visible(True)

centroid, label = kmeans2(whiten(m), 2)

mc = np.column_stack((m, label))

x1 = [t[0] for t in mc if t[2] == 1]
y1 = [t[1] for t in mc if t[2] == 1]

x2 = [t[0] for t in mc if t[2] == 0]
y2 = [t[1] for t in mc if t[2] == 0]

ax.plot(x1, y1, 'g^', x2, y2, 'bo')

logging.debug("centroid: {}".format(centroid))
logging.debug("labels: {}".format(label))

plt.show()
