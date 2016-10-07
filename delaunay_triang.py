# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 13:42:08 2016

Code to run Hongbos list of processed files based on on Cristina's verified groud truth list:
    Z:\\Breast\\gtSeg_verified\\CADlesions_wSeries.csv'

Input files:
 - listofprobmaps.txt   (619 cases ~ 50/50 lesion/normals)
     - list of mhas of lesion probability maps in Y:\\Hongbo\\segmentations_train
     - to discover in Y:\\Hongbo\\processed_data:
         - mc Left/Right according to listofprobmaps.txt syntax: 1_0002_6745896_right_2_1_NS_NF_probmap.mha
 - listofprobmaps_test.txt  (450 cases ~ 33/66 lesion/normals)
     - list of mhas of lesion probability maps in Y:\\Hongbo\\segmentations_test
     - to discover in Y:\\Hongbo\\processed_data:
         - mc Left/Right according to listofprobmaps.txt syntax: 1_0002_6745896_right_2_1_NS_NF_probmap.mha

Info about lesions:
 - currently in database textureUpdatedFeatures.db segmentation.lesion_centroid_ijk will have centroid [ijk] coords
 - centroid [ijk] coords and gt mask Y:\Hongbo\gt_data are in bilateral coords, need to split in halfs Left/Right round(half)/rest
 - db segmentation.segm_xmin/max .segm_ymin/max .segm_zmin/max spans # pixels of lesion in each dim (lesion size)
 
Size considerations:
 - breast pixels inside probmap ~ 3/4 of entire volume. So total of:
     512*512*44*0.75 = 8million pixels/Volume 
     512*512*0.75 = 200K pixels/slice

@author: DeepLearning

"""

import numpy as np
from scipy.spatial import Delaunay

points = np.random.rand(30, 2)
tri = Delaunay(points)

p = tri.points[tri.vertices]

# Triangle vertices
A = p[:,0,:].T
B = p[:,1,:].T
C = p[:,2,:].T

# See http://en.wikipedia.org/wiki/Circumscribed_circle#Circumscribed_circles_of_triangles
# The following is just a direct transcription of the formula there
a = A - C
b = B - C

def dot2(u, v):
    return u[0]*v[0] + u[1]*v[1]

def cross2(u, v, w):
    """u x (v x w)"""
    return dot2(u, w)*v - dot2(u, v)*w

def ncross2(u, v):
    """|| u x v ||^2"""
    return sq2(u)*sq2(v) - dot2(u, v)**2

def sq2(u):
    return dot2(u, u)

cc = cross2(sq2(a) * b - sq2(b) * a, a, b) / (2*ncross2(a, b)) + C

# Grab the Voronoi edges
vc = cc[:,tri.neighbors]
vc[:,tri.neighbors == -1] = np.nan # edges at infinity, plotting those would need more work...

lines = []
lines.extend(zip(cc.T, vc[:,:,0].T))
lines.extend(zip(cc.T, vc[:,:,1].T))
lines.extend(zip(cc.T, vc[:,:,2].T))

# Plot itfrom scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

lines = LineCollection(lines, edgecolor='k')

plt.hold(1)
plt.plot(points[:,0], points[:,1], '.')
plt.plot(cc[0], cc[1], '*')
plt.gca().add_collection(lines)
plt.axis('equal')
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.show()



#!/usr/bin/env python
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

def voronoi(P):
    delauny = Delaunay(P)
    triangles = delauny.points[delauny.vertices]

    lines = []

    # Triangle vertices
    A = triangles[:, 0]
    B = triangles[:, 1]
    C = triangles[:, 2]
    lines.extend(zip(A, B))
    lines.extend(zip(B, C))
    lines.extend(zip(C, A))

    circum_centers = np.array([triangle_csc(tri) for tri in triangles])

    segments = []
    for i, triangle in enumerate(triangles):
        circum_center = circum_centers[i]
        for j, neighbor in enumerate(delauny.neighbors[i]):
            if neighbor != -1:
                segments.append((circum_center, circum_centers[neighbor]))
            else:
                ps = triangle[(j+1)%3] - triangle[(j-1)%3]
                ps = np.array((ps[1], -ps[0]))

                middle = (triangle[(j+1)%3] + triangle[(j-1)%3]) * 0.5
                di = middle - triangle[j]

                ps /= np.linalg.norm(ps)
                di /= np.linalg.norm(di)

                if np.dot(di, ps) < 0.0:
                    ps *= -1000.0
                else:
                    ps *= 1000.0
                segments.append((circum_center, circum_center + ps))
    return segments

def triangle_csc(pts):
    rows, cols = pts.shape

    A = np.bmat([[2 * np.dot(pts, pts.T), np.ones((rows, 1))],
                 [np.ones((1, rows)), np.zeros((1, 1))]])

    b = np.hstack((np.sum(pts * pts, axis=1), np.ones((1))))
    x = np.linalg.solve(A,b)
    bary_coords = x[:-1]
    return np.sum(pts * np.tile(bary_coords.reshape((pts.shape[0], 1)), (1, pts.shape[1])), axis=0)

if __name__ == '__main__':
    
    ####################
    #### to generate delaunay triangulation of  a set of point locations
    ####################
    P = np.random.random((300,2))

    X,Y = P[:,0],P[:,1]

    fig = plt.figure(figsize=(4.5,4.5))
    axes = plt.subplot(1,1,1)
    plt.scatter(X, Y, marker='.')
    plt.axis([-0.05,1.05,-0.05,1.05])
    
    delauny = Delaunay(P)
    triangles = delauny.points[delauny.vertices]

    lines = []

    # Triangle vertices
    A = triangles[:, 0]
    B = triangles[:, 1]
    C = triangles[:, 2]
    lines.extend(zip(A, B))
    lines.extend(zip(B, C))
    lines.extend(zip(C, A))
    lines = matplotlib.collections.LineCollection(lines, color='r')
    plt.gca().add_collection(lines)
    
    ####################

    
    #### to do voronoi 
    segments = voronoi(P)
    lines = matplotlib.collections.LineCollection(segments, color='k')
    axes.add_collection(lines)
    plt.axis([-0.05,1.05,-0.05,1.05])
    plt.show()