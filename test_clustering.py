import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm

from asfenicsx import Sampling, Clustering


Samples = Sampling(100,2)

kmeans = Clustering(k=5)

clusters, centroids = kmeans.detect(Samples.samples)

# View results
cmap = plt.get_cmap('hsv')
scalarMap = cm.ScalarMappable(colors.Normalize(vmin=0, vmax=kmeans.k),cmap=cmap)
plt.figure()
for i in range(kmeans.k):
    plt.plot(centroids[i,0], centroids[i,1], 'x', color=scalarMap.to_rgba(i))
    plt.scatter(clusters[i][:,0], clusters[i][:,1],color=scalarMap.to_rgba(i))
plt.title("K-means clustering (2D)")

Samples = Sampling(100,3)
kmeans = Clustering(k=5)

clusters, centroids = kmeans.detect(Samples.samples)

plt.figure()
ax = plt.axes(projection='3d')
for i in range(kmeans.k):
    ax.scatter3D(clusters[i][:,0], clusters[i][:,1], clusters[i][:,2],color=scalarMap.to_rgba(i))
    ax.scatter3D(centroids[i,0], centroids[i,1], centroids[i,2], marker='x',color=scalarMap.to_rgba(i))
plt.title("K-means clustering (3D)")
plt.show()

