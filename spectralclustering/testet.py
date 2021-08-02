import igl
import numpy as np
import trimesh
from sklearn.cluster import spectral_clustering

SEGMENTATION_COLORMAP = np.array(
    ((165, 242, 12), (89, 12, 89), (165, 89, 165), (242, 242, 165),
     (242, 165, 12), (89, 12, 12), (165, 12, 12), (165, 89, 242), (12, 12, 165),
     (165, 12, 89), (12, 89, 89), (165, 165, 89), (89, 242, 12), (12, 89, 165),
     (242, 242, 89), (165, 165, 165)),
    dtype=np.float32) / 255.0

sv, sf = igl.read_triangle_mesh("cow.obj")
A = igl.adjacency_matrix(sf)
labels = spectral_clustering(A, n_clusters=9)

mesh = trimesh.Trimesh(vertices=sv, faces=sf, vertex_colors=SEGMENTATION_COLORMAP[labels])
mesh.show()
mesh.export('cow.ply')





