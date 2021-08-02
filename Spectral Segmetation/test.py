#%%
import trimesh
import numpy as np
from sklearn.cluster import spectral_clustering
from mesh2seg import adjacency_matrix, SEGMENTATION_COLORMAP

#%%
mesh = trimesh.load('meshes/cow.obj')
sf = mesh.faces # mesh face
sv = mesh.vertices # mesh vertices
se = mesh.edges # mesh edges
adj_matrix = adjacency_matrix(sf, se) # adjacency matrix

labels = spectral_clustering(adj_matrix, n_clusters=2)
#mesh = trimesh.Trimesh(vertices=sv, faces=sf, vertex_colors=SEGMENTATION_COLORMAP[labels])
#mesh.export(f'test2seg.ply')

# %%
index = np.arange(len(labels))
lab2vertices = {}
for label in np.unique(labels):
    lab2vertices[label] = index[labels == label]

# %%
