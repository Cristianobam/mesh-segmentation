import igl
import trimesh
from sklearn.cluster import spectral_clustering
from tensorflow_graphics.notebooks import mesh_viewer

sv, sf = igl.read_triangle_mesh("brenda2.ply")
A = igl.adjacency_matrix(sf)
labels = spectral_clustering(A)

mesh = trimesh.Trimesh(vertices=sv, faces=sf, vertex_colors=mesh_viewer.SEGMENTATION_COLORMAP[labels])
mesh.export('brenda2seg.ply')





