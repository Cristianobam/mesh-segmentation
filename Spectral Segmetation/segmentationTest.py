#%%
import trimesh
import pymeshlab
import argparse
import numpy as np
from scipy.sparse import dok_matrix
from sklearn.cluster import spectral_clustering

SEGMENTATION_COLORMAP = np.array(
    ((165, 242, 12), (89, 12, 89), (165, 89, 165), (242, 242, 165),
     (242, 165, 12), (89, 12, 12), (165, 12, 12), (165, 89, 242), (12, 12, 165),
     (165, 12, 89), (12, 89, 89), (165, 165, 89), (89, 242, 12), (12, 89, 165),
     (242, 242, 89), (165, 165, 165)),
    dtype=np.float32) / 255.0

#%%
def adjacency_matrix(mesh:str):
    mesh = trimesh.load(mesh)
    mesh.fill_holes()
    sfTri = mesh.faces
    svTri = mesh.vertices
    seTri = mesh.edges

    n = np.max(sfTri)+1
    S = dok_matrix((n, n), dtype=np.float32)
    for (i, j) in seTri:
        S[i, j] = 1
        S[j, i] = 1
    return sfTri, svTri, S

#%%
str2int = lambda x: int(x)

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', default='brenda2.ply', help="Path to mesh to be converted")
parser.add_argument('-c', '--clusters', default=9, type=str2int, help="Path to mesh to be converted")
args = parser.parse_args()
    
#%%
if __name__ == '__main__':
    sf, sv, adj = adjacency_matrix(args.name)
    labels = spectral_clustering(adj, n_clusters=args.clusters) # 9 for real
    mesh = trimesh.Trimesh(vertices=sv, faces=sf, vertex_colors=SEGMENTATION_COLORMAP[labels])
    mesh.export(f'{args.name.split(".")[0]}2seg.ply')
