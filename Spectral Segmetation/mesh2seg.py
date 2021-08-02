#%%
import trimesh
import argparse
import numpy as np
from scipy.sparse import dok_matrix
from sklearn.cluster import spectral_clustering
import vedo 

SEGMENTATION_COLORMAP = np.array(
    ((165, 242, 12), (89, 12, 89), (165, 89, 165), (242, 242, 165),
     (242, 165, 12), (89, 12, 12), (165, 12, 12), (165, 89, 242), (12, 12, 165),
     (165, 12, 89), (12, 89, 89), (165, 165, 89), (89, 242, 12), (12, 89, 165),
     (242, 242, 89), (165, 165, 165)),
    dtype=np.float32) / 255.0

#%%
def adjacency_matrix(sf, se):
    n = np.max(sf)+1
    adj_matrix = dok_matrix((n, n), dtype=np.float32)
    for (i, j) in se:
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1
    return adj_matrix

def mesh_by_label(sf, sv, labels, n=0):
    index = np.arange(len(labels))
    index = index[labels == n]
    svNew = sv[index]
    sfNew = sf[~np.any(np.isin(sf, index), axis=1)]
    return sfNew, svNew


#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', default='cow.ply', help="Path to mesh to be converted")
    parser.add_argument('-c', '--clusters', default=9, type=int, help="Path to mesh to be converted")
    args = parser.parse_args()
    
    mesh = trimesh.load(args.name)
    sf = mesh.faces # mesh face
    sv = mesh.vertices # mesh vertices
    se = mesh.edges # mesh edges
    adj_matrix = adjacency_matrix(sf, se) # adjacency matrix
    
    labels = spectral_clustering(adj_matrix, n_clusters=args.clusters) # 9 for real
    mesh = trimesh.Trimesh(vertices=sv, faces=sf, vertex_colors=SEGMENTATION_COLORMAP[labels])
    mesh.export(f'{args.name.split(".")[0]}2seg.ply')