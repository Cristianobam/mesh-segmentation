#%%
import trimesh
import argparse
import numpy as np
from scipy.sparse import dok_matrix
from sklearn.cluster import spectral_clustering
from trimesh.visual.color import to_rgba

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

def vertex_to_face_color(color, faces):
    vertex_colors = to_rgba(color)
    def pickColor(face):
        values, counts = np.unique(vertex_colors[face], return_counts=True, axis=0)
        return values[np.argmax(counts)]
    face_colors = np.array(list(map(pickColor, faces)))
    return face_colors.astype(np.uint8)

def cluster_adjacency(mesh, clusters):
    clustersAdjacency = np.zeros((clusters, clusters))

    clusterVertices = {}
    for cluster in range(clusters):
        face_mask = (mesh.visual.face_colors == to_rgba(SEGMENTATION_COLORMAP[cluster])).all(axis=1)
        clusterVertices[cluster] = np.unique(mesh.faces[face_mask])

    for i in range(clusters):
        for j in range(clusters):
            if i != j:
                if np.isin(clusterVertices[i], clusterVertices[j]).any():
                    clustersAdjacency[i,j] = 1
                    
    return clustersAdjacency

def segmentationExport(*cluster_list, name='segment'):
    for cluster in cluster_list:
        meshCopy = mesh.copy()
        face_mask = (meshCopy.visual.face_colors == to_rgba(SEGMENTATION_COLORMAP[cluster])).all(axis=1)
        meshCopy.update_faces(face_mask)
        meshCopy.export(f'{name}-{cluster}.ply')
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
    mesh = trimesh.Trimesh(vertices=sv, faces=sf)    
    mesh.visual.face_colors = vertex_to_face_color(SEGMENTATION_COLORMAP[labels], sf)
    mesh.export(f'{args.name.split(".")[0]}2seg.ply')
    
    cluster_adj = cluster_adjacency(mesh, args.clusters)
    selected_clusters = np.where(np.sum(cluster_adj, axis=1)>=4)[0]
    
    segmentationExport(*selected_clusters, name=args.name.split(".")[0])