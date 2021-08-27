#%%
import os
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
def adjacencyMatrix(sf, se):
    n = np.max(sf)+1
    adjMatrix = dok_matrix((n, n), dtype=np.float32)
    for (i, j) in se:
        adjMatrix[i, j] = 1
        adjMatrix[j, i] = 1
    return adjMatrix

def vertex2facecolor(color, faces):
    vertexColors = to_rgba(color)
    def pickColor(face):
        values, counts = np.unique(vertexColors[face], return_counts=True, axis=0)
        return values[np.argmax(counts)]
    faceColors = np.array(list(map(pickColor, faces)))
    return faceColors.astype(np.uint8)

def adjacencyCluster(mesh:trimesh.Mesh):
    nClusters = len(np.unique(adjacencyCluster, axis=0))
    adjCluster = np.zeros((nClusters, nClusters))

    clusterVertices = {}
    for iCluster in range(nClusters):
        clusterVertices[iCluster] = np.unique(getCluster(mesh, cluster_id=iCluster))

    for i in range(nClusters):
        for j in range(nClusters):
            if i != j:
                if np.isin(clusterVertices[i], clusterVertices[j]).any():
                    adjCluster[i,j] = 1
                    
    return adjCluster

def exportCluster(mesh, *cluster_list, name='segment'):
    for clusterID in cluster_list:
        _, meshCluster = getCluster(mesh, cluster_id=clusterID, return_mesh=True)
        meshCluster.export(f'{name}-{clusterID}.ply')

def getCluster(mesh:trimesh.Mesh, cluster_id:int, return_mesh:bool=False):
    maskCluster = (mesh.visual.face_colors == to_rgba(SEGMENTATION_COLORMAP[cluster_id])).all(axis=1)
    if return_mesh:
        meshCluster = mesh.copy()
        meshCluster.update_faces(maskCluster)
        meshCluster.remove_unreferenced_vertices()
        return maskCluster, meshCluster
    return maskCluster

def getSliceY(mesh, hbottom, htop, offset, return_mesh=False):
    mask1 = hbottom <= mesh.vertices[:,1]+offset
    mask2 = mesh.vertices[:,1]+offset <= htop
    maskSlice = np.isin(mesh.faces, np.where(np.logical_and(mask1, mask2))[0]).all(axis=1)
    if return_mesh:
        meshCopy = mesh.copy()
        meshCopy.update_faces(maskSlice)
        meshCopy.remove_unreferenced_vertices()
        return maskSlice, meshCopy
    return maskSlice

def meshBoom(mesh:trimesh.Mesh, nclusters:int) -> None:
    mesh = trimesh.load('afinese.ply')
    sf = mesh.faces # mesh face
    sv = mesh.vertices # mesh vertices
    se = mesh.edges # mesh edges
    adjMatrix = adjacencyMatrix(sf, se) # adjacency matrix

    labels = spectral_clustering(adjMatrix, n_clusters=nclusters, n_init=100)
    mesh = trimesh.Trimesh(vertices=sv, faces=sf)
    mesh.visual.face_colors = vertex2facecolor(SEGMENTATION_COLORMAP[labels], sf)

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', default='cow.ply', help="Path to mesh to be converted")
    parser.add_argument('-c', '--clusters', default=12, type=int, help="Path to mesh to be converted") # old cluster = 9
    parser.add_argument('-e', '--export_path', default='output/', help="Output folder")
    args = parser.parse_args()
    
    try: os.mkdir(args.export_path)
    except: pass
    main(args)
    
    