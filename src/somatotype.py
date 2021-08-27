#%%
import trimesh
import numpy as np

from mesh2seg import *
from utils import *

#%%
STANDARD_PROPORTION = {5:{'bottomTrunk':3.5, 'midTrunk':4.5, 'topTrunk':6},
                6:{'bottomTrunk':3.5, 'midTrunk':4.5, 'topTrunk':6},
                7:{'bottomTrunk':3.5, 'midTrunk':4.5, 'topTrunk':6}}

#%%
def getHead(mesh:trimesh.Mesh, return_mesh:bool=False):
    ymax = []
    nClusters = len(np.unique(mesh.visual.face_colors, axis=0))
    for i in range(nClusters):
        verticesID = np.unique(getCluster(mesh, cluster_id=i))
        ymax.append(np.max(mesh.vertices[verticesID][:,1]))
    headIndex = np.argmax(ymax)

    maskHead = (mesh.visual.face_colors == to_rgba(SEGMENTATION_COLORMAP[headIndex])).all(axis=1)
    if return_mesh:
        meshHead = mesh.copy()
        meshHead.update_faces(maskHead)
        meshHead.remove_unreferenced_vertices()
        return maskHead, meshHead
    return maskHead

def headThreshold(mesh, iteration:int=1E3, step=0.2, epsilon=1E-6):
    bottom = mesh.centroid.copy()
    
    bottom[1] = np.min(mesh.vertices[:,1])
    yTop = np.max(mesh.vertices[:,1])
    yBottom = bottom[1]
    
    for _ in range(int(iteration)):
        ypos = np.arange(0, yTop-yBottom, step)
        mslices = mesh.section_multiplane(plane_origin=bottom, plane_normal=[0,1,0], heights=ypos)
        maskIsland =  [False if i is None else i.body_count==2 for i in mslices]
        islandPos = yBottom+ypos[maskIsland]
        if len(islandPos)>0:
            if step <= epsilon or abs(min(islandPos)-yTop)<epsilon:
                return min(islandPos)
            else:
                yTop = min(islandPos)
                yBottom = (yTop + yBottom)/2
                bottom[1] = yBottom
        else:
            if step <= epsilon:
                return None
        step *= .5
    return None

def getWristPerimeter() -> float:
    return millimeter2inches(160)

def getHeight(mesh:trimesh.Mesh) -> float:
    height = (np.max(mesh.vertices[:,1]) - np.min(mesh.vertices[:,1]))
    return millimeter2inches(height)

def getWeight() -> float:
    return kilogram2pound(62.8)

def getTrunkIndex(bottom_trunk:trimesh.Mesh, top_trunk:trimesh.Mesh) -> float:
    return float(top_trunk.volume / bottom_trunk.volume)

def main(file_name:str):
    meshBody = trimesh.load(file_name)
    meshBoom(mesh=meshBody, ncluster=12)
    
    _, meshHead = getHead(meshBody, return_mesh=True)
    headBottom = headThreshold(meshHead).astype(float)
    headTop = np.max(meshHead[:,1]).astype(float)
    headSize = headTop-headBottom
    headProp = float((np.max(meshBody.vertices[:,1])-np.min(meshBody.vertices[:,1]))//headSize)

    offset = float(-1*np.min(meshBody.vertices[:,1]))

    hbottom = STANDARD_PROPORTION[headProp]['bottomTrunk']*headSize
    htop = STANDARD_PROPORTION[headProp]['midTrunk']*headSize
    maskBottomTrunk = getSliceY(meshBody, hbottom, htop, offset, return_mesh=False)

    hbottom = STANDARD_PROPORTION[headProp]['midTrunk']*headSize
    htop = STANDARD_PROPORTION[headProp]['topTrunk']*headSize
    maskTopTrunk = getSliceY(meshBody, hbottom, htop, offset, return_mesh=False)

    adjCluster = adjacencyCluster(meshBody)
    selectedClusters = np.where(np.sum(adjCluster, axis=1)>=3)[0]

    for n, iCluster in enumerate(selectedClusters):
        if n == 0:
            maskClusters = getCluster(meshBody, cluster_id=iCluster)
        else:
            maskClusters = np.logical_or(maskClusters, getCluster(meshBody, cluster_id=iCluster))

    meshTopTrunk = meshBody.copy()
    meshTopTrunk.update_faces(np.logical_and(maskClusters, maskTopTrunk))
    meshTopTrunk.remove_unreferenced_vertices()

    meshBottomTrunk = meshBody.copy()
    meshBottomTrunk.update_faces(np.logical_and(maskClusters, maskBottomTrunk))
    meshBottomTrunk.remove_unreferenced_vertices()
    
    trunkIndex = getTrunkIndex(meshBottomTrunk, meshTopTrunk)
    height = getHeight(meshBody)
    weight = getWeight()
    wristIndex = getWristPerimeter()/height    
    getSomatotype(trunkIndex, height, weight, wristIndex)
    
def getSomatotype(trunkIndex, height, weight, wristIndex):
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', default='meshes/afinese.ply', help="Path to mesh")
    args = parser.parse_args()
    main(args.name)
# %%
