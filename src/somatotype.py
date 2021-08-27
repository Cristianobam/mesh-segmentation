#%%
import trimesh
import numpy as np
import json

from mesh2seg import *
from utils import *

#%%
STANDARD_PROPORTION = {5:{'bottomTrunk':3.5, 'midTrunk':4.5, 'topTrunk':6},
                6:{'bottomTrunk':3.5, 'midTrunk':4.5, 'topTrunk':6},
                7:{'bottomTrunk':3.5, 'midTrunk':4.5, 'topTrunk':6}}

#%%
def getHead(mesh:trimesh.base.Trimesh, return_mesh:bool=False):
    ymax = []
    nClusters = len(np.unique(mesh.visual.face_colors, axis=0))
    for i in range(nClusters):
        ymax.append(np.max(getCluster(mesh, cluster_id=i, return_mesh=True)[1].vertices))
    headIndex = np.argmax(ymax)

    return getCluster(mesh, cluster_id=headIndex, return_mesh=return_mesh)
    
def headThreshold(mesh:trimesh.base.Trimesh, iteration:int=1000, step:float=0.2, epsilon:float=1E-6):
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

def getHeight(mesh:trimesh.base.Trimesh) -> float:
    height = (np.max(mesh.vertices[:,1]) - np.min(mesh.vertices[:,1]))
    return millimeter2inches(height)

def getWeight() -> float:
    return kilogram2pound(62.8)

def getTrunkIndex(bottom_trunk:trimesh.base.Trimesh, top_trunk:trimesh.base.Trimesh) -> float:
    return roundcap(float(top_trunk.volume / bottom_trunk.volume), 0.05, 2)

def getGender():
    return 'Female'

def main(file_name:str):
    meshBody = trimesh.load(file_name)
    meshBoom(mesh=meshBody, nclusters=12)
    
    _, meshHead = getHead(meshBody, return_mesh=True)
    headBottom = headThreshold(meshHead).astype(float)
    headTop = np.max(meshHead.vertices[:,1]).astype(float)
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
    height = roundcap(getHeight(meshBody), 0.5, 1)
    weight = getWeight()
    wristIndex = getWristPerimeter()/height
    genre = getGender()
    getSomatotype(trunkIndex, height, weight, wristIndex, genre)
    
def getSomatotype(trunkIndex, height, weight, wristIndex, genre):
    if genre == 'Female':
        with open('../references/female-somatotype.json', 'r') as fjson:
            somatotypeTable = json.load(fjson)
    else:
        with open('../references/male-somatotype.json', 'r') as fjson:
            somatotypeTable = json.load(fjson)
    
    somatotypeTable = {k:np.array(v) for k,v in zip(somatotypeTable.keys(),somatotypeTable.values())} 
    
    if ~np.isin(height, somatotypeTable['HEIGHT INCHES']):
        index = np.where(somatotypeTable['HEIGHT INCHES']==roundcap(height, 1, 1))[0]
    else:
        index = np.where(somatotypeTable['HEIGHT INCHES']==height)[0]
    
    if ~np.isin(trunkIndex, somatotypeTable['HEIGHT INCHES']):
        if ~np.isin(trunkIndex, somatotypeTable['HEIGHT INCHES']):
            index += np.where(somatotypeTable['TRUNK INDEX'][index]==roundcap())
        
# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', default='../data/afinese.ply', help="Path to mesh")
    args = parser.parse_args()
    main(args.name)

