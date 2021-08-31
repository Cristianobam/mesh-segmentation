#%%
import argparse
import trimesh
import numpy as np
import json

from somatochart import *
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

def getHeight() -> float:
    return roundcap(millimeter2inches(1700), 0.5, 1)

def getWeight() -> float:
    return kilogram2pound(62.8)

def getTrunkIndex(bottom_trunk:trimesh.base.Trimesh, top_trunk:trimesh.base.Trimesh) -> float:
    return roundcap(float(top_trunk.volume / bottom_trunk.volume), 0.05, 2)

def getGenre():
    return 'Female'

def getAge():
    return 42.

def getTrunk(meshBody:trimesh.base.Trimesh, return_mesh:bool=False):
    meshBoom(mesh=meshBody, nclusters=12)
    
    _, meshHead = getHead(meshBody, return_mesh=True)
    headBottom = headThreshold(meshHead).astype(float)
    headTop = np.max(meshHead.vertices[:,1]).astype(float)
    headSize = headTop-headBottom
    headProp = float((np.max(meshBody.vertices[:,1])-np.min(meshBody.vertices[:,1]))//headSize)

    offset = float(-1*np.min(meshBody.vertices[:,1]))

    hbottom = STANDARD_PROPORTION[headProp]['bottomTrunk']*headSize
    htop = STANDARD_PROPORTION[headProp]['midTrunk']*headSize
    maskBottomSlice = getSliceY(meshBody, hbottom, htop, offset, return_mesh=False)

    hbottom = STANDARD_PROPORTION[headProp]['midTrunk']*headSize
    htop = STANDARD_PROPORTION[headProp]['topTrunk']*headSize
    maskTopSlice = getSliceY(meshBody, hbottom, htop, offset, return_mesh=False)

    adjCluster = adjacencyCluster(meshBody)
    selectedClusters = np.where(np.sum(adjCluster, axis=1)>=3)[0]

    for n, iCluster in enumerate(selectedClusters):
        if n == 0:
            maskClusters = getCluster(meshBody, cluster_id=iCluster)
        else:
            maskClusters = np.logical_or(maskClusters, getCluster(meshBody, cluster_id=iCluster))

    maskTopTrunk = np.logical_and(maskClusters, maskTopSlice)
    maskBottomTrunk = np.logical_and(maskClusters, maskBottomSlice)
    
    if return_mesh:
        meshTopTrunk = meshBody.copy()
        meshTopTrunk.update_faces(maskTopTrunk)
        meshTopTrunk.remove_unreferenced_vertices()
        
        meshBottomTrunk = meshBody.copy()
        meshBottomTrunk.update_faces(maskBottomTrunk)
        meshBottomTrunk.remove_unreferenced_vertices()    
        return (maskBottomTrunk, maskTopTrunk), (meshBottomTrunk, meshTopTrunk)
    
    return (maskBottomTrunk, maskTopTrunk)
    
    
def getWristIndex():
    return getWristPerimeter()/getHeight()*100

def getSomatotype(mesh:trimesh.base.Trimesh):
    _ , (meshBottomTrunk, meshTopTrunk) = getTrunk(mesh, return_mesh=True)
    trunkIndex = getTrunkIndex(meshBottomTrunk, meshTopTrunk)
    height = getHeight()
    age = getAge()
    weight = getWeight()
    wristIndex = getWristIndex()
    genre = getGenre()
    
    if genre == 'Female':
        with open('../references/female-somatotype.json', 'r') as fjson:
            somatotypeTable = json.load(fjson)
    else:
        with open('../references/male-somatotype.json', 'r') as fjson:
            somatotypeTable = json.load(fjson)
    
    somatotypeTable = {k:np.array(v) for k,v in zip(somatotypeTable.keys(),somatotypeTable.values())} 
    
    with open('../references/wristHeightIndex.json', 'r') as fjson:
        wristTable = json.load(fjson)
    
    if ~np.isin(height, somatotypeTable['HEIGHT INCHES']):
        indexHeight = somatotypeTable['HEIGHT INCHES']==roundcap(height, 1, 1)
    else:
        indexHeight = somatotypeTable['HEIGHT INCHES']==height
    
    if ~np.isin(trunkIndex, somatotypeTable['TRUNK INDEX']):
        if ~np.isin(trunkIndex := roundcap(trunkIndex, 0.1, 2), somatotypeTable['TRUNK INDEX']):
            indexTrunk = somatotypeTable['TRUNK INDEX']==roundcap(trunkIndex, 1, 1)
        else:
            indexTrunk = somatotypeTable['TRUNK INDEX']==trunkIndex
    else:
        indexTrunk = somatotypeTable['TRUNK INDEX']==trunkIndex
    
    if 20 <= age < 30:
        indexWeight = somatotypeTable['MAX LBS @20']==roundcap(weight, 1, 1)
    elif 30 <= age < 40:
        indexWeight = somatotypeTable['MAX LBS @30']==roundcap(weight, 1, 1)
    elif 40 <= age < 50:
        indexWeight = somatotypeTable['MAX LBS @40']==roundcap(weight, 1, 1)
    elif age > 50:
        indexWeight = somatotypeTable['MAX LBS @50']==roundcap(weight, 1, 1)
    
    if ~any(index := indexHeight & indexTrunk & indexWeight):
        if ~any(index := indexHeight & indexTrunk):
            argMAX = np.array(wristTable['Body Type'])[list(map(lambda x: x[0]<=wristIndex<x[1], wristTable[genre]))][0]
            indexN = np.argmax(somatotypeTable[argMAX][index])

    return somatotypeTable['ENDO'][index][indexN], \
            somatotypeTable['MESO'][index][indexN], \
            somatotypeTable['ECTO'][index][indexN], \
            somatotypeTable['BALANCE'][index][indexN]
# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', default='../data/afinese.ply', help="Path to mesh")
    args = parser.parse_args()
    mesh = trimesh.load(args.name)
    endo, meso, ecto, balance = getSomatotype(mesh)
    somatochart3D(endo, meso, ecto, savefig=True)