#%%
import trimesh
import numpy as np
from mesh2seg import *
import os

#%%
meshes = list()
files = ['output/'+f for f in os.listdir('output') if not f.endswith('2seg.ply') and f.lower()!='.ds_store']

for file in files:
    meshes.append(trimesh.load(file))
                
mesh = trimesh.util.concatenate(meshes)

#%%
def legThreshold(mesh, epsilon:float=1E-12, iteration:int=10E12):
    centroid = mesh.centroid
    bottom = centroid.copy()
    top = centroid.copy()
    
    bottom[1] = np.sort(mesh.vertices[:,1])[0]
    top[1] = np.sort(mesh.vertices[:,1])[-1]
    
    for _ in range(int(iteration)):
        if top[1]-bottom[1] <= epsilon:
            break
        else:
            mslice = mesh.section(plane_origin=(top+bottom)/2, plane_normal=[0,1,0]).to_planar()[0]
            if mslice.body_count <= 1:
                top = (top+bottom)/2
            else:
                bottom = (top+bottom)/2
    return  top

# %%
#import vedo
pos = legThreshold(mesh)
mslice = mesh.section(plane_origin=pos, plane_normal=[0,1,0])
slice_2D = mslice.to_planar()[0]

#pl = vedo.Plane(pos, normal=[0,1,0], sx=400, sy=400, c='green', alpha=0.3)
#vedo.show([(mesh, pl), (slice_2D)], N=2, sharecam=False, axes=7).close()
# %%
# %%

meshCopy = mesh.copy()
vertexIndex = np.where(meshCopy.vertices[:,1]<pos[1])[0]
mask = ~np.isin(meshCopy.faces, vertexIndex).any(axis=1)
meshCopy.update_faces(mask)
meshCopy.export('bla2.ply')

# %%
