#!/usr/bin/env python
'''
Created on Feb 26, 2014

@author: AllBodyScan3D
'''
import vtk
import numpy as np
import logging
from scipy.cluster.vq import kmeans2, whiten
from vtk.util import numpy_support

logging.basicConfig(level=logging.DEBUG)

plyReader = vtk.vtkPLYReader()
plyReader.SetFileName("padrao2.ply")
plyReader.Update()

plyActor1 = vtk.vtkActor()
plyActor2 = vtk.vtkActor()
plyActor3 = vtk.vtkActor()
plyActor4 = vtk.vtkActor()
plyActor5 = vtk.vtkActor()

plyMapper1 = vtk.vtkPolyDataMapper()
plyMapper2 = vtk.vtkPolyDataMapper()
plyMapper3 = vtk.vtkPolyDataMapper()
plyMapper4 = vtk.vtkPolyDataMapper()
plyMapper5 = vtk.vtkPolyDataMapper()

#plyReader.SetFileName(plyFile)
data = plyReader.GetOutput()
#data.Update()

logging.debug("number of cells: {}".format(data.GetNumberOfCells()))
logging.debug("number of points: {}".format(data.GetNumberOfPoints()))

points = data.GetPoints()

lpoints = []

for i in range(data.GetNumberOfPoints()):
    lpoints.append(points.GetPoint(i))

apoints = np.array(lpoints)

centroid, label = kmeans2(whiten(apoints), 5)

apoints = np.column_stack((apoints, label))

g1 = [x for x in apoints if x[3] == 0]
g2 = [x for x in apoints if x[3] == 1]
g3 = [x for x in apoints if x[3] == 2]
g4 = [x for x in apoints if x[3] == 3]
g5 = [x for x in apoints if x[3] == 4]

def get_vtk_points(group):
    logging.debug("group: {}".format(len(group)))
    arr = np.array(group)
    arr = np.hsplit(arr, np.array([3,]))
    arr = arr[0]
    arr = np.array(arr, order='C')
    return numpy_support.numpy_to_vtk(arr)

arr1 = np.array(g1) 
arr1 = np.hsplit(arr1, np.array([3,]))
arr1 = arr1[0]
arr1 = np.array(arr1, order='C')
vtkarray1 = numpy_support.numpy_to_vtk(arr1)

g1points = vtk.vtkPoints()
g1points.SetNumberOfPoints(vtkarray1.GetNumberOfTuples())
g1points.SetData(vtkarray1)

g1data = vtk.vtkPolyData()
g1data.SetPoints(g1points)

vertexFilter1 = vtk.vtkVertexGlyphFilter()
vertexFilter1.SetInputData(g1data)
vertexFilter1.Update()

arr2 = np.array(g2) 
arr2 = np.hsplit(arr2, np.array([3,]))
arr2 = arr2[0]
arr2 = np.array(arr2, order='C')
vtkarray2 = numpy_support.numpy_to_vtk(arr2)

g2points = vtk.vtkPoints()
g2points.SetNumberOfPoints(vtkarray2.GetNumberOfTuples())
g2points.SetData(vtkarray2)

g2data = vtk.vtkPolyData()
g2data.SetPoints(g2points)

vertexFilter2 = vtk.vtkVertexGlyphFilter()
vertexFilter2.SetInputData(g2data)
vertexFilter2.Update()

arr3 = np.array(g3) 
arr3 = np.hsplit(arr3, np.array([3,]))
arr3 = arr3[0]
arr3 = np.array(arr3, order='C')
vtkarray3 = numpy_support.numpy_to_vtk(arr3)

g3points = vtk.vtkPoints()
g3points.SetNumberOfPoints(vtkarray3.GetNumberOfTuples())
g3points.SetData(vtkarray3)

g3data = vtk.vtkPolyData()
g3data.SetPoints(g3points)

vertexFilter3 = vtk.vtkVertexGlyphFilter()
vertexFilter3.SetInputData(g3data)
vertexFilter3.Update()

arr4 = np.array(g4) 
arr4 = np.hsplit(arr4, np.array([3,]))
arr4 = arr4[0]
arr4 = np.array(arr4, order='C')
vtkarray4 = numpy_support.numpy_to_vtk(arr4)

g4points = vtk.vtkPoints()
g4points.SetNumberOfPoints(vtkarray4.GetNumberOfTuples())
g4points.SetData(vtkarray4)

g4data = vtk.vtkPolyData()
g4data.SetPoints(g4points)

vertexFilter4 = vtk.vtkVertexGlyphFilter()
vertexFilter4.SetInputData(g4data)
vertexFilter4.Update()

arr5 = np.array(g5) 
arr5 = np.hsplit(arr5, np.array([3,]))
arr5 = arr5[0]
arr5 = np.array(arr5, order='C')
vtkarray5 = numpy_support.numpy_to_vtk(arr5)

g5points = vtk.vtkPoints()
g5points.SetNumberOfPoints(vtkarray5.GetNumberOfTuples())
g5points.SetData(vtkarray5)

g5data = vtk.vtkPolyData()
g5data.SetPoints(g5points)

vertexFilter5 = vtk.vtkVertexGlyphFilter()
vertexFilter5.SetInputData(g5data)
vertexFilter5.Update()

# plyMapper1.SetInputConnection(plyReader.GetOutputPort())
plyMapper1.SetInputConnection(vertexFilter1.GetOutputPort())
plyMapper2.SetInputConnection(vertexFilter2.GetOutputPort())
plyMapper3.SetInputConnection(vertexFilter3.GetOutputPort())
plyMapper4.SetInputConnection(vertexFilter4.GetOutputPort())
plyMapper5.SetInputConnection(vertexFilter5.GetOutputPort())
plyActor1.SetMapper(plyMapper1)
plyActor2.SetMapper(plyMapper2)
plyActor3.SetMapper(plyMapper3)
plyActor4.SetMapper(plyMapper4)
plyActor5.SetMapper(plyMapper5)

plyActor1.GetProperty().SetColor(0.5, 1, 0.5)
# plyActor1.GetProperty().SetOpacity(.75)
plyActor1.GetProperty().SetPointSize(3)

plyActor2.GetProperty().SetColor(0.75, 0.5, 0.5)
# plyActor2.GetProperty().SetOpacity(.75)
plyActor2.GetProperty().SetPointSize(3)

plyActor3.GetProperty().SetColor(0.75, 0.25, 0.75)
# plyActor3.GetProperty().SetOpacity(.75)
plyActor3.GetProperty().SetPointSize(3)

plyActor4.GetProperty().SetColor(0.25, 0.25, 0.75)
# plyActor4.GetProperty().SetOpacity(.75)
plyActor4.GetProperty().SetPointSize(3)

plyActor5.GetProperty().SetColor(0.5, 0.25, 0.25)
# plyActor5.GetProperty().SetOpacity(.75)
plyActor5.GetProperty().SetPointSize(3)

transform = vtk.vtkTransform()
transform.RotateZ(-85.0)

# plyActor1.SetUserTransform(transform) 

#create renderers and add actors of plane and cube
ren = vtk.vtkRenderer()
ren.AddActor(plyActor1)
ren.AddActor(plyActor2)
ren.AddActor(plyActor3)
ren.AddActor(plyActor4)
ren.AddActor(plyActor5)
 
#Add renderer to renderwindow and render
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
renWin.SetSize(600, 600)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
ren.SetBackground(0,0,0)
renWin.Render()
iren.Start()