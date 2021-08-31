#%%
import vedo
import argparse
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple
from numbers import Number
#%%
def sphere(r:Number, phi:Tuple[Number,Number], theta:Tuple[Number,Number], offset:Tuple[Number,Number,Number]=(0,0,0), n:int=100) -> np.ndarray:
    phi = np.linspace(phi[0], phi[1], n) if len(np.atleast_1d(phi)) == 2 else np.ones(n) * phi
    theta = np.linspace(theta[0], theta[1], n) if len(np.atleast_1d(theta)) == 2 else np.ones(n) * theta
    x = r * np.sin(theta) * np.cos(phi) + offset[0]
    y = r * np.sin(theta) * np.sin(phi) + offset[1]
    z = r * np.cos(theta) + offset[2]
    return np.c_[x, y, z]

#%%
# 2D Chart

U = lambda Endomorphy, Mesomorphy, Ectomorphy: Ectomorphy - Endomorphy
V = lambda Endomorphy, Mesomorphy, Ectomorphy: 2*Mesomorphy - (Endomorphy + Ectomorphy)

#%%
# 3D Chart
def somatochart3D(endomophy:Number, mesomorphy:Number, ectomorphy:Number, savefig:bool=True) -> None:
    _, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_axis_off()

    pnts = sphere(6, 0, (np.pi/2, 0), offset=(1,1,1))
    pnts = np.r_[sphere(6, (np.pi/2, 0), np.pi/2, offset=(1,1,1)), pnts]
    pnts = np.r_[sphere(6, np.pi/2, (0, np.pi/2), offset=(1,1,1)), pnts]
    ax.plot(pnts[:,0], pnts[:,1], pnts[:,2], color='k', alpha=1, linewidth=.5)


    points = np.array([[1,1,1],
                    [9,1,1],
                    [1,9,1],
                    [1,1,9],
                    [9,9,1],
                    [1,9,9],
                    [9,1,9]])

    for point in points[1:]:    
        ax.plot(*np.c_[points[0], point], color='grey', alpha=1, linewidth=.3, linestyle='dashed')
    
    ax.text(10,1,1,'Endomorfo',horizontalalignment='center')
    ax.text(1,10,1,'Ectomorfo',horizontalalignment='center')
    ax.text(1,1,10,'Mesomorfo',horizontalalignment='center')
    
    ax.view_init(30, 45)
    ax.plot(endomophy, ectomorphy, mesomorphy, marker='*', linestyle='none', color='k')
    if savefig:
        plt.savefig('somatochart.png', dpi=500, bbox_inches='tight', transparent=True)
        
    plt.show()

def somatochartVedo(endomophy:Number, mesomorphy:Number, ectomorphy:Number) -> None:
    line1 = vedo.Spline(sphere(6, 0, (np.pi/2, 0), offset=(1,1,1)), res=100).c('grey').lw(.5)
    line2 = vedo.Spline(sphere(6, (np.pi/2, 0), np.pi/2, offset=(1,1,1)), res=100).c('grey').lw(.5)
    line3 = vedo.Spline(sphere(6, np.pi/2, (0, np.pi/2), offset=(1,1,1)), res=100).c('grey').lw(.5)
    
    points = np.array([[1,1,1],
                    [9,1,1],
                    [1,9,1],
                    [1,1,9],
                    [9,9,1],
                    [1,9,9],
                    [9,1,9]])
    
    lines = vedo.Lines(np.repeat(points[0], 6).reshape(-1,3),
                       points[1:]).c('grey').lw(.3)
    
    txts = vedo.Text3D('Endomorfo', s=.5, justify='centered').origin(0,0,0).rotateY(45).rotateX(-30, locally=True).pos(1,1,10).c('black')
    txts += vedo.Text3D('Mesomorfo', s=.5, justify='centered').origin(0,0,0).rotateY(45).rotateX(-30, locally=True).pos(1,10,1).c('black')
    txts += vedo.Text3D('Ectomorfo', s=.5, justify='centered').origin(0,0,0).rotateY(45).rotateX(-30, locally=True).pos(10,1,1).c('black')
    
    gpts = vedo.Points([[ectomorphy, mesomorphy, endomophy]], r=10).c('black', 1)
    
    vedo.show(line1, line2, line3, lines, gpts, txts, axes=0, azimuth=45, elevation=30)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vedo', action='store_true', help="To use vedo")
    parser.add_argument('-d', '--data', type=eval, default=True, help="(Endomophic,Mesomorphy,Ectomorphy)", required=True)
    args = parser.parse_args()
    if args.vedo:
        somatochartVedo(args.data[0], args.data[1], args.data[2]) # Plot using vedo
    else:
        somatochart3D(args.data[0], args.data[1], args.data[2]) # Plot using matplotlib
    