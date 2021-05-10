import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pcl
import matplotlib.cm as cm
import scipy.signal as ss
import xtgeo
import re
from math import pi
from mpl_toolkits.mplot3d import Axes3D
from obspy.io.segy.segy import _read_segy
from polygon_inclusion import wn_PnPoly
from itertools import product
from scipy.signal import convolve2d

class dataload:
    r'''load the sleipner data'''
    def __init__(self, fn='/data/libei/co2_data/94p10/2010 processing/data/94p10nea.sgy'):
        
        r'''load the segy seismic data:
            fn-data filename'''
        
        self.d = _read_segy(fn, headonly=True) # load the data head
        self.Ntrace = len(self.d.traces) # total trace number
        self.ny = self.d.binary_file_header.number_of_data_traces_per_ensemble # trace number per ensemble
        self.nx = int(self.Ntrace/self.ny) # number of ensembles
        self.nt = self.d.binary_file_header.number_of_samples_per_data_trace # number of samples per trace
        self.dt = self.d.binary_file_header.sample_interval_in_microseconds/1000/1000 # time invertal in second
        
        # print the dataset basic information
        print(f'Total trace number: {self.Ntrace}')
        print(f'Sample number along each trace: {self.nt}')
        print(f'Sampling interval along each trace: {self.dt} s')
        print(f'data arrangement: {self.nx} (number of ensembles) x {self.ny} (trace number per ensemble) x {self.nt} (sample number per trace)')
        
    def getdata(self, idy='all', idx='all', idt='all'):
        
        r'''get the data slice;
            idy--y direction (Inline, ensemble) indices ('all' or 1D int array (ny,))
            idx--x direction (Crossline, ) indices ('all' or 1D int array (nx,))
            idt--t direction indices ('all' or 1D int array (nt,))'''
        
        if idy == 'all':
            idy = np.arange(self.ny)
        if idx == 'all':
            idx = np.arange(self.nx)
        ny = len(idy)
        nx = len(idx)
        # get the objective trace indices
        idtr = []
        for i,j in product(idx,idy):
            idtr.append(i*self.ny+j)
        # stack and reshape the objective traces
        d = np.stack([self.d.traces[k].data for k in idtr])
        d = np.reshape(d,(nx,ny,self.nt))
        # get the objective time samples along all objective traces
        if idt == 'all':
            idt = np.arange(self.nt)
        d = d[:,:,tuple(idt)]
        # get the spatial UTM coordinates of all objective traces
        xd = np.zeros(ny*nx)
        yd = np.zeros(ny*nx)
        for I,i in enumerate(idtr):
            xd[I] = self.d.traces[i].header.x_coordinate_of_ensemble_position_of_this_trace/100
            yd[I] = self.d.traces[i].header.y_coordinate_of_ensemble_position_of_this_trace/100
        xd = np.reshape(xd,(nx,ny))
        yd = np.reshape(yd,(nx,ny))
        td = idt*self.dt
        
        return d,xd,yd,td


def ind2sub(array_shape, ind):
    rows = (ind.astype('int') // array_shape[1])
    cols = (ind.astype('int') % array_shape[1]) # or numpy.mod(ind.astype('int'), array_shape[1])
    return (rows, cols)


def show3D(md, ax=None, xyz=None, xyzi=(0,0,0), ea=(30,-45), clip=1, rcstride=(1,1), clim=None, tl=None):
    r'''plot 3D cube image:
        md-3-D data volume (3darray, float, (n1,n2,n3))
        ax-plot axis (None or a given ax)
        xyz-3-D axes coordinates (list, 1darray, (3,))
        xyzi-position of three slicing image indices (tuple, int, (3,))
        ea-viewing angle (tuple, float, (2,))
        clip-image clipping (scalar, float, <1)
        rcstride-2-D plotting stride (tuple, int, (2,))
        clim-colorbar range (None or tuple, int, (2,)): if it is not None, clip is overwritten'''
    
    # get default coordinates
    nx,ny,nz = md.shape
    if xyz is None:
        xyz = [np.arange(nx),np.arange(ny),np.arange(nz)]
    # slice zero index image along each dimension
    mx = md[xyzi[0],:,:].transpose()
    my = md[:,xyzi[1],:].transpose()
    mz = md[:,:,xyzi[2]].transpose()
    MIN = min([np.amin(mx),np.amin(my),np.amin(mz)])
    MAX = max([np.amax(mx),np.amax(my),np.amax(mz)])
    if clim is None:
        cN = pcl.Normalize(vmin=MIN*clip, vmax=MAX*clip)
        rg = [MIN*clip,(MAX-MIN)*clip]
    else:
        cN = pcl.Normalize(vmin=clim[0], vmax=clim[1])
        rg = [clim[0],clim[1]-clim[0]]
    # plot the model
    if ax is None:
        fig = plt.figure(figsize=plt.figaspect(0.6))
        ax = fig.add_subplot(1,1,1,projection='3d')
    
    # plot the indicator line
    xi = xyz[0][xyzi[0]]
    yi = xyz[1][xyzi[1]]
    zi = xyz[2][xyzi[2]]
    ax.plot([xi,xi],[xyz[1][0],xyz[1][0]],[xyz[2][0],xyz[2][-1]],'r-',linewidth=2,zorder=10)
    ax.plot([xi,xi],[xyz[1][0],xyz[1][-1]],[xyz[2][0],xyz[2][0]],'r-',linewidth=2,zorder=10)
    ax.plot([xyz[0][0],xyz[0][0]],[yi,yi],[xyz[2][0],xyz[2][-1]],'r-',linewidth=2,zorder=10)
    ax.plot([xyz[0][0],xyz[0][-1]],[yi,yi],[xyz[2][0],xyz[2][0]],'r-',linewidth=2,zorder=10)
    ax.plot([xyz[0][0],xyz[0][-1]],[xyz[1][0],xyz[1][0]],[zi,zi],'r-',linewidth=2,zorder=10)
    ax.plot([xyz[0][0],xyz[0][0]],[xyz[1][0],xyz[1][-1]],[zi,zi],'r-',linewidth=2,zorder=10)
    
    # plot the three surfaces
    ax = slice_show(ax, mz, xyz, 0, rg=rg, rcstride=rcstride)
    ax = slice_show(ax, mx, xyz, 1, rg=rg, rcstride=rcstride)
    ax = slice_show(ax, my, xyz, 2, rg=rg, rcstride=rcstride)
    
    # set the axes
    ax.set_xticks(np.linspace(xyz[0][0],xyz[0][-1],5))
    ax.set_yticks(np.linspace(xyz[1][0],xyz[1][-1],5))
    ax.set_zticks(np.linspace(xyz[2][0],xyz[2][-1],5))
    ax.invert_zaxis()
    ax.invert_xaxis()
    ax.set_xlabel('x',fontsize=12)
    ax.set_ylabel('y',fontsize=12)
    ax.set_zlabel('T',fontsize=12)
    if tl is not None:
        ax.set_title(tl,fontsize=12)
    ax.view_init(elev=ea[0],azim=ea[1])
    r'''
    if cb:
        plt.colorbar(cm.ScalarMappable(norm=cN, cmap='gray'))
    plt.show()
    '''
    return ax

def slice_show(ax, ms, xyz, od, rg=None, offset=0, rcstride=(10,10)):
    r'''show specific slice of model'''
    
    if rg is None:
        shift = np.amin(ms)
        normalizer = np.amax(ms)-shift
    else:
        shift = rg[0]
        normalizer = rg[1]
    if normalizer == 0:
        msN = np.zeros_like(ms)+0.5
    else:
        msN = (ms-shift)/normalizer
    colors = plt.cm.gray(msN)
    if od == 0:
        [X,Y] = np.meshgrid(xyz[0],xyz[1])
        Z = np.zeros_like(X)+xyz[2][0]+offset
    if od == 1:
        [Y,Z] = np.meshgrid(xyz[1],xyz[2])
        X = np.zeros_like(Y)+xyz[0][0]+offset
    if od == 2:
        [X,Z] = np.meshgrid(xyz[0],xyz[2])
        Y = np.zeros_like(X)+xyz[1][0]+offset
    surf = ax.plot_surface(X, Y, Z, 
                           facecolors=colors, rstride=rcstride[0], cstride=rcstride[1], zorder=1)
    
    return ax



