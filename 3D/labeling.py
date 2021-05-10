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

def ind2sub(array_shape, ind):
    rows = (ind.astype('int') // array_shape[1])
    cols = (ind.astype('int') % array_shape[1]) # or numpy.mod(ind.astype('int'), array_shape[1])
    return (rows, cols)

def dp2ti(d,dv,v):

    r'''convert a depth into time according to given depth with corresponding interval velocity:
        d--depth of a reflector (float scalar)
        dv--depth of the base velcocity surfaces (1D float array(nv,))
        v--interval velocity corresponding to the current and previous velocity surfaces (1D float array(nv,))'''

    nv = len(dv)
    if nv!=len(v):
        raise ImportError('The dimensions of dv and v are not matching!')
    t = 0
    d0 = 0
    for i in range(nv):
        if d<=dv[i]:
            t += (d-d0)/v[i]
            return t*2
        else:
            t += (dv[i]-d0)/v[i]
            d0 = dv[i]
            
def arrivaltime(D,V,dV):

    r'''calculate the arrival time for all depth layers:
        D--surfaces depth (3D float array (nf,nx,ny))
        V--velocity surfaces (3D float array (nfv,nx,ny))
        dV--velocity surface depth (3D float array (nfv,nx,ny))'''

    nf,nx,ny = D.shape
    T_ref = np.zeros((nf,nx,ny)) # allocate memory for arrival times of all reflectors
    for i,j in product(range(nx),range(ny)):
        dv = dV[:,i,j]
        v = V[:,i,j]
        for k in range(nf):
            dk = D[k,i,j]
            T_ref[k,i,j] = dp2ti(dk,dv,v)
    
    return T_ref

class dataload:
    r'''load the sleipner data'''
    def __init__(self,fn):
        
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

class surfaces:
    r'''load depth-surface and interval-velocity files'''
    def __init__(self
                 ,dir_grid='/data/libei/co2_data/model_grid'
                 ,path_ds='DepthSurfaces_Grid/'
                 ,path_vs='HUM_Interval_Velocity_Trends/'
                 ,fnd=None
                 ,fnv=None):
        
        r'''load the velocity and depth surfaces according to the given model grid:
            fn_mdgd--modelgrid file path (str)
            path_surf--depth surface files path (str)
            fn--depth surface file names (None or list)'''
        
        self.path_ds = dir_grid+'/'+path_ds
        self.path_vs = dir_grid+'/'+path_vs
        
        ##****define depth surface information****##
        if fnd is None: # default 19 reflector surfaces (from shallow to deep)
            self.fnd = ['Top_Caprock','TopSW','ThickShale','TopUtsiraFm',\
                  'Reflector7','Base_Reflector7','Reflector6','Base_Reflector6',\
                  'Reflector5','Base_Reflector5','Reflector4','Base_Reflector4',\
                  'Reflector3','Base_Reflector3','Reflector2','Base_Reflector2',\
                  'Reflector1','Base_Reflector1','BaseUtsiraFm']
        else:
            self.fnd = fnd
        if fnv is None: # default 3 interval velocity
            self.fnv = ['1994_MSL_TopSW_Trend','1994_TopSW_TopUtsiraFm_Trend',\
                        '1994_Top_Base_Utsira_Fm_Trend']
        else:
            self.fnv = fnv
        
    def get_data(self,path,fn,flip=True):
        
        r'''Get the depth surface data or interval velocity data into 3d array:
            path--file path (str)
            fn--file names of the data (str list)'''
        
        nf = len(fn)
        zs = []
        for j in range(nf):
            with open(f'{path}{fn[j]}') as f:
                lines = f.readlines()
            # read in data dimension
            tmp = lines[7].split(',')
            tmp[-1] = tmp[-1][:-1]            
            ny, nx = int(tmp[0]),int(tmp[1])
            xyr = [float(tmp[i+2]) for i in range(4)]
            if j == 0:
                ny0,nx0 = ny,nx
                xyr0 = xyr
            else:
                if [nx0,ny0,xyr0] != [nx,ny,xyr]:
                    raise ImportError('Different data file has different dimension or range!')
            # read in data
            c = 0
            z = np.zeros((nx0*ny0),dtype=np.float32)
            for i in range(11,len(lines)): # from line 11, the depth informatin starts
                nums = re.findall(r"\d+\.?\d*",lines[i])
                num = np.array([float(x) for x in nums])
                z[c:c+num.size] = num
                c += num.size
            zs.append(np.reshape(z,(nx0,ny0)))

        d = np.stack(zs,axis=0) # original data
        if flip:
            D = np.flip(d,(2))
        else:
            D = d
        # define coordinates
        xv = np.linspace(xyr0[0],xyr0[1],nx0)
        yv = np.linspace(xyr0[2],xyr0[3],ny0)
        x,y = np.meshgrid(xv,yv,indexing='ij')

        return D,x,y

class CO2_pb:
    r'''load CO2 plume boundaries'''
    def __init__(self
                 ,dir_grid='/data/libei/co2_data/model_grid'
                 ,path_pb='Sleipner_Plumes_Boundaries/data/'
                 ,fn=None):
        
        r'''load the CO2 plume boundaries:
            path_pb--plume boundary file path (str)
            fn--plume boundary file names (str list)'''
        
        self.path_pb = dir_grid+'/'+path_pb
        if fn is None:
            self.fn = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9']
        else:
            self.fn = fn
        self.nf = len(self.fn)
        
        # get the boundary coordinates and close index
        N = 20000 # preallocated boundary node number for each layer
        self.xb = []
        self.yb = []
        self.ab = [] # negtive amplitude of the plume boundary (supposed to reflect the thickness of CO2)
        self.pn = [] # close index, indicates whether the polygon is inclusive or exclusive
        for j in range(self.nf):
            with open(f'{self.path_pb}{self.fn[j]}') as f:
                lines = f.readlines()
            c = 0
            xbj = np.zeros(N)
            ybj = np.zeros(N)
            abj = np.zeros(N)
            pnj = np.zeros(N)
            for i in range(16,len(lines)):
                nums = re.findall(r"\d+\.?\d*",lines[i])
                num = np.array([float(x) for x in nums])
                xbj[c] = num[0]
                ybj[c] = num[1]
                abj[c] = num[2]
                pnj[c] = num[-1]
                c += 1
            self.xb.append(xbj[:c])
            self.yb.append(ybj[:c])
            self.ab.append(abj[:c])
            self.pn.append(pnj[:c])
            
    def CO2mask(self,xd,yd,td,T_top,T_base):
        
        r'''find the CO2 mask:
            xd,yd--(x,y) coordinates of a given dataset (1D float arrays (nx,ny))
            td--time samples of a trace in seconds (1D float array (nt,)
            T_top--objective layer top surface arrival time (nl,nx,ny)
            T_base--objective layer base surface arrival time (nl,nx,ny)'''
        
        nt = len(td)
        nl,nx,ny = T_top.shape
        mask_CO2 = np.zeros((nt,nx,ny),dtype=np.float32)
        for k in range(nl): # loop through all layers
            xb_layer,yb_layer,pn_layer = self.xb[k],self.yb[k],self.pn[k] # get the boundary coordinates for layer k
            pnu = np.unique(pn_layer) # the indices of closed individual polygons
            b = np.zeros((nx,ny))
            for p in pnu:
                mask = pn_layer==p
                xbp,ybp = xb_layer[mask],yb_layer[mask] # get a closed individual polygon
                xybp = np.stack((xbp,ybp),axis=1) # get the polygon coordinates
                for i,j in product(range(nx),range(ny)): # loop through all traces
                    b[i,j] += wn_PnPoly([xd[i,j],yd[i,j]],xybp) # determine whether the trace is within the polygon
            for i,j in product(range(nx),range(ny)): # loop through all traces
                if b[i,j] == 1: # if b==1, means the trace is within only one polygon, then the times samples within this layer at this trace are saturated with CO2
                    mask_new = (td>T_top[k,i,j])*(td<T_base[k,i,j]) # find the corresponding time samples within in this layer
                    if np.sum(mask_new) == 0:
                        ind = np.argmin(np.abs(td-T_top[k,i,j])+np.abs(td-T_base[k,i,j]))
                        mask_new[ind] = True
                    mask_CO2[:,i,j] += mask_new # put 1s on this time samples
                    
        mask_CO2[mask_CO2>1] = 1
        
        return mask_CO2
    
class expand_xy:
    r'''expand model class'''
    def __init__(self,xmo,ymo,xd,yd):
        
        r'''expand the depth or velocity model (x,y) coordinates according to the data (x,y) coordinates:
            xmo,ymo--original model coordinates (float 2D array, (nx,ny))
            xd,yd--data coordinates (float 2D array, (nxd,nyd))'''

        xm = np.array(xmo)
        ym = np.array(ymo)
        dx = xm[1,0]-xm[0,0]
        dy = ym[0,1]-ym[0,0]
        #----expand x direction (left)-----#
        xel = xmo[0,0]-np.amin(xd)
        if xel>0:
            nxel = int(np.ceil(xel/dx))
            xm_pad = np.array([xm[0,:]-dx*(i+1) for i in range(nxel-1,-1,-1)])
            xm = np.concatenate((xm_pad,xm),axis=0)
            ym = np.pad(ym,((nxel,0),(0,0)),'edge')
        else:
            nxel = 0
        #----expand x direction (right)-----#
        xer = np.amax(xd)-xmo[-1,0]
        
        if xer>0:
            nxer = int(np.ceil(xer/dx))
            xm_pad = np.array([xm[-1,:]+dx*(i+1) for i in range(nxer)])
            xm = np.concatenate((xm,xm_pad),axis=0)
            ym = np.pad(ym,((0,nxer),(0,0)),'edge')
        else:
            nxer = 0
        #----expand y direction (up)-----#
        yet = ymo[0,0]-np.amin(yd)
        if yet>0:
            nyet = int(np.ceil(yet/dy))
            ym_pad = np.array([ym[:,0]-dy*(i+1) for i in range(nyet-1,-1,-1)]).T
            ym = np.concatenate((ym_pad,ym),axis=1)
            xm = np.pad(xm,((0,0),(nyet,0)),'edge')
        else:
            nyet = 0
        #----expand y direction (bottom)-----#
        yeb = np.amax(yd)-ymo[0,-1]
        if yeb>0:
            nyeb = int(np.ceil(yeb/dy))
            ym_pad = np.array([ym[:,-1]+dy*(i+1) for i in range(nyeb)]).T
            ym = np.concatenate((ym,ym_pad),axis=1)
            xm = np.pad(xm,((0,0),(0,nyeb)),'edge')
        else:
            nyet = 0    

        # save the class attributes
        self.x = xm
        self.y = ym
        self.xv = self.x[:,0]
        self.yv = self.y[0,:]
        self.epdim = ((nxel,nxer),(nyet,nyeb))

    def expand_m(self,d):

        r'''expand the depth/velocity surface model:
            d--model being expanded (nf,nx,ny)'''

        D = np.pad(d,((0,0),self.epdim[0],self.epdim[1]),'edge')    

        return D

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







