# my classes
import torch
import torchvision
import math
from math import pi
import torch.nn.functional as F
import scipy.stats as stats
#from IPython.display import display, clear_output
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import struct
import re

from labeling import show3D
from torchvision.transforms.functional import resize
from itertools import product
from collections import namedtuple
from collections import OrderedDict

class dataset_patch(torch.utils.data.Dataset):
    def __init__(self, root, pmf, ptf, mask=True, sliceitp=False):
        r'''The input of the neural network are:
                normalized baseline and timelapse data patch (4D float32 array, (2,Ni,Nx,Nt))
            The output of the neural network is:
                co2 mask (4d float32 array, (1,Ni,Nx,Nt))
            root: data path (str)
            pmf: json file storing general information of the dataset (str) 
            ptf: csv file storing information of each patch (str)
            mask: boolean indicates wether there are labeled masks
            sliceitp: boolean indicates wether to use slice interpretation'''
        
        super().__init__()
        self.fn_R0t = []
        self.mask = mask
        self.sliceitp = sliceitp
        self.fn_M = []
        self.fn_W = []
        # get general info
        with open(f'{root}/{pmf}','r') as load_f:
            load_dict = json.load(load_f)
            self.DD = load_dict['data_dim']
            self.osz = load_dict['patch_osize']
            self.nsz = load_dict['patch_nsize']
        # get each patch info
        self.pf = pd.read_csv(f'{root}/{ptf}')
        self.idx = self.pf['Ptch_id']
        if self.mask:
            self.midx = self.pf['Mask_id']
        self.N = len(self.idx)
        # get patch file names 
        for i in range(self.N):
            self.fn_R0t.append(f'{root}/R0t_{self.idx[i]}.dat')
            if self.mask:
                self.fn_M.append(f'{root}/Mask_{self.midx[i]}.dat')
                if self.sliceitp:
                    self.fn_W.append(f'{root}/Weight_{self.midx[i]}.dat')
    
    def __getitem__(self, idx):
        # readin the idx patches
        R0t = torch.from_numpy(np.reshape(np.fromfile(self.fn_R0t[idx],dtype=np.float32),(2,self.nsz[0],self.nsz[1],self.nsz[2])))
        if self.mask:
            M = torch.from_numpy(np.reshape(np.fromfile(self.fn_M[idx],dtype=np.float32),(1,self.nsz[0],self.nsz[1],self.nsz[2])))
            if self.sliceitp:
                W = torch.from_numpy(np.reshape(np.fromfile(self.fn_W[idx],dtype=np.float32),(1,self.nsz[0],self.nsz[1],self.nsz[2])))
                return R0t, M, W, idx
            else:
                return R0t, M, idx
        else:
            return R0t, idx

    def __len__(self):
        return self.N    

class SubsetSampler(torch.utils.data.Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices"""

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)
    
    
class Epoch():
    def __init__(self):
        self.count = 0
        self.loss = 0
        self.num_correct = 0
        self.start_time = None

class Run():
    def __init__(self):
        self.params = None
        self.count = 0
        self.data = []
        self.start_time = None

class RunBuilder():
    @staticmethod
    def get_runs(params):

        Run = namedtuple('Run',params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))
        
        return runs

class RunManager():
    def __init__(self, cuda_gpu=False):
        self.e = Epoch()
        self.r = Run()
        
        self.network = None
        self.loader = None
        self.tb = None
        self.gpu = cuda_gpu

    def begin_run(self, run, network, loader):

        self.r.start_time = time.time()
        
        self.r.params = run
        self.r.count += 1

        self.network = network
        self.loader = loader
        #self.tb = SummaryWriter(comment=f'-{run}')

        #images, labels = next(iter(self.loader))
        #grid = torchvision.utils.make_grid(images)

        #self.tb.add_image('images',grid)
        #self.tb.add_graph(self.network,images)
        
    def end_run(self):
        self.tb.close()
        self.e.count = 0

    def begin_epoch(self):
        self.e.start_time = time.time()

        self.e.count += 1
        self.e.loss = 0
        self.e.num_correct = 0

    def end_epoch(self):
        epoch_duration = time.time() - self.e.start_time
        run_duration = time.time() - self.r.start_time

        loss = self.e.loss / len(self.loader.dataset)
        #accuracy = self.e.num_correct / len(self.loader.dataset)

        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.e.count)
            self.tb.add_histogram(f'{name}.grad',param.grad, self.e.count)
        
        results = OrderedDict()
        results["run"] = self.r.count
        results["epoch"] = self.e.count
        results["loss"] = loss
        #results["accuracy"]= accuracy
        results["epoch duration"] = epoch_duration
        results["run duration"] = run_duration

        for k, v in self.r.params._asdict().items():
            results[k] = v
        
        self.r.data.append(results)

    def track_loss(self, loss, batch_size):
        self.e.loss += loss.item() * batch_size

    def save(self, filename):

        pd.DataFrame.from_dict(
            self.r.data, orient='columns'
        ).to_csv(f'{filename}.csv')

        with open(f'{filename}.json', 'w', encoding='utf-8') as f:
            json.dump(self.r.data, f, ensure_ascii=False, indent=4)

            
class patch_show:
    def __init__(self,ds,id_list,ixtid=None):
        r'''initialize the patch show class:
            ds--dataset_patch class
            id_list--patch indices to be displayed (int list)
            ixtid--indices of slices for the 3D patch to display (None or default list)'''
        self.N = len(id_list)
        if ixtid is None:
            # determine the slice indices
            if not(ds.mask):
                # if no mask, then set the middle index as the slice index
                tmp = [x//2 for x in ds.nsz]
                ixtid = [tmp for i in range(self.N)]
            else:
                # if mask available, then set according to the maximum mask amplitude
                ixtid = []
                for i in id_list:
                    if ds.sliceitp:
                        _,M,_,_ = ds.__getitem__(i)
                    else:
                        _,M,_ = ds.__getitem__(i)
                    M = M.numpy()[0]
                    I = np.argmax(np.sum(M,axis=(1,2)))
                    X = np.argmax(np.sum(M,axis=(0,2)))
                    T = np.argmax(np.sum(M,axis=(0,1)))
                    ixtid.append([I,X,T])
        self.ixtid = ixtid
        self.ds = ds
        self.id_list = id_list
        # single graphy width
        self.gw = 6
   
    def view3d(self, pred=np.zeros(0), rcstride=(5,5), clim=None):
        r'''show 3D data patches possibly with predictions:
                pred--predicted co2 mask patches (empty np.array or 4D numpy array (N,rs))'''
        # create the figure and the axis
        if pred.size !=0:
            ncol = 4
        else:
            ncol = 3
        if not(self.ds.mask):
            ncol -= 1
        fig = plt.figure(figsize=((1.2*ncol)*self.gw,(0.7*self.N)*self.gw))
        # loop to plot all 3D graphs
        c = 1
        for i in range(self.N):
            if self.ds.mask:
                if self.ds.sliceitp:
                    R0t,M,_,_ = self.ds.__getitem__(self.id_list[i])
                else:
                    R0t,M,_ = self.ds.__getitem__(self.id_list[i])
            else:
                R0t,_ = self.ds.__getitem__(self.id_list[i])
            R0 = R0t.numpy()[0]
            Rt = R0t.numpy()[1]
            if self.ds.mask:
                Mr = M.numpy()[0]
            # display R0
            ax = fig.add_subplot(self.N,ncol,c,projection='3d')
            if i == 0:
                tl = f'$R_0$'
            else:
                tl = ''
            _ = show3D(R0,ax=ax,xyzi=tuple(self.ixtid[i]),rcstride=rcstride,tl=tl)
            c += 1
            # display Rt
            ax = fig.add_subplot(self.N,ncol,c,projection='3d')
            if i == 0:
                tl = f'$R_t$'
            else:
                tl = ''
            _ = show3D(Rt,ax=ax,xyzi=tuple(self.ixtid[i]),rcstride=rcstride,tl=tl)
            c += 1
            # display reference mask (if mask)
            if self.ds.mask:
                ax = fig.add_subplot(self.N,ncol,c,projection='3d')
                if i == 0:
                    tl = f'mask_ref'
                else:
                    tl = ''
                _ = show3D(Mr,ax=ax,xyzi=tuple(self.ixtid[i]),rcstride=rcstride,tl=tl,clim=[0,1])
                c += 1
            # display predicted mask (if pred)
            if pred.size != 0:
                ax = fig.add_subplot(self.N,ncol,c,projection='3d')
                if i == 0:
                    tl = f'mask_pred'
                else:
                    tl = ''
                _ = show3D(pred[i],ax=ax,xyzi=tuple(self.ixtid[i]),rcstride=rcstride,tl=tl,clim=[0,1])
                c += 1
                
        # show image    
        plt.show(block=False)
    
def findtrace(id_list, batch_idx):
    # batch_idx is a list containing current batch indices
    # find corresponding data indices of id_list in given batch
    idx = []
    for c,i in enumerate(id_list):
        if i in batch_idx:
            p = batch_idx.index(i)
            idx.append([c,p])
    return idx
    
def adjust_learning_rate(optimizer, epoch, lr, dc=2, epn = 50):
    """Sets the learning rate to the initial LR decayed by 5 every 30 epochs"""
    lr_new = lr * (1/dc ** (epoch // epn))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new    

def track_loss_out(loss_org, loss, batch_size):
    loss_org += loss.item() * batch_size 
    return loss_org
    
def lossplot(loss, label_str, log_mode=False):
    max_loss = max(loss)
    loss = [i/max_loss for i in loss]
    if log_mode:
        loss = [math.log10(i) for i in loss]
        yl = 'log10(Loss)'
    else:
        yl = 'Loss'
    fig, ax = plt.subplots(figsize=(5,3))
    ax.plot(range(len(loss)), loss, label=label_str)
    ax.set_title('Loss Update')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(yl)
    ax.legend()  
    return fig, ax
    
def gauss2d(Nr,Nt,sigr,sigt):
    r'''2D Gaussian function:
        input:
            Nr,Nt--Size of the 2D gaussian window (int scalar)
        output:
            G--2D gaussian window (2D float array: (Nr-by-Nt))'''
    # the size of the gaussian filter is ny-by-ny
    G = np.zeros((Nr,Nt))
    # generate grid and center
    [Y, X] = np.meshgrid(range(Nr),range(Nt), indexing='ij')
    xc = (Nr/2,Nt/2)
    Gy = 1/(sigr**2*2*pi)*np.exp(-0.5*(Y-xc[0])**2/sigr**2)
    Gx = 1/(sigt**2*2*pi)*np.exp(-0.5*(X-xc[1])**2/sigt**2)
    G = Gy*Gx
    G = G/np.sum(G)
    return G
    
def hanning2d(Nr,Nt,hr,ht):
    r'''2D averaging function with hanning window at the edge
        input:
            Nr,Nt--Size of the averaging function (int scalar)
            hr,ht--hanning window half length (int scalar)
        output:
            A--average function (2D float array: (Nr-by-Nt))'''
    Ar = np.hanning(2*hr)
    Ar = Ar/np.amax(Ar)
    At = np.hanning(2*ht)
    At = At/np.amax(At)
    AR = np.concatenate((Ar[:hr],np.ones(Nr-2*hr),Ar[-hr:]))
    AT = np.concatenate((At[:ht],np.ones(Nt-2*ht),At[-ht:]))
    Y = np.expand_dims(AR,1)
    X = np.expand_dims(AT,0)
    A = Y*X
    
    return A

def hanning3d(sp,hw):
    r'''3D averaging function with hanning window at the edge
        input:
            sp--Size of the averaging function (int tuple, (3,))
            hw--hanning window half length percentage w.r.t sp (float tuple, (3,))
        output:
            A--average function (3D float array: (sp))'''
    ni,nx,nt = sp
    hi,hx,ht = [int(sp[i]*hw[i]) for i in range(3)]
    # 1D hanning window
    Ai = np.hanning(2*hi)
    Ax = np.hanning(2*hx)
    At = np.hanning(2*ht)
    Ai = Ai/np.amax(Ai)
    Ax = Ax/np.amax(Ax)
    At = At/np.amax(At)
    # expand 1D window
    AI = np.concatenate((Ai[:hi],np.ones(ni-2*hi),Ai[-hi:]))
    AX = np.concatenate((Ax[:hx],np.ones(nx-2*hx),Ax[-hx:]))
    AT = np.concatenate((At[:ht],np.ones(nt-2*ht),At[-ht:]))
    # expand into 3D
    x1 = np.expand_dims(np.expand_dims(AI,1),1)
    x2 = np.expand_dims(np.expand_dims(AX,1),0)
    x3 = np.expand_dims(np.expand_dims(AT,0),0)
    # generate 3D weight
    A = x1*x2*x3
    
    return A
    
class resize3d:
    r'''resize 3D tensors'''
    def __init__(self,N,D=128,H=128,W=128):
        r'''initialize the resize class:
            D,H,W--resized 3D tensor dimension (int)'''
        self.rs = (D,H,W)
        self.N = N
        x = np.linspace(-1,1,D,dtype=np.float32)
        y = np.linspace(-1,1,H,dtype=np.float32)
        z = np.linspace(-1,1,W,dtype=np.float32)
        meshx, meshy, meshz = np.meshgrid(x, y, z, indexing='ij')
        grid0 = np.stack((meshz, meshy, meshx), 3)
        self.grid = torch.tensor(np.tile(grid0,(self.N,1,1,1,1)))
        
    def resize(self, x, NP=False):
        r'''resize the 5D tensor into (N,C,self.rs):
            x--5D tensor (N,C,Din,Hin,Win)'''
        y = F.grid_sample(x,self.grid,padding_mode="border",align_corners=True)
        if NP: # whether to convert to numpy array
            y = y.numpy()
        return y
    
def patch_combine_3D(D,ds,rs3,hwp=(0.2,0.2,0.2)):
    r'''combine patches into complete 3D dataset:
            D--predicted data (5D tensor, (N,1,rs))
            ds--dataset class
            rs3--resize3d class
            hwp--half hanning window length w.r.t patch size (float tuple (3,))'''
    
    # patch related dimension
    Ni,Nx,Nt = ds.osz
    hNi,hNx,hNt = Ni//2,Nx//2,Nt//2
    hNi1,hNx1,hNt1 = Ni-hNi,Nx-hNx,Nt-hNt
    # create 3D hanning weight
    A = hanning3d(ds.osz,hwp)
    # preallocate the combined 3D data
    Z = torch.zeros(ds.DD,dtype=torch.float32)
    W = torch.zeros(ds.DD,dtype=torch.float32)
    # total number of patches
    N = D.shape[0]
    # resize the predicted data
    Dr = rs3.resize(D,NP=True)
    # loop through all patches
    for i in range(N):
        cts = ds.pf['ct'][i]
        y = [int(x) for x in re.findall(r'\d+',cts)]
        Z[y[0]-hNi:y[0]+hNi1,y[1]-hNx:y[1]+hNx1,y[2]-hNt:y[2]+hNt1] += Dr[i,0]*A
        W[y[0]-hNi:y[0]+hNi1,y[1]-hNx:y[1]+hNx1,y[2]-hNt:y[2]+hNt1] += A
    W[W==0] = 1
    return (Z/W).numpy()     

def mute_top(d,t1,t2):
    
    r'''mute the top of a data in the last dimension with hanning window:
        d--seismic data (float 3D array,(nx,ny,nt))
        t1,t2--mute window limits (t1<t2, int)'''
    
    Ai = np.hanning((t2-t1)*2+1)
    Ai = np.concatenate((np.zeros(t1),Ai[:t2-t1+1]),axis=0)
    d[:,:,:t2+1] *= np.expand_dims(np.expand_dims(Ai,0),0)
    
    return d


def BBCE(output,target,beta=None):
    r'''calculate the balanced BCE loss:
        output: NN output tensor (bs,1,64,64,64)
        target: label tensor (bs,1,64,64,64)'''

    thd = -100
    thdx = np.exp(thd)
    N = torch.numel(target)
    if beta is None:
        beta = torch.sum(1-target,dim=[0,1,2,3,4])/N
        #beta[beta>0.999] = 0.999

    o1 = torch.clone(output)
    
    M1 = torch.isnan(o1)
    o1[M1] = 0
    
    o1[o1<thdx] = thdx
    o1[o1>1-thdx] = 1-thdx
    l1 = torch.log(o1)
    l0 = torch.log(1-o1)
    
    y = -beta*torch.sum(target*l1,dim=[1,2,3,4])-(1-beta)*torch.sum((1-target)*l0,dim=[1,2,3,4])
    
    return torch.mean(y)/N
    
    
    