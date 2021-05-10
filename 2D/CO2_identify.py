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

from torchvision.transforms.functional import resize
from itertools import product
from collections import namedtuple
from collections import OrderedDict

class dataset_patch(torch.utils.data.Dataset):
    def __init__(self, root, pmf, ptf, mask=True):
        r'''The input of the neural network are:
                normalized baseline and timelapse data patch (3D float32 array, (2,Nh,Nt))
            The output of the neural network is:
                co2 mask (3d float32 array, (1,Nh,Nt))
            root: data path (str)
            pmf: json file storing general information of the dataset (str) 
            ptf: csv file storing information of each patch (str)
            mask: boolean indicates wether there are labeled masks'''
        
        super().__init__()
        self.fn_R0t = []
        self.mask = mask
        self.fn_M = []
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
    
    def __getitem__(self, idx):
        # readin the idx patches
        R0t = torch.from_numpy(np.reshape(np.fromfile(self.fn_R0t[idx],dtype=np.float32),(2,self.nsz[0],self.nsz[1])))
        if self.mask:
            M = torch.from_numpy(np.reshape(np.fromfile(self.fn_M[idx],dtype=np.float32),(1,self.nsz[0],self.nsz[1])))
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
    def __init__(self,ds,id_list):
        r'''initialize the patch show class:
            ds--dataset_patch class
            id_list--patch indices to be displayed (int list)'''
        self.N = len(id_list)
        self.ds = ds
        self.id_list = id_list
        # single graphy width
        self.gw = 4
   
    def view2d(self, pred=np.zeros(0)):
        r'''show 2D data patches possibly with predictions:
                pred--predicted co2 mask patches (empty np.array or 3D numpy array (N,rs))'''
        # create the figure and the axis
        if pred.size !=0:
            ncol = 4
        else:
            ncol = 3
        if not(self.ds.mask):
            ncol -= 1
        fig = plt.figure()
        # loop to plot all 2D graphs
        figaxs = plt.subplots(self.N,ncol,figsize=(ncol*self.gw,1.1*self.N*self.gw))
        fig, axs = figaxs
        fig.subplots_adjust(hspace = 0.3, wspace = 0.2)
        # add outter subplot for common labels
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axes
        plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        plt.grid(False)
        plt.ylabel(f'Time (8 ms)')
        plt.xlabel(f'Trace (25 m)')
        
        for i in range(self.N):
            c = 0
            if self.ds.mask:
                R0t,M,_ = self.ds.__getitem__(self.id_list[i])
            else:
                R0t,_ = self.ds.__getitem__(self.id_list[i])
            R0 = R0t.numpy()[0]
            Rt = R0t.numpy()[1]
            if self.ds.mask:
                Mr = M.numpy()[0]

            # display R0
            if i == 0:
                tl = f'$R_0$'
            else:
                tl = ''
            axs[i][c].imshow(R0.T, cmap=plt.cm.gray)
            axs[i][c].set_title(tl)
            c += 1
            # display Rt
            if i == 0:
                tl = f'$R_t$'
            else:
                tl = ''
            axs[i][c].imshow(Rt.T, cmap=plt.cm.gray)
            axs[i][c].set_title(tl)
            c += 1
            # display reference mask (if mask)
            if self.ds.mask:
                if i == 0:
                    tl = f'mask_ref'
                else:
                    tl = ''
                axs[i][c].imshow(Mr.T, cmap=plt.cm.gray, vmin=0, vmax=1)
                axs[i][c].set_title(tl)
                c += 1
            # display predicted mask (if pred)
            if pred.size != 0:
                if i == 0:
                    tl = f'mask_pred'
                else:
                    tl = ''
                axs[i][c].imshow(pred[i].T, cmap=plt.cm.gray)
                axs[i][c].set_title(tl)
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

def hanning2d(Nrt,ph):
    r'''2D averaging function with hanning window at the edge
        input:
            Nr,Nt--Size of the averaging function (int scalar)
            phr,pht--hanning window half length in percentage of Nr and Nt, resp (float, [0,0.5])
        output:
            A--average function (2D float array: (Nr-by-Nt))'''
    Nr,Nt = Nrt
    phr,pht = ph
    hr,ht = int(Nr*phr),int(Nt*pht)
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
    
def patch_combine_2D(D,ds,hwp=(0.2,0.2),ixswitch=0):
    r'''combine patches into complete 3D dataset:
            D--predicted data (3D tensor, (N,rs))
            ds--dataset class
            hwp--half hanning window length w.r.t patch size (float tuple (2,),[0,0.5])
            ixswitch--inline, crossline switch idx for given patches (int)'''
    
    # patch related dimension
    Nh,Nt = ds.osz
    hNh,hNt = Nh//2,Nt//2
    hNh1,hNt1 = Nh-hNh,Nt-hNt
    # create 2D hanning weight
    A = hanning2d(ds.osz,hwp)
    # preallocate the combined 3D data
    Z = np.zeros(ds.DD,dtype=np.float32)
    W = np.zeros(ds.DD,dtype=np.float32)
    # total number of patches
    N = D.shape[0]
    # resize the predicted data
    Dr = resize(D,ds.osz).numpy()
    # loop through all patches
    for i in range(N):
        cts = ds.pf['ct'][i]
        y = [int(x) for x in re.findall(r'\d+',cts)]
        if i<ixswitch:
            Z[y[0],y[1]-hNh:y[1]+hNh1,y[2]-hNt:y[2]+hNt1] += Dr[i]*A
            W[y[0],y[1]-hNh:y[1]+hNh1,y[2]-hNt:y[2]+hNt1] += A
        else:
            Z[y[0]-hNh:y[0]+hNh1,y[1],y[2]-hNt:y[2]+hNt1] += Dr[i]*A
            W[y[0]-hNh:y[0]+hNh1,y[1],y[2]-hNt:y[2]+hNt1] += A
    W[W==0] = 1
    return Z/W 

    
    
    