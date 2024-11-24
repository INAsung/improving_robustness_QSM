import os
import csv, pandas

import numpy as np
from numpy.fft import fftn, ifftn, fftshift
from math import log10, sqrt
from skimage.metrics import structural_similarity as ssim

import scipy.ndimage
import matplotlib.pyplot as plt
from matplotlib.pyplot import colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
##################################################################################################################################################
##### dataset 
##################################################################################################################################################
def dipole_kernel(matrix_size, voxel_size, B0_dir):
    """
    Args:
        matrix_size (array_like): should be length of 3.
        voxel_size (array_like): should be length of 3.
        B0_dir (array_like): should be length of 3.
        
    Returns:
        D (ndarray): 3D dipole kernel matrix in Fourier domain.  
    """    
    x = np.arange(-matrix_size[1]/2, matrix_size[1]/2, 1)
    y = np.arange(-matrix_size[0]/2, matrix_size[0]/2, 1)
    z = np.arange(-matrix_size[2]/2, matrix_size[2]/2, 1)
    Y, X, Z = np.meshgrid(x, y, z)
    
    X = X/(matrix_size[0]*voxel_size[0])
    Y = Y/(matrix_size[1]*voxel_size[1])
    Z = Z/(matrix_size[2]*voxel_size[2])
    
    D = 1/3 - (X*B0_dir[0] + Y*B0_dir[1] + Z*B0_dir[2])**2/(X**2 + Y**2 + Z**2)
    D[np.isnan(D)] = 0
    D = fftshift(D)
    return D


class dataset():
    def __init__(self, CONFIG):
        pinput, plabel, pmask, pscale, _, _, _    = load_data(CONFIG["DATA_TYPE"], 'train_patch', CONFIG["DATA_PATH"], getlist=False)
        data, sl, matrix_size, voxel_size, B0_dir = load_data(CONFIG["DATA_TYPE"], 'val', CONFIG["DATA_PATH"], getlist=False)
        
        # (16800, 64, 64, 64)
        self.trfield = pinput
        self.trmask  = pmask
        self.trsusc  = plabel
        
        # (176, 176, 160, 5)
        self.tefield = data[0].field
        self.tesusc  = data[0].label
        self.temask  = data[0].mask
        
        self.X_mean = pscale[0]
        self.X_std = pscale[1]
        self.Y_mean = pscale[2]
        self.Y_std = pscale[3]
        
        # (12, 64, 64, 64, 1)
        self.dipole = dipole_kernel([64, 64, 64], voxel_size=voxel_size, B0_dir=B0_dir)
        self.dipole = np.expand_dims(self.dipole, axis=0)
        self.dipole = np.expand_dims(self.dipole, axis=4)
        self.dipole = np.tile(self.dipole, (CONFIG["TRAIN_BATCH"], 1, 1, 1, 1))
        
        
##################################################################################################################################################
##### log 
##################################################################################################################################################
class Logger():
    def __init__(self, path):
        self.path = path
        if not os.path.isfile(self.path):
            print(f'create log file : {self.path}')
            f=open(self.path, 'w')
            f.close()
        else:
            print("already exist, remove and make new one...")
            os.remove(self.path)
            f=open(self.path, 'w')
            f.close()
            
    def log(self, items):
        with open(self.path, 'a') as f:
            try:
                writer=csv.DictWriter(f, self.fieldnames)
            except AttributeError:
                self.fieldnames=list(items.keys())
                writer=csv.DictWriter(f, self.fieldnames)
                writer.writeheader()
            writer.writerow(items)
            
class display_data():
    def __init__(self):
        # data
        self.data = []
        self.label = []
        self.v = []
        
        # for common setting
        self.linewidth = 2
        self.figsize = (12,4)
        self.grid = False
        
        # for plots
        self.title = 'Title'
        self.ylabel = 'yAxis'
        self.xlabel = 'xAxis'     
        self.alpha = []
        self.color = []
        self.legend = 'upper right'    
        
    def __len__(self):
        return len(self.data)
    
def display_images(n, m, data, p=False):
    # n,m : integer
    # data : display_data
    # p : bool (print or not)
    fig, axes = plt.subplots(n,m,figsize=data.figsize)
    plt.rcParams['lines.linewidth'] = data.linewidth
    plt.rcParams['axes.grid'] = data.grid
    
    if n==1 and m==1:
        axes = np.array([[axes]])
    elif n==1 or m==1:
        axes = axes[np.newaxis,...] 
    fig.subplots_adjust(wspace=0.3)
    
    for i in range(n):
        for j in range(m):
            ax = axes[i,j]
            idx = i*m+j
            
            try:
                im = ax.imshow(data.data[idx], vmin=data.v[idx][0], vmax=data.v[idx][1], cmap=plt.cm.gray)
                ax_divider = make_axes_locatable(ax)
                cax = ax_divider.append_axes("right", size="5%", pad="1%")
                cb = colorbar(im, cax=cax)
                ax.axis('off')
                ax.set_title(data.label[idx])
            except IndexError:
                im = ax.imshow(np.ones((3,3)), vmin=0, vmax=1, cmap=plt.cm.gray)                
                ax.axis('off')
    if p:
        plt.show()
    plt.close()
    return fig

def display_plots(data, p=False):
    # data : display_data
    # p : bool (print or not)
    fig, ax = plt.subplots(1,1,figsize=data.figsize)
    plt.rcParams['lines.linewidth'] = data.linewidth
    plt.rcParams['axes.grid'] = data.grid
    
    for i in range(len(data)):
        x, y = data.data[i][0], data.data[i][1]
        label = data.label[i]
        alpha = 1 if not data.alpha else data.alpha[i]
        color = np.random.rand(3) if not data.color else data.color[i]
        im = ax.plot(x,y,label=label, alpha=alpha, color=color)
    
    ax.set_xlim(data.v[0:2])
    ax.set_ylim(data.v[2:4])
    ax.set_title(data.title)
    ax.set_xlabel(data.xlabel)
    ax.set_ylabel(data.ylabel)
    ax.legend(loc=data.legend)   
    
    if p:
        plt.show()
    plt.close()
    return fig
            
def print_options(parser, opt):
    """Print and save options
    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    taken from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/options/base_options.py
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

            
##################################################################################################################################################
##### QSM metrics
##################################################################################################################################################
def compute_all(im1, im2, mask):
    return compute_nrmse(im1,im2,mask), compute_psnr(im1,im2,mask), compute_ssim(im1,im2,mask), compute_hfen(im1,im2,mask)

def compute_nrmse(im1, im2, mask):
    mse = np.mean((im1[mask]-im2[mask])**2)
    nrmse = sqrt(mse)/sqrt(np.mean(im2[mask]**2))
    return 100*nrmse

def compute_psnr(im1, im2, mask):
    mse = np.mean((im1[mask]-im2[mask])**2)
    if mse == 0:
        return 100
    #PIXEL_MAX = max(im2[mask])
    PIXEL_MAX = 1
    return 20 * log10(PIXEL_MAX / sqrt(mse))

def compute_ssim(im1, im2, mask):    
    im1 = np.pad(im1,((5,5),(5,5),(5,5)),'constant',constant_values=(0))   
    im2 = np.pad(im2,((5,5),(5,5),(5,5)),'constant',constant_values=(0)) 
    mask = np.pad(mask,((5,5),(5,5),(5,5)),'constant',constant_values=(0)).astype(bool) 

    im1 = np.copy(im1); im2 = np.copy(im2);
    min_im = np.min([np.min(im1),np.min(im2)])
    im1[mask] = im1[mask] - min_im
    im2[mask] = im2[mask] - min_im
    
    max_im = np.max([np.max(im1),np.max(im2)])
    im1 = 255*im1/max_im
    im2 = 255*im2/max_im

    _, ssim_map =ssim(im1, im2, data_range=255, gaussian_weights=True, K1=0.01, K2=0.03, full=True)
    return np.mean(ssim_map[mask])

def compute_hfen(im1, im2, mask):
    sigma=1.5
    [x,y,z]=np.mgrid[-7:8,-7:8,-7:8]
    h=np.exp(-(x**2+y**2+z**2)/(2*sigma**2))
    h=h/np.sum(h)
    
    arg=(x**2+y**2+z**2)/(sigma**4)-(1+1+1)/(sigma**2)
    H=arg*h
    H=H-np.sum(H)/(15**3)
    
    im1_log = scipy.ndimage.correlate(im1, H, mode='constant')
    im2_log = scipy.ndimage.correlate(im2, H, mode='constant')
    return compute_nrmse(im1_log, im2_log, mask)




