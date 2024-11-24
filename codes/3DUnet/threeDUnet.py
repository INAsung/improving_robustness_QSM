import os    
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from torch.autograd import Variable

#########################
########  model  ########
#########################
class Conv3d(nn.Module):
    def __init__(self, c_in, c_out, ker, act_func, slope):
        super(Conv3d, self).__init__()
        self.conv=nn.Conv3d(c_in,  c_out, kernel_size=ker, stride=1, padding=int(ker/2), dilation=1)
        self.bn  =nn.BatchNorm3d(c_out)
        if act_func == 'relu':
            self.act = nn.ReLU()
        if act_func == 'leaky_relu':
            self.act = nn.LeakyReLU(negative_slope=slope)
        nn.init.xavier_uniform_(self.conv.weight)
    
    def forward(self,x):
        return self.act(self.bn(self.conv(x)))

class Conv(nn.Module):
    def __init__(self, c_in, c_out):
        super(Conv, self).__init__()
        self.conv=nn.Conv3d(c_in,  c_out, kernel_size=1, stride=1, padding=0, dilation=1)
        nn.init.xavier_uniform_(self.conv.weight)
    
    def forward(self,x):
        return self.conv(x)

class Pool3d(nn.Module):
    def __init__(self, pooling):
        super(Pool3d, self).__init__()
        if pooling == 'max':
            self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1)
        if pooling == 'avg':
            self.pool = nn.AvgPool3d(kernel_size=2, stride=2, padding=0)
    
    def forward(self,x):
        return self.pool(x)

class Deconv3d(nn.Module):
    def __init__(self, c_in, c_out):
        super(Deconv3d, self).__init__()
        self.deconv=nn.ConvTranspose3d(c_in, c_out, kernel_size=2, stride=2, padding=0, dilation=1)
        nn.init.xavier_uniform_(self.deconv.weight)
    
    def forward(self,x):
        return self.deconv(x)

def Concat(x, y):
    return torch.cat((x,y),1)

class Unet_4(nn.Module):
    def __init__(self, CONFIG):
        super(Unet_4,self).__init__()
        c, k, a, s, p = CONFIG["NET_CHA"], CONFIG["NET_KER"], CONFIG["NET_ACT"], CONFIG["NET_SLP"], CONFIG["NET_POOL"]

        self.conv11 = Conv3d(1, c, k, a, s)
        self.conv12 = Conv3d(c, c, k, a, s)
        self.pool1  = Pool3d(p)
        
        self.conv21 = Conv3d(c, 2*c, k, a, s)
        self.conv22 = Conv3d(2*c, 2*c, k, a, s)
        self.pool2  = Pool3d(p)
        
        self.conv31 = Conv3d(2*c, 4*c, k, a, s)
        self.conv32 = Conv3d(4*c, 4*c, k, a, s)
        self.pool3  = Pool3d(p)
        
        self.conv41 = Conv3d(4*c, 8*c, k, a, s)
        self.conv42 = Conv3d(8*c, 8*c, k, a, s)
        self.pool4  = Pool3d(p)        
        
        self.l_conv1 = Conv3d(8*c, 16*c, k, a, s)
        self.l_conv2 = Conv3d(16*c, 16*c, k, a, s)
        
        self.deconv4 = Deconv3d(16*c, 8*c)
        self.conv51  = Conv3d(16*c, 8*c, k, a, s)
        self.conv52  = Conv3d(8*c, 8*c, k, a, s)
        
        self.deconv3 = Deconv3d(8*c, 4*c)
        self.conv61  = Conv3d(8*c, 4*c, k, a, s)
        self.conv62  = Conv3d(4*c, 4*c, k, a, s)
        
        self.deconv2 = Deconv3d(4*c, 2*c)
        self.conv71  = Conv3d(4*c, 2*c, k, a, s)
        self.conv72  = Conv3d(2*c, 2*c, k, a, s)
        
        self.deconv1 = Deconv3d(2*c, c)
        self.conv81  = Conv3d(2*c, c, k, a, s)
        self.conv82  = Conv3d(c, c, k, a, s)        
        
        self.out = Conv(c, 1)
                
    def forward(self, x):
        e1 = self.conv12(self.conv11(x))
        e2 = self.conv22(self.conv21(self.pool1(e1)))
        e3 = self.conv32(self.conv31(self.pool2(e2)))
        e4 = self.conv42(self.conv41(self.pool3(e3)))
        m1 = self.l_conv2(self.l_conv1(self.pool4(e4)))
        d4 = self.conv52(self.conv51(Concat(self.deconv4(m1),e4)))
        d3 = self.conv62(self.conv61(Concat(self.deconv3(d4),e3)))
        d2 = self.conv72(self.conv71(Concat(self.deconv2(d3),e2)))
        d1 = self.conv82(self.conv81(Concat(self.deconv1(d2),e1)))
        x  = self.out(d1)        
        return x

######################
### loss functions ###
######################
def l1_loss(x, y):
    return torch.abs(x-y).mean()

def l2_loss(x, y):
    return torch.abs((x-y)**2).mean()

def save_checkpoint(epoch, step, model, optimizer, scheduler, MODEL_DIR, TAG):
    """
    Args:
        epoch (integer): current epoch.
        step (integer): current step.
        model (class): pytorch neural network model.
        optimizer (torch.optim): pytorch optimizer
        scheduler (torch.optim): pytorch learning rate scheduler
        MODEL_DIR (string): directory path of model save
    """    
    torch.save({
        'epoch': epoch,
        'step' : step,
        'model' : model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }, f'{MODEL_DIR}/{TAG}')
    print(f'Save {TAG} checkpoint done!')

def load_checkpoint(model, optimizer, scheduler, MODEL_PATH):
    """
    Args:
        model (class): pytorch neural network model.
        optimizer (torch.optim): pytorch optimizer
        scheduler (torch.optim): pytorch learning rate scheduler
        MODEL_PATH (string): path of model save

    Returns:
        step (integer): step of saved model
    """        
    checkpoint = torch.load(MODEL_PATH)
    epoch = checkpoint['epoch']
    step = checkpoint['step']
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    del(checkpoint); torch.cuda.empty_cache();
    print(f"Load {MODEL_PATH.split('/')[-1]} checkpoint done!")
    return epoch, step
    
def load_weights(model, MODEL_PATH):
    """
    Args:
        model (class): pytorch neural network model.
        MODEL_PATH (string): path of model save
    """        
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['model'])
    del(checkpoint); torch.cuda.empty_cache();
    print(f"Load {MODEL_PATH.split('/')[-1]} network!")

