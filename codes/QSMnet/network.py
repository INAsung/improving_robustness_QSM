import os    
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import numpy as np

##################################################################################################################################################
##### QSM utils
##################################################################################################################################################
def l1_loss(x, y):
    return torch.abs(x-y).mean()

def model_loss(p, x, m, d, input_std,input_mean,label_std,label_mean):
    p = p * label_std + label_mean
    num_batch = p.shape[0]
    dtype = p.dtype; device = p.device;
    p = torch.stack((p,torch.zeros(p.shape, dtype=dtype, device=device)),dim=-1)
    k = torch.fft.fft(p,3)
    
    d = d[np.newaxis, np.newaxis, ...]
    d = torch.tensor(d, dtype=dtype, device=device).repeat(num_batch, 1, 1, 1, 1)
    d = torch.stack((d,torch.zeros(d.shape, dtype=dtype, device=device)),dim=-1)
    
    y = torch.zeros(p.shape, dtype=dtype, device=device)
    y[...,0] = k[...,0]*d[...,0] - k[...,1]*d[...,1]
    y[...,1] = k[...,0]*d[...,1] + k[...,1]*d[...,0]
    
    y = torch.fft.ifft(y,3)
    y = y[...,0]
    
    x = x*input_std+input_mean
    
    return l1_loss(x*m, y*m)

def grad_loss(x, y):
    device = x.device
    x_cen = x[:,:,1:-1,1:-1,1:-1]
    grad_x = torch.zeros(x_cen.shape, device=device)
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                x_slice = x[:,:,i+1:i+x.shape[2]-1,j+1:j+x.shape[3]-1,k+1:k+x.shape[4]-1]
                s = np.sqrt(i*i+j*j+k*k)
                if s == 0:
                    temp = torch.zeros(x_cen.shape, device=device)
                else:
                    temp = torch.relu(x_slice-x_cen)/s
                grad_x = grad_x + temp
    
    y_cen = y[:,:,1:-1,1:-1,1:-1]
    grad_y = torch.zeros(y_cen.shape, device=device)
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                y_slice = y[:,:,i+1:i+x.shape[2]-1,j+1:j+x.shape[3]-1,k+1:k+x.shape[4]-1]
                s = np.sqrt(i*i+j*j+k*k)
                if s == 0:
                    temp = torch.zeros(y_cen.shape, device=device)
                else:
                    temp = torch.relu(y_slice-y_cen)/s
                grad_y = grad_y + temp
    
    return l1_loss(grad_x, grad_y)

def total_loss(p, x, y, m, d, w1, w2, input_std, input_mean, label_std, label_mean):
    """
    Args:
        p (torch.tensor): (batch_size, 1, s1, s2, s3) size matrix. predicted susceptability map.
        x (torch.tensor): (batch_size, 1, s1, s2, s3) size matrix. local field map.
        y (torch.tensor): (batch_size, 1, s1, s2, s3) size matrix. susceptability map (label).
        m (torch.tensor): (batch_size, 1, s1, s2, s3) size matrix. mask.
        d (ndarray): (s1, s2, s3) size matrix. dipole kernel.
        w1 (float): weighting factor for sum losses
        w2 (float): weighting factor for sum losses
        
    Returns:
        l1loss (torch.float): L1 loss. 
        mdloss (torch.float): model loss
        gdloss (torch.float): gradient loss
        tloss (torch.float): total loss. sum of above three losses with weighting factor
    """        
    l1loss = l1_loss(p, y)
    #l1loss = l1_loss(p*m, y*m)
    mdloss = model_loss(p, x, m, d, input_std, input_mean, label_std, label_mean)
    gdloss = grad_loss(p, y)
    #gdloss = grad_loss(p*m, y*m)
    tloss = l1loss + mdloss * w1 + gdloss * w2
    return l1loss, mdloss, gdloss, tloss


##################################################################################################################################################
##### QSM utils
##################################################################################################################################################
def l1_loss2(x, y):
    return torch.abs(x-y).mean(dim=4).mean(dim=3).mean(dim=2).mean(dim=1)

def model_loss2(p, x, m, d, input_std,input_mean,label_std,label_mean):
    p = p * label_std + label_mean
    num_batch = p.shape[0]
    dtype = p.dtype; device = p.device;
    p = torch.stack((p,torch.zeros(p.shape, dtype=dtype, device=device)),dim=-1)
    k = torch.fft(p,3)
    
    d = d[np.newaxis, np.newaxis, ...]
    d = torch.tensor(d, dtype=dtype, device=device).repeat(num_batch, 1, 1, 1, 1)
    d = torch.stack((d,torch.zeros(d.shape, dtype=dtype, device=device)),dim=-1)
    
    y = torch.zeros(p.shape, dtype=dtype, device=device)
    y[...,0] = k[...,0]*d[...,0] - k[...,1]*d[...,1]
    y[...,1] = k[...,0]*d[...,1] + k[...,1]*d[...,0]
    
    y = torch.ifft(y,3)
    y = y[...,0]
    
    x = x*input_std+input_mean
        
    return l1_loss2(x*m, y*m)

def grad_loss2(x, y):
    device = x.device
    x_cen = x[:,:,1:-1,1:-1,1:-1]
    grad_x = torch.zeros(x_cen.shape, device=device)
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                x_slice = x[:,:,i+1:i+x.shape[2]-1,j+1:j+x.shape[3]-1,k+1:k+x.shape[4]-1]
                s = np.sqrt(i*i+j*j+k*k)
                if s == 0:
                    temp = torch.zeros(x_cen.shape, device=device)
                else:
                    temp = torch.relu(x_slice-x_cen)/s
                grad_x = grad_x + temp
    
    y_cen = y[:,:,1:-1,1:-1,1:-1]
    grad_y = torch.zeros(y_cen.shape, device=device)
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                y_slice = y[:,:,i+1:i+x.shape[2]-1,j+1:j+x.shape[3]-1,k+1:k+x.shape[4]-1]
                s = np.sqrt(i*i+j*j+k*k)
                if s == 0:
                    temp = torch.zeros(y_cen.shape, device=device)
                else:
                    temp = torch.relu(y_slice-y_cen)/s
                grad_y = grad_y + temp
    
    return l1_loss2(grad_x, grad_y)

def total_loss2(p, x, y, m, d, w1, w2, input_std, input_mean, label_std, label_mean):
    """
    Args:
        p (torch.tensor): (batch_size, 1, s1, s2, s3) size matrix. predicted susceptability map.
        x (torch.tensor): (batch_size, 1, s1, s2, s3) size matrix. local field map.
        y (torch.tensor): (batch_size, 1, s1, s2, s3) size matrix. susceptability map (label).
        m (torch.tensor): (batch_size, 1, s1, s2, s3) size matrix. mask.
        d (ndarray): (s1, s2, s3) size matrix. dipole kernel.
        w1 (float): weighting factor for sum losses
        w2 (float): weighting factor for sum losses
        
    Returns:
        l1loss (torch.float): L1 loss. 
        mdloss (torch.float): model loss
        gdloss (torch.float): gradient loss
        tloss (torch.float): total loss. sum of above three losses with weighting factor
    """        
    l1loss = l1_loss2(p, y)
    #l1loss = l1_loss(p*m, y*m)
    mdloss = model_loss2(p, x, m, d, input_std, input_mean, label_std, label_mean)
    gdloss = grad_loss2(p, y)
    #gdloss = grad_loss(p*m, y*m)
    tloss = l1loss + mdloss * w1 + gdloss * w2
    return l1loss, mdloss, gdloss, tloss


##################################################################################################################################################
##### neural networks
################################################################################################################################################## 
def load_network(CONFIG, p=False):
    l = CONFIG["NET_LAY"]
    if l==4:
        model = Unet_4(CONFIG).cuda()
    elif l==3:
        model = Unet_3(CONFIG).cuda()
    elif l==2:
        model = Unet_2(CONFIG).cuda()
    if p:
        summary(model, input_size=(1, 64, 64, 64))     
    return model
  

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
        #print('input: ', np.shape(x))
        #print('conv11(x): ', np.shape(self.conv11(x)))
        e1 = self.conv12(self.conv11(x))
        #print('e1:', np.shape(e1))
        e2 = self.conv22(self.conv21(self.pool1(e1)))
        #print('e2:', np.shape(e2))
        e3 = self.conv32(self.conv31(self.pool2(e2)))
        #print('e3:',np.shape(e3))
        e4 = self.conv42(self.conv41(self.pool3(e3)))
        #print('e4:',np.shape(e4))
        m1 = self.l_conv2(self.l_conv1(self.pool4(e4)))
        #print('m1:',np.shape(m1))
        #print('deconv m1:',np.shape(self.deconv4(m1)))
        d4 = self.conv52(self.conv51(Concat(self.deconv4(m1),e4)))
        #print('d4:',np.shape(d4))
        #print('deconv d4: ', np.shape(self.deconv3(d4)))
        d3 = self.conv62(self.conv61(Concat(self.deconv3(d4),e3)))
        #print('d3: ', np.shape(d3))
        #print('deconv d3: ', np.shape(self.deconv2(d3)))
        d2 = self.conv72(self.conv71(Concat(self.deconv2(d3),e2)))
        #print('d2:', np.shape(d2))
        #print('deconv d2: ', np.shape(self.deconv1(d2)))
        d1 = self.conv82(self.conv81(Concat(self.deconv1(d2),e1)))
        #print('d1: ', np.shape(d1))
        x  = self.out(d1)
        #print('x: ',np.shape(x))
        return x

class Unet_3(nn.Module):
    def __init__(self, CONFIG):
        super(Unet_3,self).__init__()
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
        
        self.l_conv1 = Conv3d(4*c, 8*c, k, a, s)
        self.l_conv2 = Conv3d(8*c, 8*c, k, a, s)
        
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
        m1 = self.l_conv2(self.l_conv1(self.pool3(e3)))
        d3 = self.conv62(self.conv61(Concat(self.deconv3(m1),e3)))
        d2 = self.conv72(self.conv71(Concat(self.deconv2(d3),e2)))
        d1 = self.conv82(self.conv81(Concat(self.deconv1(d2),e1)))
        x  = self.out(d1)        
        return x
    
class Unet_2(nn.Module):
    def __init__(self, CONFIG):
        super(Unet_2,self).__init__()
        c, k, a, s, p = CONFIG["NET_CHA"], CONFIG["NET_KER"], CONFIG["NET_ACT"], CONFIG["NET_SLP"], CONFIG["NET_POOL"]

        self.conv11 = Conv3d(1, c, k, a, s)
        self.conv12 = Conv3d(c, c, k, a, s)
        self.pool1  = Pool3d(p)
        
        self.conv21 = Conv3d(c, 2*c, k, a, s)
        self.conv22 = Conv3d(2*c, 2*c, k, a, s)
        self.pool2  = Pool3d(p)
        
        self.l_conv1 = Conv3d(2*c, 4*c, k, a, s)
        self.l_conv2 = Conv3d(4*c, 4*c, k, a, s)
        
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
        m1 = self.l_conv2(self.l_conv1(self.pool2(e2)))
        d2 = self.conv72(self.conv71(Concat(self.deconv2(m1),e2)))
        d1 = self.conv82(self.conv81(Concat(self.deconv1(d2),e1)))
        x  = self.out(d1)        
        return x

           
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

