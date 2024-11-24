import scipy.io
import h5py
import nibabel as nib
import numpy as np
import glob, os

def load_traindata(network, path, load_scale=False):
    
    if network in ['2DUnet', '2DResidualUNet', '2DDenseUnet']:
        m = scipy.io.loadmat(f'{path}/training_data_patch_norm_factor_2D.mat')
    else:
        m = scipy.io.loadmat(f'{path}/training_data_patch_norm_factor_3D.mat')

    input_mean = m['input_mean'].item()
    label_mean = m['label_mean'].item()
    input_std = m['input_std'].item()
    label_std = m['label_std'].item()
    n_element = m['n_element']
    
    if load_scale:
        return {'input_mean':input_mean,'input_std':input_std,'label_mean':label_mean,'label_std':label_std}
    
    if network in ['2DUnet', '2DResidualUNet', '2DDenseUnet']:
        h = scipy.io.loadmat(f'{path}/training_data_patch_2D.hdf5', 'r')
    else:
        h = scipy.io.loadmat(f'{path}/training_data_patch_3D.hdf5', 'r')

    pfield = h['pfield']
    plabel = h['plabel']
    pmask  = h['pmask' ]

    val=scipy.io.loadmat(f'{path}/train7/phscos_final.mat')
    vfield=val['multiphs_final']
    vlabel=val['multicos_final']
    vmask =val['multimask_final'].astype(bool)

    return {'pfield':pfield,'plabel':plabel,'pmask':pmask,'patch_size':pmask.shape,'voxel_size':(1,1,1),
            'vfield':vfield,'vlabel':vlabel,'vmask':vmask,'matrix_size':vmask.shape,'slice':73,
            'input_mean':input_mean,'input_std':input_std,'label_mean':label_mean,'label_std':label_std}
        
        
        
def load_testdata(path):
  
    noisy = []; clean = []; mask = []; 
        
    for subj in [1, 2, 3, 4]:
        for std in [0, 3, 5, 7, 9]:
            m = scipy.io.loadmat(f'{path}/std{std}/test{subj}/phscos_final.mat')
            if std==0:
                clean.append(m['multiphs'])
                mask.append(m['multimask'].astype(bool))
            else:
                noisy.append(m['multiphs'])

    return {'num':4, 'clean':clean,'noisy':noisy,'tmask':mask,'matrix_size':mask[0].shape,'slice':80}