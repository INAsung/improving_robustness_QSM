import scipy.io
import h5py
import nibabel as nib
import numpy as np
import glob, os

def load_traindata(path, load_scale=False):  
        
    m = scipy.io.loadmat(f'{path}/std0/training_data_patch_norm_factor_QSMnetori.mat')
    input_mean = m['input_mean'].item()
    label_mean = m['label_mean'].item()
    input_std = m['input_std'].item()
    label_std = m['label_std'].item()
    n_element = m['n_element']
    if load_scale:
        return {'input_mean':input_mean,'input_std':input_std,'label_mean':label_mean,'label_std':label_std}
    
    h = h5py.File(f'{path}/training_data_patch_QSMnetori.hdf5','r')
    pfield = h['pfield']
    plabel = h['plabel']
    pmask  = h['pmask' ]

    ## validation data...
    val=scipy.io.loadmat('/fast_storage/intern1/QSMnet_data/std0/train7/phscos_final.mat')
    vfield=val['multiphs']
    vlabel=val['multicos']
    vmask =val['multimask'].astype(bool)


    return {'pfield':pfield,'plabel':plabel,'pmask':pmask,'patch_size':pmask.shape,'voxel_size':(1,1,1),
            'vfield':vfield,'vlabel':vlabel,'vmask':vmask,'matrix_size':vmask.shape,'slice':73,
            'input_mean':input_mean,'input_std':input_std,'label_mean':label_mean,'label_std':label_std}
        
# def load_testdata(path):
  
#     tfield = []; tlabel = []; tmask = []; 
        
#     for idx in range(1, 6): ## test1 .. test5
#         sub = f'test{idx}'
#         m = scipy.io.loadmat(f'{path}/{sub}/phscos_final.mat')
#         tfield.append(m[f'multiphs'])
#         tlabel.append(m[f'multicos'])
#         tmask.append(m[f'multimask'].astype(bool))
        
#     return {'num':5,'tfield':tfield,'tlabel':tlabel,'tmask':tmask,'matrix_size':tmask[0].shape,'slice':73}

def load_testdata(path):
  
    local_field = []; cosmos = []; mask = []; 
        
    for subj in [1, 2, 3, 4]:
        for std in [0, 3, 5, 7, 9]:
            m = scipy.io.loadmat(f'{path}/std{std}/test{subj}/phscos_final.mat')
            if std==0:
                cosmos.append(m['multicos'])
                mask.append(m['multimask'].astype(bool))
            local_field.append(m['multiphs'])

    return {'num':4, 'tfield':local_field,'tlabel':cosmos,'tmask':mask,'matrix_size':mask[0].shape,'slice':80}

def load_label(path):
  
    local_field = []; cosmos = []; mask = []; 
        
    for subj in [1, 2, 3, 4]:
        m = scipy.io.loadmat(f'{path}/std0/test{subj}/phscos_final.mat')
        cosmos.append(m['multicos'])
        mask.append(m['multimask'].astype(bool))

    return {'num':4, 'tlabel':cosmos,'tmask':mask,'matrix_size':mask[0].shape,'slice':80}

def load_testdata_denoiser(path):
  
    local_field = []; cosmos = []; mask = []; 
        
    for subj in [1, 2, 3, 4]:
        m = scipy.io.loadmat(f'/fast_storage/intern1/QSMnet_data/std0/test{subj}/phscos_final.mat')
        cosmos.append(m['multicos'])
        mask.append(m['multimask'].astype(bool))
        for std in [0, 3, 5, 7, 9]:
            m = scipy.io.loadmat(f'{path}/std{std}/test{subj}/phscos_final.mat')
            local_field.append(m['multiphs'])

    return {'num':4, 'tfield':local_field,'tlabel':cosmos,'tmask':mask,'matrix_size':mask[0].shape,'slice':80}