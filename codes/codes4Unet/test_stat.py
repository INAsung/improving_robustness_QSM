import scipy.io
import numpy as np
import h5py
import time
import os
import sys
import random
label_all = []
mask_all = []
std3_all = []
std5_all = []
std7_all = []
std9_all = []
dpath = '/fast_storage/intern1/QSMnet_data'
for s in range(4):
    s += 1

    m = scipy.io.loadmat(os.path.join(dpath, f'./std0/test{s}/phscos_final.mat'))
    label = m['multiphs']
    mask = m['multimask']
    n1 = scipy.io.loadmat(os.path.join(dpath, f'./std3/test{s}/phscos_final.mat'))
    n1_field = n1['multiphs']
    n2 = scipy.io.loadmat(os.path.join(dpath, f'./std5/test{s}/phscos_final.mat'))
    n2_field = n2['multiphs']
    n3 = scipy.io.loadmat(os.path.join(dpath, f'./std7/test{s}/phscos_final.mat'))
    n3_field = n3['multiphs']
    n4 = scipy.io.loadmat(os.path.join(dpath, f'./std9/test{s}/phscos_final.mat'))
    n4_field = n4['multiphs']
    label_all.append(label)
    mask_all.append(mask)
    std3_all.append(n1_field)
    std5_all.append(n2_field)
    std7_all.append(n3_field)
    std9_all.append(n4_field)
    
label_all = np.array(label_all, dtype='float32', copy=False)
mask_all = np.array(mask_all, dtype='bool', copy=False)
std3_all= np.array(std3_all, dtype='float32', copy=False)
std5_all = np.array(std5_all, dtype='float32', copy=False)
std7_all = np.array(std7_all, dtype='float32', copy=False)
std9_all = np.array(std9_all, dtype='float32', copy=False)

std0_mean = np.mean(label_all[mask_all> 0])
std0_std  = np.std( label_all[mask_all > 0])
std3_mean = np.mean(std3_all[mask_all > 0])
std3_std  = np.std( std3_all[mask_all > 0])
std5_mean = np.mean(std5_all[mask_all > 0])
std5_std  = np.std( std5_all[mask_all > 0])
std7_mean = np.mean(std7_all[mask_all > 0])
std7_std  = np.std( std7_all[mask_all > 0])
std9_mean = np.mean(std9_all[mask_all > 0])
std9_std  = np.std( std9_all[mask_all > 0])

print(std0_mean, std0_std, std3_mean, std3_std, std5_mean, std5_std , std7_mean, std7_std, std9_mean, std9_std)

scipy.io.savemat(os.path.join(dpath, f'./test_stat.mat'),
                 mdict={'std3_mean': std3_mean,         'std3_std': std3_std,
                 'std5_mean': std5_mean,         'std5_std': std5_std,
                 'std7_mean': std7_mean,         'std7_std': std7_std,
                 'std9_mean': std9_mean,         'std9_std': std9_std,
                 'std0_mean': std0_mean,
                 'std0_std': std0_std})