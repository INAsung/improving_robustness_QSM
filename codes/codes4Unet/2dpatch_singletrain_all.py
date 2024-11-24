import scipy.io
import numpy as np
import h5py
import time
import os
import sys
import random

dpath = '/fast_storage/intern1/QSMnet_data'
result_file = h5py.File(os.path.join(dpath, f'./noisepatch_2D_all.hdf5'), 'w')

# Patch the input & mask file ----------------------------------------------------------------
print("####patching input####")
patches_std3 = []
patches_std5 = []
patches_std7 = []
patches_std9 = []
patches_label = []
patches_mask = []
for s in [2,3]:
    m = scipy.io.loadmat(os.path.join(dpath, f'./std0/train{s}/phscos_final.mat'))
    mask = m['multimask']
    label = m['multiphs']
    n1 = scipy.io.loadmat(os.path.join(dpath, f'./std3/train{s}/phscos_final.mat'))
    n1_field = n1['multiphs']
    n2 = scipy.io.loadmat(os.path.join(dpath, f'./std5/train{s}/phscos_final.mat'))
    n2_field = n2['multiphs']
    n3 = scipy.io.loadmat(os.path.join(dpath, f'./std7/train{s}/phscos_final.mat'))
    n3_field = n3['multiphs']
    n4 = scipy.io.loadmat(os.path.join(dpath, f'./std9/train{s}/phscos_final.mat'))
    n4_field = n4['multiphs']

    print([np.shape(mask), np.shape(n1_field), np.shape(n2_field), np.shape(n3_field)])

    
    matrix_size = np.shape(mask)    

    for idx in range(matrix_size[-1]): # orientation
        #print(f'orientation{idx} now processing')           
        for slicenum in range(matrix_size[2]): # z axis
            patches_label.append(label[:,:,slicenum,idx])
            patches_mask.append(mask[:,:,slicenum,idx])
            patches_std3.append(n1_field[:,:,slicenum,idx])             
            patches_std5.append(n2_field[:,:,slicenum,idx])             
            patches_std7.append(n3_field[:,:,slicenum,idx])
            patches_std9.append(n4_field[:,:,slicenum,idx])             
          
print("Done!")

patches_label = np.array(patches_label, dtype='float32', copy=False)
patches_std3 = np.array(patches_std3, dtype='float32', copy=False)
patches_std5 = np.array(patches_std5, dtype='float32', copy=False)
patches_std7 = np.array(patches_std7, dtype='float32', copy=False)
patches_std9 = np.array(patches_std9, dtype='float32', copy=False)
patches_mask = np.array(patches_mask, dtype='bool', copy=False)
print("Final input data size : " + str(np.shape(patches_std3)))
print("Final label data size : " + str(np.shape(patches_label)))

label_mean = np.mean(patches_label[patches_mask > 0])
label_std  = np.std( patches_label[patches_mask > 0])
std3_mean = np.mean(patches_std3[patches_mask > 0])
std3_std  = np.std( patches_std3[patches_mask > 0])
std5_mean = np.mean(patches_std5[patches_mask > 0])
std5_std  = np.std( patches_std5[patches_mask > 0])
std7_mean = np.mean(patches_std7[patches_mask > 0])
std7_std  = np.std( patches_std7[patches_mask > 0])
std9_mean = np.mean(patches_std9[patches_mask > 0])
std9_std  = np.std( patches_std9[patches_mask > 0])

print(label_mean, label_std, std3_mean, std3_std, std5_mean, std5_std , std7_mean, std7_std, std9_mean, std9_std)

patches_label = (patches_label - label_mean) / label_std
patches_std3 = (patches_std3 - std3_mean) / std3_std
patches_std5 = (patches_std5 - std5_mean) / std5_std
patches_std7 = (patches_std7 - std7_mean) / std7_std
patches_std9 = (patches_std9 - std9_mean) / std9_std


result_file.create_dataset('std3', data=patches_std3)
result_file.create_dataset('std5', data=patches_std5)
result_file.create_dataset('std7', data=patches_std7)
result_file.create_dataset('std9', data=patches_std9)
result_file.create_dataset('clean', data=patches_label)
result_file.create_dataset('pmask', data=patches_mask)
result_file.close()

result_file.close()

scipy.io.savemat(os.path.join(dpath, f'./noisepatch_norm_factor_2D_all.mat'),
                 mdict={'std3_mean': std3_mean,         'std3_std': std3_std,
                 'std5_mean': std5_mean,         'std5_std': std5_std,
                 'std7_mean': std7_mean,         'std7_std': std7_std,
                 'std9_mean': std9_mean,         'std9_std': std9_std,
                 'label_mean': label_mean,
                 'label_std': label_std})



dpath = '/fast_storage/intern1/QSMnet_data'
result_file = h5py.File(os.path.join(dpath, f'./noisepatch_val_2D_all.hdf5'), 'w')