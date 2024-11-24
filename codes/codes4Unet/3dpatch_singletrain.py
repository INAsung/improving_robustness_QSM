import scipy.io
import numpy as np
import h5py
import time
import os
import sys
import random

PS = 32
patch_num = [6,8,7]

dpath = '/fast_storage/intern1/QSMnet_data'
result_file = h5py.File(os.path.join(dpath, f'./training_data_noisepatch_3D.hdf5'), 'w')


# Patch the input & mask file ----------------------------------------------------------------
print("####patching input####")
patches_field = []
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
    
    print([np.shape(mask), np.shape(n1_field), np.shape(n2_field), np.shape(n3_field), np.shape(n4_field)])
    matrix_size = np.shape(mask)    
    strides = [(matrix_size[i] - PS) // (patch_num[i] - 1) for i in range(3)]; 
    
    for idx in range(matrix_size[-1]): # orientation
        for i in range(patch_num[0]): # x axis
            for j in range(patch_num[1]): # y axis
                for k in range(patch_num[2]): # z axis

                    patches_label.append(label[i * strides[0]:i * strides[0] + PS,
                                    j * strides[1]:j * strides[1] + PS,
                                    k * strides[2]:k * strides[2] + PS,
                                    idx])
                    patches_mask.append(mask[
                                    i * strides[0]:i * strides[0] + PS,
                                    j * strides[1]:j * strides[1] + PS,
                                    k * strides[2]:k * strides[2] + PS,
                                    idx])

                    ## for various noise level, let's shuffle stdXX datas
                    randi = random.randint(1, 4)
                    if(randi==1):
                        patches_field.append(n1_field[
                                    i * strides[0]:i * strides[0] + PS,
                                    j * strides[1]:j * strides[1] + PS,
                                    k * strides[2]:k * strides[2] + PS,
                                    idx])             
                    elif(randi==2):
                        patches_field.append(n2_field[
                                    i * strides[0]:i * strides[0] + PS,
                                    j * strides[1]:j * strides[1] + PS,
                                    k * strides[2]:k * strides[2] + PS,
                                    idx])     
                    elif(randi==3):
                        patches_field.append(n3_field[
                                    i * strides[0]:i * strides[0] + PS,
                                    j * strides[1]:j * strides[1] + PS,
                                    k * strides[2]:k * strides[2] + PS,
                                    idx])             
                    else:
                        patches_field.append(n4_field[
                                    i * strides[0]:i * strides[0] + PS,
                                    j * strides[1]:j * strides[1] + PS,
                                    k * strides[2]:k * strides[2] + PS,
                                    idx])             
print("Done!")

patches_label = np.array(patches_label, dtype='float32', copy=False)
patches_field = np.array(patches_field, dtype='float32', copy=False)
patches_mask = np.array(patches_mask, dtype='bool', copy=False)
print("Final input data size : " + str(np.shape(patches_field)))
print("Final label data size : " + str(np.shape(patches_label)))


result_file.create_dataset('noisy', data=patches_field)
result_file.create_dataset('clean', data=patches_label)
result_file.create_dataset('pmask', data=patches_mask)
result_file.close()

input_mean = np.mean(patches_field[patches_mask > 0])
input_std = np.std(patches_field[patches_mask > 0])
label_mean = np.mean(patches_label[patches_mask > 0])
label_std  = np.std( patches_label[patches_mask > 0])
n_element = np.sum(patches_mask)

print('input_mean: ', input_mean, 'input_std: ',input_std, 'label_mean: ', label_mean, 'label_std: ', label_std, 'n: ', n_element)


result_file.close()

scipy.io.savemat(os.path.join(dpath, f'./training_data_noisepatch_norm_factor_3D.mat'),
                 mdict={'input_mean': input_mean, 'input_std': input_std,
                        'label_mean': label_mean, 'label_std': label_std, 'n_element': n_element})


del patches_field, patches_label, patches_mask 