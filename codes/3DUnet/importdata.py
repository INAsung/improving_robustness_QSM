import scipy.io
import h5py
import nibabel as nib
import numpy as np
import glob, os
import random
import torch.nn as nn
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(Dataset,self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.datanum =  len(h5py.File(data_dir)['clean'])

    def __len__(self):
        return self.datanum # total subject #
        
    def __getitem__(self, index):
        with h5py.File(self.data_dir, 'r') as h5file:
            
            # loading data
            clean = np.array(h5file['clean'][index], dtype=np.float32)
            # pick random noise
            randi = random.randint(1, 4)
            if(randi==1):
                noisy = np.array(h5file['std3'][index], dtype=np.float32)
            elif(randi==2):
                noisy = np.array(h5file['std5'][index], dtype=np.float32)
            elif(randi==3):
                noisy = np.array(h5file['std7'][index], dtype=np.float32)
            else:
                noisy = np.array(h5file['std9'][index], dtype=np.float32)

            mask = np.array(h5file['pmask'][index], dtype=np.float32)
                
            clean = (clean * mask)
            noisy = (noisy * mask)
                
            # If no channel dimension
            if clean.ndim == 3:
                clean =clean[np.newaxis, :, :, :]
            if noisy.ndim == 3:
                noisy = noisy[np.newaxis, :, :, :]
            if mask.ndim == 3:
                mask = mask[np.newaxis, :, :, :]
                    
            data = {'clean': clean, 'noisy': noisy, 'mask': mask}
                
            if self.transform:
                data = self.transform(data)
                
            return data


class Dataset_val(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(Dataset_val,self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.datanum =  len(h5py.File(data_dir)['clean'])

    def __len__(self):
        return self.datanum # total subject #
        
    def __getitem__(self, index):
        with h5py.File(self.data_dir, 'r') as h5file:
            # loading data
            clean = np.array(h5file['clean'][index], dtype=np.float32)
            # pick random noise
            noisy = []
            noisy.append(np.array(h5file['std3'][index], dtype=np.float32))
            noisy.append(np.array(h5file['std5'][index], dtype=np.float32))
            noisy.append(np.array(h5file['std7'][index], dtype=np.float32))
            noisy.append(np.array(h5file['std9'][index], dtype=np.float32))
            
            mask = np.array(h5file['pmask'][index], dtype=np.float32)
                
            clean = (clean * mask)
            for i, item in enumerate(noisy):
                noisy[i] = item * mask
                
            # If no channel dimension
            if clean.ndim == 3:
                clean =clean[np.newaxis, :, :, :]
            for i, noisy_each in enumerate(noisy):
                if noisy_each.ndim == 3:
                    noisy[i] = noisy_each[np.newaxis, :, :, :]
            if mask.ndim == 3:
                mask = mask[np.newaxis, :, :, :]
                    
            data = {'clean': clean, 'noisy': noisy, 'mask': mask}
                
            if self.transform:
                data = self.transform(data)
            return data

def load_testdata(path):
      
    clean = []; noisy = []; mask = []; 
        
    for subj in [1, 2, 3, 4]:
        for idx, std in enumerate([0, 3, 5, 7, 9]):
            m = scipy.io.loadmat(f'{path}/std{std}/test{subj}/phscos_final.mat')
            if std == 0:
                clean.append(m[f'multiphs'])
                mask.append(m[f'multimask'].astype(bool))
            else:
                noisy.append(m[f'multiphs'])
            
    return {'num':4,'clean':clean,'noisy':noisy,'mask':mask,'matrix_size':mask[0].shape,'slice':80}        




class ToTensor(object):
    def __call__(self, data):
        clean, noisy, mask = data['clean'], data['noisy'], data['mask']
        
        clean = clean.copy()
        mask = mask.copy()

        if type(noisy) == list:
            for i, noisy_each in enumerate(noisy):
                noisy_each = noisy_each.copy()
                noisy[i] = torch.from_numpy(noisy_each)
            data = {'clean': torch.from_numpy(clean), 'noisy': noisy, 'mask': torch.from_numpy(mask)}

        else:
            noisy = noisy.copy()
            data = {'clean': torch.from_numpy(clean), 'noisy': torch.from_numpy(noisy), 'mask': torch.from_numpy(mask)}
        
        
        return data        

class Normalization(object):
    def __init__(self):
        pass

    def __call__(self, data):
        clean, noisy, mask = data['clean'], data['noisy'], data['mask']

        max_clean = clean.max()
        min_clean = clean.min()
        max_noisy = noisy.max()
        min_noisy = noisy.min()

        clean = (clean - min_clean) / (max_clean - min_clean)  
        noisy = (noisy - min_noisy) / (max_noisy - min_noisy)  

        data = {'clean': clean, 'noisy': noisy, 'mask': mask}

        return data
    
class RandomCrop3D(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, data):
        clean, noisy, mask = data['clean'], data['noisy'], data['mask']

        valid_crop = False
        while not valid_crop:
            # 시작 지점을 랜덤하게 선택, 3차원이기에 3개의 식 존재
            start_x = np.random.randint(0, clean.shape[2] - self.output_size[0]) #zxy순서
            start_y = np.random.randint(0, clean.shape[3] - self.output_size[1])
            start_z = np.random.randint(0, clean.shape[1] - self.output_size[2])

            # 시작 지점에서 output_size만큼 잘라냅니다.
            cropped_clean = clean[:, start_z:start_z + self.output_size[2], start_y:start_y + self.output_size[1], start_x:start_x + self.output_size[0]]
            cropped_noisy = noisy[:, start_z:start_z + self.output_size[2], start_y:start_y + self.output_size[1], start_x:start_x + self.output_size[0]]
            cropped_mask = mask[:, start_z:start_z + self.output_size[2], start_y:start_y + self.output_size[1], start_x:start_x + self.output_size[0]]

            # 모든 텐서의 값이 0이 아닌지 확인합니다.
            if not (torch.all(cropped_clean == 0) or torch.all(cropped_noisy == 0) or torch.all(cropped_mask == 0)):
                valid_crop = True

        data = {'clean': cropped_clean, 'noisy': cropped_noisy, 'mask': cropped_mask}
        return data
    
class CenterCrop3D(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, data):
        clean, noisy, mask = data['clean'], data['noisy'], data['mask']

        # 입력 텐서의 중심 지점을 계산합니다.
        center_x = clean.shape[2] // 2
        center_y = clean.shape[3] // 2
        center_z = clean.shape[1] // 2

        # 중심 지점을 기준으로 output_size 만큼의 크기로 잘라냅니다.
        start_x = center_x - self.output_size[0] // 2
        start_y = center_y - self.output_size[1] // 2
        start_z = center_z - self.output_size[2] // 2

        end_x = start_x + self.output_size[0]
        end_y = start_y + self.output_size[1]
        end_z = start_z + self.output_size[2]

        cropped_clean = clean[:, start_z:end_z, start_y:end_y, start_x:end_x]
        cropped_noisy = noisy[:, start_z:end_z, start_y:end_y, start_x:end_x]
        cropped_mask = mask[:, start_z:end_z, start_y:end_y, start_x:end_x]

        data = {'clean': cropped_clean, 'noisy': cropped_noisy, 'mask': cropped_mask}
        return data

class RandomFlip(object):
    def __call__(self, data):
        clean, noisy, mask = data['clean'], data['noisy'], data['mask']
        
        # 수평 뒤집기 (가로축)
        if np.random.rand() > 0.5:
            clean = torch.flip(clean, [2])  # 2D 이미지라면 [1] 사용
            noisy = torch.flip(noisy, [2])  # [2]는 3차원 이미지의 가로축
            mask = torch.flip(mask, [2])

        # 수직 뒤집기 (세로축)
        if np.random.rand() > 0.5:
            clean = torch.flip(clean, [1])  # 2D 이미지라면 [0] 사용
            noisy = torch.flip(noisy, [1])  # [1]은 3차원 이미지의 세로축
            mask = torch.flip(mask, [1])

        data = {'clean': clean, 'noisy': noisy, 'mask': mask}

        return data
    
class RandomRotate(object):
    def __init__(self, angle_range=(-30, 30)):
        self.angle_range = angle_range

    def __call__(self, data):
        clean, noisy, mask = data['clean'], data['noisy'], data['mask']

        # 임의의 각도를 선택
        angle = np.random.uniform(self.angle_range[0], self.angle_range[1])
        
        # 선택한 각도로 데이터를 회전
        clean = rotate(clean, angle, reshape=False, mode='nearest')
        noisy = rotate(noisy, angle, reshape=False, mode='nearest')
        mask = rotate(mask, angle, reshape=False, mode='nearest')

        data = {'clean': clean, 'noisy': noisy, 'mask': mask}

        return data