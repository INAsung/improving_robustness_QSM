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
            if clean.ndim == 2:
                clean =clean[np.newaxis, :, :]
            if noisy.ndim == 2:
                noisy = noisy[np.newaxis, :, :]
            if mask.ndim == 2:
                mask = mask[np.newaxis, :, :]
                    
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
            clean = np.array(h5file['clean'][index][np.newaxis,:,:,:], dtype=np.float32)
            clean =np.transpose(clean, (-1,0,1,2))

            noisy = []
            noisy.append(np.array(h5file['std3'][index][np.newaxis,:,:,:], dtype=np.float32))
            noisy.append(np.array(h5file['std5'][index][np.newaxis,:,:,:], dtype=np.float32))
            noisy.append(np.array(h5file['std7'][index][np.newaxis,:,:,:], dtype=np.float32))
            noisy.append(np.array(h5file['std9'][index][np.newaxis,:,:,:], dtype=np.float32))
            
            mask = np.array(h5file['pmask'][index][np.newaxis, :,:,:], dtype=np.float32)
            mask =np.transpose(mask,(-1,0,1,2))
            clean = (clean * mask)
            for i, item in enumerate(noisy):
                noisy[i] = np.transpose(noisy[i],(-1,0,1,2))
                noisy[i] = noisy[i] * mask
            data = {'clean': clean, 'noisy': noisy, 'mask': mask}
                
            if self.transform:
                data = self.transform(data)

            return data

def load_testdata(path):
    clean = []; noisy = []; mask = []; 
        
    for subj in [1, 2, 3, 4]:
        for idx, std in enumerate([0, 3, 5, 7, 9]):
            m = scipy.io.loadmat(f'{path}/std{std}/test{subj}/phscos_final.mat')
            for ori in range(5):
                if std == 0:
                    clean.append(np.transpose(m[f'multiphs'][np.newaxis,:,:,:,ori], (-1,0,1,2)))
                    mask.append(np.transpose(m[f'multimask'][np.newaxis,:,:,:,ori].astype(bool),(-1,0,1,2)))
                else:
                    noisy.append(np.transpose(m[f'multiphs'][np.newaxis,:,:,:,ori],(-1,0,1,2)))
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
    
class Random_or_Center_Crop2D(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, data):
        clean, noisy, mask = data['clean'], data['noisy'], data['mask']

        valid_crop = False
        while not valid_crop:
            if np.random.rand() > 0.7: # randomcrop 
                start_x = np.random.randint(0, clean.shape[1] - self.output_size[0])
                start_y = np.random.randint(0, clean.shape[2] - self.output_size[1])

                # 시작 지점에서 output_size만큼 잘라냅니다.
                cropped_clean = clean[:, start_x:start_x + self.output_size[0], start_y:start_y + self.output_size[1]]
                cropped_noisy = noisy[:, start_x:start_x + self.output_size[0], start_y:start_y + self.output_size[1]]
                cropped_mask = mask[:, start_x:start_x + self.output_size[0], start_y:start_y + self.output_size[1]]

                # 모든 텐서의 값이 0이 아닌지 확인합니다.
                if (torch.all(cropped_clean == 0) or torch.all(cropped_noisy == 0) or torch.all(cropped_mask == 0)):
                    continue
                else:  return {'clean': cropped_clean, 'noisy': cropped_noisy, 'mask': cropped_mask}

            else: # centercrop
                # 입력 텐서의 중심 지점을 계산합니다.
                center_x = clean.shape[1] // 2
                center_y = clean.shape[2] // 2

                # 중심 지점을 기준으로 output_size 만큼의 크기로 잘라냅니다.
                start_x = center_x - self.output_size[0] // 2
                start_y = center_y - self.output_size[1] // 2

                end_x = start_x + self.output_size[0]
                end_y = start_y + self.output_size[1]

                cropped_clean = clean[:,start_x:end_x, start_y:end_y]
                cropped_noisy = noisy[:,start_x:end_x, start_y:end_y]
                cropped_mask = mask[:, start_x:end_x, start_y:end_y]

                data = {'clean': cropped_clean, 'noisy': cropped_noisy, 'mask': cropped_mask}
                return data

class CenterCrop2D(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, data):
        clean, noisy, mask = data['clean'], data['noisy'], data['mask']

        # 입력 텐서의 중심 지점을 계산합니다.
        center_x = clean.shape[1] // 2
        center_y = clean.shape[2] // 2

        # 중심 지점을 기준으로 output_size 만큼의 크기로 잘라냅니다.
        start_x = center_x - self.output_size[0] // 2
        start_y = center_y - self.output_size[1] // 2

        end_x = start_x + self.output_size[0]
        end_y = start_y + self.output_size[1]

        cropped_clean = clean[:,start_x:end_x, start_y:end_y]
        cropped_noisy = noisy[:,start_x:end_x, start_y:end_y]
        cropped_mask = mask[:, start_x:end_x, start_y:end_y]

        data = {'clean': cropped_clean, 'noisy': cropped_noisy, 'mask': cropped_mask}
        return data

class RandomFlip(object):
    def __call__(self, data):
        clean, noisy, mask = data['clean'], data['noisy'], data['mask']
        
        # 수평 뒤집기 (가로축)
        if np.random.rand() > 0.5:
            clean = torch.flip(clean, [1])  # 2D 이미지라면 [1] 사용
            noisy = torch.flip(noisy, [1])  # [2]는 3차원 이미지의 가로축
            mask = torch.flip(mask, [1])

        # 수직 뒤집기 (세로축)
        if np.random.rand() > 0.5:
            clean = torch.flip(clean, [0])  # 2D 이미지라면 [0] 사용
            noisy = torch.flip(noisy, [0])  # [1]은 3차원 이미지의 세로축
            mask = torch.flip(mask, [0])

        data = {'clean': clean, 'noisy': noisy, 'mask': mask}

        return data
    