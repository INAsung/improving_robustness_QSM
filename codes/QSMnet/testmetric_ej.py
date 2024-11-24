## 1. parse input
## 2. initiate
## 3. train with 중간그림들 저장
## 4. test
## 5. testmeric
## 6. retrain
import argparse, os, json
from cmath import inf
import numpy as np
import pandas as pd
import torch, random
from util import *
from network import *
from importdata import *
from scipy.ndimage import rotate
from datetime import datetime


import argparse
parser = argparse.ArgumentParser(description='QSMnet arguments')
parser.add_argument('-p', '--test_data_path',type=str,    required=True,  help='directory where inferenced maps are stored')

parser.add_argument('--TRAIN_DATA',    type=str,    default='/fast_storage/intern1/QSMnet_data',   help='path for training data')
parser.add_argument('--TEST_DATA',    type=str,    default='/fast_storage/intern1/QSMnet_data',   help='path for validation data')


# settings for reproducibility
parser.add_argument('--BATCH_ORDER_SEED',   type=int,    default=0)
parser.add_argument('--WEIGHT_INIT_SEED',   type=int,    default=0)
parser.add_argument('--CUDA_deterministic', type=str,    default='True')
parser.add_argument('--CUDA_benchmark',     type=str,    default='False')

args = parser.parse_args()
print_options(parser,args)
args = vars(args)

## 2. initiate
if 2:
    os.environ['PYTHONHASHSEED'] = "0"
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.random.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    if args['CUDA_deterministic'] == 'True':
        torch.backends.cudnn.deterministic=True
    elif args['CUDA_deterministic'] == 'False':
        torch.backends.cudnn.deterministic=False
    else:
        print('wrong args[CUDA_deterministic]')
        raise ValueError
    if args['CUDA_benchmark'] == 'True':
        torch.backends.cudnn.benchmark=True
    elif args['CUDA_benchmark'] == 'False':
        torch.backends.cudnn.benchmark=False
    else:
        print('wrong args[CUDA_benchmark]')
        raise ValueError

logger_test = Logger(f'{args["test_data_path"]}/log_test.csv')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device=torch.device("cuda")
dt = load_label(args["TEST_DATA"])

for subj in range(4):
    print('now on ', args["test_data_path"], 'subj: ', subj)
    #cosmos = torch.tensor(dt['tlabel'][subj], device=device, dtype=torch.float)
    #tmask  = torch.tensor(dt['tmask'][subj], device=device, dtype=torch.bool)
    cosmos =dt['tlabel'][subj]
    tmask  = dt['tmask'][subj]

    for std in [0,3,5,7,9]:
        print('std: ',std)
        #tpred  = torch.tensor(scipy.io.loadmat(f'{args["test_data_path"]}/test{subj+1}_std{std}.mat')['pred'], device=device, dtype=torch.float)
        tpred  = scipy.io.loadmat(f'{args["test_data_path"]}/test{subj+1}_std{std}.mat')['pred']       
        for ori in range(dt['matrix_size'][-1]):
            #nrmse, psnr, ssim, hfen = compute_all_torch(tpred[...,ori], cosmos[...,ori], tmask[...,ori])
            nrmse, psnr, ssim, hfen = compute_all(tpred[...,ori], cosmos[...,ori], tmask[...,ori])
            logger_test.log({'subj': subj+1, 'std':std, 'ori':ori, 'nrmse':nrmse, 'psnr':psnr, 'ssim':ssim, 'hfen':hfen})