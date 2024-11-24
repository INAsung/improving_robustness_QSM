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

def setargues():
    if 1:
        parser = argparse.ArgumentParser(description='compareqsmnet arguments')
        #parser.add_argument('-g', '--GPU',     type=int,    required=True,  help='GPU device to use (0, 1, ...)')
        parser.add_argument('-s', '--save_dir',type=str,    default='/fast_storage/intern1/codes/FairComparison-main/save/final',  help='directory where trained networks are stored')
        parser.add_argument('--TRAIN_DATA',    type=str,    default='/fast_storage/intern1/datas/QSMnet_data/std0',   help='path for training data')
        parser.add_argument('--VAL_DATA',      type=str,    default='/fast_storage/intern1/datas/QSMnet_data/std0',   help='path for validation data')
        # hyperparameters
        parser.add_argument('--NET_CHA',       type=int,    default=32)
        parser.add_argument('--NET_KER',       type=int,    default=5)
        parser.add_argument('--NET_ACT',       type=str,    default='leaky_relu')
        parser.add_argument('--NET_SLP',       type=float,  default=0.1)
        parser.add_argument('--NET_POOL',      type=str,    default='max')
        parser.add_argument('--NET_LAY',       type=int,    default=4)
        parser.add_argument('--TRAIN_EPOCH',   type=int,    default=14)
        parser.add_argument('--MAX_STEP',      type=int,    default=inf)
        parser.add_argument('--TRAIN_BATCH',   type=int,    default=15)
        parser.add_argument('--TRAIN_LR',      type=float,  default=0.001)
        parser.add_argument('--TRAIN_W1',      type=float,  default=0.5)
        parser.add_argument('--TRAIN_W2',      type=float,  default=0.1)
        parser.add_argument('--TRAIN_OPT',     type=str,    default='RMSProp')
        # settings for reproducibility
        parser.add_argument('--BATCH_ORDER_SEED',   type=int,    default=0)
        parser.add_argument('--WEIGHT_INIT_SEED',   type=int,    default=0)
        parser.add_argument('--CUDA_deterministic', type=str,    default='True')
        parser.add_argument('--CUDA_benchmark',     type=str,    default='False')

        args = parser.parse_args()
        args = vars(args)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    device=torch.device("cuda")

    dt = load_traindata(args['TRAIN_DATA'])
    print("finished load patched train data")
    print(dt['voxel_size'])
    d = dipole_kernel((64,64,64), dt['voxel_size'], (0,0,1))
    model = load_network(args)
    model = nn.DataParallel(model)

    ind = list(range(len(dt['pfield']))) 
    num_batch = int(len(ind)/args["TRAIN_BATCH"])
    i = ii = 1
    ind_batch = sorted(ind[i*args["TRAIN_BATCH"]:(i+1)*args["TRAIN_BATCH"]])
    x_batch = torch.tensor(dt['pfield'][ind_batch,...], device=device, dtype=torch.float).unsqueeze(1)
    x_batch_val = torch.tensor(dt['vfield'][np.newaxis,np.newaxis,...,ii], device=device, dtype=torch.float)

    return [model, x_batch, x_batch_val]