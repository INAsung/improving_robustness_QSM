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

## 1. parse input
if 1:
    parser = argparse.ArgumentParser(description='compareqsmnet arguments')
    parser.add_argument('-g', '--GPU',      type=str,    default="0,1",  help='GPU device to use (0, 1, ...)')
    parser.add_argument('-s', '--save_dir', default='codes/FairComparison-main/save/final',type=str,  help='directory where inferenced maps are stored')
    parser.add_argument('-o', '--output_dir', required=True, type=str, help='directory where inferenced maps are stored')
    parser.add_argument('-e', '--net_epoch',type=int,    default=25,  help='the training epoch of the network')    

    parser.add_argument('--TRAIN_DATA',    type=str,    default='/fast_storage/intern1/QSMnet_data',   help='path for training data')
    parser.add_argument('--TEST_DATA',    type=str,    required=True,   help='path for validation data')

    # hyperparameters
    parser.add_argument('--NET_CHA',       type=int,    default=32)
    parser.add_argument('--NET_KER',       type=int,    default=5)
    parser.add_argument('--NET_ACT',       type=str,    default='leaky_relu')
    parser.add_argument('--NET_SLP',       type=float,  default=0.1)
    parser.add_argument('--NET_POOL',      type=str,    default='max')
    parser.add_argument('--NET_LAY',       type=int,    default=4)
    parser.add_argument('--TRAIN_EPOCH',   type=int,    default=25)
    parser.add_argument('--MAX_STEP',      type=int,    default=inf)
    parser.add_argument('--TRAIN_BATCH',   type=int,    default=12)
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
    if os.path.exists(f'{args.save_dir}/../CONFIG.txt'):
        with open(f'{args.save_dir}/../CONFIG.txt','r') as f:
            CONFIG=json.loads(f.read())
        args.NET_CHA = CONFIG['NET_CHA']
        args.NET_KER = CONFIG['NET_KER']
        args.NET_ACT = CONFIG['NET_ACT']
        args.NET_SLP = CONFIG['NET_SLP']
        args.NET_POOL = CONFIG['NET_POOL']
        args.NET_LAY = CONFIG['NET_LAY']
        args.TRAIN_BATCH = CONFIG['TRAIN_BATCH']
        args.TRAIN_LR = CONFIG['TRAIN_LR']
        args.TRAIN_W1 = CONFIG['TRAIN_W1']
        args.TRAIN_W2 = CONFIG['TRAIN_W2']
        args.TRAIN_OPT = CONFIG['TRAIN_OPT']
    print_options(parser,args)
    args = vars(args)

## 2. initiate
if 2:
    os.makedirs(args["save_dir"], exist_ok=True)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args['GPU'])
    device=torch.device("cuda")

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


# 3. load data

stat = scipy.io.loadmat('/fast_storage/intern1/QSMnet_data/test_stat.mat')
std0_mean = torch.tensor(stat['std0_mean']).to(device)
std0_std = torch.tensor(stat['std0_std']).to(device)
std3_mean = torch.tensor(stat['std3_mean']).to(device)
std3_std = torch.tensor(stat['std3_std']).to(device)
std5_mean = torch.tensor(stat['std5_mean']).to(device)
std5_std = torch.tensor(stat['std5_std']).to(device)
std7_mean = torch.tensor(stat['std7_mean']).to(device)
std7_std = torch.tensor(stat['std7_std']).to(device)
std9_mean = torch.tensor(stat['std9_mean']).to(device)
std9_std = torch.tensor(stat['std9_std']).to(device)
mean_all = [std0_mean, std3_mean, std5_mean, std7_mean, std9_mean]
std_all = [std0_std, std3_std, std5_std, std7_std, std9_std]

model = load_network(args).eval().to(device)

checkpoint = torch.load(f'/fast_storage/intern1/codes/FairComparison-main/save/final/model/ep025.pth')
loaded_state_dict = checkpoint['model']
new_state_dict = {}
for n, v in loaded_state_dict.items():
    name = n.replace("module.","")
    new_state_dict[name] = v
    
# load_weights(model,f'{args["save_dir"]}/../model/ep{args["net_epoch"]:03d}.pth')
model.load_state_dict(new_state_dict)

print(f'Load {args["net_epoch"]} network!')

dt = load_traindata(args['TRAIN_DATA'],load_scale=True)
label_mean = dt['label_mean']; label_std = dt['label_std'];

dt = load_testdata_denoiser(args["TEST_DATA"])

print(len(dt['tfield']))
print(len(dt['tmask']))
print(len(dt['tlabel']))


local_field_idx = 0
for subj in range(4):
    print('subj', subj)
    tlabel = dt['tlabel'][subj]
    tmask  = dt['tmask'][subj]
    for std_idx, std in enumerate([0, 3, 5, 7, 9]): # std
        tfield = dt['tfield'][local_field_idx]
        local_field_idx += 1
        tpred  = np.empty(dt['matrix_size'])

        with torch.no_grad():
            for ori in range(dt['matrix_size'][-1]):
                x = torch.tensor(tfield[np.newaxis,np.newaxis,...,ori], device=device, dtype=torch.float)
                x = ( x - mean_all[std_idx] ) / std_all[std_idx]
                pred = model(x).squeeze().cpu().numpy()*label_std+label_mean 
            
                tpred[...,ori] = pred*tmask[...,ori]
                print('std', std, 'ori', ori)

            dtimg = display_data(); sl=dt['slice'];
            dtimg.figsize = (12,8)
            dtimg.data.append(rotate(tfield[...,sl,0], -90)); dtimg.v.append((-0.05,0.05)); dtimg.label.append('field')
            dtimg.data.append(rotate(tpred[ ...,sl,0], -90)); dtimg.v.append((-0.15,0.15));   dtimg.label.append('predict')
            dtimg.data.append(rotate(tlabel[...,sl,0], -90)); dtimg.v.append((-0.15,0.15));   dtimg.label.append('cosmos')
            fig=display_images(1, 3, dtimg, p=False)
            fig.savefig(f'{args["output_dir"]}/test{subj+1}_std{std}.png')
            scipy.io.savemat(f'{args["output_dir"]}/test{subj+1}_std{std}.mat', mdict={'pred':tpred})