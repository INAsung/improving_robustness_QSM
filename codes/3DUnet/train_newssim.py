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
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms, datasets
from util import *
from threeDUnet import *
from importdata import *
from scipy.ndimage import rotate
from datetime import datetime
from torch.autograd import Variable
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import ast

## 1. parse input
if 1:
    parser = argparse.ArgumentParser(description='compareqsmnet arguments')
    parser.add_argument('-g', '--GPU',     type=str,    default="0,1",  help='GPU device to use (0, 1, ...)')
    parser.add_argument('-s', '--save_dir',type=str,    default='/fast_storage/intern1/codes/denoising_agents/',  help='directory where trained networks are stored')
    parser.add_argument('-w', '--weight_list',type=ast.literal_eval,    required = True, help='weight list (l1-l2-ssim)')
    parser.add_argument('--TRAIN_DATA',    type=str,    default='/fast_storage/intern1/QSMnet_data',  help='path for training data')
    parser.add_argument('--VAL_DATA',      type=str,    default='/fast_storage/intern1/QSMnet_data',  help='path for validation data')

    # hyperparameters
    parser.add_argument('--NET_CHA',       type=int,    default=32)
    parser.add_argument('--NET_KER',       type=int,    default=5)
    parser.add_argument('--NET_ACT',       type=str,    default='leaky_relu')
    parser.add_argument('--NET_SLP',       type=float,  default=0.1)
    parser.add_argument('--NET_POOL',      type=str,    default='max')
    parser.add_argument('--NET_LAY',       type=int,    default=4)
    parser.add_argument('--TRAIN_EPOCH',   type=int,    default=50)
    parser.add_argument('--MAX_STEP',      type=int,    default=inf)
    parser.add_argument('--TRAIN_BATCH',   type=int,    default=11)
    parser.add_argument('--TRAIN_LR',      type=float,  default=1e-4*5)
    parser.add_argument('--TRAIN_W1',      type=float,  default=0.5)
    parser.add_argument('--TRAIN_W2',      type=float,  default=0.1)
    parser.add_argument('--TRAIN_OPT',     type=str,    default='Adam')

    # settings for reproducibility
    parser.add_argument('--BATCH_ORDER_SEED',   type=int,    default=0)
    parser.add_argument('--WEIGHT_INIT_SEED',   type=int,    default=0)
    parser.add_argument('--CUDA_deterministic', type=str,    default='True')
    parser.add_argument('--CUDA_benchmark',     type=str,    default='False')

    args = parser.parse_args()
    if os.path.exists(f'{args.save_dir}/CONFIG.txt'):
        with open(f'{args.save_dir}/CONFIG.txt','r') as f:
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
    print("#1 ended")


## 2. initiate
if 2:
    my_list = list(args["weight_list"])
    formatted_weights = '-'.join(map(str, my_list))

    print(formatted_weights)

    args["save_dir"] = f'/fast_storage/intern1/codes/denoising_agents/3DUnet_ver2/save/{formatted_weights}'
    os.makedirs(f'{args["save_dir"]}/model', exist_ok=True)
    os.makedirs(f'{args["save_dir"]}/val', exist_ok=True)
    os.makedirs(f'{args["save_dir"]}/loss_graph', exist_ok=True)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    device=torch.device("cuda")

    os.environ['PYTHONHASHSEED'] = "0"
    random.seed(args["BATCH_ORDER_SEED"])
    np.random.seed(0)
    torch.manual_seed(0)
    torch.random.manual_seed(args["WEIGHT_INIT_SEED"])
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
    print("#2-1 ended")

## load training & validation datas
transform = transforms.Compose([ToTensor(),RandomFlip()])
transform_origin = transforms.Compose([ToTensor()])

dataset_train = Dataset('/fast_storage/intern1/QSMnet_data/noisepatch_3D_all.hdf5', transform=transform)
loader_train = DataLoader(dataset_train, batch_size = args['TRAIN_BATCH'], shuffle=True, num_workers=6) # patch size
    
dataset_val = Dataset_val('/fast_storage/intern1/QSMnet_data/noisepatch_val_3D_all.hdf5', transform=transform_origin)
loader_val = DataLoader(dataset_val, batch_size = 1, shuffle=False, num_workers=6) # original size 


val_stat = scipy.io.loadmat('/fast_storage/intern1/QSMnet_data/noisepatch_norm_factor_val_3D_all.mat')
val_label_mean = val_stat['label_mean']
val_label_std = val_stat['label_std']
val_std3_mean = val_stat['std3_mean']
val_std3_std = val_stat['std3_std']
val_std5_mean = val_stat['std5_mean']
val_std5_std = val_stat['std5_std']
val_std7_mean = val_stat['std7_mean']
val_std7_std = val_stat['std7_std']
val_std9_mean = val_stat['std9_mean']
val_std9_std = val_stat['std9_std']

# 5. training
step = 0

model = Unet_4(args).cuda()
model = nn.DataParallel(model)

if args["TRAIN_OPT"] == "RMSProp":
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args["TRAIN_LR"])    
if args["TRAIN_OPT"] == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args["TRAIN_LR"]) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=600, gamma=0.95, last_epoch=-1)
os.makedirs(f'{args["save_dir"]}/model', exist_ok=True)
os.makedirs(f'{args["save_dir"]}/val', exist_ok=True)
logger_train = Logger(f'{args["save_dir"]}/log_train.csv')
logger_val = Logger(f'{args["save_dir"]}/log_val.csv')

ssim_loss = SSIM(channel=1)

w_l1 = args['weight_list'][0]
w_l2 = args['weight_list'][1]
w_ssim = args['weight_list'][2]
print(w_l1, w_l2, w_ssim)
print(len(loader_train))

for epoch in range(args["TRAIN_EPOCH"]): 
    model.train()
    for batch, data in enumerate(loader_train, 1):
    ## 1. train dataset
        clean = data['clean'].to(device)
        noisy = data['noisy'].to(device)
        mask = data['mask'].to(device)
        output = model(noisy)*mask 
        #backward pass
        l1_loss_ =  l1_loss(clean[mask>0], output[mask>0])
        l2_loss_ =  l2_loss(clean[mask>0], output[mask>0])
        # ssim loss
        ssim_out = 0
        equalcount = 0
        # for batchnum in range(output.shape[0]):
        #     mask_pick = mask[batchnum,...]
        #     output_pick = output[batchnum,...]
        #     clean_pick = clean[batchnum,...]
        #     if (torch.equal(output_pick,clean_pick)):
        #         equalcount += 1
        #         continue
        #     nowssim = ssim_loss(output_pick.permute(-1, 0, 1, 2), clean_pick.permute(-1, 0, 1, 2))
        #     ssim_out += nowssim
        # ssim_out /= (output.shape[0] - equalcount)
        # ssim_loss_ = 1 - ssim_out
        for batchnum in range(output.shape[0]):
            mask_pick = mask[batchnum,...]
            output_pick = output[batchnum,...]
            clean_pick = clean[batchnum,...]
            nowssim = ssim(output_pick.permute([-1,0,1,2]), clean_pick.permute([-1,0,1,2]), mask_pick.permute([-1,0,1,2]))
            ssim_out += nowssim
        ssim_out /= output.shape[0]
        ssim_loss_ = 1 - ssim_out

        total_loss_ = l1_loss_ * w_l1 + l2_loss_ * w_l2 + ssim_loss_ * w_ssim 

        optimizer.zero_grad()
        total_loss_.backward()
        optimizer.step()

        ## scheduler step
        scheduler.step()
        
        t_totalloss = total_loss_.item(); t_l1loss=l1_loss_.item(); t_l2loss = l2_loss_.item(); t_ssimloss=ssim_loss_.item(); 
        logger_train.log({'epoch':epoch+1, 'step':step,
                        'total_loss':t_totalloss, 'l1loss':t_l1loss, 'l2loss': t_l2loss, 'ssimloss':t_ssimloss, 
                        'lr':optimizer.param_groups[0]['lr']})        
    if (epoch > 15):
        save_checkpoint(epoch, step, model, optimizer, scheduler, f'{args["save_dir"]}/model', f'ep{epoch+1:03d}.pth')
    
    # 2. validation
    with torch.no_grad():
        model.eval()

        for batch, data in enumerate(loader_val):
            for std_now, noisy_each in enumerate(data['noisy']):
                # forward pass
                std_now = std_now*2+3
                clean = data['clean'].to(device).squeeze(0)
                noisy = noisy_each.to(device)
                mask = data['mask'].to(device)
                
                output = model(noisy)*mask
                output = output.squeeze(0)
                noisy = noisy.squeeze(0)
                mask = mask.squeeze(0)

                ##손실함수 계산
                psnr_norm = compute_psnr(clean, output, mask)
                nrmse_norm = compute_nrmse(clean, output, mask)
                l1_loss_norm = l1_loss(clean[mask>0], output[mask>0])
                l2_loss_norm = l2_loss(clean[mask>0], output[mask>0])
                ssim_norm = ssim(clean.permute(-1, 0, 1, 2), output.permute(-1, 0, 1, 2), mask.permute(-1,0,1,2))
                total_norm = l1_loss_norm * w_l1 + l2_loss_norm * w_l2 + ssim_norm * w_ssim 

                clean_ori = clean.clone().detach() * torch.tensor(val_label_std).to(clean.device) + torch.tensor(val_label_mean).to(clean.device)
                output_ori = output.clone().detach() * torch.tensor(val_label_std).to(clean.device) + torch.tensor(val_label_mean).to(clean.device)

                psnr_ori = compute_psnr(clean_ori, output_ori, mask)
                nrmse_ori = compute_nrmse(clean_ori, output_ori, mask)
                ssim_ori = ssim(clean_ori.permute(-1, 0, 1, 2), output_ori.permute(-1, 0, 1, 2),mask.permute(-1,0,1,2))
                l1_loss_ori = l1_loss(clean_ori[mask>0], output_ori[mask>0])
                l2_loss_ori = l2_loss(clean_ori[mask>0], output_ori[mask>0])
                total_ori = l1_loss_ori * w_l1 + l2_loss_ori * w_l2 + ssim_ori * w_ssim 

                logger_val.log({'epoch':epoch+1, 'std': std_now, 'ori':batch,
                'total_norm': total_norm.item(), 'l1_loss_norm': l1_loss_norm.item(), 'l2_loss_norm': l2_loss_norm.item(), 'ssim_norm': ssim_norm.item(), 'psnr_norm':psnr_norm,
                'total_ori': total_ori.item(), 'l1_loss_ori': l1_loss_ori.item(), 'l2_loss_ori': l2_loss_ori.item(), 'ssim_ori': ssim_ori.item(), 'psnr_ori':psnr_ori})
                            
            if batch==0 and std_now == 9:
                noisy_ori = noisy.clone().detach() * torch.tensor(val_std9_std).to(clean.device) + torch.tensor(val_std9_mean).to(clean.device)

                dtimg = display_data(); sl=80;
                dtimg.figsize = (12,8)
                dtimg.data.append(rotate(noisy_ori[0,...,sl].cpu().detach().numpy(), -90)); dtimg.v.append((-0.05,0.05)); dtimg.label.append(f'noisy_std9')
                dtimg.data.append(rotate(output_ori[0,...,sl].cpu().detach().numpy(), -90)); dtimg.v.append((-0.05,0.05));   dtimg.label.append('predicted_denoised')
                dtimg.data.append(rotate(clean_ori[0,...,sl].cpu().detach().numpy(), -90)); dtimg.v.append((-0.05,0.05));   dtimg.label.append('clean')
                fig=display_images(1, 3, dtimg, p=False)
                fig.savefig(f'{args["save_dir"]}/val/{epoch+1:03d}.png')          
                print(f"{datetime.now().strftime('%y-%m-%d %H:%M:%S')}   Epoch: {epoch+1:04d}")    