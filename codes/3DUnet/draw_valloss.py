import pandas as pd
import matplotlib.pyplot as plt
import argparse, os, json
import csv


parser = argparse.ArgumentParser(description='compareqsmnet arguments')
parser.add_argument('-i', '--inputfile',     type=str,    required=True,  help='enter path of input file')
args = parser.parse_args()
args = vars(args)

basepath = '/fast_storage/intern1/codes/denoising_agents/3DUnet_ver2/save/'
filepath = os.path.join(basepath, args["inputfile"], 'log_val.csv')
output_png = os.path.join(basepath, args["inputfile"],'loss_graph/validation_loss')
output_csv = os.path.join(basepath, args["inputfile"],'loss_graph/validation_loss_sorted.csv')
df = pd.read_csv(filepath)

num_epoch = df['epoch'].max()
value_norm_list = ['l1_loss_norm', 'l2_loss_norm', 'ssim_norm']
value_ori_list = ['l1_loss_ori', 'l2_loss_ori', 'ssim_ori']
grouped = df.groupby(['std', 'epoch']).mean()
for i, std in enumerate([3, 5, 7, 9]):
    l1_history = grouped['l1_loss_norm'][i*num_epoch:(i+1)*num_epoch]
    l2_history = grouped['l2_loss_norm'][i*num_epoch:(i+1)*num_epoch]
    ssim_history = grouped['ssim_norm'][i*num_epoch:(i+1)*num_epoch]
    psnr_history = grouped['psnr_norm'][i*num_epoch:(i+1)*num_epoch]

    all_history = [ l1_history, l2_history, ssim_history]
    best_l1_epoch = l1_history.idxmin()
    best_l2_epoch = l2_history.idxmin()
    best_ssim_epoch = ssim_history.idxmax()
    best_psnr_epoch = psnr_history.idxmax()
    print(best_l1_epoch, best_l2_epoch, best_ssim_epoch, best_psnr_epoch )
    plt.figure(figsize = (10,8))
    plt.rc('font', size=20)
    for i, value in enumerate(['l1', 'l2']):
        plt.plot([j for j in range(num_epoch)], all_history[i], alpha = 0.4)
       # plt.ylim([0,1])
        plt.xlabel('epochs')
        plt.legend(('l1','l2', 'ssim'),loc='center right')
        plt.title(f'validation l1l2 std{std}')
    plt.savefig(f'{output_png}_l1l2_std{std}.png')

    plt.figure(figsize = (10,8))
    plt.rc('font', size=20)
    for i, value in enumerate(['psnr']):
        plt.plot([j for j in range(num_epoch)],psnr_history, alpha = 0.4)
       # plt.ylim([0,1])
        plt.xlabel('epochs')
        plt.title(f'validation psnr std{std}')
        plt.savefig(f'{output_png}_{value}_std_{std}.png')

    plt.figure(figsize = (10,8))
    plt.rc('font', size=20)
    for i, value in enumerate(['ssim']):
        plt.plot([j for j in range(num_epoch)], ssim_history, alpha = 0.4)
       # plt.ylim([0,1])
        plt.xlabel('epochs')
        plt.title(f'validation ssim std{std}')
        plt.savefig(f'{output_png}_{value}_std_{std}.png')