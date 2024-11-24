import pandas as pd
import matplotlib.pyplot as plt
import argparse, os, json
import csv


parser = argparse.ArgumentParser(description='compareqsmnet arguments')
parser.add_argument('-i', '--inputfile',     type=str,    required=True,  help='enter path of input file')
args = parser.parse_args()
args = vars(args)

basepath = '/fast_storage/intern1/codes/denoising_agents/2DUnet/save/'
filepath = os.path.join(basepath, args["inputfile"], 'log_val.csv')
output_png = os.path.join(basepath, args["inputfile"],'loss_graph/validation_loss')
output_csv = os.path.join(basepath, args["inputfile"],'loss_graph/validation_loss_all.csv')
df = pd.read_csv(filepath)

num_epoch = df['epoch'].max()
grouped = df.groupby(['std', 'epoch']).mean()
print(grouped)
for i in range(4):  
    if i == 0:
        l1_history = grouped['l1_loss_norm'][i*num_epoch:(i+1)*num_epoch]
        l2_history = grouped['l2_loss_norm'][i*num_epoch:(i+1)*num_epoch]
        ssim_history = grouped['ssim_norm'][i*num_epoch:(i+1)*num_epoch]
        psnr_history = grouped['psnr_norm'][i*num_epoch:(i+1)*num_epoch]    
    else:
        l1_history = [x+y for x,y in zip(grouped['l1_loss_norm'][i*num_epoch:(i+1)*num_epoch], l1_history)]
        l2_history = [x+y for x,y in zip(grouped['l2_loss_norm'][i*num_epoch:(i+1)*num_epoch], l2_history)]
        ssim_history = [x+y for x,y in zip(grouped['ssim_norm'][i*num_epoch:(i+1)*num_epoch], ssim_history)]
        psnr_history = [x+y for x,y in zip(grouped['psnr_norm'][i*num_epoch:(i+1)*num_epoch], psnr_history)]
    print(l1_history)
    print(len(l1_history))
l1_history = [x / 4 for x in l1_history]
l2_history = [x / 4 for x in l2_history]
ssim_history = [x / 4 for x in ssim_history]
psnr_history = [x / 4 for x in psnr_history]

best_l1_epoch = l1_history.index(min(l1_history))
best_l2_epoch = l2_history.index(min(l2_history))
best_ssim_epoch = ssim_history.index(max(ssim_history))
best_psnr_epoch = psnr_history.index(max(psnr_history))

print(best_l1_epoch, best_l2_epoch, best_ssim_epoch, best_psnr_epoch)


f = open(output_csv, "w")
writer = csv.writer(f)

for row in range(num_epoch):
    if row==0:
        writer.writerow(['epoch', 'l1', 'l2', 'ssim', 'psnr'])
    writer.writerow([row+1, l1_history[row], l2_history[row], ssim_history[row], psnr_history[row]])
writer.writerow(['best', best_l1_epoch, best_l2_epoch, best_ssim_epoch, best_psnr_epoch])
f.close()

plt.figure(figsize = (10,8))
plt.rc('font', size=20)
for i, std in enumerate([3, 5, 7, 9]):
    plt.plot([j for j in range(num_epoch)], grouped['l1_loss_norm'][i*num_epoch:(i+1)*num_epoch], alpha = 0.4)
plt.plot([j for j in range(num_epoch)], l1_history, alpha = 0.4)
plt.xlabel('epochs')
plt.legend((('std3', 'std5', 'std7', 'std9', 'all')),loc='center right')
plt.title(f'validation l1')
plt.savefig(f'{output_png}_l1_all.png')
    
plt.figure(figsize = (10,8))
plt.rc('font', size=20)
for i, std in enumerate([3, 5, 7, 9]):
    plt.plot([j for j in range(num_epoch)], grouped['l2_loss_norm'][i*num_epoch:(i+1)*num_epoch], alpha = 0.4)
plt.plot([j for j in range(num_epoch)], l2_history, alpha = 0.4)
plt.xlabel('epochs')
plt.legend((('std3', 'std5', 'std7', 'std9', 'all')),loc='center right')
plt.title(f'validation l2')
plt.savefig(f'{output_png}_l2_all.png')

plt.figure(figsize = (10,8))
plt.rc('font', size=20)
for i, std in enumerate([3, 5, 7, 9]):
    plt.plot([j for j in range(num_epoch)], grouped['ssim_norm'][i*num_epoch:(i+1)*num_epoch], alpha = 0.4)
plt.plot([j for j in range(num_epoch)], ssim_history, alpha = 0.4)
plt.xlabel('epochs')
plt.legend((('std3', 'std5', 'std7', 'std9', 'all')),loc='center right')
plt.title(f'validation ssim')
plt.savefig(f'{output_png}_ssim_all.png')

plt.figure(figsize = (10,8))
plt.rc('font', size=20)
for i, std in enumerate([3, 5, 7, 9]):
    plt.plot([j for j in range(num_epoch)], grouped['psnr_norm'][i*num_epoch:(i+1)*num_epoch], alpha = 0.4)
plt.plot([j for j in range(num_epoch)], psnr_history, alpha = 0.4)
plt.xlabel('epochs')
plt.legend((('std3', 'std5', 'std7', 'std9', 'all')),loc='center right')
plt.title(f'validation psnr')
plt.savefig(f'{output_png}_psnr_all.png')