import pandas as pd
import matplotlib.pyplot as plt
import argparse, os, json
import csv


parser = argparse.ArgumentParser(description='compareqsmnet arguments')
parser.add_argument('-i', '--inputfile',     type=str,    required=True,  help='enter path of input file')
args = parser.parse_args()
args = vars(args)

basepath = '/fast_storage/intern1/codes/denoising_agents/3DUnet_ver2/save/'
filepath = os.path.join(basepath, args["inputfile"], 'log_train.csv')
output_png = os.path.join(basepath, args["inputfile"],'loss_graph/loss_all.png')
output_csv = os.path.join(basepath, args["inputfile"],'loss_graph/loss_sorted.csv')
df = pd.read_csv(filepath)

grouped = df.groupby('epoch')

averages = grouped.mean()
print(averages)

total_loss_history = averages['total_loss']
l1_loss_history = averages['l1loss']
l2_loss_history = averages['l2loss']
ssimloss_history = averages['ssimloss']

best_tot_epoch = total_loss_history.idxmin()
best_l1_epoch = l1_loss_history.idxmin()
best_l2_epoch =  l2_loss_history.idxmin()
best_ssim_epoch =  ssimloss_history.idxmin()

print(best_l1_epoch, best_l2_epoch, best_tot_epoch, best_ssim_epoch)
average_with_best_epoch = averages
average_with_best_epoch.loc[ len(average_with_best_epoch)+1]=[0,best_tot_epoch,best_l1_epoch,best_l2_epoch,best_ssim_epoch,0]
del average_with_best_epoch['step']
del average_with_best_epoch['lr']
average_with_best_epoch.to_csv(output_csv)

all_history = [total_loss_history, l1_loss_history, l2_loss_history, ssimloss_history]
style_list = ['.', 'o', 'v', '^']
plt.figure(figsize = (10,8))
plt.rc('font', size=20)
for i in range(0,4):
    plt.plot([j for j in range(len(all_history[0]))], all_history[i], marker = style_list[i], alpha = 0.4)
    plt.ylim([0,1])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(('total_loss','l1loss', 'l2loss','ssimloss'),loc='center right')
plt.title('loss_all')
plt.savefig(output_png)