import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
import sys,os

np.set_printoptions(suppress=True)

"""
April 10

4 numpy arrays as input,
x_train
yen_train
yf_train
yt_train

eliminate non interacting data from each

"""

parser = argparse.ArgumentParser(description="")
non_opt = parser.add_argument_group("mandatory arguments")
non_opt.add_argument('-i', metavar="<dat>", type=str, dest="input_datas",nargs='+', required=True, help="-i x y " )

args = parser.parse_args()
input_datas = args.input_datas
low_limit = 0.05

x = np.load(input_datas[0])
yen_f = np.load(input_datas[1])
yt = np.load(input_datas[2])

yen_abs = np.abs(np.copy(yen_f[:,0]))
f_mag = np.linalg.norm(yen_f[:,1:],axis=1)
t_mag = np.linalg.norm(yt[:,:],axis=1) # corrected line that was not used in trv101

tt = torch.from_numpy(yen_abs)
tt = torch.reshape(tt,(-1,1))

labels = torch.zeros_like(tt,dtype=torch.long)

N_before = len(labels[labels==0])
mask = ((yen_abs>low_limit) | (f_mag>0.1)) | (t_mag>0.1)

labels[mask] = 1
N_after = len(labels[labels==1])

print(N_after/N_before)

x = x.astype(np.float32)
x = torch.from_numpy(x)
torch.save(x,'x_train_for_bool.pt')

torch.save(labels,'labels.pt')

print("DONE")











exit()
