import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
import sys,os

np.set_printoptions(suppress=True)

"""
July 5 2023

Label interacting and non interacting data
also arccos thing
"""

parser = argparse.ArgumentParser(description="")
non_opt = parser.add_argument_group("mandatory arguments")
non_opt.add_argument('-i', metavar="<dat>", type=str, dest="input_datas",nargs='+', required=True, help="-i x y " )

args = parser.parse_args()
input_datas = args.input_datas
low_limit = 0.05

x = np.load(input_datas[0])
print(np.max(x,axis=0))
print(np.min(x,axis=0))
yen_f = np.load(input_datas[1])
yt = np.load(input_datas[2])
yen_abs = np.abs(np.copy(yen_f[:,0]))
f_mag = np.linalg.norm(yen_f[:,1:],axis=1)
t_mag = np.linalg.norm(yt[:,1:],axis=1)

N_before = len(x)
mask = ((yen_abs>low_limit) | (f_mag>0.1)) | (t_mag>0.1)

x = x[mask]
yen_f = yen_f[mask]
yt = yt[mask]


np.save('x_train_int.npy',x)
del(x)

# yf = np.load(input_datas[2])
np.save('yf_train_int.npy',yen_f)
del(yen_f)

np.save('yt_train_int.npy',yt)
del(yt)


















exit()
