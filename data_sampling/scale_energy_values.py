import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
import sys,os

np.set_printoptions(suppress=True,precision=4)

"""
April 10

2 numpy arrays as input
yf_train_int
yt_train_int

you should have an idea on how the data looks like by using
fortor_data_distro_analysis.py before hand

Here just type margins or values to cap
Cap the 6 columns of data coming from the two files
Min Max Scale (0.0,1.0)
Convert to torch
"""

parser = argparse.ArgumentParser(description="")
non_opt = parser.add_argument_group("mandatory arguments")
non_opt.add_argument('-i', metavar="<dat>", type=str, dest="input_datas",nargs='+', required=True, help="-i x y " )

args = parser.parse_args()
input_datas = args.input_datas

margin = 0.01
forces = np.load(input_datas[0])
x = np.load(input_datas[1])
f_scaled = np.zeros_like(forces)
N = len(forces)
ind1 = int(N*margin)
ind2 = int(N*(1-margin))

en = forces[:,0]
# ind = np.argsort(en)
# en = en[ind]
# x = x[ind]
# print(np.min(en,axis=0))
# print(np.max(en,axis=0))
# print(en[-10:])
# print(x[-10:])
#
min = -5.1
max = 17.0
en = (en -min) / (max - min)
en[en>1.0] = 1.0
en[en<0.0] = 0.0

f_name = input_datas[0][:-4] + '_mm01.pt'
en = torch.from_numpy(en)
torch.save(en, f_name)

exit()
""" energy and forces """
# mins = np.array([-4.2,-26.00,-9.75,-4.0,-3.25,-11.25,-13.25]) # v3
# maxs = np.array([12.50,9.3,2.25,1.25,3.25,11.25,13.25])
mins = np.array([-4.2,-26.00,-9.75,-4.0,-1.50,-13.80,-6.00]) # v4 with very low kT (0.2 - 0.5)
maxs = np.array([12.50,14.00,2.25,1.25,4.5,9.00,17.50])

# fx = np.sort(forces[:,0])
# fx_min = fx[ind1]
# fx_max = fx[ind2]
# print(fx_min,fx_max)
f_scaled[:,0] = (forces[:,0] - mins[0]) / (maxs[0] - mins[0])

# fy = np.sort(forces[:,1])
# fy_min = fy[ind1]
# fy_max = fy[ind2]
# print(fy_min,fy_max)
f_scaled[:,1] = (forces[:,1] - mins[1]) / (maxs[1] - mins[1])

# fz = np.sort(forces[:,2])
# fz_min = fz[ind1]
# fz_max = fz[ind2]
# print(fz_min,fz_max)
f_scaled[:,2] = (forces[:,2] - mins[2]) / (maxs[2] - mins[2])

# fz = np.sort(forces[:,3])
# fz_min = fz[ind1]
# fz_max = fz[ind2]
# print(fz_min,fz_max)
# exit()
f_scaled[:,3] = (forces[:,3] - mins[3]) / (maxs[3] - mins[3])

f_scaled[f_scaled>1.0] = 1.0
f_scaled[f_scaled<0.0] = 0.0

f_name = input_datas[0][:-4] + '_mm01.pt'
f_scaled = torch.from_numpy(f_scaled)
torch.save(f_scaled, f_name)

del(forces,f_scaled)


""" torks """
torks = np.load(input_datas[1])
f_scaled = np.zeros_like(torks)
# fx = np.sort(torks[:,0])
# fx_min = fx[ind1]
# fx_max = fx[ind2]
# print(fx_min,fx_max)
f_scaled[:,0] = (torks[:,0] - mins[4]) / (maxs[4] - mins[4])

# fy = np.sort(torks[:,1])
# fy_min = fy[ind1]
# fy_max = fy[ind2]
# print(fy_min,fy_max)
f_scaled[:,1] = (torks[:,1] - mins[5]) / (maxs[5] - mins[5])

# fz = np.sort(torks[:,2])
# fz_min = fz[ind1]
# fz_max = fz[ind2]
# print(fz_min,fz_max)
# exit()
f_scaled[:,2] = (torks[:,2] - mins[6]) / (maxs[6] - mins[6])

f_scaled[f_scaled>1.0] = 1.0
f_scaled[f_scaled<0.0] = 0.0

f_name = input_datas[1][:-4] + '_mm01.pt'
f_scaled = torch.from_numpy(f_scaled)
torch.save(f_scaled, f_name)

del(torks,f_scaled)

exit()
