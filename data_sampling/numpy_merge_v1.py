import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import numpy as np
import re
import time
import torch

"""
April 5

Takes in numpy arrays and merges them and saves them as numpy array again
Need to specify the name in the script
"""

parser = argparse.ArgumentParser(description="")
non_opt = parser.add_argument_group("mandatory arguments")
non_opt.add_argument('-i', '--input', metavar="<dat>", type=str, dest="input_datas",nargs="+", required=True, help="-i d1_configs_reduced_filtered.npy d2_configs_reduced_filtered.npy" )
non_opt.add_argument('--savename', metavar="<dat>", type=str, dest="savename", required=True )


args = parser.parse_args()
input_datas = args.input_datas
savename = args.savename

for k in input_datas:
    print(k)

N_files = len(input_datas)

config_data = np.load(input_datas[0])
print(config_data.shape)

for i in range(1,N_files):
    new_data = np.load(input_datas[i])
    config_data = np.vstack((config_data,new_data))

config_data = config_data.astype(np.float32)
print(config_data.shape)
np.save(savename,config_data)


#
# x_data = torch.from_numpy(config_data)
#
# print("Size of x_data : ",x_data.size())
# # x_out = "x_27_d1234_training.pt"
# x_out = "x_39_d35_test.pt"
#
# torch.save(x_data, x_out)




exit()
