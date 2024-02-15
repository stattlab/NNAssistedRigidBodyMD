import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import numpy as np
import re
import time
import torch

"""
April 10

convert x_train_int.npy to .pt
"""

parser = argparse.ArgumentParser(description="")
non_opt = parser.add_argument_group("mandatory arguments")
non_opt.add_argument('-i', '--input', metavar="<dat>", type=str, dest="input_data", required=True, help="-i d1_configs_reduced_filtered.npy d2_configs_reduced_filtered.npy" )


args = parser.parse_args()
input_data = args.input_data


data = np.load(input_data)
data = data.astype(np.float32)
pt_data = torch.from_numpy(data)


outname = input_data[:-4] + '.pt'
torch.save(pt_data, outname)




exit()
