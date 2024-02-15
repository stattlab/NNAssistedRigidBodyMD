from sys import argv
import sys
import os
import matplotlib.pyplot as plt
import argparse
import numpy as np
import gsd
import gsd.hoomd
from GetPairsUtils import GetPairsUtils
import re


"""
Takes a .gsd snapshot
Get positions and orientation coordinates of every interacting cubes and their
interaction torque and central force (on cube 1)
"""

parser = argparse.ArgumentParser(description="Cluster analysis of a gsd file for a single frame")
non_opt = parser.add_argument_group("mandatory arguments")
non_opt.add_argument('-i', '--input', metavar="<dat>", type=str, dest="input_file",
required=True, help=".gsd files " )

non_opt.add_argument('--cutoff', metavar="<float>", type=float, dest="cutoff",
required=True, help=" cutoff for DBSCAN ", default = -1 )


args = parser.parse_args()
input_file = args.input_file
cutoff = args.cutoff

in_file=gsd.fl.GSDFile(name=input_file, mode='rb',application="hoomd",schema="hoomd", schema_version=[1,0])
N_frames = in_file.nframes


frames = np.arange(N_frames)
# frames = np.arange(100)
configs = []
torfors = []
sizes = []
last_size = 0

k = 666 # Nbead per shape - this array is calculated here to save time
indexing = np.zeros(k*k,dtype=np.int32)
for jj in range(k):
    indexing[jj*k:(jj+1)*k] = np.arange(0,k*k,k,dtype=np.int32)+jj

for i_f,target_frame in enumerate(frames):
    print("******************",i_f,"***********************")
    frame_data = ExtractDataUtils(input_file,target_frame)
    frame_data.detect_pairs(cutoff)

    # energy, config = frame_data.calculate_pairwise_interactions_HPHC_cupy()
    torfor, config = frame_data.calculate_pairwise_torques(indexing)
    torfors.append(torfor)
    configs.append(config)
    sizes.append(len(torfor)+last_size)
    last_size = sizes[-1]

final_size = sizes[-1]

"""method fast"""
torfor_array2 = np.zeros((final_size,10))
configs_array2 = np.zeros((final_size,16))

torfor_array2[0:sizes[0],:] = torfors[0]
configs_array2[0:sizes[0],:] = configs[0]

for i in range(1,len(torfors)):
    torfor_array2[sizes[i-1]:sizes[i],:] = torfors[i]
    configs_array2[sizes[i-1]:sizes[i],:] = configs[i]

name_config = input_file[:-4] + '_configs.txt'
name_energ = input_file[:-4] + '_torqueforce.txt'
np.savetxt(name_energ,torfor_array2,fmt='%.5f')
np.savetxt(name_config,configs_array2,fmt='%.7f',header='COM1 (3), COM2 (3), QUAT1 (4), QUAT2 (4), Lx(1), Frame(1) -> 16 columns in total, positions are in absolute')


exit()
