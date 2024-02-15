import numpy as np
from ProcessCubeUtils import *
import argparse

np.set_printoptions(suppress=True)

"""

(1) For each pair generate the 6 face points relative to COM
(2) Find the closest pair of face points for each pair of cubes
(3) (2) also determines the face vectors
(4) Rotate the system such that interacting direction of cube 1 is towards [1.0,0.0,0.0]
(5) Rotate the system such that cube 1 is well-posed in it's frame
(6) reflect the cube 2 such that it's com (or interacting face) is at y>0 z>0


(TODO) It is possible to get rid of the for loops and use cupy for significant speedup. 
"""

parser = argparse.ArgumentParser(description="Cluster analysis of a gsd file for a single frame")
non_opt = parser.add_argument_group("mandatory arguments")
non_opt.add_argument('--configs', metavar="<dat>", type=str, dest="config_file",
required=True, help="*config.txt file")

non_opt.add_argument('--torfors', metavar="<dat>", type=str, dest="torfor_file",
required=True, help="*torqueforce.txt file")

args = parser.parse_args()
config_file = args.config_file
torfor_file = args.torfor_file

data = np.loadtxt(config_file,dtype=np.float32)
torfor_data = np.loadtxt(torfor_file,dtype=np.float32)
torfor_data = torfor_data[data[:,-1]>0.01] ## This is done because calculations don't work for pairs with unit quaternions
data = data[data[:,-1]>0.01]

force_abs = np.copy(torfor_data[:,-3:])
tork_abs = np.copy(torfor_data[:,1:4])
energy_data = np.copy(torfor_data[:,0])
del(torfor_data)
N_data = len(data)
Lx = np.copy(data[0,-2])
print(N_data)
print(config_file)
print("------")

QUAT1 = data[:,6:10]
QUAT2 = data[:,10:14]
COM2 = data[:,3:6]
COM1 = data[:,0:3]


""" FIND INTERACTING FACE 1 """
true_faces1 = np.array([
[0.0,0.0,0.0],
[1.0,0.0,0.0],
[-1.0,0.0,0.0],
[0.0,1.0,0.0],
[0.0,-1.0,0.0],
[0.0,0.0,1.0],
[0.0,0.0,-1.0]
])

true_faces1 = np.tile(true_faces1,(N_data,1))
q1_faces = np.repeat(QUAT1,7,axis=0)
com1_faces = np.repeat(COM1,7,axis=0)
com2_faces = np.repeat(COM2,7,axis=0)
faces1 = rotate(q1_faces,true_faces1)
faces1_abs = faces1 + com1_faces
com2_faces1_rel = wrap_pbc(com2_faces-faces1_abs,Lx)
dist2faces1 = np.linalg.norm(com2_faces1_rel,axis=1)
dist2faces1 = dist2faces1.reshape(-1,7)
face1_index = np.argmin(dist2faces1,axis=1)


""" FIND INTERACTING FACE 2 """

true_faces2 = np.array([
[0.0,0.0,0.0],
[1.0,0.0,0.0],
[-1.0,0.0,0.0],
[0.0,1.0,0.0],
[0.0,-1.0,0.0],
[0.0,0.0,1.0],
[0.0,0.0,-1.0]
])

true_faces2 = np.tile(true_faces2,(N_data,1))
q2_faces = np.repeat(QUAT2,7,axis=0)
com1_faces = np.repeat(COM1,7,axis=0)
com2_faces = np.repeat(COM2,7,axis=0)
faces2 = rotate(q2_faces,true_faces2)
faces2_abs = faces2 + com2_faces

com1_faces2_rel = wrap_pbc(faces2_abs - com1_faces,Lx)
dist2faces2 = np.linalg.norm(com1_faces2_rel,axis=1)
dist2faces2 = dist2faces2.reshape(-1,7)
face2_index = np.argmin(dist2faces2,axis=1)

""" Rotate everything such that interacting face1 is [1.0,0.0,0.0] """
right_true = np.array([1.0,0.0,0.0])
right_true = np.tile(right_true,(N_data,1))

faces1i = faces1.reshape(-1,7,3)
faces1_inter = np.zeros((N_data,3))
for i in range(N_data):
    faces1_inter[i] = np.copy(faces1i[i,face1_index[i],:])

q_rot1 = quat_from_two_vectors(faces1_inter,right_true)
force_r1 = rotate(q_rot1,force_abs)
tork_r1 = rotate(q_rot1,tork_abs)
q_rot1 = np.repeat(q_rot1,7,axis=0)
faces1_r1 = rotate(q_rot1,faces1)
faces2_r1 = rotate(q_rot1,com1_faces2_rel)


"""
Rotate everything such that cube1 is aligned with coordinate axis
ie. rotate a face that is perp to inter face 1 such that it faces [0.0,1.0,0.0]
"""

face1p_index = (face1_index + 2)%7
face1p_index[face1p_index==0] = 1

forward_true = np.array([0.0,1.0,0.0])
forward_true = np.tile(forward_true,(N_data,1))
faces1p_inter = np.zeros((N_data,3))
faces1ri = faces1_r1.reshape(-1,7,3)

for i in range(N_data):
    faces1p_inter[i] = np.copy(faces1ri[i,face1p_index[i],:])

q_rot2 = quat_from_two_vectors(faces1p_inter,forward_true)

force_r2 = rotate(q_rot2,force_r1)
tork_r2 = rotate(q_rot2,tork_r1)

q_rot2 = np.repeat(q_rot2,7,axis=0)
faces1_r2 = rotate(q_rot2,faces1_r1)
faces2_r2 = rotate(q_rot2,faces2_r1)

"""
Reflect every interacting face 2 together with the rest of cube 2
to the first quadrant of y,z (i.e. make sure theya re both positive)

For the inter face 2 there are 4 options equally likely:
1 - y > 0 & z > 0 ==> Do nothing
2 - y < 0 & z > 0 ==> Only reflect across y axis i.e. multiply only y vals by -1
3 - y > 0 & z < 0 ==> Only reflect across z axis i.e. multiply only z vals by -1
4 - y < 0 & z < 0 ==> Reflect across both y and z axis i.e. multiply z and y vals by -1

Reflection makes sure that we keep the interaction energy the same

Don't know how to make this work with forces though
"""
faces2ri = faces2_r2.reshape(-1,7,3)
faces2_r2_com = faces2ri[:,0,:]

multiplier = np.ones_like(faces2_r2)
y_signs = np.sign(faces2_r2_com[:,1])
z_signs = np.sign(faces2_r2_com[:,2])

multiplier_force = np.ones_like(force_r2)
multiplier_force[:,1] = y_signs
multiplier_force[:,2] = z_signs
force_r2 = force_r2*multiplier_force

multiplier_tork = np.ones_like(force_r2)
multiplier_tork[:,0] = y_signs*z_signs
multiplier_tork[:,1] = z_signs
multiplier_tork[:,2] = y_signs
tork_r2 = tork_r2*multiplier_tork


y_signs = np.repeat(y_signs,7)
z_signs = np.repeat(z_signs,7)

multiplier[:,1] = y_signs
multiplier[:,2] = z_signs

faces2_r2 = faces2_r2*multiplier

faces2ri = faces2_r2.reshape(-1,7,3)


"""
ADDED FOR V2 :
If z > y for faces2_com than switch z and y

"""
faces2_r2_com = faces2ri[:,0,:]

switch_index = np.where(faces2_r2_com[:,2]>faces2_r2_com[:,1])
switch_index = switch_index[0]


for i in range(N_data):
    if(i in switch_index):
        faces2ri[i,:,[1,2]] = faces2ri[i,:,[2,1]]
        force_r2[i,[1,2]] = force_r2[i,[2,1]]
        tork_r2[i,[1,2]] = -tork_r2[i,[2,1]]
        tork_r2[i,0] = -tork_r2[i,0]

faces2_r2 = faces2ri.reshape(-1,3)
faces2_inter = np.zeros((N_data,3))
for i in range(N_data):
    faces2_inter[i] = np.copy(faces2ri[i,face2_index[i],:])

faces2_r2_com = faces2ri[:,0,:]

"""
Why I have face2_dirs with x > 0.0, that shouldn't happen
Maybe because being closest to the com1 is not the same thing as
being closest to the face inter 1

-> See 70 kup yes it can happen for some weakly interacting configs
-> For this shape (cube_v2) and interaction (GLJ-18) I've decided to neglect
pairs with face2dir.x > 0.0. This may change for other shapes or interactions.
"""

""" Get the first 2 angles of face inter 2  """
faces2_r2_com = faces2ri[:,0,:]
faces2_r2_com7 = np.repeat(faces2_r2_com,7,axis=0)

faces2_r2_relcom2 = faces2_r2 - faces2_r2_com7


faces2_inter_relcom2 = np.zeros((N_data,3))
faces2_r2_relcom2_3d = faces2_r2_relcom2.reshape(-1,7,3)
# print(faces2_r2)

for i in range(N_data):
    faces2_inter_relcom2[i] = np.copy(faces2_r2_relcom2_3d[i,face2_index[i],:])

""" Be careful we take the cosine from the opposite direction """

""" Very rarely arccos won't work because the x dimension will be larger than 1 due to numerical errors"""
faces2_inter_relcom2[faces2_inter_relcom2[:,0]<-1.0,0] = -1.0
faces2_inter_relcom2[faces2_inter_relcom2[:,0]>1.0,0] = 1.0

xcos_angle2 = np.arccos(-faces2_inter_relcom2[:,0]) # 0 ile pi arasi gelir bu
yztan_angle2 = np.arctan2(-faces2_inter_relcom2[:,2],-faces2_inter_relcom2[:,1])

"""
- Only the last angle of 2 is left
(1) Find the quat that will rotate face2_inter_relcom2 towards [-1,0,0]
(2) Apply the quat to the faces2_r2_relcom2
(3) See that opposite of the face2_inter_relcom2 is towards [1,0,0]
(4) Find the angle to rotate other faces towards z and y axis
"""

true_left = np.array([-1.0,0.0,0.0])
true_left = np.tile(true_left,(len(faces2_inter_relcom2),1))
q21 = quat_from_two_vectors(faces2_inter_relcom2,true_left)
q21 = np.repeat(q21,7,axis=0)
faces2_r2_r21_relcom2 = rotate(q21,faces2_r2_relcom2)

faces2p_inter = np.zeros((N_data,3))
faces2r21_3d = faces2_r2_r21_relcom2.reshape(-1,7,3)

shape_2_same_dir = np.zeros(N_data,dtype=np.int32)

for i in range(N_data):
    ff = np.copy(faces2r21_3d[i])
    ff = ff[1:]
    ff = ff[np.abs(ff[:,0])<0.001]
    ff = ff[ff[:,2]>=0.0]
    ff = ff[ff[:,1]>0.000]
    if(len(ff)!=1):
        print("ppl")
        print(faces2r21_3d[i])
        print(ff)
        shape_2_same_dir[i] = 1
        faces2p_inter[i] = np.array([0.0,1.0,0.0])
    else:
        faces2p_inter[i] = ff[0]


## last angle is correct
last_angle = np.arctan2(faces2p_inter[:,2],faces2p_inter[:,1])
reduced_configs = np.zeros((len(last_angle),6))
reduced_configs[:,:3] = faces2_r2_com
# reduced_configs[:,:3] = COM2_REL
reduced_configs[:,3] = xcos_angle2
reduced_configs[:,4] = yztan_angle2
reduced_configs[:,5] = last_angle

energy_force_rel = np.zeros((len(force_r2),4))
energy_force_rel[:,0] = energy_data
energy_force_rel[:,1:] = force_r2

# savename_force = torfor_file[:-4] + "_reducedforce.npy"
# savename_tork = torfor_file[:-4] + "_reducedtork.npy"
# savename_configs = config_file[:-4] + "_reduced6_v4.npy"
savename_force = torfor_file + "_reducedforce.npy"
savename_tork = torfor_file + "_reducedtork.npy"
savename_configs = config_file + "_reduced.npy"

np.save(savename_force,energy_force_rel)
np.save(savename_tork,tork_r2)
np.save(savename_configs,reduced_configs)


exit()
