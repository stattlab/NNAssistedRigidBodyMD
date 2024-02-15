import numpy as np
import gsd.hoomd
from hoomd.data import make_snapshot, boxdim
import os.path
from scipy.spatial import cKDTree as KDTree
import matplotlib.pyplot as plt
from scipy.spatial import distance
from textwrap import wrap
import cupy as cp
np.set_printoptions(suppress=True,precision=5,linewidth=150)


def in_deg(x):
    return (180.0*x)/np.pi

def wrap_pbc(x, Box):
    delta = np.where(x > 0.5 * Box, x- Box, x)
    delta = np.where(delta <- 0.5 * Box, Box + delta, delta)
    return delta


def com(a,Box):
    theta = np.divide(a + 0.5 * Box, Box)*np.multiply(2,np.pi)
    xi_average = np.average(np.cos(theta), axis = 0)
    zeta_average = np.average(np.sin(theta), axis = 0)
    theta_average = np.arctan2(-zeta_average,-xi_average) + np.pi
    com = np.multiply(Box,theta_average)/np.multiply(2,np.pi)-0.5 * Box
    return com

def HalfPowerHalfCosine(r,pwr,A,b,c):
    cutoff_right = (-np.pi+b*c)/c + np.pi*2/c
    pot = -A*np.cos(c*r-b*c)-A
    pot[r<b] = 0.0
    pot2 = np.power(b-r,pwr) - A*2.0
    pot2[r>b] = 0.0
    pot = pot + pot2
    pot[r>cutoff_right] = 0.0
    return pot


class GetPairsUtils():

    def __init__(self,input,frame):
        self.read_system(input,frame)
        self.calculate_beads_per_shape()
        self.noise_id = 0


    def read_system(self,input,target_frame):
        """
        Read in a snapshot from a gsd file or snapshot.
        """
        self.target_frame = target_frame
        self.frame = 0
        try:
            with gsd.hoomd.open(name=input, mode='rb') as f:
                if (target_frame==-1):
                    frame = f.read_frame(len(f)-1)
                    self.frame = len(f)-1
                else:
                    self.frame = target_frame
                    frame = f.read_frame(target_frame)
                self.positions = (frame.particles.position).copy()
                self.velocities = (frame.particles.velocity).copy()
                self.bodi = (frame.particles.body).copy()
                self.moment_inertia = (frame.particles.moment_inertia).copy()
                self.orientations = (frame.particles.orientation).copy()
                self.mass = (frame.particles.mass).copy()
                self.angmom = (frame.particles.angmom).copy()

                self.types = (frame.particles.types).copy()
                self.typeid = (frame.particles.typeid).copy()

                self.Lx,self.Ly,self.Lz = frame.configuration.box[0:3]
                self.box = frame.configuration.box

        except:
            self.positions = (input.particles.position).copy()
            self.velocities = (input.particles.velocity).copy()
            self.types = (input.particles.types).copy()
            self.typeid = (input.particles.typeid).copy()
            self.bodi = (input.particles.body).copy()
            self.moment_inertia = (input.particles.moment_inertia).copy()
            self.orientations = (input.particles.orientation).copy()
            self.mass = (input.particles.mass).copy()
            self.angmom = (input.particles.angmom).copy()

            self.Lx = input.box.Lx
            self.Ly = input.box.Lx
            self.Lz = input.box.Lx

        """
        snapshot.particles.body is broken, non body particles should have -1
        but the container I think only holds unsigned ints so -1 defaults to
        4294967295 which is very inconvenient for the rest oof the class
        so here I swithc it back to -1
        """
        self.body = np.array(self.bodi,dtype=int)
        self.body[self.body>9999999]= -1.0


    def calculate_beads_per_shape(self):
        """
        Assumes every shape is the same
        There are shape beads only in the shape
        The constituent bead typeid is 1, central particle typeid is 0
        """
        self.Nbead_per_shape = len(self.positions[self.typeid==1])//len(self.positions[self.typeid==0]) + 1
        self.N_shape = len(self.positions[self.typeid==0])

    def detect_pairs(self,cutoff):
        """
        For pairs use rigid body centers, if they're closer than a given distance
        than we have a pair
        For each shape :
        Index 0 is the rigid center
        Index 1 is the top plate center
        so the first two points are enough to define all of them

        """
        self.center_distance = cutoff ### depends on the shape should be an input
        ## 5.6 for the silindir_01.gsd

        pos_tree = self.positions[self.typeid==0] + self.Lx*0.5
        tree = KDTree(data=pos_tree, leafsize=12, boxsize=self.Lx+0.0001)
        pairs = tree.query_pairs(r=self.center_distance)
        self.pairs_body = np.zeros((len(pairs),2),dtype=np.int32)
        self.Npairs = len(self.pairs_body)
        pair1_body = self.pairs_body[:,0]
        self.pair1_pos = np.zeros((self.Npairs,self.Nbead_per_shape,3),dtype=np.float32)
        self.pair2_pos = np.zeros((self.Npairs,self.Nbead_per_shape,3),dtype=np.float32)

        self.pair1_ori = np.zeros((self.Npairs,self.Nbead_per_shape,4),dtype=np.float32)
        self.pair2_ori = np.zeros((self.Npairs,self.Nbead_per_shape,4),dtype=np.float32)

        self.pairs_body = np.zeros((len(pairs),2),dtype=np.int32)
        for i,pair in enumerate(pairs):
            # self.pairs_body[i,0] = pair[0]*self.Nbead_per_shape
            # self.pairs_body[i,1] = pair[1]*self.Nbead_per_shape
            self.pair1_pos[i] = self.positions[self.body==pair[0]*self.Nbead_per_shape]
            self.pair2_pos[i] = self.positions[self.body==pair[1]*self.Nbead_per_shape]

            self.pair1_ori[i] = self.orientations[self.body==pair[0]*self.Nbead_per_shape]
            self.pair2_ori[i] = self.orientations[self.body==pair[1]*self.Nbead_per_shape]





    def calculate_pairwise_torques(self,indexing):
        """
        Output data :
        COM1 (3), COM2 (3), QUAT1 (4), QUAT2 (4), Lx(1), Frame(1) -> 16 columns in total
        """

        """pair configurations (no need for cupy)"""
        self.pair_configurations = np.zeros((len(self.pairs_body),16))
        self.pair_configurations[:,15] = self.target_frame
        self.pair_configurations[:,14] = self.Lx
        self.pair_configurations[:,:3] = self.pair1_pos[:,0,:]
        self.pair_configurations[:,3:6] = self.pair2_pos[:,0,:]
        self.pair_configurations[:,6:10] = self.pair1_ori[:,0,:]
        self.pair_configurations[:,10:14] = self.pair2_ori[:,0,:]

        """
        cupy implementation

        pair1_pos :
        p00 p01 p02
        p00 p01 p02
        ...

        pair2_pos :
        p01 p02 p03
        p11 p12 p13
        ...

        """
        dev1 = cp.cuda.Device(1)
        dev1.use()
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()

        self.pair1_pos = cp.asarray(self.pair1_pos)
        self.pair2_pos = cp.asarray(self.pair2_pos)
        self.pair2_pos_ = cp.tile(self.pair2_pos,(1,self.Nbead_per_shape,1))
        self.pair1_pos_ = cp.repeat(self.pair1_pos,self.Nbead_per_shape,axis=1)

        delta = self.pair1_pos_ - self.pair2_pos_
        self.pair1_pos_ = 0.0
        self.pair2_pos_ = 0.0
        Box = self.Lx
        mempool.free_all_blocks()
        delta = cp.where(delta > 0.5 * Box, delta- Box, delta)
        delta = cp.where(delta <- 0.5 * Box, Box + delta, delta)
        dists = cp.linalg.norm(delta,axis=2)

        """
        Energy Calculation
        """
        pw = 2.5
        A = 0.00035
        b = 1.46375
        c = 3.3833
        bc = b*c
        depth = 2.0*A
        cutoff_right = (-np.pi+bc)/c + np.pi*2/c
        cutoff_left = b
        pot = -A*cp.cos(c*dists-bc)-A
        pot[dists<cutoff_left] = 0.0
        pot2 = cp.power(-dists+cutoff_left,pw) - depth
        pot2[dists>cutoff_left] = 0.0
        pot = pot + pot2
        pot[dists>cutoff_right] = 0.0
        pair_energies = cp.sum(pot,axis=1).get()
        pair_energies = np.expand_dims(pair_energies,axis=1)
        mempool.free_all_blocks()
        del(pot,pot2)
        mempool.free_all_blocks()

        """
        Force Calculation

        cutoff_right = (-np.pi+b*c)/c + np.pi*2/c
        force = -c*A*np.sin(c*(r-b))
        force[r<b] = 0.0
        force2 = pwr*np.power(b-r,pwr-1.0)
        force2[r>b] = 0.0
        force = force + force2
        force[r>cutoff_right] = 0.0
        """
        cutoff_right = (-np.pi+bc)/c + np.pi*2/c
        cutoff_left = b
        force = -c*A*cp.sin(c*dists-bc)
        force[dists<cutoff_left] = 0.0
        force2 = pw*cp.power(-dists+cutoff_left,pw-1.0)
        force2[dists>cutoff_left] = 0.0
        force = force + force2
        force[dists>cutoff_right] = 0.0
        Np,Ncomb = force.shape
        force = force.reshape(Np,Ncomb,1)
        dists = dists.reshape(Np,Ncomb,1)
        delta = cp.multiply(delta,force)
        delta = cp.divide(delta,dists)
        del(force,dists)
        mempool.free_all_blocks()
        """ on 1 """
        delta_on_1 = delta.reshape(Np,self.Nbead_per_shape,self.Nbead_per_shape,3)
        net_force_vectors_on_1 = cp.sum(delta_on_1,axis=2)

        """
        on 2
        indexing is done in the main script
        """
        # k = self.Nbead_per_shape
        # indexing = np.zeros(k*k,dtype=np.int32)
        # for jj in range(k):
        #     indexing[jj*k:(jj+1)*k] = np.arange(0,k*k,k,dtype=np.int32)+jj
        delta = -delta[:,indexing,:]
        delta_on_2 = delta.reshape(Np,self.Nbead_per_shape,self.Nbead_per_shape,3)
        net_force_vectors_on_2 = cp.sum(delta_on_2,axis=2)

        """
        Translation force is simply the sum of all forces on every bead
        Torque is the sum of the cross products (rxf) on all beads
        """

        total_force_vectors_on_1 = cp.sum(net_force_vectors_on_1,axis=1).get()
        total_force_vectors_on_2 = cp.sum(net_force_vectors_on_2,axis=1).get()

        self.pair1_pos_COM = cp.asarray(self.pair1_pos)
        self.pos_COMs = cp.copy(self.pair1_pos_COM[:,0,:])
        self.pos_COMs = cp.expand_dims(self.pos_COMs,axis=1)
        self.pos_COMs = cp.repeat(self.pos_COMs,self.Nbead_per_shape,axis=1)
        self.pair1_pos_COM =  self.pair1_pos_COM - self.pos_COMs
        self.pair1_pos_COM = cp.where(self.pair1_pos_COM > 0.5 * Box, self.pair1_pos_COM- Box, self.pair1_pos_COM)
        self.pair1_pos_COM = cp.where(self.pair1_pos_COM <- 0.5 * Box, Box + self.pair1_pos_COM, self.pair1_pos_COM)

        self.pair2_pos_COM = cp.asarray(self.pair2_pos)
        self.pos2_COMs = cp.copy(self.pair2_pos_COM[:,0,:])
        self.pos2_COMs = cp.expand_dims(self.pos2_COMs,axis=1)
        self.pos2_COMs = cp.repeat(self.pos2_COMs,self.Nbead_per_shape,axis=1)
        self.pair2_pos_COM =  self.pair2_pos_COM - self.pos2_COMs
        self.pair2_pos_COM = cp.where(self.pair2_pos_COM > 0.5 * Box, self.pair2_pos_COM- Box, self.pair2_pos_COM)
        self.pair2_pos_COM = cp.where(self.pair2_pos_COM <- 0.5 * Box, Box + self.pair2_pos_COM, self.pair2_pos_COM)

        ### take the cross product to find the torque vectors
        net_force_vectors_on_1 = net_force_vectors_on_1.reshape(-1,3)
        self.pair1_pos_COM = self.pair1_pos_COM.reshape(-1,3)
        torques_on_1 = cp.cross(self.pair1_pos_COM,net_force_vectors_on_1)
        torques_on_1 = torques_on_1.reshape(self.Npairs,-1,3)
        total_torque_on_1 = cp.sum(torques_on_1,axis=1).get()

        net_force_vectors_on_2 = net_force_vectors_on_2.reshape(-1,3)
        self.pair2_pos_COM = self.pair2_pos_COM.reshape(-1,3)
        torques_on_2 = cp.cross(self.pair2_pos_COM,net_force_vectors_on_2)
        torques_on_2 = torques_on_2.reshape(self.Npairs,-1,3)
        total_torque_on_2 = cp.sum(torques_on_2,axis=1).get()

        force_torque_info = np.hstack((pair_energies,total_torque_on_1,total_torque_on_2,total_force_vectors_on_1))

        return force_torque_info,self.pair_configurations




    def dump_snap(self,name):
        snap = gsd.hoomd.Snapshot()
        snap.configuration.step = 0
        snap.configuration.box = [self.Lx, self.Ly, self.Lz, 0, 0, 0]

        # particles
        snap.particles.N = len(self.positions)
        snap.particles.position = self.positions[:]
        snap.particles.velocity  = self.velocities[:]
        snap.particles.body = self.body[:]
        snap.particles.types  = self.types[:]
        snap.particles.typeid  = self.typeid[:]
        snap.particles.moment_inertia = self.moment_inertia[:]
        snap.particles.orientation = self.orientations[:]
        snap.particles.mass = self.mass[:]
        snap.particles.angmom = self.angmom[:]

        with gsd.hoomd.open(name=name, mode='wb') as f:
            f.append(snap)
