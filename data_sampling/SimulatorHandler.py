import numpy as np
import gsd.hoomd
from hoomd.data import make_snapshot, boxdim
import os.path
import sys
import time
from sklearn.cluster import DBSCAN, OPTICS, MeanShift, estimate_bandwidth
from scipy.spatial import cKDTree as KDTree
from scipy.spatial import distance
import string
np.set_printoptions(suppress=True,threshold=sys.maxsize)

def matrix_to_axisangle(rotmax):
    r = Rot.from_dcm(rotmax)
    rv = r.as_rotvec() # vector is the axis you turn around, norm is the angle
    return rv


def check_overlap2(pts1,pts2,margin,L):
    """
    Two groups of points returns False if they are closer than margin
    in the box
    """
    pts1 = pts1 + L*0.5
    pts2 = pts2 + L*0.5
    tree1 = KDTree(data=pts1, leafsize=12, boxsize=L+0.0001)
    tree2 = KDTree(data=pts2, leafsize=12, boxsize=L+0.0001)
    n = tree2.count_neighbors(tree1, margin)
    if(n==0):
        answer = True
    else:
        answer = False
    return answer

def translate(pos,v,L):
    pos = pos + v
    pos = pbc(pos,L)
    return pos

def rotatePbc(pos,rot_max,L):
    """
    Assumes the first is the central particle ie. CofM
    """
    cm = np.copy(pos[0])
    pos = pos - cm
    pos = pbc(pos,L)
    pos = np.matmul(rot_max,np.transpose(pos))
    pos = np.transpose(pos)
    pos = pos + cm
    pos = pbc(pos,L)
    return pos

def random_quat():
    """
    Not sure the first or last term is the scalar
    probably last
    """
    rands = np.random.uniform(size=3)
    quat = np.array([np.sqrt(1.0-rands[0])*np.sin(2*np.pi*rands[1]),
            np.sqrt(1.0-rands[0])*np.cos(2*np.pi*rands[1]),
            np.sqrt(rands[0])*np.sin(2*np.pi*rands[2]),
            np.sqrt(rands[0])*np.cos(2*np.pi*rands[2])])
    return quat

def quat_to_matrix(quat):
    """
    Convert a quaternion (assuming last term is scalar)
    to a rotation matrix that rotates columns when multiplies
    from left.
    https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Euler_rotations
    """
    i = quat[0]
    j = quat[1]
    k = quat[2]
    r = quat[3]
    matrix = np.zeros((3,3))
    matrix[0,0] = -1.0 + 2.0*i*i + 2.0*r*r
    matrix[0,1] = 2.0*(i*j-k*r)
    matrix[0,2] = 2.0*(i*k+j*r)
    matrix[1,0] = 2.0*(i*j+k*r)
    matrix[1,1] = -1.0 + 2.0*j*j + 2.0*r*r
    matrix[1,2] = 2.0*(j*k-i*r)
    matrix[2,0] = 2.0*(i*k-j*r)
    matrix[2,1] = 2.0*(j*k+i*r)
    matrix[2,2] = -1.0 + 2.0*k*k + 2.0*r*r
    return matrix

def quat_converter(quat):
    """
    HOOMD : scalar is first (ie default is 1,0,0,0)
    MCSim : scalar is last (ie default is 0,0,0,1)
    Converts mc_sim quats -> hoomd quats
    ovito also uses mc_sim quats
    """
    a,b,c,d = quat
    quat_new = np.array([d,a,b,c])
    return quat_new

def pbc(pos,L):
    pos = pos - ((pos - L*0.5)//L + 1)*L
    return pos

def rotate_rows(matrix,pos):
    """
    takes a position matrix (N,3)-> rows are positions
    and a rotation matrix to rotate the positions
    - simple r_mat *matmul* pos_mat rotates the columns only
    which is not what we want. It is possible to take
    transpose of row-wise position vector do the matmul
    and than retranspose it again to row wise to get the
    same result.
    """
    pos = np.matmul(matrix,np.transpose(pos))
    return np.transpose(pos)


def check_overlap(p,pts,margin,L):
    """
    Returns true if there is no overlap btw a point p and a  group of points pts
    Useful for placing noms
    """
    pts = np.array(pts)
    dists_v = pts - p
    dists_v = pbc(dists_v,L)
    dists = np.linalg.norm(dists_v,axis=1)
    min_dist = np.min(dists)
    if(min_dist>margin):
        answer = True
    else:
        answer = False
    return answer


def matrix_to_quat(m):
    """
    take a rotation matrix and return corresponding quat
    onenote 6.0
    !!! Different Quat Notation !!
    Unit quat is = [1,0,0,0]
    (before it was [0,0,0,1])
    This one  fails if the one of the euler angles are too close to
    zero(Not so sure about that) (quat returns NaN since inside the sqrt becomes
    negative) Scipy handles it better
    """
    q = np.zeros(4)
    q[0] = 0.5*np.sqrt(1.0+m[0,0]+m[1,1]+m[2,2])
    q[1] = (1.0/(4.0*q[0]))*(m[2,1]-m[1,2])
    q[2] = (1.0/(4.0*q[0]))*(m[0,2]-m[2,0])
    q[3] = (1.0/(4.0*q[0]))*(m[1,0]-m[0,1])
    return q

def matrix_to_quat2(m):
    """
    Try with scipy
    """
    r = Rot.from_dcm(m)
    q = r.as_quat()
    return q


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

def getMOI(pos):
    """
    Copied from : https://scipython.com/book/chapter-6-numpy/problems/p65/the-moment-of-inertia-tensor/
    Inertia products ie. non-diagonal terms can be negative
    """
    masses = np.ones(len(pos))
    x, y, z = pos.T
    Ixx = np.sum(masses * (y**2 + z**2))
    Iyy = np.sum(masses * (x**2 + z**2))
    Izz = np.sum(masses * (x**2 + y**2))
    Ixy = -np.sum(masses * x * y)
    Iyz = -np.sum(masses * y * z)
    Ixz = -np.sum(masses * x * z)
    I = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
    # print(I)
    # exit()
    return I

class Shape():
    """
    Similar to MoleculeTemplate class in mcsim code.
    Any necessary info about the shape that is independent of the actual simulation box

    positions include the central particle at first index
    typeids are always 0 for central 1 for the constituents
    """
    def __init__(self,filename):
        self.positions = []
        self.typeids = []
        self.filename = filename
        self.read_shape(filename,0)


    def set_positions(self):
        """
        centers the shape to origin
        Adds the central particle
        """
        n_type = len(self.types) - 1
        alphabet_string = string.ascii_uppercase
        alphabet_list = list(alphabet_string)
        self.types = alphabet_list[:n_type]

        self.positions = self.positions - np.average(self.positions,axis=0)
        new_pos = np.zeros((len(self.positions)+1,3))
        new_pos[1:,:] = self.positions
        self.positions = new_pos
        self.typeid = np.ones(len(self.positions),dtype=int)
        self.typeid[0] = 0

        self.confirmMOI()

    def confirmMOI(self):
        """
        Not only confirms moi but sets it and sets masses etc.
        I don't rotate my shapes to diagonalize my MOI anymore. That should
        be done when the shape is made. Here I just make sure that it is diagonal.
        """
        matrix_moi = getMOI(self.positions[1:])

        ### Check ###
        sum_all = np.sum(np.abs(matrix_moi))
        sum_diag = np.abs(matrix_moi[0,0]) + np.abs(matrix_moi[1,1]) + np.abs(matrix_moi[2,2])
        if((sum_all - sum_diag) > 0.001 ):
            print("Error 223-OP")
            exit()

        moi = np.diag(matrix_moi)
        N_beads = len(self.positions)
        self.mass = np.ones(len(self.positions),dtype=float)/N_beads
        self.mass[0] = 1.0
        self.MOI = np.zeros_like(self.positions)
        self.MOI[0,:] = np.copy(moi)/N_beads
        print("moi, ",moi/N_beads)
        self.body = np.zeros_like(self.typeid,dtype=int)
        a = np.array([1.0,0.0,0.0,0.0])
        a = np.tile(a,(len(self.positions),1))
        self.orientation = a


    def getMOI2(self):
        return np.copy(self.diagonalMOI)

    def getPositions(self):
        return np.copy(self.positions)
    def getTypeids(self):
        return np.copy(self.typeid)
    def getmass(self):
        return np.copy(self.mass)
    def getbody(self):
        return np.copy(self.body)
    def getMOI(self):
        return np.copy(self.MOI)
    def getorientation(self):
        return np.copy(self.orientation)

    def read_shape(self,input,target_frame):
        try:
            with gsd.hoomd.open(name=input, mode='rb') as f:
                if (target_frame==-1):
                    frame = f.read_frame(len(f)-1)
                    self.frame = len(f)-1
                    print("Reading last frame ")
                else:
                    self.frame = target_frame
                    frame = f.read_frame(target_frame)
                    print("Reading frame ", target_frame)
                self.positions = (frame.particles.position).copy()
                self.types = (frame.particles.types).copy()
                self.typeid = (frame.particles.typeid).copy()

        except:
            self.positions = (input.particles.position).copy()
            self.types = (input.particles.types).copy()
            self.typeid = (input.particles.typeid).copy()


class SimulatorHandler():
    def __init__(self,input):
        self.shape_filename = input
        self.setShape()

    def setShape(self):
        """
        Create the shape class
        """
        self.shape = Shape(self.shape_filename)
        self.shape.set_positions()

    def setSeed(self,seed):
        np.random.seed(seed)

    def setBox(self,Box_size,N_mp):
        self.Lx = Box_size
        self.Ly = Box_size
        self.Lz = Box_size
        self.N_mp = N_mp
        self.placeMps()


    def placeMps(self):
        """
        First MP goes to origin without rotation,
        Than set the rigid body stuff since this is how it was done in rras_v6
        the rest is added using AddShape
        function from p_shear
        """
        self.positions = self.shape.getPositions()
        self.typeid = self.shape.getTypeids()
        self.types = ['Re','Os']
        self.velocities = np.zeros_like(self.positions)
        self.angmom = np.zeros((len(self.positions),4))
        self.setRigidBody()
        print("Placing Shapes ...")
        for i in range(self.N_mp-1):
            # print("%d/%d" %(i,self.N_mp-1))
            self.addShape()

    def setRigidBody(self):
        """
        Sets rigid body related things like mass, body, inertia etc.
        """
        mass_shape = self.shape.getmass()
        body_shape = self.shape.getbody()
        inertia_shape = self.shape.getMOI()
        orientation_shape = self.shape.getorientation()
        # print(orientation_shape)
        # exit()
        # mass_noms = np.ones(self.NomPerShape)
        # body_noms = np.ones(self.NomPerShape)*-1.0

        # inertia_noms = np.zeros((self.NomPerShape,3))
        # orientation_noms = np.tile([1.0,0.0,0.0,0.0],(self.NomPerShape,1))

        self.mass = mass_shape
        self.body = body_shape
        self.moment_inertia = inertia_shape
        self.orientation = orientation_shape


    def getMOIShape(self):
        return self.shape.getMOI2()

    def addShape(self):
        """
        add the new shape away from the previous shapes
        Central particles must have a body tag identical to their contiguous tag.
        """

        t = 0
        current_body = len(self.positions)
        pos_new_shape = self.shape.getPositions()
        typeid_new_shape = self.shape.getTypeids()
        velocity_new_shape = np.zeros_like(pos_new_shape)
        angmom_new_shape = np.zeros((len(pos_new_shape),4))
        pos_old = self.positions
        good = False
        margin = 1.0
        while(good is False):
            pos_new_shape = self.shape.getPositions()
            t += 1
            p = (np.random.rand(3) - 0.5)*self.Lx ### 0.70 to make encounter earlier see 28.1
            # p = np.array([3.0,3.0,3.0]) ## when debugging with 2 shapes
            ### turn these on for old style - this is for approach2
            # quat = random_quat()
            # rot_max = quat_to_matrix(quat)
            # quat_hoomd = quat_converter(quat)
            pos_new_shape = translate(pos_new_shape,p,self.Lx)
            # pos_new_shape = rotatePbc(pos_new_shape,rot_max,self.Lx)
            good = check_overlap2(pos_new_shape,pos_old,margin,self.Lx)
            if(t>100000):
                print("AddShape tried too much!!!")
                exit()

        self.positions = np.vstack((self.positions,pos_new_shape))
        self.typeid = np.hstack((self.typeid,typeid_new_shape))
        self.velocities = np.vstack((self.velocities,velocity_new_shape))
        self.angmom = np.vstack((self.angmom,angmom_new_shape))


        mass_shape = self.shape.getmass()
        body_shape = np.ones_like(mass_shape)*current_body
        inertia_shape = self.shape.getMOI()
        orientation_shape = self.shape.getorientation()
        # orientation_shape[0,:] = quat_hoomd
        # print(quat_hoomd)
        self.mass = np.hstack((self.mass,mass_shape))
        self.body = np.hstack((self.body,body_shape))
        self.moment_inertia = np.vstack((self.moment_inertia,inertia_shape))
        self.orientation = np.vstack((self.orientation,orientation_shape))



    def getRigidConstituents(self):
        """
        Return the shape template positions except the central
        """
        pos = self.shape.getPositions()
        pos = pos[1:]
        return pos

    def get_snap(self,context):
        with context:


            snap = make_snapshot(N=len(self.positions),
                                particle_types=self.types,
                                # bond_types=self.bond_types,
                                box=boxdim(Lx=self.Lx,Ly=self.Ly,Lz=self.Lz))


            for k in range(len(self.positions)):
                snap.particles.position[k] = self.positions[k]
                snap.particles.typeid[k] = self.typeid[k]
                snap.particles.body[k] = self.body[k]
                snap.particles.moment_inertia[k] = self.moment_inertia[k]
                snap.particles.orientation[k] = self.orientation[k]
                snap.particles.mass[k] = self.mass[k]
                snap.particles.velocity[k] = self.velocities[k]
                snap.particles.angmom[k] = self.angmom[k]


        return snap
