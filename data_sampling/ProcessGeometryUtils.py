import numpy as np
import gsd.hoomd
import os.path
import time
from scipy.spatial import cKDTree as KDTree
import matplotlib.pyplot as plt
from scipy.spatial import distance
from textwrap import wrap
import cupy as cp
np.set_printoptions(suppress=True,precision=5,linewidth=150)




def quat_from_axis_angle(axis,angle):
    """
    q.w == cos(angle / 2)
    q.x == sin(angle / 2) * axis.x
    q.y == sin(angle / 2) * axis.y
    q.z == sin(angle / 2) * axis.z

    """
    q = np.zeros((len(angle),4))
    q[:,0] = np.cos(angle*0.5)
    q[:,1] = np.sin(angle*0.5)*axis[:,0]
    q[:,2] = np.sin(angle*0.5)*axis[:,1]
    q[:,3] = np.sin(angle*0.5)*axis[:,2]
    return q

def quat_around_z(teta):
    """
    quaternion that rotates around z axis by teta
    q.w == cos(angle / 2)
    q.x == sin(angle / 2) * axis.x
    q.y == sin(angle / 2) * axis.y
    q.z == sin(angle / 2) * axis.z

    """
    q = np.zeros((len(teta),4))
    q[:,0] = np.cos(teta*0.5)
    q[:,3] = np.sin(teta*0.5)
    return q


def rotate_180_around_axis(axis):
    """
    axis must be unit vector
    Rotate the quaternion that will rotate 180 around the axis given
    Used at the beginning

    q.w == cos(angle / 2)
    q.x == sin(angle / 2) * axis.x
    q.y == sin(angle / 2) * axis.y
    q.z == sin(angle / 2) * axis.z

    with sin 90 = 1 and cos 90 = 0
    """
    q = np.zeros((len(axis),4))
    q[:,1:] = axis
    return q


def quat_from_two_vectors(v_orig,v):
    """
    My vectors are normalized

    v_original is the vector that happens when orientation quaternion is [1.0,0.0,0.0,0.0]
    https://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another
    Quaternion q;
    rotation from v1 to v2
    vector a = crossproduct(v1, v2);
    q.xyz = a;
    q.w = sqrt((v1.Length ^ 2) * (v2.Length ^ 2)) + dotproduct(v1, v2);

    Don't forget to normalize it
    """

    vect = np.cross(v_orig,v)
    scalar = 1.0 + np.sum(v_orig*v,axis=1)
    quat = np.zeros((len(vect),4))
    quat[:,0] = scalar
    quat[:,1:] = vect
    quat = renormalize_quat(quat)
    return quat

def renormalize_quat(q):
    """
    At the end of the first step of the rotation integration
    Good for stability
    q = q*(Scalar(1.0)/slow::sqrt(norm2(q)));

    template < class Real >
    DEVICE inline Real norm2(const quat<Real>& a)
    {
    return (a.s*a.s + dot(a.v,a.v));
    }

    """
    #  q_norm = np.sum(q*q,axis=1) was wrong in the previous version
    q_norm = np.sqrt(np.sum(q*q,axis=1))
    q = q/q_norm.reshape(-1,1)
    return q



def rotate(q,v):
    """
    rotate vector v by quat t, same as reconstruct_top_pos_from_orientations
    """
    coef1 = q[:,0]*q[:,0] - np.sum(q[:,1:]*q[:,1:],axis=1)
    term1 = coef1.reshape(-1,1)*v

    term2 = 2.0*q[:,0].reshape(-1,1)*np.cross(q[:,1:],v)
    term3 = 2.0*np.sum(q[:,1:]*v,axis=1).reshape(-1,1)*q[:,1:]

    res = term1 + term2 + term3

    return res

def quaternion_division(q1_relativize,q2_absolute):
    """
    Divide q1 by q2, the result should be quaternion_1 that is transformed into
    the frame of quaternion2. (This function is called from relativize_forces_v2quat.py
    In that script q2 and q1 is switched relative to this one)

    We actually don't know if quaternion division is what I think it is.
    Source1 : http://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/notations/scalarAndVector/index.htm

    (sa,va) / (sb,vb) = (sa*sb+va•vb,-va × vb - sa*vb + sb*va)

    Quaternion division is essentially multiplication by inverse
    For unit quaternions the inverse of a quaternion is also the conjugate i.e. q^-1 = conj(q)
    First index is scalar for my quaternions

    Can chcek that this works correctly by using quat conjugation and multiplication
    """
    a = q1_relativize
    b = q2_absolute

    scalar = a[:,0]*b[:,0] + np.sum(a[:,1:]*b[:,1:],axis=1)
    vector = np.cross(-a[:,1:],b[:,1:]) - a[:,0].reshape(-1,1)*b[:,1:] + b[:,0].reshape(-1,1)*a[:,1:]
    q_res = np.zeros_like(a)
    q_res[:,0] = scalar
    q_res[:,1:] = vector
    return q_res

def conjugate_quat(q):
    q = np.copy(-q)
    q[:,0] = np.copy(-q[:,0])
    return q

def quaternion_multiplication(a,b):
    """
    Be Careful - NOT COMMUTATIVE - q1*q2!=q2*q1

    Source : HOOMD - VectorMath.h
    Scalar : a.s*b.s - dot(a.v, b.v)
    Vector : a.s*b.v + b.s * a.v + cross(a.v,b.v)
    """
    scalar = a[:,0]*b[:,0] - np.sum(a[:,1:]*b[:,1:],axis=1)
    vector = a[:,0].reshape(-1,1)*b[:,1:] + b[:,0].reshape(-1,1)*a[:,1:] + np.cross(a[:,1:],b[:,1:])
    q_res = np.zeros_like(a)
    q_res[:,0] = scalar
    q_res[:,1:] = vector
    return q_res


def rotate_torks_to_body_frame(q,t):
    """
    HOOMD gets the torks in box frame and converts (rotate) them into body frame
    with t = rotate(conj(q),t); where t is tork vector and q is orientation quat

    conj : return quat<Real>(a.s, -a.v);
    rotate: (a quat, b vector)
    return (a.s*a.s - dot(a.v, a.v)) *b + 2*a.s*cross(a.v,b) + 2*dot(a.v,b)*a.v;
    """
    q = np.copy(-q)
    q[:,0] = np.copy(-q[:,0])

    coef1 = q[:,0]*q[:,0] - np.sum(q[:,1:]*q[:,1:],axis=1)
    term1 = coef1.reshape(-1,1)*t

    term2 = 2.0*q[:,0].reshape(-1,1)*np.cross(q[:,1:],t)
    term3 = 2.0*np.sum(q[:,1:]*t,axis=1).reshape(-1,1)*q[:,1:]

    res = term1 + term2 + term3

    return res


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




def rotZ(teta):
    """returns a rotation matrix that rotates around z axis"""
    c = np.cos(teta)
    s = np.sin(teta)
    mat = np.array([
    [c,-s,0.0],
    [s,c,0.0],
    [0.0,0.0,1.0]
    ])
    return mat
