#%% Importing libraries
import h5py as h5
import numpy as np
import cupy as cp
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from modules.mlsarray import MLSarray,Slicelist,irft2np,rft2np,irftnp,rftnp
from modules.gamma import gam_max
import os
import glob
from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#%% Load the HDF5 file

Npx=512
# Npx=1024
datadir=f'data/{Npx}/'

# fname = datadir + 'out_kapt_0_4_D_0_1_H_3_6_em6.h5'
# fname = datadir + 'out_kapt_2_0_D_0_1_H_0_0_e0.h5'
# fname = datadir + 'out_kapt_2_0_D_0_1_H_8_6_em6.h5'
# fname = datadir + 'out_kapt_0_2_hyper_D_1_0_em5_H_8_0_em6.h5'
fname = datadir + 'out_kapt_2_0_hyper_D_5_0_em6_H_1_1_em5.h5'

with h5.File(fname, 'r', swmr=True) as fl:
    t = fl['fields/t'][:]
    nt = len(t)
    kx = fl['data/kx'][:]
    ky = fl['data/ky'][:]
    Lx = fl['params/Lx'][()]
    Ly = fl['params/Ly'][()]
    Npx= fl['params/Npx'][()]
    Npy= fl['params/Npy'][()]
    kapn = fl['params/kapn'][()]
    kapt = fl['params/kapt'][()]
    kapb = fl['params/kapb'][()]
    if 'D' in fl['params']:
        D = fl['params/D'][()]
    elif 'chi' in fl['params']:
        chi = fl['params/chi'][()]
        D = chi
    if 'H' in fl['params']:
        H = fl['params/H'][()]
    elif 'HP' in fl['params']:
        HP = fl['params/HP'][()]
        H = HP

Nx,Ny=2*Npx//3,2*Npy//3  
sl=Slicelist(Nx,Ny)
slbar=np.s_[int(Ny/2)-1:int(Ny/2)*int(Nx/2)-1:int(Nx/2)]
gammax=gam_max(kx,ky,kapn,kapt,kapb,D,H)
t=t*gammax
q = np.sqrt(np.abs(kx)**2 + np.abs(ky)**2)

nt = len(t)
if rank == 0:
    print(f"nt: {nt}")

#%% Functions for energy and enstrophy

def Phi2S(phik, q, k, delk):
    ''' Returns the var(phi) spectrum'''
    phi2k = np.abs(phik)**2
    
    Phi2k = np.zeros(len(k))
    for i in range(len(k)):
        Phi2k[i] = np.sum(phi2k[np.where(np.logical_and(q>=k[i]-delk/2, q<k[i]+delk/2))])*delk
    return Phi2k

def Phi2S_ZF(phik, q, k, delk, slbar):
    ''' Returns the zonal var(phi) spectrum'''   
    phi2k_ZF = np.abs(phik[slbar])**2
    
    Phi2k_ZF = np.zeros(len(k))
    for i in range(len(k)):
        Phi2k_ZF[i] = np.sum(phi2k_ZF[np.where(np.logical_and(q[slbar]>=k[i]-delk/2, q[slbar]<k[i]+delk/2))])*delk
    return Phi2k_ZF

def P2S(pk, q, k, delk):
    ''' Returns the var(P) spectrum'''
    p2k = np.abs(pk)**2
    
    P2k = np.zeros(len(k))
    for i in range(len(k)):
        P2k[i] = np.sum(p2k[np.where(np.logical_and(q>=k[i]-delk/2, q<k[i]+delk/2))])*delk
    return P2k

def P2S_ZF(pk, q, k, delk, slbar):
    ''' Returns the zonal var(P) spectrum'''   
    pk_ZF = np.abs(pk[slbar])**2
    
    P2k_ZF = np.zeros(len(k))
    for i in range(len(k)):
        P2k_ZF[i] = np.sum(pk_ZF[np.where(np.logical_and(q[slbar]>=k[i]-delk/2, q[slbar]<k[i]+delk/2))])*delk
    return P2k_ZF

def ES(omk, q, k, delk):
    ''' Returns the total energy spectrum'''
    sigk=np.sign(ky)
    Lk = sigk+q**2
    ek = Lk*np.abs(omk)**2/q**4

    Ek = np.zeros(len(k))
    for i in range(len(k)):
        Ek[i] = np.sum(ek[np.where(np.logical_and(q>k[i]-delk/2,q<k[i]+delk/2))])*delk
    return Ek

def ES_ZF(omk, q, k, delk, slbar):
    ''' Returns the zonal total energy spectrum'''  
    sigk=np.sign(ky[slbar])
    Lk = sigk+q[slbar]**2 
    ek_ZF = Lk*np.abs(omk[slbar])**2/q[slbar]**4
    
    Ek_ZF = np.zeros(len(k))
    for i in range(len(k)):
        Ek_ZF[i] = np.sum(ek_ZF[np.where(np.logical_and(q[slbar]>=k[i]-delk/2, q[slbar]<k[i]+delk/2))])*delk
    return Ek_ZF

def KS(omk, q, k, delk):
    ''' Returns the kinetic energy spectrum'''
    ek = np.abs(omk)**2/q**2

    Ek = np.zeros(len(k))
    for i in range(len(k)):
        Ek[i] = np.sum(ek[np.where(np.logical_and(q>k[i]-delk/2,q<k[i]+delk/2))])*delk
    return Ek

def KS_ZF(omk, q, k, delk, slbar):
    ''' Returns the zonal kinetic energy spectrum'''  
    ek_ZF = np.abs(omk[slbar])**2/q[slbar]**2
    
    Ek_ZF = np.zeros(len(k))
    for i in range(len(k)):
        Ek_ZF[i] = np.sum(ek_ZF[np.where(np.logical_and(q[slbar]>=k[i]-delk/2, q[slbar]<k[i]+delk/2))])*delk
    return Ek_ZF

def WS(omk, q, k , delk):
    ''' Returns the enstrophy spectrum'''    
    wk = np.abs(omk)**2 

    Wk = np.zeros(len(k))
    for i in range(len(k)):
        Wk[i] = np.sum(wk[np.where(np.logical_and(q>=k[i]-delk/2, q<k[i]+delk/2))])*delk
    return Wk
    
def WS_ZF(omk, q, k, delk, slbar):
    ''' Returns the zonal enstrophy spectrum'''    
    wk_ZF = np.abs(omk[slbar])**2

    Wk_ZF = np.zeros(len(k))
    for i in range(len(k)):
        Wk_ZF[i] = np.sum(wk_ZF[np.where(np.logical_and(q[slbar]>=k[i]-delk/2, q[slbar]<k[i]+delk/2))])*delk
    return Wk_ZF

def GS(omk, pk, q, k, delk):
    ''' Returns the generalized energy spectrum'''
    sigk=np.sign(ky)
    phik=omk/q**2
    ek = np.abs(sigk*phik+pk)**2+q**2*np.abs(phik+pk)**2

    Ek = np.zeros(len(k))
    for i in range(len(k)):
        Ek[i] = np.sum(ek[np.where(np.logical_and(q>k[i]-delk/2,q<k[i]+delk/2))])*delk
    return Ek

def GS_ZF(omk, pk, q, k, delk, slbar):
    ''' Returns the zonal generalized energy spectrum'''  
    sigk=np.sign(ky)
    phik=omk/q**2
    ek_ZF = np.abs(sigk[slbar]*phik[slbar]+pk[slbar])**2+q[slbar]**2*np.abs(phik[slbar]+pk[slbar])**2
    
    Ek_ZF = np.zeros(len(k))
    for i in range(len(k)):
        Ek_ZF[i] = np.sum(ek_ZF[np.where(np.logical_and(q[slbar]>=k[i]-delk/2, q[slbar]<k[i]+delk/2))])*delk
    return Ek_ZF

def GKS(omk, pk, q, k, delk):
    ''' Returns the generalized kinetic energy spectrum'''
    phik=omk/q**2
    ek = q**2*np.abs(phik+pk)**2

    Ek = np.zeros(len(k))
    for i in range(len(k)):
        Ek[i] = np.sum(ek[np.where(np.logical_and(q>k[i]-delk/2,q<k[i]+delk/2))])*delk
    return Ek

def GKS_ZF(omk, pk, q, k, delk, slbar):
    ''' Returns the zonal generalized kinetic energy spectrum'''  
    phik=omk/q**2
    ek_ZF = q[slbar]**2*np.abs(phik[slbar]+pk[slbar])**2
    
    Ek_ZF = np.zeros(len(k))
    for i in range(len(k)):
        Ek_ZF[i] = np.sum(ek_ZF[np.where(np.logical_and(q[slbar]>=k[i]-delk/2, q[slbar]<k[i]+delk/2))])*delk
    return Ek_ZF

#%% compute quantities

delk = ky[1] - ky[0]
# k = np.linspace(np.min(q), np.max(q), num=int(np.max(q)/delk))
k = np.linspace(0, np.max(q), num=int(np.max(q)/delk)+1)

# MPI parallelization for time series calculation
nt2 = int(nt/2)
nt2 = nt2 - (nt2 % size)
if rank == 0:
    indices = np.array_split(range(nt2), size)
else:
    indices = None
local_indices = comm.scatter(indices, root=0)

# Local arrays for each process
reynolds_power_local = np.zeros(len(local_indices), dtype=np.float64)
Phi2k_local = np.zeros((len(local_indices), len(k)))
Phi2k_ZF_local = np.zeros((len(local_indices), len(k)))
P2k_local = np.zeros((len(local_indices), len(k)))
P2k_ZF_local = np.zeros((len(local_indices), len(k)))
Ek_local = np.zeros((len(local_indices), len(k)))
Ek_ZF_local = np.zeros((len(local_indices), len(k)))
Kk_local = np.zeros((len(local_indices), len(k)))
Kk_ZF_local = np.zeros((len(local_indices), len(k)))
Wk_local = np.zeros((len(local_indices), len(k)))
Wk_ZF_local = np.zeros((len(local_indices), len(k)))
Gk_local = np.zeros((len(local_indices), len(k)))
Gk_ZF_local = np.zeros((len(local_indices), len(k)))
GKk_local = np.zeros((len(local_indices), len(k)))
GKk_ZF_local = np.zeros((len(local_indices), len(k)))

with h5.File(fname, 'r', swmr=True) as fl:
    for idx, it in enumerate(local_indices):
        print(f"Rank {rank} processing time step {it}")
        Omk = fl['fields/Omk'][it+nt//2]
        Pk = fl['fields/Pk'][it+nt//2]
        kpsq_loc  = kx**2 + ky**2
        Phik_loc  = -Omk / kpsq_loc
        Om_loc    = irft2np(Omk, Npx=Npx, Npy=Npy, Nx=Nx, sl=sl)
        vx_loc    = irft2np(-1j*ky*Phik_loc, Npx=Npx, Npy=Npy, Nx=Nx, sl=sl)
        vy_loc    = irft2np(1j*kx*Phik_loc, Npx=Npx, Npy=Npy, Nx=Nx, sl=sl)
        wx_loc    = irft2np(-1j*ky*Pk, Npx=Npx, Npy=Npy, Nx=Nx, sl=sl)
        Ombar_loc = np.mean(Om_loc, axis=1)
        RPhi_loc  = np.mean(vy_loc*vx_loc, axis=1)
        RP_loc    = np.mean(vy_loc*wx_loc, axis=1)
        reynolds_power_local[idx] = np.mean((RPhi_loc + RP_loc) * Ombar_loc)
        Phi2k_local[idx,:] = Phi2S(Pk, q, k, delk)
        Phi2k_ZF_local[idx,:] = Phi2S_ZF(Pk, q, k, delk, slbar)
        P2k_local[idx,:] = P2S(Pk, q, k, delk)
        P2k_ZF_local[idx,:] = P2S_ZF(Pk, q, k, delk, slbar)
        Ek_local[idx,:] = ES(Omk, q, k, delk)
        Ek_ZF_local[idx,:] = ES_ZF(Omk, q, k, delk, slbar)
        Kk_local[idx,:] = KS(Omk, q, k, delk)
        Kk_ZF_local[idx,:] = KS_ZF(Omk, q, k, delk, slbar)
        Wk_local[idx,:] = WS(Omk, q, k, delk)
        Wk_ZF_local[idx,:] = WS_ZF(Omk, q, k, delk, slbar)
        Gk_local[idx,:] = GS(Omk, Pk, q, k, delk)
        Gk_ZF_local[idx,:] = GS_ZF(Omk, Pk, q, k, delk, slbar)
        GKk_local[idx,:] = GKS(Omk, Pk, q, k, delk)
        GKk_ZF_local[idx,:] = GKS_ZF(Omk, Pk, q, k, delk, slbar)

# Gather results from all processes
Phi2k_t = Phi2k_ZF_t = P2k_t = P2k_ZF_t = Ek_t = Ek_ZF_t = Kk_t = Kk_ZF_t = Wk_t = Wk_ZF_t = Gk_t = Gk_ZF_t = GKk_t = GKk_ZF_t = None
reynolds_power_gathered = None
if rank == 0:
    reynolds_power_gathered = np.zeros(nt2)
    Phi2k_t = np.zeros((nt2, len(k)))
    Phi2k_ZF_t = np.zeros((nt2, len(k)))
    P2k_t = np.zeros((nt2, len(k)))
    P2k_ZF_t = np.zeros((nt2, len(k)))
    Ek_t = np.zeros((nt2, len(k)))
    Ek_ZF_t = np.zeros((nt2, len(k)))
    Kk_t = np.zeros((nt2, len(k)))
    Kk_ZF_t = np.zeros((nt2, len(k)))
    Wk_t = np.zeros((nt2, len(k)))
    Wk_ZF_t = np.zeros((nt2, len(k)))
    Gk_t = np.zeros((nt2, len(k)))
    Gk_ZF_t = np.zeros((nt2, len(k)))
    GKk_t = np.zeros((nt2, len(k)))
    GKk_ZF_t = np.zeros((nt2, len(k)))

comm.Gather(Phi2k_local, Phi2k_t, root=0)
comm.Gather(Phi2k_ZF_local, Phi2k_ZF_t, root=0)
comm.Gather(P2k_local, P2k_t, root=0)
comm.Gather(P2k_ZF_local, P2k_ZF_t, root=0)
comm.Gather(Ek_local, Ek_t, root=0)
comm.Gather(Ek_ZF_local, Ek_ZF_t, root=0)
comm.Gather(Kk_local, Kk_t, root=0)
comm.Gather(Kk_ZF_local, Kk_ZF_t, root=0)
comm.Gather(Wk_local, Wk_t, root=0)
comm.Gather(Wk_ZF_local, Wk_ZF_t, root=0)
comm.Gather(Gk_local, Gk_t, root=0)
comm.Gather(Gk_ZF_local, Gk_ZF_t, root=0)
comm.Gather(GKk_local, GKk_t, root=0)
comm.Gather(GKk_ZF_local, GKk_ZF_t, root=0)
comm.Gather(reynolds_power_local, reynolds_power_gathered, root=0)

if rank == 0:
    print("Gathered")

    Phi2k_turb_t = Phi2k_t - Phi2k_ZF_t
    P2k_turb_t = P2k_t - P2k_ZF_t
    Ek_turb_t = Ek_t - Ek_ZF_t
    Kk_turb_t = Kk_t - Kk_ZF_t
    Wk_turb_t = Wk_t - Wk_ZF_t
    Gk_turb_t = Gk_t - Gk_ZF_t
    GKk_turb_t = GKk_t - GKk_ZF_t

    Phi2k = np.mean(Phi2k_t, axis=0)
    Phi2k_ZF = np.mean(Phi2k_ZF_t, axis=0)
    Phi2k_turb = Phi2k - Phi2k_ZF
    P2k = np.mean(P2k_t, axis=0)
    P2k_ZF = np.mean(P2k_ZF_t, axis=0)
    P2k_turb = P2k - P2k_ZF
    Ek = np.mean(Ek_t, axis=0)
    Ek_ZF = np.mean(Ek_ZF_t, axis=0)
    Ek_turb = Ek - Ek_ZF
    Kk = np.mean(Kk_t, axis=0)
    Kk_ZF = np.mean(Kk_ZF_t, axis=0)
    Kk_turb = Kk - Kk_ZF
    Wk = np.mean(Wk_t, axis=0)
    Wk_ZF = np.mean(Wk_ZF_t, axis=0)
    Wk_turb = Wk - Wk_ZF
    Gk = np.mean(Gk_t, axis=0)
    Gk_ZF = np.mean(Gk_ZF_t, axis=0)
    Gk_turb = Gk - Gk_ZF
    GKk = np.mean(GKk_t, axis=0)
    GKk_ZF = np.mean(GKk_ZF_t, axis=0)
    GKk_turb = GKk - GKk_ZF

    #%% Save computed spectra
    savefile = fname.replace('out_', 'spectrum/spectrum_')
    with h5.File(savefile, 'w') as fl:
        fl.create_dataset('k', data=k)
        fl.create_dataset('idx_saved', data=np.arange(nt//2, nt//2 + nt2))
        fl.create_dataset('Phi2k', data=Phi2k)
        fl.create_dataset('Phi2k_ZF', data=Phi2k_ZF)
        fl.create_dataset('Phi2k_turb', data=Phi2k_turb)
        fl.create_dataset('Phi2k_t', data=Phi2k_t)
        fl.create_dataset('Phi2k_ZF_t', data=Phi2k_ZF_t)
        fl.create_dataset('P2k', data=P2k)
        fl.create_dataset('P2k_ZF', data=P2k_ZF)
        fl.create_dataset('P2k_turb', data=P2k_turb)
        fl.create_dataset('P2k_t', data=P2k_t)
        fl.create_dataset('P2k_ZF_t', data=P2k_ZF_t)
        fl.create_dataset('Ek', data=Ek)
        fl.create_dataset('Ek_ZF', data=Ek_ZF)
        fl.create_dataset('Ek_turb', data=Ek_turb)
        fl.create_dataset('Ek_t', data=Ek_t)
        fl.create_dataset('Ek_ZF_t', data=Ek_ZF_t)
        fl.create_dataset('Kk', data=Kk)
        fl.create_dataset('Kk_ZF', data=Kk_ZF)
        fl.create_dataset('Kk_turb', data=Kk_turb)
        fl.create_dataset('Kk_t', data=Kk_t)
        fl.create_dataset('Kk_ZF_t', data=Kk_ZF_t)
        fl.create_dataset('Wk', data=Wk)
        fl.create_dataset('Wk_ZF', data=Wk_ZF)
        fl.create_dataset('Wk_turb', data=Wk_turb)
        fl.create_dataset('Wk_t', data=Wk_t)
        fl.create_dataset('Wk_ZF_t', data=Wk_ZF_t)
        fl.create_dataset('Gk', data=Gk)
        fl.create_dataset('Gk_ZF', data=Gk_ZF)
        fl.create_dataset('Gk_turb', data=Gk_turb)
        fl.create_dataset('Gk_t', data=Gk_t)
        fl.create_dataset('Gk_ZF_t', data=Gk_ZF_t)
        fl.create_dataset('GKk', data=GKk)
        fl.create_dataset('GKk_ZF', data=GKk_ZF)
        fl.create_dataset('GKk_turb', data=GKk_turb)
        fl.create_dataset('GKk_t', data=GKk_t)
        fl.create_dataset('GKk_ZF_t', data=GKk_ZF_t)
    print(f"Saved spectra to {savefile}")
