#%% Importing libraries
import h5py as h5
import numpy as np
import cupy as cp
from modules.mlsarray import MLSarray, Slicelist
from modules.mlsarray import irft2np as original_irft2np, rft2np as original_rft2np, irftnp as original_irftnp, rftnp as original_rftnp
from modules.gamma_2d3c import gam_max, ky_max
from functools import partial
from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#%% Load the HDF5 file

Npx=1024
datadir=f'data_2d3c/{Npx}'
fname = datadir + 'out_2d3c_kapt_2_0_D_0_1_kz_0_1.h5'
# kapt=0.3
# chi=0.1
# pattern = datadir + f'out_2d3c_kapt_{str(kapt).replace(".", "_")}_chi_{str(chi).replace(".", "_")}*.h5'
# files = glob.glob(pattern)
# if not files:
#     print(f"No file found for kappa_T = {kapt}")
# else:
#     fname = files[0]

with h5.File(fname, 'r', swmr=True) as fl:
    t = fl['fields/t'][:]
    nt = len(t)
    kx = fl['data/kx'][:]
    ky = fl['data/ky'][:]
    Lx = fl['params/Lx'][()]
    Ly = fl['params/Ly'][()]
    Npx = fl['params/Npx'][()]
    Npy = fl['params/Npy'][()]
    kapn = fl['params/kapn'][()]
    kapt = fl['params/kapt'][()]
    kapb = fl['params/kapb'][()]
    D = fl['params/D'][()]
    kz = fl['params/kz'][()]

Nx, Ny = 2*Npx//3, 2*Npy//3
sl = Slicelist(Nx, Ny)
slbar = np.s_[int(Ny/2)-1:int(Ny/2)*int(Nx/2)-1:int(Nx/2)]
gammax = gam_max(kx, ky, kapn, kapt, kapb, D, kz)
t = t * gammax
k_lin = ky_max(kx, ky, kapn, kapt, kapb, D, kz)

#%% Functions

irft2np = partial(original_irft2np, Npx=Npx, Npy=Npy, Nx=Nx, sl=sl)
rft2np = partial(original_rft2np, sl=sl)
irftnp = partial(original_irftnp, Npx=Npx, Nx=Nx)
rftnp = partial(original_rftnp, Nx=Nx)

def Eflux(Omk, Pk, kx, ky, k, delk, flag='Pik'):
    ''' Returns the RHS of the model equations'''
    kpsq = kx**2 + ky**2
    Phik = -Omk/kpsq
    dyphi = irft2np(1j*ky*Phik)
    dxphi = irft2np(1j*kx*Phik)
    dyP = irft2np(1j*ky*Pk)
    dxP = irft2np(1j*kx*Pk)
    sigk = np.sign(ky)
    Wk = sigk + kpsq
    dxnOmg = irft2np(1j*kx*Wk*Phik)
    dynOmg = irft2np(1j*ky*Wk*Phik)

    nltermOmg = rft2np(dxphi*dynOmg - dyphi*dxnOmg)
    nltermP = -kx**2*(rft2np(dxphi*dyP)) + kx*ky*(rft2np(dxphi*dxP)) - kx*ky*(rft2np(dyphi*dyP)) + ky**2*(rft2np(dyphi*dxP))

    ak = np.zeros_like(Omk)
    Ak = np.zeros(len(k))
    if flag == 'Pik_phi':
        ak = np.real(np.conj(Phik)*nltermOmg)
        for i in range(len(k)):
            Ak[i] = np.sum(ak[np.where(q <= k[i])])
    elif flag == 'Pik_d':
        ak = np.real(np.conj(Phik)*nltermP)
        for i in range(len(k)):
            Ak[i] = np.sum(ak[np.where(q <= k[i])])
    elif flag == 'fk':
        ak = np.real(np.conj(Phik)*kapb*1j*ky*Pk)
        for i in range(len(k)):
            Ak[i] = np.sum(ak[np.where(np.logical_and(q > k[i], q <= k[i]+delk))])/delk
    elif flag == 'dk':
        ak = np.real(np.conj(Phik)*D*kpsq*Phik )
        for i in range(len(k)):
            Ak[i] = np.sum(ak[np.where(np.logical_and(q > k[i], q <= k[i]+delk))])/delk
    return Ak

def Gflux(Omk, Pk, kx, ky, k, delk, flag='PiGk_P'):
    ''' Returns spectral fluxes for the generalized energy G '''
    kpsq = kx**2 + ky**2
    Phik = -Omk/kpsq
    dyphi = irft2np(1j*ky*Phik)
    dxphi = irft2np(1j*kx*Phik)
    dyP = irft2np(1j*ky*Pk)
    dxP = irft2np(1j*kx*Pk)
    sigk = np.sign(ky)
    Wk = sigk + kpsq
    dxnOmg = irft2np(1j*kx*Wk*Phik)
    dynOmg = irft2np(1j*ky*Wk*Phik)

    nltermOmg = rft2np(dxphi*dynOmg - dyphi*dxnOmg)
    nltermP = -kx**2*(rft2np(dxphi*dyP)) + kx*ky*(rft2np(dxphi*dxP)) - kx*ky*(rft2np(dyphi*dyP)) + ky**2*(rft2np(dyphi*dxP))
    nltermP_pb = rft2np(dxphi*dyP - dyphi*dxP)

    ak = np.zeros_like(Omk)
    Ak = np.zeros(len(k))
    if flag == 'PiGk_P':
        ak = np.real((1 + kpsq)*np.conj(Pk)*nltermP_pb + Wk*np.conj(Phik)*nltermP_pb)
        for i in range(len(k)):
            Ak[i] = np.sum(ak[np.where(q <= k[i])])
    elif flag == 'PiGk_phi':
        ak = np.real((np.conj(Phik) + np.conj(Pk))*nltermOmg)
        for i in range(len(k)):
            Ak[i] = np.sum(ak[np.where(q <= k[i])])
    elif flag == 'PiGk_d':
        ak = np.real((np.conj(Phik) + np.conj(Pk))*nltermP)
        for i in range(len(k)):
            Ak[i] = np.sum(ak[np.where(q <= k[i])])
    elif flag == 'fGk':
        ak = np.real(np.conj(Phik)*(kapb + 2*kapn + kapt)*1j*ky*Pk)
        for i in range(len(k)):
            Ak[i] = np.sum(ak[np.where(np.logical_and(q > k[i], q <= k[i]+delk))])/delk
    elif flag == 'dGk':
        ak = D*kpsq * ((sigk + kpsq)*np.abs(Phik + Pk)**2 + (1 - sigk)*np.abs(Pk)**2)
        for i in range(len(k)):
            Ak[i] = np.sum(ak[np.where(np.logical_and(q > k[i], q <= k[i]+delk))])/delk
    return Ak

#%% Calculate quantities

delk = ky[1] - ky[0]
q = np.sqrt(np.abs(kx)**2 + np.abs(ky)**2)
k = np.linspace(np.min(q), np.max(q), num=int(np.max(q)/delk))

# MPI parallelization

nt2 = int(nt/2)
nt2 = nt2 - (nt2 % size)
if rank == 0:
    indices = np.array_split(range(nt2), size)
else:
    indices = None

local_indices = comm.scatter(indices, root=0)
count_local = len(local_indices)

Pik_phi = np.zeros(len(k))
Pik_d = np.zeros(len(k))
fk = np.zeros(len(k))
dk = np.zeros(len(k))
PiGk_P = np.zeros(len(k))
PiGk_phi = np.zeros(len(k))
PiGk_d = np.zeros(len(k))
fGk = np.zeros(len(k))
dGk = np.zeros(len(k))

with h5.File(fname, 'r', swmr=True) as fl:
    for idx, it in enumerate(local_indices):
        print(f"Rank {rank} processing time step {it}")
        Omk = fl['fields/Omk'][it+nt//2]
        Pk = fl['fields/Pk'][it+nt//2]
        Pik_phi += Eflux(Omk, Pk, kx, ky, k, delk, flag='Pik_phi')
        Pik_d += Eflux(Omk, Pk, kx, ky, k, delk, flag='Pik_d')
        fk += Eflux(Omk, Pk, kx, ky, k, delk, flag='fk')
        dk += Eflux(Omk, Pk, kx, ky, k, delk, flag='dk')
        PiGk_P += Gflux(Omk, Pk, kx, ky, k, delk, flag='PiGk_P')
        PiGk_phi += Gflux(Omk, Pk, kx, ky, k, delk, flag='PiGk_phi')
        PiGk_d += Gflux(Omk, Pk, kx, ky, k, delk, flag='PiGk_d')
        fGk += Gflux(Omk, Pk, kx, ky, k, delk, flag='fGk')
        dGk += Gflux(Omk, Pk, kx, ky, k, delk, flag='dGk')

Pik_phi = comm.allreduce(Pik_phi, op=MPI.SUM)/nt2
Pik_d = comm.allreduce(Pik_d, op=MPI.SUM)/nt2
fk = comm.allreduce(fk, op=MPI.SUM)/nt2
dk = comm.allreduce(dk, op=MPI.SUM)/nt2
PiGk_P = comm.allreduce(PiGk_P, op=MPI.SUM)/nt2
PiGk_phi = comm.allreduce(PiGk_phi, op=MPI.SUM)/nt2
PiGk_d = comm.allreduce(PiGk_d, op=MPI.SUM)/nt2
fGk = comm.allreduce(fGk, op=MPI.SUM)/nt2
dGk = comm.allreduce(dGk, op=MPI.SUM)/nt2

if rank == 0:
    Pik = Pik_phi + Pik_d
    idx_k = np.argmax(fk)
    k_f = k[idx_k]
    k_Gf = k[np.argmax(fGk)]
    PiGk = PiGk_P + PiGk_phi + PiGk_d

    out_file = fname.replace('out_', 'flux/spectral_flux_')
    with h5.File(out_file, 'w') as fl:
        fl.create_dataset('k', data=k)
        fl.create_dataset('k_f', data=k_f)
        fl.create_dataset('k_Gf', data=k_Gf)
        fl.create_dataset('k_lin', data=k_lin)
        fl.create_dataset('Pik', data=Pik)
        fl.create_dataset('Pik_phi', data=Pik_phi)
        fl.create_dataset('Pik_d', data=Pik_d)
        fl.create_dataset('fk', data=fk)
        fl.create_dataset('dk', data=dk)
        fl.create_dataset('PiGk', data=PiGk)
        fl.create_dataset('PiGk_P', data=PiGk_P)
        fl.create_dataset('PiGk_phi', data=PiGk_phi)
        fl.create_dataset('PiGk_d', data=PiGk_d)
        fl.create_dataset('fGk', data=fGk)
        fl.create_dataset('dGk', data=dGk)
    print(f"Saved to {out_file}")
