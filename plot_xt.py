#%% Importing libraries
import h5py as h5
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from modules.mlsarray import MLSarray,Slicelist,irft2np,rft2np,irftnp,rftnp
from modules.plot_basics import symmetrize_y_axis
from modules.gamma import gam_max
import os
import glob 

from modules.plot_basics import apply_style, savename as _savename
from functools import partial
apply_style()

#%% Load the HDF5 file

# Npx=512
Npx=1024
datadir=f'data/{Npx}/'

# fname = datadir + 'out_kapt_0_4_D_0_1_H_3_6_em6.h5'
fname = datadir + 'out_kapt_2_0_D_0_1_H_8_6_em6.h5'
# fname = datadir + 'out_kapt_2_0_D_0_1_H_1_7_em5.h5'

# kapt=0.2
# D=0.1
# pattern = datadir + f'out_kapt_{str(kapt).replace(".", "_")}_D_{str(D).replace(".", "_")}*_1024_1024.h5'
# files = glob.glob(pattern)
# if not files:
#     print(f"No file found for kappa_T = {kapt}")
# else:
#     fname = files[0]

# Downsample time axis to reduce memory usage
stride = 4
with h5.File(fname, 'r', swmr=True) as fl:
    t = fl['fluxes/t'][::stride].astype(np.float32)
    kx = fl['data/kx'][:]
    ky = fl['data/ky'][:]
    Lx = fl['params/Lx'][()]
    Ly = fl['params/Ly'][()]
    Npx= fl['params/Npx'][()]
    Npy= fl['params/Npy'][()]
    kapn = fl['params/kapn'][()]
    kapt = fl['params/kapt'][()]
    kapb = fl['params/kapb'][()]
    D = fl['params/D'][()]
    if 'H' in fl['params']:
        H = fl['params/H'][()]
    elif 'HP' in fl['params']:
        HP = fl['params/HP'][()]
        H=HP

Nx,Ny=2*Npx//3,2*Npy//3
sl=Slicelist(Nx,Ny)
slbar=np.s_[int(Ny/2)-1:int(Ny/2)*int(Nx/2)-1:int(Nx/2)]
gammax=gam_max(kx,ky,kapn,kapt,kapb,D,H)
t=t*gammax
nt = len(t)
nt_data = nt

xl,yl=np.arange(0,Lx,Lx/Npx),np.arange(0,Ly,Ly/Npy)
x,y=np.meshgrid(np.array(xl),np.array(yl),indexing='ij')
xm, tm = np.meshgrid(x[:, 0], t[:nt_data])

savename = partial(_savename, datadir, fname)

#%% vbar

with h5.File(fname, 'r', swmr=True) as fl:
    vbar_t = fl['zonal/vbar'][::stride].astype(np.float32)

vbar_lim = float(np.percentile(np.abs(vbar_t), 75))
plt.figure(figsize=(16, 9))
plt.pcolormesh(xm, tm, vbar_t[:nt_data,:], vmin=-vbar_lim, vmax=vbar_lim, cmap='seismic')
plt.xlabel('x')
plt.ylabel(r'$\gamma t$')
plt.title(r'Zonal flow: $\partial_x\overline{\phi}$')
plt.colorbar()
plt.tight_layout(pad=0.5)
plt.savefig(savename('vbar_xt'), dpi=100, bbox_inches='tight')
plt.show()
plt.close()
del vbar_t

#%% dxvbar

with h5.File(fname, 'r', swmr=True) as fl:
    dxvbar_t = fl['zonal/Ombar'][::stride].astype(np.float32)

dxvbar_lim = float(np.percentile(np.abs(dxvbar_t), 75))
plt.figure(figsize=(16, 9))
plt.pcolormesh(xm, tm, dxvbar_t[:nt_data,:], vmin=-dxvbar_lim, vmax=dxvbar_lim, cmap='seismic')
plt.xlabel('x')
plt.ylabel(r'$\gamma t$')
plt.title(r'Zonal shear: $\partial_x^2\overline{\phi}$')
plt.colorbar()
plt.tight_layout(pad=0.5)
plt.savefig(savename('dxvbar_xt'), dpi=100, bbox_inches='tight')
plt.show()
plt.close()
del dxvbar_t

#%% RPhi

with h5.File(fname, 'r', swmr=True) as fl:
    Rphi_t = fl['fluxes/RPhi'][::stride].astype(np.float32)

RPhi_lim = float(np.percentile(np.abs(Rphi_t), 75))
plt.figure(figsize=(16, 9))
plt.pcolormesh(xm, tm, Rphi_t[:nt_data,:], vmin=-RPhi_lim, vmax=RPhi_lim, cmap='seismic')
plt.xlabel('x')
plt.ylabel(r'$\gamma t$')
plt.title(r'$R_{\mathrm{\phi}}$')
plt.colorbar()
plt.tight_layout(pad=0.5)
plt.savefig(savename('RPhi_xt'), dpi=100, bbox_inches='tight')
plt.show()
plt.close()
del Rphi_t

#%% Rd

with h5.File(fname, 'r', swmr=True) as fl:
    Rd_t = fl['fluxes/RP'][::stride].astype(np.float32)

RP_lim = float(np.percentile(np.abs(Rd_t), 75))
plt.figure(figsize=(16, 9))
plt.pcolormesh(xm, tm, Rd_t[:nt_data,:], vmin=-RP_lim, vmax=RP_lim, cmap='seismic')
plt.xlabel('x')
plt.ylabel(r'$\gamma t$')
plt.title(r'$R_{\mathrm{d}}$')
plt.colorbar()
plt.tight_layout(pad=0.5)
plt.savefig(savename('Rd_xt'), dpi=100, bbox_inches='tight')
plt.show()
plt.close()
del Rd_t

#%% R

with h5.File(fname, 'r', swmr=True) as fl:
    Rphi_t = fl['fluxes/RPhi'][::stride].astype(np.float32)
    Rd_t   = fl['fluxes/RP'][::stride].astype(np.float32)

R_t = Rphi_t + Rd_t
del Rphi_t, Rd_t
R_lim = float(np.percentile(np.abs(R_t), 75))
plt.figure(figsize=(16, 9))
plt.pcolormesh(xm, tm, R_t[:nt_data,:], vmin=-R_lim, vmax=R_lim, cmap='seismic')
plt.xlabel('x')
plt.ylabel(r'$\gamma t$')
plt.title(r'$R = R_{\mathrm{\phi}} + R_{\mathrm{d}}$')
plt.colorbar()
plt.tight_layout(pad=0.5)
plt.savefig(savename('R_xt'), dpi=100, bbox_inches='tight')
plt.show()
plt.close()
del R_t

#%% PPhi

with h5.File(fname, 'r', swmr=True) as fl:
    Rphi_t   = fl['fluxes/RPhi'][::stride].astype(np.float32)
    dxvbar_t = fl['zonal/Ombar'][::stride].astype(np.float32)

PPhi_t   = Rphi_t[:nt_data] * dxvbar_t[:nt_data]
del Rphi_t, dxvbar_t
PPhi_lim = float(np.percentile(np.abs(PPhi_t), 75))
plt.figure(figsize=(16, 9))
plt.pcolormesh(xm, tm, PPhi_t[:nt_data,:], vmin=-PPhi_lim, vmax=PPhi_lim, cmap='seismic')
plt.xlabel('x')
plt.ylabel(r'$\gamma t$')
plt.title(r'$P_{\mathrm{\phi}}$')
plt.colorbar()
plt.tight_layout(pad=0.5)
plt.savefig(savename('PPhi_xt'), dpi=100, bbox_inches='tight')
plt.show()
plt.close()
del PPhi_t

#%% Pd

with h5.File(fname, 'r', swmr=True) as fl:
    Rd_t     = fl['fluxes/RP'][::stride].astype(np.float32)
    dxvbar_t = fl['zonal/Ombar'][::stride].astype(np.float32)

PP_t   = Rd_t[:nt_data] * dxvbar_t[:nt_data]
del Rd_t, dxvbar_t
PP_lim = float(np.percentile(np.abs(PP_t), 75))
plt.figure(figsize=(16, 9))
plt.pcolormesh(xm, tm, PP_t[:nt_data,:], vmin=-PP_lim, vmax=PP_lim, cmap='seismic')
plt.xlabel('x')
plt.ylabel(r'$\gamma t$')
plt.title(r'$P_{\mathrm{d}}$')
plt.colorbar()
plt.tight_layout(pad=0.5)
plt.savefig(savename('Pd_xt'), dpi=100, bbox_inches='tight')
plt.show()
plt.close()
del PP_t

#%% P

with h5.File(fname, 'r', swmr=True) as fl:
    Rphi_t   = fl['fluxes/RPhi'][::stride].astype(np.float32)
    Rd_t     = fl['fluxes/RP'][::stride].astype(np.float32)
    dxvbar_t = fl['zonal/Ombar'][::stride].astype(np.float32)

P_t   = (Rphi_t[:nt_data] + Rd_t[:nt_data]) * dxvbar_t[:nt_data]
del Rphi_t, Rd_t, dxvbar_t
P_lim = float(np.percentile(np.abs(P_t), 75))
plt.figure(figsize=(16, 9))
plt.pcolormesh(xm, tm, P_t[:nt_data,:], vmin=-P_lim, vmax=P_lim, cmap='seismic')
plt.xlabel('x')
plt.ylabel(r'$\gamma t$')
plt.title(r'$P = P_{\mathrm{\phi}} + P_{\mathrm{d}}$')
plt.colorbar()
plt.tight_layout(pad=0.5)
plt.savefig(savename('P_xt'), dpi=100, bbox_inches='tight')
plt.show()
plt.close()
del P_t

#%% Heat flux

with h5.File(fname, 'r', swmr=True) as fl:
    Q_t = fl['fluxes/Q'][::stride].astype(np.float32)

Q_lim = float(np.percentile(np.abs(Q_t), 90))
plt.figure(figsize=(16, 9))
plt.pcolormesh(xm, tm, Q_t[:nt_data,:], vmin=-Q_lim, vmax=Q_lim, cmap='seismic')
plt.xlabel('x')
plt.ylabel(r'$\gamma t$')
plt.title('Heat flux: $Q$')
plt.colorbar()
plt.tight_layout(pad=0.5)
plt.savefig(savename('Q_xt'), dpi=100, bbox_inches='tight')
plt.show()
plt.close()
del Q_t, xm, tm
