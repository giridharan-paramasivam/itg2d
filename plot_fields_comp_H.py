#%% Importing libraries
import h5py as h5
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from modules.mlsarray import irft2np as original_irft2np, rft2np as original_rft2np, irftnp as original_irftnp, rftnp as original_rftnp
from modules.mlsarray import Slicelist
import cupy as cp
from functools import partial
import warnings
from modules.gamma import gam_max

from modules.plot_basics import apply_style, savename as _savename, figsize_single, figsize_double
apply_style()

#%% Load the HDF5 file

Npx=512
# Npx=1024
datadir=f'data/{Npx}/'
subdir = 'fields/'

fname = 'out_kapt_2_0_D_0_1_H_0_0_e0.h5'
# fname = 'out_kapt_2_0_D_0_1_H_8_6_em6.h5'

evol_fname = datadir + 'evol/' + fname.replace('out_', 'evol_')
try:
    with h5.File(evol_fname, 'r') as flevol:
        t_evol = flevol['t'][:]
        Qbox_t = flevol['Qbox_t'][:]
        it = int(np.argmax(Qbox_t))
        t_spike = t_evol[it]
except Exception as e:
    warnings.warn(f"Could not load Qbox_t from {evol_fname}: {e}\nDefaulting to last time index.")
    it = -1
    t_spike = None

with h5.File(datadir+fname, 'r', swmr=True) as fl:
    Omk = fl['fields/Omk'][it]
    Pk = fl['fields/Pk'][it]
    Ombar = fl['zonal/Ombar'][it]
    Pbar = fl['zonal/Pbar'][it]
    vbar = fl['zonal/vbar'][it]
    t = fl['fields/t'][:]
    kx = fl['data/kx'][:]
    ky = fl['data/ky'][:]
    Lx = fl['params/Lx'][()]
    Ly = fl['params/Ly'][()]
    Npx= fl['params/Npx'][()]
    Npy= fl['params/Npy'][()]
    kapt = fl['params/kapt'][()]
    kapn = fl['params/kapn'][()]
    kapb = fl['params/kapb'][()]
    D = fl['params/D'][()]
    if 'H' in fl['params']:
        H = fl['params/H'][()]
    elif 'HP' in fl['params']:
        HP = fl['params/HP'][()]
        H = HP

    gammax = gam_max(kx, ky, kapn, kapt, kapb, D, H)
    if it >= 0 and t_spike is not None:
        print(f"Plotting at heat flux spike: t = {t_spike:.3f}, gamma*t = {gammax * t_spike:.3f}, index = {it}")

Nx,Ny=2*Npx//3,2*Npy//3
sl=Slicelist(Nx,Ny)
slbar=np.s_[int(Ny/2)-1:int(Ny/2)*int(Nx/2)-1:int(Ny/2)]
nt = len(t)
print("nt: ", nt)

xl,yl=np.linspace(0,Lx,Npx),np.linspace(0,Ly,Npy)
x,y=np.meshgrid(np.array(xl),np.array(yl),indexing='ij')

#%% Functions

irft2np = partial(original_irft2np,Npx=Npx,Npy=Npy,Nx=Nx,sl=sl)
rft2np = partial(original_rft2np,sl=sl)
irftnp = partial(original_irftnp,Npx=Npx,Nx=Nx)
rftnp = partial(original_rftnp,Nx=Nx)

savename = partial(_savename, datadir+subdir, fname)

#%% Calculate Om and P in real space

Om = irft2np(Omk)
P = irft2np(Pk)

#%% Plot: Om

fig, ax = plt.subplots(figsize=figsize_single)
c = ax.pcolormesh(x, y, Om, cmap='seismic', vmin=-np.max(np.abs(Om)), vmax=np.max(np.abs(Om)), rasterized=True)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.colorbar(c, ax=ax)
plt.savefig(savename('fields_Om_streamer'), bbox_inches='tight')
plt.show()

#%% Plot: P

fig, ax = plt.subplots(figsize=figsize_single)
c = ax.pcolormesh(x, y, P, cmap='seismic', vmin=-np.max(np.abs(P)), vmax=np.max(np.abs(P)), rasterized=True)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.colorbar(c, ax=ax)
plt.savefig(savename('fields_P_streamer'), bbox_inches='tight')
plt.show()
# %%
