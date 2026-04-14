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

# Npx=512
Npx=1024
datadir=f'data/{Npx}/'
subdir = 'fields/'

# fname = 'out_kapt_0_4_D_0_1_H_3_6_em6.h5'
fname = 'out_kapt_2_0_D_0_1_H_8_6_em6.h5'
# fname = 'out_kapt_2_0_D_0_1_H_1_7_em5.h5'

plot_zonal = False
it = -1
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

#%% Plot: Om and P

fig, axs = plt.subplots(1, 2, figsize=figsize_double, sharey=True)

c0 = axs[0].pcolormesh(x, y, Om, cmap='seismic', vmin=-np.max(np.abs(Om)), vmax=np.max(np.abs(Om)), rasterized=True)
if plot_zonal:
    axs[0].plot(x[:,0], 0.5*(y[:,-1]+y[:,0])+0.25*(y[:,-1]-y[:,0])*vbar/np.max(np.abs(vbar)),'w',linewidth=5)
    axs[0].plot(x[:,0], 0.5*(y[:,-1]+y[:,0])+0.25*(y[:,-1]-y[:,0])*vbar/np.max(np.abs(vbar)),'k',label=r'$\overline{v}_y$')
    axs[0].legend(loc='upper right')
axs[0].set_title(r'$\nabla^2 \phi$')
axs[0].set_xlabel('$x$')
axs[0].set_ylabel('$y$')
plt.colorbar(c0, ax=axs[0])

c1 = axs[1].pcolormesh(x, y, P, cmap='seismic', vmin=-np.max(np.abs(P)), vmax=np.max(np.abs(P)), rasterized=True)
if plot_zonal:
    axs[1].plot(x[:,0], 0.5*(y[:,-1]+y[:,0])+0.25*(y[:,-1]-y[:,0])*Pbar/np.max(np.abs(Pbar)),'w',linewidth=5)
    axs[1].plot(x[:,0], 0.5*(y[:,-1]+y[:,0])+0.25*(y[:,-1]-y[:,0])*Pbar/np.max(np.abs(Pbar)),'k',label=r'$\overline{P}$')
    axs[1].legend(loc='upper right')
axs[1].set_title('$P$')
axs[1].set_xlabel('$x$')
plt.colorbar(c1, ax=axs[1])
fig.tight_layout(pad=0.5) # Use tight_layout with reduced padding

plt.savefig(savename('fields'), bbox_inches='tight')
plt.show()

#%% Plot: Om

fig, ax = plt.subplots(figsize=figsize_single)
c = ax.pcolormesh(x,y,Om, cmap='seismic', vmin=-np.max(np.abs(Om)), vmax=np.max(np.abs(Om)), rasterized=True)
if plot_zonal:
    ax.plot(x[:,0], 0.5*(y[:,-1]+y[:,0])+0.25*(y[:,-1]-y[:,0])*vbar/np.max(np.abs(vbar)),'w',linewidth=5)
    ax.plot(x[:,0], 0.5*(y[:,-1]+y[:,0])+0.25*(y[:,-1]-y[:,0])*vbar/np.max(np.abs(vbar)),'k',label=r'$\overline{v}_y$')
    ax.legend(loc='upper right')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.colorbar(c, ax=ax)
fig.tight_layout(pad=0.5)
plt.savefig(savename('fields_Om'), bbox_inches='tight')
plt.show()

#%% Plot: P

fig, ax = plt.subplots(figsize=figsize_single)
c = ax.pcolormesh(x,y,P, cmap='seismic', vmin=-np.max(np.abs(P)), vmax=np.max(np.abs(P)), rasterized=True)
if plot_zonal:
    ax.plot(x[:,0], 0.5*(y[:,-1]+y[:,0])+0.25*(y[:,-1]-y[:,0])*Pbar/np.max(np.abs(Pbar)),'w',linewidth=5)
    ax.plot(x[:,0], 0.5*(y[:,-1]+y[:,0])+0.25*(y[:,-1]-y[:,0])*Pbar/np.max(np.abs(Pbar)),'k',label=r'$\overline{P}$')
    ax.legend(loc='upper right')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.colorbar(c, ax=ax)
fig.tight_layout(pad=0.5)
plt.savefig(savename('fields_P'), bbox_inches='tight')
plt.show()
# %%
