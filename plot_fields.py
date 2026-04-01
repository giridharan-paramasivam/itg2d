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

from modules.plot_basics import apply_style, figsize_single, figsize_double
apply_style()

#%% Load the HDF5 file

# Npx=512
Npx=1024
datadir=f'data/{Npx}/'


# fname = datadir + 'out_kapt_0_4_D_0_1_H_3_6_em6.h5'
fname = datadir + 'out_kapt_2_0_D_0_1_H_8_6_em6.h5'
# fname = datadir + 'out_kapt_2_0_D_0_1_H_1_7_em5.h5'

STREAMER = True  # Set to True to pick Qbox_t spike, False for last time index

if STREAMER:
    subdir = 'evol/'
    evol_fname = datadir + subdir + fname.split('/')[-1].replace('out_', 'evol_')
    try:
        with h5.File(evol_fname, 'r') as flevol:
            t_evol = flevol['t'][:]
            Qbox_t = flevol['Qbox_t'][:]
            it = int(np.argmax(Qbox_t))
            t_spike = t_evol[it]
    except Exception as e:
        warnings.warn(f"Could not load Qbox_t from {evol_fname}: {e}\nDefaulting to last time index.")
        it = -1
else:
    it = -1

with h5.File(fname, 'r', swmr=True) as fl:
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
    if STREAMER and it >= 0:
        print(f"Plotting at heat flux spike: t = {t_spike:.3f}, gamma*t = {gammax * t_spike:.3f}, index = {it}")

Nx,Ny=2*Npx//3,2*Npy//3  
sl=Slicelist(Nx,Ny)
slbar=np.s_[int(Ny/2)-1:int(Ny/2)*int(Nx/2)-1:int(Ny/2)]
nt = len(t)
print("nt: ", nt)

xl,yl=np.linspace(0,Lx,Npx),np.linspace(0,Ly,Npy)
x,y=np.meshgrid(np.array(xl),np.array(yl),indexing='ij')

#%% Plots

irft2np = partial(original_irft2np,Npx=Npx,Npy=Npy,Nx=Nx,sl=sl)
rft2np = partial(original_rft2np,sl=sl)
irftnp = partial(original_irftnp,Npx=Npx,Nx=Nx)
rftnp = partial(original_rftnp,Nx=Nx)

Om = irft2np(Omk)
P = irft2np(Pk)

# Plotting function
def plot_colormesh(dat, dat_bar, title, lab_bar, ax):
    c = ax.pcolormesh(x,y,dat, cmap='seismic', vmin=-np.max(np.abs(dat)), vmax=np.max(np.abs(dat)))
    # ax.plot(x[:,0], 0.5*(y[:,-1]+y[:,0])+0.25*(y[:,-1]-y[:,0])*dat_bar/np.max(np.abs(dat_bar)),'w',linewidth=5)
    # ax.plot(x[:,0], 0.5*(y[:,-1]+y[:,0])+0.25*(y[:,-1]-y[:,0])*dat_bar/np.max(np.abs(dat_bar)),'k',label=lab_bar)
    ax.set_title(title)
    ax.set_xlabel('$x$')
    # ax.legend(loc='upper right')
    plt.colorbar(c, ax=ax)

#%% Plot: Om and P
fig, axs = plt.subplots(1, 2, figsize=figsize_double, sharey=True)

# Plot each dataset
plot_colormesh(Om, vbar, r'$\Omega$', r'$\overline{v}_y$', axs[0])
axs[0].set_ylabel('$y$')

plot_colormesh(P, vbar, '$P$', r'$\overline{v}_y$', axs[1])
fig.tight_layout(pad=0.5) # Use tight_layout with reduced padding

# Add bbox_inches='tight' to savefig calls
if fname.endswith('out.h5'):
    plt.savefig(datadir+'fields.svg', dpi=100, bbox_inches='tight')
else:
    plt.savefig(datadir+fname.split('/')[-1].replace('out_', 'fields_').replace('.h5', '.svg'), dpi=100, bbox_inches='tight')
plt.show()

#%% Plot: Om

# fig, ax = plt.subplots(figsize=figsize_single)
# c =ax.pcolormesh(x,y,Om, cmap='seismic', vmin=-np.max(np.abs(Om)), vmax=np.max(np.abs(Om)))
# ax.plot(x[:,0], 0.5*(y[:,-1]+y[:,0])+0.25*(y[:,-1]-y[:,0])*vbar/np.max(np.abs(vbar)),'w',linewidth=5)
# ax.plot(x[:,0], 0.5*(y[:,-1]+y[:,0])+0.25*(y[:,-1]-y[:,0])*vbar/np.max(np.abs(vbar)),'k',label=r'$\overline{v}_y$')
# ax.legend(loc='upper right')
# ax.set_title(rf'$\Omega$ for $\kappa_T={kapt}$')
# ax.set_xlabel('$x$')
# ax.set_ylabel('$y$')
# plt.colorbar(c, ax=ax)
# fig.tight_layout(pad=0.5)
# plt.savefig(datadir+fname.split('/')[-1].replace('out_', 'fields_Om_').replace('.h5', '.svg'), bbox_inches='tight')
# plt.show()

#%% Plot: P
# fig, ax = plt.subplots(figsize=figsize_single)
# plot_colormesh(P, Pbar, '$P$', r'$\overline{P}$', ax)
# fig.tight_layout(pad=0.5)
# plt.savefig(datadir+fname.split('/')[-1].replace('out_', 'fields_P_').replace('.h5', '.svg'), bbox_inches='tight')
# plt.show()