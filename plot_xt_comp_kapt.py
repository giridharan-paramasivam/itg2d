#%% Importing libraries
import h5py as h5
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from modules.mlsarray import MLSarray,Slicelist,irft2np,rft2np,irftnp,rftnp
from modules.gamma import gam_max
from modules.basics import format_exp
from modules.plot_basics import apply_style, savename as _savename, figsize_double
from functools import partial
apply_style()

#%% Load the HDF5 files

Npx=1024
datadir=f'data/{Npx}/'
subdir = 'xt_maps/'

fname1 = datadir + 'out_kapt_0_4_D_0_1_H_3_6_em6.h5'
fname2 = datadir + 'out_kapt_2_0_D_0_1_H_8_6_em6.h5'

stride = 10

def load_params(fname):
    with h5.File(fname, 'r', swmr=True) as fl:
        t    = fl['fluxes/t'][::stride].astype(np.float32)
        kx   = fl['data/kx'][:]
        ky   = fl['data/ky'][:]
        Lx   = fl['params/Lx'][()]
        Npx_ = fl['params/Npx'][()]
        kapn = fl['params/kapn'][()]
        kapt = fl['params/kapt'][()]
        kapb = fl['params/kapb'][()]
        D    = fl['params/D'][()]
        if 'H' in fl['params']:
            H = fl['params/H'][()]
        elif 'HP' in fl['params']:
            H = fl['params/HP'][()]
    return t, kx, ky, Lx, Npx_, kapn, kapt, kapb, D, H

t1, kx1, ky1, Lx1, Npx1, kapn1, kapt1, kapb1, D1, H1 = load_params(fname1)
t2, kx2, ky2, Lx2, Npx2, kapn2, kapt2, kapb2, D2, H2 = load_params(fname2)

gammax1 = gam_max(kx1, ky1, kapn1, kapt1, kapb1, D1, H1)
gammax2 = gam_max(kx2, ky2, kapn2, kapt2, kapb2, D2, H2)

t1 = t1 * gammax1
t2 = t2 * gammax2

xl1 = np.arange(0, Lx1, Lx1/Npx1)
xl2 = np.arange(0, Lx2, Lx2/Npx2)

xm1, tm1 = np.meshgrid(xl1, t1)
xm2, tm2 = np.meshgrid(xl2, t2)

savename = partial(_savename, datadir+subdir, fname1)

#%% Plot: vbar comparison

with h5.File(fname1, 'r', swmr=True) as fl:
    vbar1 = fl['zonal/vbar'][::stride].astype(np.float32)

with h5.File(fname2, 'r', swmr=True) as fl:
    vbar2 = fl['zonal/vbar'][::stride].astype(np.float32)

vbar1_lim = float(np.percentile(np.abs(vbar1), 75))
vbar2_lim = float(np.percentile(np.abs(vbar2), 75))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize_double, sharey=True)

im1 = ax1.pcolormesh(xm1, tm1, vbar1, vmin=-vbar1_lim, vmax=vbar1_lim, cmap='seismic', rasterized=True)
ax1.set_xlabel('x')
ax1.set_ylabel(r'$\gamma t$')
ax1.set_title(rf'$\kappa_T={kapt1}$')

im2 = ax2.pcolormesh(xm2, tm2, vbar2, vmin=-vbar2_lim, vmax=vbar2_lim, cmap='seismic', rasterized=True)
ax2.set_xlabel('x')
ax2.set_title(rf'$\kappa_T={kapt2}$')

fig.colorbar(im1, ax=ax1)
fig.colorbar(im2, ax=ax2)
plt.tight_layout()
plt.savefig(datadir+subdir+'vbar_xt_kapt_comp.svg', bbox_inches='tight')
plt.show()
plt.close()
del vbar1, vbar2

# %%
