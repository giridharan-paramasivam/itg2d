#%% Import modules
import gc
import os
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import torch 
import h5py

from modules.plot_basics import apply_style
apply_style()

#%% Initialize

datadir='data_linear/'
os.makedirs(datadir, exist_ok=True)

# Load datasets
with h5py.File(datadir + 'gammax_vals_kapt_kz_scan_itg2d3c.h5', 'r') as fl:
    gammax_kapt_kz = fl['gammax_vals'][:]
    kapt_vals = fl['kapt_vals'][:]
    kz_vals = fl['kz_vals'][:]
    kapb = fl['kapb'][()]

#%% Colormesh of gam(kapt,kz)

Kapt, Kz = np.meshgrid(kapt_vals, kz_vals)
plt.figure()
plt.pcolormesh(Kapt, Kz, gammax_kapt_kz.T, vmax=1.0, vmin=-1.0, cmap='seismic', rasterized=True, shading='auto')
plt.xlabel(r'$\kappa_T$')
plt.ylabel('$k_z$')
plt.title(rf"$\gamma_{{max}}$ for $\kappa_B$={kapb:.2f}")
plt.colorbar()
plt.savefig(datadir + 'gammax_kapt_kz_itg2d3c.png', dpi=100)
plt.show()
del gammax_kapt_kz