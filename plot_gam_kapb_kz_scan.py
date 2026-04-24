#%% Import modules
import gc
import os
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import torch 
import h5py

from modules.plot_basics import apply_style, FIGSIZE_DOUBLE, FIGSIZE_SINGLE
apply_style()

#%% Initialize

datadir='data_linear/'
os.makedirs(datadir, exist_ok=True)

# Load datasets
with h5py.File(datadir + 'gammax_vals_kapb_kz_scan_itg2d3c.h5', 'r') as fl:
    gammax_kapb_kz = fl['gammax_vals'][:]
    kapb_vals = fl['kapb_vals'][:]
    kz_vals = fl['kz_vals'][:]
    kapt = fl['kapt'][()]

#%% Colormesh of gam(kapt,kz)

Kapb, Kz = np.meshgrid(kapb_vals, kz_vals)
plt.figure(figsize=FIGSIZE_DOUBLE)
plt.pcolormesh(Kapb, Kz, gammax_kapb_kz.T, vmax=1.0, vmin=-1.0, cmap='seismic', rasterized=True, shading='auto')
plt.xlabel(r'$\kappa_B$')
plt.ylabel('$k_z$')
plt.title(rf"$\gamma_{{max}}$ for $\kappa_T$={kapt:.2f}")
plt.colorbar()
plt.tight_layout()
plt.savefig(datadir + 'gammax_kapb_kz_itg2d3c.png', dpi=100)
plt.show()
del gammax_kapb_kz
