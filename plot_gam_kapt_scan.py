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

kapn=0.2 
kapb=0.02
# Load datasets
fname = datadir+f'gammax_vals_kapt_scan_kapn_{str(kapn).replace(".", "_")}_kapb_{str(kapb).replace(".", "_")}_itg2d.h5'
with h5py.File(fname, 'r') as fl:
    gammax_kapt = fl['gammax_vals'][:]
    kapt_vals = fl['kapt_vals'][:]
    kapb = fl['kapb'][()]

#%% Colormesh of gam(kapt,kz)

plt.figure(figsize=FIGSIZE_DOUBLE)
plt.plot(kapt_vals, gammax_kapt)
plt.xlabel(r'$\kappa_T$')
plt.ylabel(r'$\gamma_{max}$')
plt.title(rf"$\gamma_{{max}}$ for $\kappa_n$={kapn:.2f} $\kappa_B$={kapb:.2f}")
plt.savefig(fname.replace('gammax_vals_kapt', 'gammax_kapt').replace('.h5', '.png'), dpi=100)
plt.tight_layout()
plt.show()