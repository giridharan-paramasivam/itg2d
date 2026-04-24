#%% Import modules
import gc
import os
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import torch 
import h5py
from modules.plot_basics import apply_style, figsize_single
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

kapt_th_idx = np.argmax(gammax_kapt > 0)
kapt_th = kapt_vals[kapt_th_idx] if gammax_kapt[kapt_th_idx] > 0 else None

plt.figure(figsize=figsize_single)
plt.plot(kapt_vals, gammax_kapt)
plt.xlim(right=1.61)
if kapt_th is not None:
    plt.axvline(kapt_th, color='k', linestyle='--', lw=1.5, label = rf'$\kappa_{{T,\mathrm{{lin}}}}$={kapt_th:.2f}')
plt.xlabel(r'$\kappa_T$')
plt.ylabel(r'$\gamma_{max}$')
plt.title(rf"$\gamma_{{max}}$ for $\kappa_n$={kapn:.2f} $\kappa_B$={kapb:.2f}")
plt.legend()
plt.savefig(fname.replace('gammax_vals_kapt', 'gammax_kapt').replace('.h5', '.svg'), bbox_inches='tight')
plt.tight_layout()
plt.show()