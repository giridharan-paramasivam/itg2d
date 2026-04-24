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

kapb=0.02
fname = datadir + f'lin_kapn_kapt_scan_kapb_{str(kapb).replace(".", "_")}_itg2d.h5'
base_name = fname.replace(datadir+'lin_', '').replace('_scan', '').replace('.h5', '.svg')

# Load datasets
with h5py.File(fname, 'r') as fl:
    gammax_kapn_kapt = fl['gammax_vals'][:]
    Dturbmax_kapn_kapt = fl['Dturbmax_vals'][:]
    kapn_vals = fl['kapn_vals'][:]
    kapt_vals = fl['kapt_vals'][:]
    kapb = fl['kapb'][()]

Kapn, Kapt = np.meshgrid(kapn_vals, kapt_vals)
zero_or_negative = gammax_kapn_kapt <= 0
kapn_zero = kapn_vals[zero_or_negative.any(axis=0)]
kapt_zero = kapt_vals[zero_or_negative.any(axis=1)]
# print(f"kapn values where gamma <= 0: {kapn_zero}")
# print(f"kapt values where gamma <= 0: {kapt_zero}")

# Compute kapt_thresh
kapt_thresh = np.full(len(kapn_vals), np.nan)
for i, _ in enumerate(kapn_vals):
    gam_slice = gammax_kapn_kapt[i, :]
    sign_changes = np.where(np.diff(np.sign(gam_slice)))[0]
    if len(sign_changes) > 0:
        idx = sign_changes[0]
        g0, g1 = gam_slice[idx], gam_slice[idx + 1]
        k0, k1 = kapt_vals[idx], kapt_vals[idx + 1]
        kapt_thresh[i] = k0 - g0 * (k1 - k0) / (g1 - g0)

#%% Plot: Colormesh of gam(kapn,kapt)

plt.figure(figsize=figsize_single)
gammax_vmax = np.max(np.abs(gammax_kapn_kapt))
im_gam = plt.pcolormesh(Kapn, Kapt, gammax_kapn_kapt.T, vmax=gammax_vmax, vmin=-gammax_vmax, cmap='seismic', rasterized=True, shading='auto')
plt.contour(Kapn, Kapt, gammax_kapn_kapt.T, levels=[0.0], colors='k', linewidths=2)
plt.plot([], [], color='k', linewidth=2, label=r"$\gamma=0$")
kapn_curve = kapn_vals**2 / (4 * kapb) - kapn_vals
kapn_mask = (kapn_vals < 10 * kapb) & (kapn_curve <= np.max(kapt_vals))
plt.plot(kapn_vals[kapn_mask], kapn_curve[kapn_mask], label=r"$\kappa_T=\kappa_n^2/4\kappa_B - \kappa_n$", color='k', linestyle='--', linewidth=2)
plt.axhline(y=0, linewidth=1, color='black')
plt.axvline(x=0, linewidth=1, color='black')
plt.xlim((-0.4, np.max(kapn_vals)))
plt.ylim((-0.4, np.max(kapt_vals)))    
plt.xlabel(r'$\kappa_n$')
plt.ylabel(r'$\kappa_T$')
plt.legend(loc='lower left',fontsize=24)
plt.colorbar(im_gam)
plt.tight_layout()
plt.savefig(datadir + fname.replace(datadir+'lin_', 'gammax_').replace('.h5', '.svg'), bbox_inches='tight')
plt.show()

#%% Plot: Colormesh of Dturb(kapn,kapt)

plt.figure(figsize=figsize_single)
Dturbmax_vmax = np.max(np.abs(Dturbmax_kapn_kapt))
im_dturb = plt.pcolormesh(Kapn, Kapt, Dturbmax_kapn_kapt.T, vmax=Dturbmax_vmax, vmin=-Dturbmax_vmax, cmap='seismic', rasterized=True, shading='auto')
plt.contour(Kapn, Kapt, Dturbmax_kapn_kapt.T, levels=[0.0], colors='k', linewidths=2)
plt.plot([], [], color='k', linewidth=2, label=r"$D_\mathrm{turb}=0$")
plt.axhline(y=0, linewidth=1, color='black')
plt.axvline(x=0, linewidth=1, color='black')
plt.xlim((-0.4, np.max(kapn_vals)))
plt.ylim((-0.4, np.max(kapt_vals))) 
plt.xlabel(r'$\kappa_n$')
plt.ylabel(r'$\kappa_T$')
plt.legend(loc='lower left',fontsize=24)
plt.colorbar(im_dturb)
plt.tight_layout()
plt.savefig(datadir + fname.replace(datadir+'lin_', 'Dturbmax_').replace('.h5', '.svg'), bbox_inches='tight')
# plt.show()
