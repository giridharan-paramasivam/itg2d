#%% Importing libraries
import numpy as np
import h5py as h5
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from functools import partial
from modules.plot_basics import apply_style, savename as _savename, figsize_single

apply_style()
xtick_fontsize = matplotlib.rcParams.get('xtick.labelsize', 32)

#%% Load computed flux data

# Npx=512
Npx=1024
datadir=f'data/{Npx}/'
subdir='spectral_flux/'

fname = 'out_kapt_0_4_D_0_1_H_3_6_em6.h5'
# fname = 'out_kapt_2_0_D_0_1_H_8_6_em6.h5'
# fname = 'out_kapt_2_0_D_0_1_H_1_7_em5.h5'

flux_file = datadir + subdir + fname.replace('out_', 'spectral_flux_')
with h5.File(flux_file, 'r') as fl:
    k          = fl['k'][:]
    k_f_G      = float(fl['k_f_G'][()])
    k_lin      = float(fl['k_lin'][()])
    k_D_G      = float(fl['k_D_G'][()])
    PiGk       = fl['PiGk'][:]
    PiGk_P     = fl['PiGk_P'][:]
    PiGk_phi   = fl['PiGk_phi'][:]
    PiGk_d     = fl['PiGk_d'][:]
    fGk        = fl['fGk'][:]
    dGk_D      = fl['dGk_D'][:]
    dGk_H      = fl['dGk_H'][:]
    PiGk_P_t   = fl['PiGk_P_t'][:]
    PiGk_phi_t = fl['PiGk_phi_t'][:]
    PiGk_d_t   = fl['PiGk_d_t'][:]

#%% Functions

savename = partial(_savename, datadir+subdir, fname)

#%% Derived quantities for PDFs

idx_k_f_G = np.argmin(np.abs(k - k_f_G))
PiGk_P_series   = PiGk_P_t[:, idx_k_f_G]
PiGk_phi_series = PiGk_phi_t[:, idx_k_f_G]
PiGk_d_series   = PiGk_d_t[:, idx_k_f_G]
PiGk_series     = PiGk_P_series + PiGk_phi_series + PiGk_d_series

PiGk_P_series_norm   = (PiGk_P_series - np.mean(PiGk_P_series)) / np.std(PiGk_P_series)
PiGk_phi_series_norm = (PiGk_phi_series - np.mean(PiGk_phi_series)) / np.std(PiGk_phi_series)
PiGk_d_series_norm   = (PiGk_d_series - np.mean(PiGk_d_series)) / np.std(PiGk_d_series)
PiGk_series_norm     = (PiGk_series - np.mean(PiGk_series)) / np.std(PiGk_series)

idx_k_lin = np.argmin(np.abs(k - k_lin))
PiGk_P_series_max   = PiGk_P_t[:, idx_k_lin]
PiGk_phi_series_max = PiGk_phi_t[:, idx_k_lin]
PiGk_d_series_max   = PiGk_d_t[:, idx_k_lin]
PiGk_series_max     = PiGk_P_series_max + PiGk_phi_series_max + PiGk_d_series_max

PiGk_P_series_max_norm   = (PiGk_P_series_max - np.mean(PiGk_P_series_max)) / np.std(PiGk_P_series_max)
PiGk_phi_series_max_norm = (PiGk_phi_series_max - np.mean(PiGk_phi_series_max)) / np.std(PiGk_phi_series_max)
PiGk_d_series_max_norm   = (PiGk_d_series_max - np.mean(PiGk_d_series_max)) / np.std(PiGk_d_series_max)
PiGk_series_max_norm     = (PiGk_series_max - np.mean(PiGk_series_max)) / np.std(PiGk_series_max)

idx_k_1 = np.argmin(np.abs(k - 1))
PiGk_P_series_1   = PiGk_P_t[:, idx_k_1]
PiGk_phi_series_1 = PiGk_phi_t[:, idx_k_1]
PiGk_d_series_1   = PiGk_d_t[:, idx_k_1]
PiGk_series_1     = PiGk_P_series_1 + PiGk_phi_series_1 + PiGk_d_series_1

PiGk_P_series_1_norm   = (PiGk_P_series_1 - np.mean(PiGk_P_series_1)) / np.std(PiGk_P_series_1)
PiGk_phi_series_1_norm = (PiGk_phi_series_1 - np.mean(PiGk_phi_series_1)) / np.std(PiGk_phi_series_1)
PiGk_d_series_1_norm   = (PiGk_d_series_1 - np.mean(PiGk_d_series_1)) / np.std(PiGk_d_series_1)
PiGk_series_1_norm     = (PiGk_series_1 - np.mean(PiGk_series_1)) / np.std(PiGk_series_1)

savename = partial(_savename, datadir+subdir, fname)

#%% Gk-flux

plt.figure(figsize=figsize_single)
plt.plot(k[1:-1], PiGk[1:-1],     label=r'$\Pi_{G,k}$')
plt.plot(k[1:-1], PiGk_phi[1:-1], label=r'$\Pi_{G,k}^{\left(\phi\right)}$')
plt.plot(k[1:-1], PiGk_d[1:-1],   label=r'$\Pi_{G,k}^{\left(d\right)}$')
plt.plot(k[1:-1], PiGk_P[1:-1],   label=r'$\Pi_{G,k}^{\left(P\right)}$')
plt.axhline(0, color='k', linestyle='-', linewidth=1)
plt.axvline(x=1,     color='k', linestyle='--', linewidth=2)
plt.axvline(x=k_f_G,  color='k', linestyle=':',  linewidth=2)
plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2)
ymin, ymax = plt.ylim()
offset = 0.025 * (ymax - ymin)
plt.text(k_f_G, ymin - offset, r'$k_{G,f}$', ha='center', va='top', fontsize=xtick_fontsize)
plt.text(k_lin, ymin - offset, r'$k_{\mathrm{lin}}$', ha='center', va='top', fontsize=xtick_fontsize)
plt.xscale('log')
plt.xlabel('$k$')
plt.ylabel(r'$\Pi_{G,k}$')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.legend()
plt.tight_layout()
plt.savefig(savename('Gk_flux'), bbox_inches='tight')
plt.show()

#%% Gk-flux: injection

plt.figure(figsize=figsize_single)
plt.plot(k[1:-1], fGk[1:-1], label=r'$f_{G,k}$')
plt.axhline(0, color='k', linestyle='-', linewidth=1)
plt.axvline(x=1, color='k', linestyle='--', linewidth=2)
plt.axvline(x=k_f_G, color='k', linestyle=':', linewidth=2)
plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2)
ymin, ymax = plt.ylim()
offset = 0.025 * (ymax - ymin)
plt.text(k_f_G, ymin - offset, r'$k_{G,f}$', ha='center', va='top', fontsize=xtick_fontsize)
plt.text(k_lin, ymin - offset, r'$k_{\mathrm{lin}}$', ha='center', va='top', fontsize=xtick_fontsize)
plt.xscale('log')
plt.xlabel('$k$')
plt.ylabel(r'$f_{G,k}$')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.legend()
plt.tight_layout()
plt.savefig(savename('Gk_injection'), bbox_inches='tight')
plt.show()

#%% Gk-flux: dissipation

plt.figure(figsize=figsize_single)
plt.plot(k[1:-1], dGk_D[1:-1], label=r'$d_{G,k}^{(D)}$')
plt.plot(k[1:-1], dGk_H[1:-1], label=r'$d_{G,k}^{(H)}$')
plt.axhline(0, color='k', linestyle='-', linewidth=1)
plt.axvline(x=1, color='k', linestyle='--', linewidth=2)
plt.axvline(x=k_f_G, color='k', linestyle=':', linewidth=2)
plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2)
plt.axvline(x=k_D_G, color='k', linestyle=(0, (3, 1, 1, 1, 1, 1)), linewidth=2)
ymin, ymax = plt.ylim()
offset = 0.025 * (ymax - ymin)
plt.text(k_f_G, ymin - offset, r'$k_{G,f}$', ha='center', va='top', fontsize=xtick_fontsize)
plt.text(k_lin, ymin - offset, r'$k_{\mathrm{lin}}$', ha='center', va='top', fontsize=xtick_fontsize)
plt.text(k_D_G, ymin - offset, r'$k_D$', ha='center', va='top', fontsize=xtick_fontsize)
plt.xscale('log')
plt.xlabel('$k$')
plt.ylabel(r'$d_{G,k}$')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.legend()
plt.tight_layout()
plt.savefig(savename('Gk_dissipation'), bbox_inches='tight')
plt.show()