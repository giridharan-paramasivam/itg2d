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

Npx=512
# Npx=1024
datadir=f'data/{Npx}/'
subdir = 'spectral_flux/'

# fname = 'out_kapt_0_4_D_0_1_H_3_6_em6.h5'
# fname =  'out_kapt_2_0_D_0_1_H_0_0_e0.h5'
fname = 'out_kapt_2_0_D_0_1_H_8_6_em6.h5'
# fname = 'out_kapt_0_2_hyper_D_1_0_em5_H_8_0_em6.h5'
# fname = 'out_kapt_2_0_hyper_D_5_0_em6_H_1_1_em5.h5'

flux_file = datadir + subdir + fname.replace('out_', 'spectral_flux_')
with h5.File(flux_file, 'r') as fl:
    k         = fl['k'][:]
    k_f_E       = float(fl['k_f_E'][()])
    k_lin     = float(fl['k_lin'][()])
    k_D_E       = float(fl['k_D_E'][()])
    Pik       = fl['Pik'][:]
    Pik_phi   = fl['Pik_phi'][:]
    Pik_d     = fl['Pik_d'][:]
    fk        = fl['fk'][:]
    dk_D      = fl['dk_D'][:]
    dk_H      = fl['dk_H'][:]
    Pik_phi_t = fl['Pik_phi_t'][:]
    Pik_d_t   = fl['Pik_d_t'][:]

#%% Functions

savename = partial(_savename, datadir+subdir, fname)

#%% Derived quantities for PDFs

idx_k_f_E = np.argmin(np.abs(k - k_f_E))
Pik_phi_series = Pik_phi_t[:, idx_k_f_E]
Pik_d_series   = Pik_d_t[:, idx_k_f_E]
Pik_series     = Pik_phi_series + Pik_d_series

Pik_phi_series_norm = (Pik_phi_series - np.mean(Pik_phi_series)) / np.std(Pik_phi_series)
Pik_d_series_norm   = (Pik_d_series - np.mean(Pik_d_series)) / np.std(Pik_d_series)
Pik_series_norm     = (Pik_series - np.mean(Pik_series)) / np.std(Pik_series)

idx_k_lin = np.argmin(np.abs(k - k_lin))
Pik_phi_series_max = Pik_phi_t[:, idx_k_lin]
Pik_d_series_max   = Pik_d_t[:, idx_k_lin]
Pik_series_max     = Pik_phi_series_max + Pik_d_series_max

Pik_phi_series_max_norm = (Pik_phi_series_max - np.mean(Pik_phi_series_max)) / np.std(Pik_phi_series_max)
Pik_d_series_max_norm   = (Pik_d_series_max - np.mean(Pik_d_series_max)) / np.std(Pik_d_series_max)
Pik_series_max_norm     = (Pik_series_max - np.mean(Pik_series_max)) / np.std(Pik_series_max)

idx_k_1 = np.argmin(np.abs(k - 1))
Pik_phi_series_1 = Pik_phi_t[:, idx_k_1]
Pik_d_series_1   = Pik_d_t[:, idx_k_1]
Pik_series_1     = Pik_phi_series_1 + Pik_d_series_1

Pik_phi_series_1_norm = (Pik_phi_series_1 - np.mean(Pik_phi_series_1)) / np.std(Pik_phi_series_1)
Pik_d_series_1_norm   = (Pik_d_series_1 - np.mean(Pik_d_series_1)) / np.std(Pik_d_series_1)
Pik_series_1_norm     = (Pik_series_1 - np.mean(Pik_series_1)) / np.std(Pik_series_1)

# Cumulative integral: Pi(k) = int_0^k f(k') dk' = cumsum(f*dk)
# cumsum[j] = Pi(k[j+1]), so [:-2] aligns with the plot x-axis k[1:-1]
# k[0]=0 shell is included; last shell [:-1] is excluded (may have partial points)
dk = np.gradient(k)
cumsum_fk = np.cumsum(fk * dk)[:-2]
cumsum_dk_D = np.cumsum(dk_D * dk)[:-2]
cumsum_dk_H = np.cumsum(dk_H * dk)[:-2]

#%% Ek-flux

plt.figure(figsize=figsize_single)
plt.plot(k[1:-1], Pik[1:-1], label = r'$\Pi_{k}$')
plt.plot(k[1:-1], Pik_phi[1:-1], label = r'$\Pi_{k}^{\left(\phi\right)}$')
plt.plot(k[1:-1], Pik_d[1:-1], label = r'$\Pi_{k}^{\left(d\right)}$')
plt.axhline(0,color='k', linestyle='-', linewidth=1)
plt.axvline(x=k_f_E, color='k', linestyle=':', linewidth=2)
plt.axvline(x=k_D_E, color='k', linestyle=(0, (3, 1, 1, 1, 1, 1)), linewidth=2)
plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2)
ax = plt.gca()
ax.text(k_f_E, 1.01, r'$k_f$', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=xtick_fontsize)
ax.text(k_D_E, 1.01, r'$k_D$', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=xtick_fontsize)
ax.text(k_lin, 1.01, r'$k_{\mathrm{lin}}$', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=xtick_fontsize)
plt.xscale('log')
plt.xlabel('$k$')
plt.ylabel(r'$\Pi_k$')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.legend()
plt.tight_layout()
plt.savefig(savename('Ek_flux'), bbox_inches='tight')
plt.show()

#%% Ek-flux with cumsum of dissipation

plt.figure(figsize=figsize_single)
plt.plot(k[1:-1], Pik[1:-1], label = r'$\Pi_{k}$')
plt.plot(k[1:-1], cumsum_fk, label = r'$\mathcal{F}_k$')
plt.plot(k[1:-1], -cumsum_dk_D, label = r'$-\mathcal{D}_k^{(D)}$')
plt.plot(k[1:-1], -cumsum_dk_H, label = r'$-\mathcal{D}_k^{(H)}$')
plt.plot(k[1:-1], cumsum_fk-cumsum_dk_D-cumsum_dk_H, label = r'$\mathcal{F}_k - \mathcal{D}_k$')
plt.axhline(0,color='k', linestyle='-', linewidth=1)
plt.axvline(x=k_f_E, color='k', linestyle=':', linewidth=2)
plt.axvline(x=k_D_E, color='k', linestyle=(0, (3, 1, 1, 1, 1, 1)), linewidth=2)
plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2)
ax = plt.gca()
ax.text(k_f_E, 1.01, r'$k_f$', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=xtick_fontsize)
ax.text(k_D_E, 1.01, r'$k_D$', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=xtick_fontsize)
ax.text(k_lin, 1.01, r'$k_{\mathrm{lin}}$', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=xtick_fontsize)
plt.xscale('log')
plt.xlabel('$k$')
plt.ylabel(r'$\Pi_k$')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.legend(fontsize=20)
plt.tight_layout()
plt.savefig(savename('Ek_flux_with_dissipation'), bbox_inches='tight')
plt.show()

#%% Ek-flux: injection

plt.figure(figsize=figsize_single)
plt.plot(k[1:-1], fk[1:-1])
plt.axhline(0,color='k', linestyle='-', linewidth=1)
plt.axvline(x=k_f_E, color='k', linestyle=':', linewidth=2)
plt.axvline(x=k_D_E, color='k', linestyle=(0, (3, 1, 1, 1, 1, 1)), linewidth=2)
plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2)
ax = plt.gca()
ax.text(k_f_E, 1.01, r'$k_f$', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=xtick_fontsize)
ax.text(k_D_E, 1.01, r'$k_D$', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=xtick_fontsize)
ax.text(k_lin, 1.01, r'$k_{\mathrm{lin}}$', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=xtick_fontsize)
plt.xscale('log')
plt.xlabel('$k$')
plt.ylabel(r'$f_k$')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.tight_layout()
plt.savefig(savename('Ek_injection'), bbox_inches='tight')
plt.show()

#%% Ek-flux: dissipation

plt.figure(figsize=figsize_single)
plt.plot(k[1:-1], dk_D[1:-1], label=r'$d_k^{(D)}$')
plt.plot(k[1:-1], dk_H[1:-1], label=r'$d_k^{(H)}$')
plt.axhline(0,color='k', linestyle='-', linewidth=1)
plt.axvline(x=k_f_E, color='k', linestyle=':', linewidth=2)
plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2)
plt.axvline(x=k_D_E, color='k', linestyle=(0, (3, 1, 1, 1, 1, 1)), linewidth=2)
ax = plt.gca()
ax.text(k_f_E, 1.01, r'$k_f$', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=xtick_fontsize)
ax.text(k_lin, 1.01, r'$k_{\mathrm{lin}}$', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=xtick_fontsize)
ax.text(k_D_E, 1.01, r'$k_D$', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=xtick_fontsize)
plt.xscale('log')
plt.xlabel('$k$')
plt.ylabel(r'$d_k$')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.legend()
plt.tight_layout()
plt.savefig(savename('Ek_dissipation'), bbox_inches='tight')
plt.show()
# %%
