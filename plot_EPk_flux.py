#%% Importing libraries
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
subdir = 'spectral_flux/'

# fname = 'out_kapt_0_4_D_0_1_H_3_6_em6.h5'
fname = 'out_kapt_2_0_D_0_1_H_8_6_em6.h5'
# fname = 'out_kapt_2_0_D_0_1_H_1_7_em5.h5'

flux_file = datadir + subdir + fname.replace('out_', 'spectral_flux_')
with h5.File(flux_file, 'r') as fl:
    k         = fl['k'][:]
    k_f_P     = float(fl['k_f_P'][()])
    k_lin     = float(fl['k_lin'][()])
    k_D_P     = float(fl['k_D_P'][()])
    PiPk      = fl['PiPk'][:]
    fPk       = fl['fPk'][:]
    dPk_D     = fl['dPk_D'][:]
    dPk_H     = fl['dPk_H'][:]

#%% Functions

savename = partial(_savename, datadir+subdir, fname)

#%% Pk-flux

plt.figure(figsize=figsize_single)
plt.plot(k[1:-1], PiPk[1:-1], label = r'$\Pi_{P,k}$')
plt.axhline(0,color='k', linestyle='-', linewidth=1)
plt.axvline(x=k_f_P, color='k', linestyle=':', linewidth=2)
plt.axvline(x=k_D_P, color='k', linestyle=(0, (3, 1, 1, 1, 1, 1)), linewidth=2)
plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2)
ax = plt.gca()
ax.text(k_f_P, 1.01, r'$k_{P,f}$', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=xtick_fontsize)
ax.text(k_D_P, 1.01, r'$k_D$', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=xtick_fontsize)
ax.text(k_lin, 1.01, r'$k_{\mathrm{lin}}$', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=xtick_fontsize)
plt.xscale('log')
plt.xlabel('$k$')
plt.ylabel(r'$\Pi_{P,k}$')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.tight_layout()
plt.savefig(savename('EPk_flux'), bbox_inches='tight')
plt.show()

#%% Pk-flux: injection

plt.figure(figsize=figsize_single)
plt.plot(k[1:-1], fPk[1:-1], label=r'$f_{P,k}$')
plt.axhline(0,color='k', linestyle='-', linewidth=1)
plt.axvline(x=k_f_P, color='k', linestyle=':', linewidth=2)
plt.axvline(x=k_D_P, color='k', linestyle=(0, (3, 1, 1, 1, 1, 1)), linewidth=2)
plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2)
ax = plt.gca()
ax.text(k_f_P, 1.01, r'$k_{P,f}$', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=xtick_fontsize)
ax.text(k_D_P, 1.01, r'$k_D$', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=xtick_fontsize)
ax.text(k_lin, 1.01, r'$k_{\mathrm{lin}}$', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=xtick_fontsize)
plt.xscale('log')
plt.xlabel('$k$')
plt.ylabel(r'$f_{P,k}$')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.tight_layout()
plt.savefig(savename('EPk_injection'), bbox_inches='tight')
plt.show()

#%% Pk-flux: dissipation

plt.figure(figsize=figsize_single)
plt.plot(k[1:-1], dPk_D[1:-1], label=r'$d_{P,k}^{(D)}$')
plt.plot(k[1:-1], dPk_H[1:-1], label=r'$d_{P,k}^{(H)}$')
plt.axhline(0,color='k', linestyle='-', linewidth=1)
plt.axvline(x=k_f_P, color='k', linestyle=':', linewidth=2)
plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2)
plt.axvline(x=k_D_P, color='k', linestyle=(0, (3, 1, 1, 1, 1, 1)), linewidth=2)
ax = plt.gca()
ax.text(k_f_P, 1.01, r'$k_{P,f}$', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=xtick_fontsize)
ax.text(k_lin, 1.01, r'$k_{\mathrm{lin}}$', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=xtick_fontsize)
ax.text(k_D_P, 1.01, r'$k_D$', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=xtick_fontsize)
plt.xscale('log')
plt.xlabel('$k$')
plt.ylabel(r'$d_{P,k}$')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.tight_layout()
plt.savefig(savename('EPk_dissipation'), bbox_inches='tight')
plt.show()