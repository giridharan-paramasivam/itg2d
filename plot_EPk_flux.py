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
    k_Pf      = float(fl['k_Pf'][()])
    k_lin     = float(fl['k_lin'][()])
    PiPk      = fl['PiPk'][:]
    fPk       = fl['fPk'][:]
    dPk       = fl['dPk'][:]

#%% Functions

savename = partial(_savename, datadir+subdir, fname)

#%% Pk-flux

plt.figure(figsize=figsize_single)
plt.plot(k[1:-1], PiPk[1:-1], label = r'$\Pi_{P,k}$')
plt.axhline(0,color='k', linestyle='-', linewidth=1)
plt.axvline(x=1, color='k', linestyle='--', linewidth=2)
plt.axvline(x=k_Pf, color='k', linestyle=':', linewidth=2)
plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2)
ymin, ymax = plt.ylim()
offset = 0.025 * (ymax - ymin)
plt.text(k_Pf, ymin - offset, r'$k_{P,f}$', ha='center', va='top', fontsize=xtick_fontsize)
plt.text(k_lin, ymin - offset, r'$k_{\mathrm{lin}}$', ha='center', va='top', fontsize=xtick_fontsize)
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
plt.axvline(x=1, color='k', linestyle='--', linewidth=2)
plt.axvline(x=k_Pf, color='k', linestyle=':', linewidth=2)
plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2)
ymin, ymax = plt.ylim()
offset = 0.025 * (ymax - ymin)
plt.text(k_Pf, ymin - offset, r'$k_{P,f}$', ha='center', va='top', fontsize=xtick_fontsize)
plt.text(k_lin, ymin - offset, r'$k_{\mathrm{lin}}$', ha='center', va='top', fontsize=xtick_fontsize)
plt.xscale('log')
plt.xlabel('$k$')
plt.ylabel(r'$f_{P,k}$')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.tight_layout()
plt.savefig(savename('EPk_injection'), bbox_inches='tight')
plt.show()

#%% Pk-flux: dissipation

plt.figure(figsize=figsize_single)
plt.plot(k[1:-1], dPk[1:-1], label=r'$d_{P,k}$')
plt.axhline(0,color='k', linestyle='-', linewidth=1)
plt.axvline(x=1, color='k', linestyle='--', linewidth=2)
plt.axvline(x=k_Pf, color='k', linestyle=':', linewidth=2)
plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2)
ymin, ymax = plt.ylim()
offset = 0.025 * (ymax - ymin)
plt.text(k_Pf, ymin - offset, r'$k_{P,f}$', ha='center', va='top', fontsize=xtick_fontsize)
plt.text(k_lin, ymin - offset, r'$k_{\mathrm{lin}}$', ha='center', va='top', fontsize=xtick_fontsize)
plt.xscale('log')
plt.xlabel('$k$')
plt.ylabel(r'$d_{P,k}$')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.tight_layout()
plt.savefig(savename('EPk_dissipation'), bbox_inches='tight')
plt.show()