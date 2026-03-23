#%% Importing libraries
import numpy as np
import h5py as h5
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

from modules.plot_basics import apply_style, savename as _savename
from functools import partial
apply_style()

#%% Load computed flux data

Npx=1024
datadir=f'data/{Npx}/'

fname = datadir + 'out_kapt_0_4_D_0_1_H_3_6_em6.h5'
# fname = datadir + 'out_kapt_2_0_D_0_1_H_8_6_em6.h5'
# fname = datadir + 'out_kapt_2_0_D_0_1_H_1_7_em5.h5'

flux_file = fname.replace('out_', 'energy_flux_')
with h5.File(flux_file, 'r') as fl:
    k          = fl['k'][:]
    k_f        = float(fl['k_f'][()])
    k_lin      = float(fl['k_lin'][()])
    fk         = fl['fk'][:]
    Pik_phi_t  = fl['Pik_phi_t'][:]
    Pik_d_t    = fl['Pik_d_t'][:]
    PiGk_P_t   = fl['PiGk_P_t'][:]
    PiGk_phi_t = fl['PiGk_phi_t'][:]
    PiGk_d_t   = fl['PiGk_d_t'][:]

#%% Compute skewness and flatness vs k

Pik_t     = Pik_phi_t + Pik_d_t
PiGk_t    = PiGk_P_t + PiGk_phi_t + PiGk_d_t

Pik_skew       = skew(Pik_t, axis=0)
Pik_phi_skew   = skew(Pik_phi_t, axis=0)
Pik_d_skew     = skew(Pik_d_t, axis=0)

Pik_flat       = kurtosis(Pik_t, axis=0, fisher=False)
Pik_phi_flat   = kurtosis(Pik_phi_t, axis=0, fisher=False)
Pik_d_flat     = kurtosis(Pik_d_t, axis=0, fisher=False)

PiGk_skew      = skew(PiGk_t, axis=0)
PiGk_P_skew    = skew(PiGk_P_t, axis=0)
PiGk_phi_skew  = skew(PiGk_phi_t, axis=0)
PiGk_d_skew    = skew(PiGk_d_t, axis=0)

PiGk_flat      = kurtosis(PiGk_t, axis=0, fisher=False)
PiGk_P_flat    = kurtosis(PiGk_P_t, axis=0, fisher=False)
PiGk_phi_flat  = kurtosis(PiGk_phi_t, axis=0, fisher=False)
PiGk_d_flat    = kurtosis(PiGk_d_t, axis=0, fisher=False)

#%% Plots

savename = partial(_savename, datadir, fname)

# E-flux skewness
plt.figure(figsize=(16, 9))
plt.plot(k[1:-1], Pik_skew[1:-1],     label=r'$\Pi_{k}$')
plt.plot(k[1:-1], Pik_phi_skew[1:-1], label=r'$\Pi_{k,\phi}$')
plt.plot(k[1:-1], Pik_d_skew[1:-1],   label=r'$\Pi_{k,d}$')
plt.axhline(0, color='k', linestyle='-', linewidth=1)
plt.axvline(x=1,     color='k', linestyle='--', linewidth=2, label='$k=1$')
plt.axvline(x=k_f,   color='k', linestyle=':',  linewidth=2, label=f'$k_f={k_f:.2f}$')
plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2, label=f'$k_{{lin}}={k_lin:.2f}$')
plt.xscale('log')
plt.xlabel('$k$')
plt.ylabel(r'$S(\Pi_k)$')
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(savename('E_flux_skewness'), dpi=100)
plt.show()

# E-flux flatness
plt.figure(figsize=(16, 9))
plt.plot(k[1:-1], Pik_flat[1:-1],     label=r'$\Pi_{k}$')
plt.plot(k[1:-1], Pik_phi_flat[1:-1], label=r'$\Pi_{k,\phi}$')
plt.plot(k[1:-1], Pik_d_flat[1:-1],   label=r'$\Pi_{k,d}$')
plt.axhline(3, color='k', linestyle='--', linewidth=1, label='Gaussian ($F=3$)')
plt.axvline(x=1,     color='k', linestyle='--', linewidth=2, label='$k=1$')
plt.axvline(x=k_f,   color='k', linestyle=':',  linewidth=2, label=f'$k_f={k_f:.2f}$')
plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2, label=f'$k_{{lin}}={k_lin:.2f}$')
plt.xscale('log')
plt.xlabel('$k$')
plt.ylabel(r'$F(\Pi_k)$')
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(savename('E_flux_flatness'), dpi=100)
plt.show()

# G-flux skewness
plt.figure(figsize=(16, 9))
plt.plot(k[1:-1], PiGk_skew[1:-1],     label=r'$\Pi_{G,k}$')
plt.plot(k[1:-1], PiGk_P_skew[1:-1],   label=r'$\Pi_{G,k,P}$')
plt.plot(k[1:-1], PiGk_phi_skew[1:-1], label=r'$\Pi_{G,k,\phi}$')
plt.plot(k[1:-1], PiGk_d_skew[1:-1],   label=r'$\Pi_{G,k,d}$')
plt.axhline(0, color='k', linestyle='-', linewidth=1)
plt.axvline(x=1,     color='k', linestyle='--', linewidth=2, label='$k=1$')
plt.axvline(x=k_f,   color='k', linestyle=':',  linewidth=2, label=f'$k_f={k_f:.2f}$')
plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2, label=f'$k_{{lin}}={k_lin:.2f}$')
plt.xscale('log')
plt.xlabel('$k$')
plt.ylabel(r'$S(\Pi_{G,k})$')
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(savename('G_flux_skewness'), dpi=100)
plt.show()

# G-flux flatness
plt.figure(figsize=(16, 9))
plt.plot(k[1:-1], PiGk_flat[1:-1],     label=r'$\Pi_{G,k}$')
plt.plot(k[1:-1], PiGk_P_flat[1:-1],   label=r'$\Pi_{G,k,P}$')
plt.plot(k[1:-1], PiGk_phi_flat[1:-1], label=r'$\Pi_{G,k,\phi}$')
plt.plot(k[1:-1], PiGk_d_flat[1:-1],   label=r'$\Pi_{G,k,d}$')
plt.axhline(3, color='k', linestyle='--', linewidth=1, label='Gaussian ($F=3$)')
plt.axvline(x=1,     color='k', linestyle='--', linewidth=2, label='$k=1$')
plt.axvline(x=k_f,   color='k', linestyle=':',  linewidth=2, label=f'$k_f={k_f:.2f}$')
plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2, label=f'$k_{{lin}}={k_lin:.2f}$')
plt.xscale('log')
plt.xlabel('$k$')
plt.ylabel(r'$F(\Pi_{G,k})$')
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(savename('G_flux_flatness'), dpi=100)
plt.show()
