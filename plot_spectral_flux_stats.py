#%% Importing libraries
import numpy as np
import h5py as h5
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from modules.plot_basics import apply_style, savename as _savename, figsize_single
from functools import partial
apply_style()
xtick_fontsize = matplotlib.rcParams.get('xtick.labelsize', 32)

#%% Load computed flux data

Npx=1024
datadir=f'data/{Npx}/'
subdir='spectral_flux/'

# fname = 'out_kapt_0_4_D_0_1_H_3_6_em6.h5'
fname = 'out_kapt_2_0_D_0_1_H_8_6_em6.h5'
# fname = 'out_kapt_2_0_D_0_1_H_1_7_em5.h5'

flux_file = datadir + subdir + fname.replace('out_', 'spectral_flux_')
with h5.File(flux_file, 'r') as fl:
    k          = fl['k'][:]
    k_f        = float(fl['k_f'][()])
    k_Pf       = float(fl['k_Pf'][()])
    k_Gf       = float(fl['k_Gf'][()])
    k_lin      = float(fl['k_lin'][()])
    fk         = fl['fk'][:]
    PiPk_t     = fl['PiPk_t'][:]
    Pik_phi_t  = fl['Pik_phi_t'][:]
    Pik_d_t    = fl['Pik_d_t'][:]
    PiGk_P_t   = fl['PiGk_P_t'][:]
    PiGk_phi_t = fl['PiGk_phi_t'][:]
    PiGk_d_t   = fl['PiGk_d_t'][:]

#%% Compute skewness and flatness vs k

Pik_t     = Pik_phi_t + Pik_d_t
PiGk_t    = PiGk_P_t + PiGk_phi_t + PiGk_d_t

Pik_var     = np.var(Pik_t, axis=0)
Pik_phi_var = np.var(Pik_phi_t, axis=0)
Pik_d_var   = np.var(Pik_d_t, axis=0)

PiPk_var    = np.var(PiPk_t, axis=0)

PiGk_var    = np.var(PiGk_t, axis=0)
PiGk_P_var  = np.var(PiGk_P_t, axis=0)
PiGk_phi_var= np.var(PiGk_phi_t, axis=0)
PiGk_d_var  = np.var(PiGk_d_t, axis=0)

Pik_skew       = skew(Pik_t, axis=0)
Pik_phi_skew   = skew(Pik_phi_t, axis=0)
Pik_d_skew     = skew(Pik_d_t, axis=0)

PiPk_skew      = skew(PiPk_t, axis=0)

PiGk_skew      = skew(PiGk_t, axis=0)
PiGk_P_skew    = skew(PiGk_P_t, axis=0)
PiGk_phi_skew  = skew(PiGk_phi_t, axis=0)
PiGk_d_skew    = skew(PiGk_d_t, axis=0)

Pik_flat       = kurtosis(Pik_t, axis=0, fisher=False)
Pik_phi_flat   = kurtosis(Pik_phi_t, axis=0, fisher=False)
Pik_d_flat     = kurtosis(Pik_d_t, axis=0, fisher=False)

PiPk_flat      = kurtosis(PiPk_t, axis=0, fisher=False)

PiGk_flat      = kurtosis(PiGk_t, axis=0, fisher=False)
PiGk_P_flat    = kurtosis(PiGk_P_t, axis=0, fisher=False)
PiGk_phi_flat  = kurtosis(PiGk_phi_t, axis=0, fisher=False)
PiGk_d_flat    = kurtosis(PiGk_d_t, axis=0, fisher=False)

savename = partial(_savename, datadir+subdir, fname)

#%% Plot: Ek-flux variance

plt.figure(figsize=figsize_single)
plt.plot(k[1:-1], Pik_var[1:-1],     label=r'$\Pi_{k}$', color='C0')
plt.plot(k[1:-1], Pik_phi_var[1:-1], label=r'$\Pi_{k}^{\left(\phi\right)}$', color='C1')
plt.plot(k[1:-1], Pik_d_var[1:-1],   label=r'$\Pi_{k}^{\left(d\right)}$', color='C2')
plt.axhline(0, color='k', linestyle='-', linewidth=1)
plt.axvline(x=1,     color='k', linestyle='--', linewidth=2)
plt.axvline(x=k_f,   color='k', linestyle=':',  linewidth=2)
plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2)
ymin, ymax = plt.ylim()
offset = 0.025 * (ymax - ymin)
plt.text(k_f, ymin - offset, r'$k_f$', ha='center', va='top', fontsize=xtick_fontsize)
plt.text(k_lin, ymin - offset, r'$k_{\mathrm{lin}}$', ha='center', va='top', fontsize=xtick_fontsize)
plt.xscale('log')
plt.xlabel('$k$')
plt.ylabel(r'$\mathrm{Var}\left(\Pi_k\right)$')
plt.legend()
plt.tight_layout()
plt.savefig(savename('E_flux_variance'), dpi=100, bbox_inches='tight')
plt.show()

#%% Plot: Ek-flux skewness

plt.figure(figsize=figsize_single)
plt.plot(k[1:-1], Pik_skew[1:-1],     label=r'$\Pi_{k}$', color='C0')
plt.plot(k[1:-1], Pik_phi_skew[1:-1], label=r'$\Pi_{k}^{\left(\phi\right)}$', color='C1')
plt.plot(k[1:-1], Pik_d_skew[1:-1],   label=r'$\Pi_{k}^{\left(d\right)}$', color='C2')
plt.axhline(0, color='k', linestyle='-', linewidth=1)
plt.axvline(x=1,     color='k', linestyle='--', linewidth=2)
plt.axvline(x=k_f,   color='k', linestyle=':',  linewidth=2)
plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2)
ymin, ymax = plt.ylim()
offset = 0.025 * (ymax - ymin)
plt.text(k_f, ymin - offset, r'$k_f$', ha='center', va='top', fontsize=xtick_fontsize)
plt.text(k_lin, ymin - offset, r'$k_{\mathrm{lin}}$', ha='center', va='top', fontsize=xtick_fontsize)
plt.xscale('log')
plt.xlabel('$k$')
plt.ylabel(r'$\mathrm{Skew}\left[\Pi_k\right]$')
plt.legend()
plt.tight_layout()
plt.savefig(savename('E_flux_skewness'), dpi=100, bbox_inches='tight')
plt.show()

#%% Plot: Ek-flux kurtosis (flatness)

plt.figure(figsize=figsize_single)
plt.plot(k[1:-1], Pik_flat[1:-1],     label=r'$\Pi_{k}$', color='C0')
plt.plot(k[1:-1], Pik_phi_flat[1:-1], label=r'$\Pi_{k}^{\left(\phi\right)}$', color='C1')
plt.plot(k[1:-1], Pik_d_flat[1:-1],   label=r'$\Pi_{k}^{\left(d\right)}$', color='C2')
plt.axvline(x=1,     color='k', linestyle='--', linewidth=2)
plt.axvline(x=k_f,   color='k', linestyle=':',  linewidth=2)
plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2)
ymin, ymax = plt.ylim()
offset = 0.025 * (ymax - ymin)
plt.text(k_f, ymin - offset, r'$k_f$', ha='center', va='top', fontsize=xtick_fontsize)
plt.text(k_lin, ymin - offset, r'$k_{\mathrm{lin}}$', ha='center', va='top', fontsize=xtick_fontsize)
plt.xscale('log')
plt.xlabel('$k$')
plt.ylabel(r'$\mathrm{Kurt}\left[\Pi_k\right]$')
plt.legend()
plt.tight_layout()
plt.savefig(savename('E_flux_kurtosis'), dpi=100, bbox_inches='tight')
plt.show()

#%% Plot: EPk-flux variance

plt.figure(figsize=figsize_single)
plt.plot(k[1:-1], PiPk_var[1:-1], label=r'$\Pi_{P,k}$', color='C0')
plt.axhline(0, color='k', linestyle='-', linewidth=1)
plt.axvline(x=1,     color='k', linestyle='--', linewidth=2)
plt.axvline(x=k_Pf,  color='k', linestyle=':',  linewidth=2)
plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2)
ymin, ymax = plt.ylim()
offset = 0.025 * (ymax - ymin)
plt.text(k_Pf, ymin - offset, r'$k_{P,f}$', ha='center', va='top', fontsize=xtick_fontsize)
plt.text(k_lin, ymin - offset, r'$k_{\mathrm{lin}}$', ha='center', va='top', fontsize=xtick_fontsize)
plt.xscale('log')
plt.xlabel('$k$')
plt.ylabel(r'$\mathrm{Var}\left(\Pi_{P,k}\right)$')
plt.legend()
plt.tight_layout()
plt.savefig(savename('EPk_flux_variance'), dpi=100, bbox_inches='tight')
plt.show()

#%% Plot: EPk-flux skewness

plt.figure(figsize=figsize_single)
plt.plot(k[1:-1], PiPk_skew[1:-1], label=r'$\Pi_{P,k}$', color='C0')
plt.axhline(0, color='k', linestyle='-', linewidth=1)
plt.axvline(x=1,     color='k', linestyle='--', linewidth=2)
plt.axvline(x=k_Pf,  color='k', linestyle=':',  linewidth=2)
plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2)
ymin, ymax = plt.ylim()
offset = 0.025 * (ymax - ymin)
plt.text(k_Pf, ymin - offset, r'$k_{P,f}$', ha='center', va='top', fontsize=xtick_fontsize)
plt.text(k_lin, ymin - offset, r'$k_{\mathrm{lin}}$', ha='center', va='top', fontsize=xtick_fontsize)
plt.xscale('log')
plt.xlabel('$k$')
plt.ylabel(r'$\mathrm{Skew}\left[\Pi_{P,k}\right]$')
plt.legend()
plt.tight_layout()
plt.savefig(savename('EPk_flux_skewness'), dpi=100, bbox_inches='tight')
plt.show()

#%% Plot: EPk-flux kurtosis (flatness)

plt.figure(figsize=figsize_single)
plt.plot(k[1:-1], PiPk_flat[1:-1], label=r'$\Pi_{P,k}$', color='C0')
plt.axvline(x=1,     color='k', linestyle='--', linewidth=2)
plt.axvline(x=k_Pf,  color='k', linestyle=':',  linewidth=2)
plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2)
ymin, ymax = plt.ylim()
offset = 0.025 * (ymax - ymin)
plt.text(k_Pf, ymin - offset, r'$k_{P,f}$', ha='center', va='top', fontsize=xtick_fontsize)
plt.text(k_lin, ymin - offset, r'$k_{\mathrm{lin}}$', ha='center', va='top', fontsize=xtick_fontsize)
plt.xscale('log')
plt.xlabel('$k$')
plt.ylabel(r'$\mathrm{Kurt}\left[\Pi_{P,k}\right]$')
plt.legend()
plt.tight_layout()
plt.savefig(savename('EPk_flux_kurtosis'), dpi=100, bbox_inches='tight')
plt.show()

#%% Plot: Gk-flux variance

plt.figure(figsize=figsize_single)
plt.plot(k[1:-1], PiGk_var[1:-1],     label=r'$\Pi_{G,k}$', color='C0')
plt.plot(k[1:-1], PiGk_phi_var[1:-1], label=r'$\Pi_{G,k}^{\left(\phi\right)}$', color='C1')
plt.plot(k[1:-1], PiGk_d_var[1:-1],   label=r'$\Pi_{G,k}^{\left(d\right)}$', color='C2')
plt.plot(k[1:-1], PiGk_P_var[1:-1],   label=r'$\Pi_{G,k}^{\left(P\right)}$', color='C3')
plt.axhline(0, color='k', linestyle='-', linewidth=1)
plt.axvline(x=1,      color='k', linestyle='--', linewidth=2)
plt.axvline(x=k_Gf,  color='k', linestyle=':',  linewidth=2)
plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2)
ymin, ymax = plt.ylim()
offset = 0.025 * (ymax - ymin)
plt.text(k_Gf, ymin - offset, r'$k_{G,f}$', ha='center', va='top', fontsize=xtick_fontsize)
plt.text(k_lin, ymin - offset, r'$k_{\mathrm{lin}}$', ha='center', va='top', fontsize=xtick_fontsize)
plt.xscale('log')
plt.xlabel('$k$')
plt.ylabel(r'$\mathrm{Var}\left(\Pi_{G,k}\right)$')
plt.legend()
plt.tight_layout()
plt.savefig(savename('G_flux_variance'), dpi=100, bbox_inches='tight')
plt.show()

#%% Plot: Gk-flux skewness

plt.figure(figsize=figsize_single)
plt.plot(k[1:-1], PiGk_skew[1:-1],     label=r'$\Pi_{G,k}$', color='C0')
plt.plot(k[1:-1], PiGk_phi_skew[1:-1], label=r'$\Pi_{G,k}^{\left(\phi\right)}$', color='C1')
plt.plot(k[1:-1], PiGk_d_skew[1:-1],   label=r'$\Pi_{G,k}^{\left(d\right)}$', color='C2')
plt.plot(k[1:-1], PiGk_P_skew[1:-1],   label=r'$\Pi_{G,k}^{\left(P\right)}$', color='C3')
plt.axhline(0, color='k', linestyle='-', linewidth=1)
plt.axvline(x=1,      color='k', linestyle='--', linewidth=2)
plt.axvline(x=k_Gf,  color='k', linestyle=':',  linewidth=2)
plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2)
ymin, ymax = plt.ylim()
offset = 0.025 * (ymax - ymin)
plt.text(k_Gf, ymin - offset, r'$k_{G,f}$', ha='center', va='top', fontsize=xtick_fontsize)
plt.text(k_lin, ymin - offset, r'$k_{\mathrm{lin}}$', ha='center', va='top', fontsize=xtick_fontsize)
plt.xscale('log')
plt.xlabel('$k$')
plt.ylabel(r'$\mathrm{Skew}\left[\Pi_{G,k}\right]$')
plt.legend()
plt.tight_layout()
plt.savefig(savename('G_flux_skewness'), dpi=100, bbox_inches='tight')
plt.show()

#%% Plot: Gk-flux flatness
plt.figure(figsize=figsize_single)
plt.plot(k[1:-1], PiGk_flat[1:-1],     label=r'$\Pi_{G,k}$', color='C0')
plt.plot(k[1:-1], PiGk_phi_flat[1:-1], label=r'$\Pi_{G,k}^{\left(\phi\right)}$', color='C1')
plt.plot(k[1:-1], PiGk_d_flat[1:-1],   label=r'$\Pi_{G,k}^{\left(d\right)}$', color='C2')
plt.plot(k[1:-1], PiGk_P_flat[1:-1],   label=r'$\Pi_{G,k}^{\left(P\right)}$', color='C3')
plt.axvline(x=1,      color='k', linestyle='--', linewidth=2)
plt.axvline(x=k_Gf,  color='k', linestyle=':',  linewidth=2)
plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2)
ymin, ymax = plt.ylim()
offset = 0.025 * (ymax - ymin)
plt.text(k_Gf, ymin - offset, r'$k_{G,f}$', ha='center', va='top', fontsize=xtick_fontsize)
plt.text(k_lin, ymin - offset, r'$k_{\mathrm{lin}}$', ha='center', va='top', fontsize=xtick_fontsize)
plt.xscale('log')
plt.xlabel('$k$')
plt.ylabel(r'$\mathrm{Kurt}\left[\Pi_{G,k}\right]$')
plt.legend()
plt.tight_layout()
plt.savefig(savename('G_flux_kurtosis'), dpi=100, bbox_inches='tight')
plt.show()