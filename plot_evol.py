#%% Importing libraries
import h5py as h5
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from modules.plot_basics import apply_style, savename as _savename, figsize_single
from functools import partial
apply_style()

#%% Load the computed HDF5 file (produced by compute_evol.py)

Npx=512
# Npx=1024
datadir=f'data/{Npx}/'
subdir='evol/'

# fname = 'out_kapt_0_2_D_0_1_H_1_5_em6.h5'
# fname = 'out_kapt_0_4_D_0_1_H_3_6_em6.h5'
# fname = 'out_kapt_2_0_D_0_1_H_0_0_e0.h5'
# fname = 'out_kapt_2_0_D_0_1_H_8_6_em6.h5'

# fname = 'out_kapt_0_2_hyper_D_1_0_em5_H_8_0_em6.h5'
fname = 'out_kapt_2_0_hyper_D_5_0_em6_H_1_1_em5.h5'

evol_fname = datadir + subdir + fname.split('/')[-1].replace('out_', 'evol_')
with h5.File(evol_fname, 'r') as fl:
    t                        = fl['t'][:]
    P2_t                     = fl['P2_t'][:]
    P2_ZF_t                  = fl['P2_ZF_t'][:]
    energy_t                 = fl['energy_t'][:]
    energy_ZF_t              = fl['energy_ZF_t'][:]
    kin_energy_t             = fl['kin_energy_t'][:]
    kin_energy_ZF_t          = fl['kin_energy_ZF_t'][:]
    enstrophy_t              = fl['enstrophy_t'][:]
    enstrophy_ZF_t           = fl['enstrophy_ZF_t'][:]
    gen_energy_t             = fl['gen_energy_t'][:]
    gen_energy_ZF_t          = fl['gen_energy_ZF_t'][:]
    entropy_t                = fl['entropy_t'][:]
    Ombar_t                  = fl['Ombar_t'][:]
    Qbox_t                   = fl['Qbox_t'][:]
    electric_reynolds_power_t   = fl['electric_reynolds_power_t'][:]
    diamagnetic_reynolds_power_t = fl['diamagnetic_reynolds_power_t'][:]
    reynolds_power_t         = fl['reynolds_power_t'][:]

nt = len(t)
savename = partial(_savename, datadir+subdir, fname)

#%% Calculate derived turbulent quantities and plot

P2_turb_t = P2_t - P2_ZF_t
energy_turb_t = energy_t - energy_ZF_t
kin_energy_turb_t = kin_energy_t - kin_energy_ZF_t
enstrophy_turb_t = enstrophy_t - enstrophy_ZF_t
gen_energy_turb_t = gen_energy_t - gen_energy_ZF_t

#%% Plot: variance(P) vs time
plt.figure(figsize=figsize_single)
plt.semilogy(t, P2_t, label = r'$P_{\mathrm{total}}^2$')
plt.semilogy(t, P2_ZF_t, label = r'$P_{\mathrm{ZF}}^2$')
plt.semilogy(t, P2_turb_t, label = r'$P_{\mathrm{turb}}^2$')
plt.xlabel(r'$\gamma t$')
plt.ylabel(r'$\langle P^2\rangle$')
plt.legend()
plt.tight_layout()
plt.savefig(savename('P2_vs_t'), bbox_inches='tight')
plt.show()

#%% Plot: total energy vs time
plt.figure(figsize=figsize_single)
plt.semilogy(t, energy_t, label = r'$E_{\mathrm{total}}$')
plt.semilogy(t, energy_ZF_t, label = r'$E_{\mathrm{ZF}}$')
plt.semilogy(t, energy_turb_t, label = r'$E_{\mathrm{turb}}$')
plt.xlabel(r'$\gamma t$')
plt.ylabel(r'$E$')
plt.legend()
plt.tight_layout()
plt.savefig(savename('energy_vs_t'), bbox_inches='tight')
plt.show()

#%% Plot: zonal energy fraction vs time
zonal_frac = energy_ZF_t / energy_t
zonal_frac_mean = np.mean(zonal_frac[nt//2:])
plt.figure(figsize=figsize_single)
plt.semilogy(t, zonal_frac)
plt.axhline(zonal_frac_mean, color='k', linestyle='--', linewidth=2.5, label=rf'$\langle E_{{\mathrm{{ZF}}}}/E \rangle_{{T/2}} = {zonal_frac_mean:.3f}$')
plt.xlabel(r'$\gamma t$')
plt.ylabel(r'$E_{\mathrm{ZF}}/E$')
plt.legend()
plt.tight_layout()
plt.savefig(savename('zonal_energy_fraction_vs_t'), bbox_inches='tight')
plt.show()

#%% Plot: kinetic energy vs time
# plt.figure(figsize=figsize_single)
# plt.semilogy(t, kin_energy_t, label = r'$E_{\mathrm{kin,\mathrm{total}}}$')
# plt.semilogy(t, kin_energy_ZF_t, label = r'$E_{\mathrm{kin,\mathrm{ZF}}}$')
# plt.semilogy(t, kin_energy_turb_t, label = r'$E_{\mathrm{kin,\mathrm{turb}}}$')
# plt.xlabel(r'$\gamma t$')
# plt.ylabel(r'$E_{\mathrm{kin}}$')
# plt.grid()
# plt.legend()
# plt.tight_layout()
# if fname.endswith('out.h5'):
#     plt.savefig(datadir+'kinetic_energy_vs_t.svg',dpi=100)
# else:
#     plt.savefig(datadir+fname.split('/')[-1].replace('out_', 'kinetic_energy_vs_t_').replace('.h5', '.svg'),dpi=100)
# plt.show()

#%% Plot: generalized energy vs time
plt.figure(figsize=figsize_single)
plt.semilogy(t, gen_energy_t, label = r'$G$')
plt.semilogy(t, gen_energy_ZF_t, label = r'$G_{\mathrm{ZF}}$')
plt.semilogy(t, gen_energy_turb_t, label = r'$G_{\mathrm{turb}}$')
plt.xlabel(r'$\gamma t$')
plt.ylabel(r'$G$')
plt.legend()
plt.tight_layout()
plt.savefig(savename('generalized_energy_vs_t'), bbox_inches='tight')
plt.show()

# #%% Plot: hyd. entropy vs time
# plt.figure(figsize=figsize_single)
# plt.semilogy(t, entropy_t, label = r'$S$')
# plt.xlabel(r'$\gamma t$')
# plt.ylabel(r'$S=-\sum_{\mathbf{k}}p_{\mathbf{k}}\log p_{\mathbf{k}}$')
# plt.grid()
# plt.legend()
# plt.tight_layout()
# if fname.endswith('out.h5'):
#     plt.savefig(datadir+'entropy_vs_t.svg',dpi=100)
# else:
#     plt.savefig(datadir+fname.split('/')[-1].replace('out_', 'entropy_vs_t_').replace('.h5', '.svg'))
# plt.show()

#%% Plot: Q vs time
plt.figure(figsize=figsize_single)
plt.plot(t, Qbox_t, '-', label = r'$Q_{\mathrm{box}}$')
plt.xlabel(r'$\gamma t$')
plt.ylabel(r'$Q_{\mathrm{box}}$')
# plt.legend()
plt.tight_layout()
plt.savefig(savename('Qbox_vs_t'), bbox_inches='tight')
plt.show()

#%% Plot: log(Q) vs time
Qbox_mean = np.mean(Qbox_t[nt//2:])

fig, ax = plt.subplots(figsize=figsize_single)
ax.semilogy(t, np.abs(Qbox_t), label=r'$Q_{\mathrm{box}}$')
# Shade the region where Qbox is negative using axvspan
neg_idx = np.where(Qbox_t < 0)[0]
for i in neg_idx:
    ax.axvspan(t[i], t[min(i + 1, len(t) - 1)], alpha=0.7, facecolor='grey', edgecolor='grey', linewidth=1.5)
ax.set_xlabel(r'$\gamma t$')
ax.set_ylabel(r'$|Q_{\mathrm{box}}|$')
ax.legend(fontsize=20)
plt.tight_layout()
plt.savefig(savename('Qbox_log_vs_t'), bbox_inches='tight')
plt.show()

#%% Plot: Reynolds power vs time
plt.figure(figsize=figsize_single)
plt.plot(t, electric_reynolds_power_t, '-', label = 'electric')
plt.plot(t, diamagnetic_reynolds_power_t, '-', label = 'diamagnetic')
plt.plot(t, reynolds_power_t, '-', label = 'total')
plt.xlabel(r'$\gamma t$')
plt.ylabel(r'$<R \partial_x \bar{v}_y>$')
plt.legend()
plt.tight_layout()
plt.savefig(savename('reynolds_power_vs_t'), bbox_inches='tight')
plt.show()

#%% Plot: Cumulative Reynolds power vs time
plt.figure(figsize=figsize_single)
plt.plot(t, np.cumsum(electric_reynolds_power_t), '-', label = 'electric')
plt.plot(t, np.cumsum(diamagnetic_reynolds_power_t), '-', label = 'diamagnetic')
plt.plot(t, np.cumsum(reynolds_power_t), '-', label = 'total')
plt.xlabel(r'$\gamma t$')
plt.ylabel(r'$\int \mathrm{d}t<R \partial_x \bar{v}_y>$')
plt.legend()
plt.tight_layout()
plt.savefig(savename('cum_reynolds_power_vs_t'), bbox_inches='tight')
plt.show()
# %%