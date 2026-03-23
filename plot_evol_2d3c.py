#%% Importing libraries
import h5py as h5
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from modules.plot_basics import apply_style
apply_style()

#%% Load the computed HDF5 file (produced by compute_evol_2d3c.py)

Npx=1024
datadir=f'data_2d3c/{Npx}/'
fname = datadir+'out_2d3c_kapt_2_0_D_0_1_kz_0_1.h5'

evol_fname = datadir + fname.split('/')[-1].replace('out_', 'evol_')

with h5.File(evol_fname, 'r') as fl:
    t               = fl['t'][:]
    P2_t            = fl['P2_t'][:]
    P2_ZF_t         = fl['P2_ZF_t'][:]
    V2_t            = fl['V2_t'][:]
    V2_ZF_t         = fl['V2_ZF_t'][:]
    energy_t        = fl['energy_t'][:]
    energy_ZF_t     = fl['energy_ZF_t'][:]
    kin_energy_t    = fl['kin_energy_t'][:]
    kin_energy_ZF_t = fl['kin_energy_ZF_t'][:]
    enstrophy_t     = fl['enstrophy_t'][:]
    enstrophy_ZF_t  = fl['enstrophy_ZF_t'][:]
    gen_energy_t    = fl['gen_energy_t'][:]
    gen_energy_ZF_t = fl['gen_energy_ZF_t'][:]
    entropy_t       = fl['entropy_t'][:]
    Q_t             = fl['Q_t'][:]
    electric_reynolds_power_t    = fl['electric_reynolds_power_t'][:]
    diamagnetic_reynolds_power_t = fl['diamagnetic_reynolds_power_t'][:]
    reynolds_power_t             = fl['reynolds_power_t'][:]

nt = len(t)

#%% Calculate derived turbulent quantities

P2_turb_t        = P2_t - P2_ZF_t
V2_turb_t        = V2_t - V2_ZF_t
energy_turb_t    = energy_t - energy_ZF_t
kin_energy_turb_t = kin_energy_t - kin_energy_ZF_t
enstrophy_turb_t = enstrophy_t - enstrophy_ZF_t
gen_energy_turb_t = gen_energy_t - gen_energy_ZF_t

#%% Plots

# Plot variance(P) vs time
plt.figure(figsize=(16, 9))
plt.semilogy(t, P2_t, label = r'$\left<P_{\mathrm{total}}^2\right>$')
plt.semilogy(t, P2_ZF_t, label = r'$\left<P_{\mathrm{ZF}}^2\right>$')
plt.semilogy(t, P2_turb_t, label = r'$\left<P_{\mathrm{turb}}^2\right>$')
plt.xlabel(r'$\gamma t$')
plt.ylabel('$P^2$')
plt.grid()
plt.legend()
plt.tight_layout()
if fname.endswith('out.h5'):
    plt.savefig(datadir+'P2_vs_t.pdf')
else:
    plt.savefig(datadir+fname.split('/')[-1].replace('out_', 'P2_vs_t_').replace('.h5', '.png'), dpi=100)
plt.show()

# Plot variance(V) vs time
plt.figure(figsize=(16, 9))
plt.semilogy(t, V2_t, label = r'$\left<V_{\mathrm{total}}^2\right>$')
plt.semilogy(t, V2_ZF_t, label = r'$\left<V_{\mathrm{ZF}}^2\right>$')
plt.semilogy(t, V2_turb_t, label = r'$\left<V_{\mathrm{turb}}^2\right>$')
plt.xlabel(r'$\gamma t$')
plt.ylabel('$V^2$')
plt.grid()
plt.legend()
plt.tight_layout()
if fname.endswith('out.h5'):
    plt.savefig(datadir+'V2_vs_t.pdf')
else:
    plt.savefig(datadir+fname.split('/')[-1].replace('out_', 'V2_vs_t_').replace('.h5', '.png'), dpi=100)
plt.show()

# Plot energy vs time
plt.figure(figsize=(16, 9))
plt.semilogy(t, energy_t, label = r'$E_{\mathrm{total}}$')
plt.semilogy(t, energy_ZF_t, label = r'$E_{\mathrm{ZF}}$')
plt.semilogy(t, energy_turb_t, label = r'$E_{\mathrm{turb}}$')
plt.xlabel(r'$\gamma t$')
plt.ylabel(r'$E$')
plt.grid()
plt.legend()
plt.tight_layout()
if fname.endswith('out.h5'):
    plt.savefig(datadir+'energy_vs_t.pdf')
else:
    plt.savefig(datadir+fname.split('/')[-1].replace('out_', 'energy_vs_t_').replace('.h5', '.png'), dpi=100)
plt.show()

# Plot zonal energy fraction vs time
plt.figure(figsize=(16, 9))
plt.semilogy(t, energy_ZF_t/energy_t, label = r'$E_{\mathrm{ZF}}/E$')
plt.xlabel(r'$\gamma t$')
plt.ylabel(r'$E_{\mathrm{ZF}}/E$')
plt.grid()
plt.legend()
plt.tight_layout()
if fname.endswith('out.h5'):
    plt.savefig(datadir+'zonal_energy_fraction_vs_t.pdf')
else:
    plt.savefig(datadir+fname.split('/')[-1].replace('out_', 'zonal_energy_fraction_vs_t_').replace('.h5', '.png'), dpi=100)
plt.show()

# # Plot kinetic energy vs time
# plt.figure(figsize=(8,6))
# plt.semilogy(t, kin_energy_t, label = '$E_{\\mathrm{kin,\mathrm{total}}}$')
# plt.semilogy(t, kin_energy_ZF_t, label = '$E_{\\mathrm{kin,\mathrm{ZF}}}$')
# plt.semilogy(t, kin_energy_turb_t, label = '$E_{\\mathrm{kin,\mathrm{turb}}}$')
# plt.xlabel('$\\gamma t$')
# plt.ylabel('$E_{\\mathrm{kin}}$')
# plt.grid()
# plt.legend()
# plt.tight_layout()
# if fname.endswith('out.h5'):
#     plt.savefig(datadir+'kinetic_energy_vs_t.pdf')
# else:
#     plt.savefig(datadir+fname.split('/')[-1].replace('out_', 'kinetic_energy_vs_t_').replace('.h5', '.png'), dpi=100)
# plt.show()

# # Plot enstrophy vs time
# plt.figure(figsize=(8,6))
# plt.semilogy(t, enstrophy_t, label = '$W_{\\mathrm{total}}$')
# plt.semilogy(t, enstrophy_ZF_t, label = '$W_{\\mathrm{ZF}}$')
# plt.semilogy(t, enstrophy_turb_t, label = '$W_{\\mathrm{turb}}$')
# plt.xlabel('$\\gamma t$')
# plt.ylabel('$W$')
# plt.grid()
# plt.legend()
# plt.tight_layout()
# if fname.endswith('out.h5'):
#     plt.savefig(datadir+'enstrophy_vs_t.pdf')
# else:
#     plt.savefig(datadir+fname.split('/')[-1].replace('out_', 'enstrophy_vs_t_').replace('.h5', '.png'), dpi=100)
# plt.show()

# Plot generalized energy vs time
plt.figure(figsize=(16, 9))
plt.semilogy(t, gen_energy_t, label = r'$G_{\mathrm{total}}$')
plt.semilogy(t, gen_energy_ZF_t, label = r'$G_{\mathrm{ZF}}$')
plt.semilogy(t, gen_energy_turb_t, label = r'$G_{\mathrm{turb}}$')
plt.xlabel(r'$\gamma t$')
plt.ylabel(r'$G$')
plt.grid()
plt.legend()
plt.tight_layout()
if fname.endswith('out.h5'):
    plt.savefig(datadir+'generalized_energy_vs_t.pdf')
else:
    plt.savefig(datadir+fname.split('/')[-1].replace('out_', 'generalized_energy_vs_t_').replace('.h5', '.png'), dpi=100)
plt.show()

# # Plot hyd. entropy vs time
# plt.figure(figsize=(8,6))
# plt.semilogy(t, entropy_t, label = '$\S$')
# plt.xlabel('$\\gamma t$')
# plt.ylabel('$\S=-\\sum_{\\mathbf{k}}p_{\\mathbf{k}}\\log p_{\\mathbf{k}}$')
# plt.grid()
# plt.legend()
# plt.tight_layout()
# if fname.endswith('out.h5'):
#     plt.savefig(datadir+'entropy_vs_t.pdf')
# else:
#     plt.savefig(datadir+fname.split('/')[-1].replace('out_', 'entropy_vs_t_').replace('.h5', '.png'), dpi=100)
# plt.show()

# Plot Q vs time
plt.figure(figsize=(16, 9))
plt.plot(t, Q_t, '-', label = r'$Q$')
plt.xlabel(r'$\gamma t$')
plt.ylabel(r'$Q$')
plt.grid()
plt.legend()
plt.tight_layout()
if fname.endswith('out.h5'):
    plt.savefig(datadir+'Q_vs_t.pdf')
else:
    plt.savefig(datadir+fname.split('/')[-1].replace('out_', 'Q_vs_t_').replace('.h5', '.png'), dpi=100)
plt.show()

# Plot Reynolds power vs time
rp_half   = reynolds_power_t[nt//2:]
rp_median = np.median(rp_half)
rp_mad    = np.median(np.abs(rp_half - rp_median))

plt.figure(figsize=(16, 9))
plt.plot(t, electric_reynolds_power_t, '-', label=r'$\langle R_{\phi} \partial_x \bar{v}_y \rangle$')
plt.plot(t, diamagnetic_reynolds_power_t, '-', label=r'$\langle R_{\mathrm{d}} \partial_x \bar{v}_y \rangle$')
plt.plot(t, reynolds_power_t, '-', label=r'$\langle R \partial_x \bar{v}_y \rangle$')
plt.axhline(rp_median + 24*rp_mad, color='k', lw=1.5, ls='--', label=r'$\mathrm{median} \pm 24\,\mathrm{MAD}$')
plt.axhline(rp_median - 24*rp_mad, color='k', lw=1.5, ls='--')
plt.xlabel(r'$\gamma t$')
plt.ylabel('Reynolds power')
plt.grid()
plt.legend()
plt.tight_layout()
if fname.endswith('out.h5'):
    plt.savefig(datadir+'reynolds_power_vs_t.pdf')
else:
    plt.savefig(datadir+fname.split('/')[-1].replace('out_', 'reynolds_power_vs_t_').replace('.h5', '.png'), dpi=100)
plt.show()