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

#%% Load computed spectra
Npx=1024
datadir=f'data_2d3c/{Npx}/'
subdir='spectrum/'

fname = 'out_2d3c_kapt_2_0_D_0_1_kz_0_1.h5'

spectrum_file = datadir + subdir + fname.replace('out_', 'spectrum_')
with h5.File(spectrum_file, 'r') as fl:
    k        = fl['k'][:]
    P2k      = fl['P2k'][:];      P2k_ZF      = fl['P2k_ZF'][:];      P2k_turb      = fl['P2k_turb'][:]
    V2k      = fl['V2k'][:];      V2k_ZF      = fl['V2k_ZF'][:];      V2k_turb      = fl['V2k_turb'][:]
    Ek       = fl['Ek'][:];       Ek_ZF       = fl['Ek_ZF'][:];       Ek_turb       = fl['Ek_turb'][:]
    Kk       = fl['Kk'][:];       Kk_ZF       = fl['Kk_ZF'][:];       Kk_turb       = fl['Kk_turb'][:]
    Wk       = fl['Wk'][:];       Wk_ZF       = fl['Wk_ZF'][:];       Wk_turb       = fl['Wk_turb'][:]
    Gk       = fl['Gk'][:];       Gk_ZF       = fl['Gk_ZF'][:];       Gk_turb       = fl['Gk_turb'][:]
    GKk      = fl['GKk'][:];      GKk_ZF      = fl['GKk_ZF'][:];      GKk_turb      = fl['GKk_turb'][:]

flux_file = datadir + 'spectral_flux/' + fname.replace('out_', 'spectral_flux_')
with h5.File(flux_file, 'r') as fl:
    k_f       = float(fl['k_f'][()])
    k_lin     = float(fl['k_lin'][()])

savename = partial(_savename, datadir+'spectrum/', fname)
k1 = np.argmin(np.abs(k - 1))

#%% Plot pressure spectrum

plt.figure(figsize=figsize_single)
plt.loglog(k, P2k, label = r'$\left|P_{k}\right|^2$')
plt.loglog(k[P2k_ZF>0], P2k_ZF[P2k_ZF>0], label = r'$\left|P_{k,\mathrm{ZF}}\right|^2$')
plt.loglog(k, P2k_turb, label = r'$\left|P_{k,\mathrm{turb}}\right|^2$')
plt.loglog(k, P2k[k1]*k**(-3), 'k--')
plt.loglog(k, P2k[k1]*k**(-4), 'r--')
plt.axvline(x=1, color='k', linestyle='--', linewidth=2)
plt.axvline(x=k_f, color='k', linestyle=':', linewidth=2)
plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2)
ax = plt.gca()
ax.text(k_f, -0.025, r'$k_f$', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=xtick_fontsize)
ax.text(k_lin, -0.025, r'$k_{\mathrm{lin}}$', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=xtick_fontsize)
plt.xlabel('$k$')
plt.ylabel(r'$\left|P_k\right|^2$')
plt.legend()
# annotate power-law lines as text
x_pos = k[int(0.6*len(k))]
plt.text(x_pos, P2k[k1]*x_pos**(-3)*1.2, '$k^{-3}$', color='k', fontsize=32)
plt.text(x_pos, P2k[k1]*x_pos**(-4)*0.8, '$k^{-4}$', color='r', fontsize=32)
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(savename('EP_spectrum'), bbox_inches='tight')
plt.show()

#%% Plot parallel velocity spectrum

plt.figure(figsize=figsize_single)
plt.loglog(k, V2k, label = r'$\left|V_{k}\right|^2$')
plt.loglog(k[V2k_ZF>0], V2k_ZF[V2k_ZF>0], label = r'$\left|V_{k,\mathrm{ZF}}\right|^2$')
plt.loglog(k, V2k_turb, label = r'$\left|V_{k,\mathrm{turb}}\right|^2$')
plt.loglog(k, V2k[k1]*k**(-2), 'k--')
plt.loglog(k, V2k[k1]*k**(-3), 'r--')
plt.axvline(x=1, color='k', linestyle='--', linewidth=2)
plt.axvline(x=k_f, color='k', linestyle=':', linewidth=2)
plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2)
ax = plt.gca()
ax.text(k_f, -0.025, r'$k_f$', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=xtick_fontsize)
ax.text(k_lin, -0.025, r'$k_{\mathrm{lin}}$', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=xtick_fontsize)
plt.xlabel('$k$')
plt.ylabel(r'$\left|V_k\right|^2$')
plt.legend()
# annotate power-law lines as text
x_pos = k[int(0.6*len(k))]
plt.text(x_pos, V2k[k1]*x_pos**(-2)*1.2, '$k^{-2}$', color='k', fontsize=32)
plt.text(x_pos, V2k[k1]*x_pos**(-3)*0.8, '$k^{-3}$', color='r', fontsize=32)
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(savename('V2_spectrum'), bbox_inches='tight')
plt.show()

#%% Plot energy spectrum

plt.figure(figsize=figsize_single)
plt.loglog(k, Ek, label = r'$E_{k}$')
plt.loglog(k[Ek_ZF>0], Ek_ZF[Ek_ZF>0], label = r'$E_{k,\mathrm{ZF}}$')
plt.loglog(k, Ek_turb, label = r'$E_{k,\mathrm{turb}}$')
plt.loglog(k, Ek[k1]*k**(-5/3), 'r--')
plt.loglog(k, Ek[k1]*k**(-3), 'k--')
plt.axvline(x=1, color='k', linestyle='--', linewidth=2)
plt.axvline(x=k_f, color='k', linestyle=':', linewidth=2)
plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2)
ax = plt.gca()
ax.text(k_f, -0.025, r'$k_f$', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=xtick_fontsize)
ax.text(k_lin, -0.025, r'$k_{\mathrm{lin}}$', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=xtick_fontsize)
plt.xlabel('$k$')
plt.ylabel(r'$E_k$')
plt.legend()
# annotate power-law lines as text
x_pos = k[int(0.6*len(k))]
plt.text(x_pos, Ek[k1]*x_pos**(-5/3)*1.2, '$k^{-5/3}$', color='r', fontsize=32)
plt.text(x_pos, Ek[k1]*x_pos**(-3)*0.8, '$k^{-3}$', color='k', fontsize=32)
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(savename('E_spectrum'), bbox_inches='tight')
plt.show()

#%% Plot kinetic energy spectrum

# plt.figure(figsize=figsize_single)
# plt.loglog(k, Kk, label = '$E_{kin,k}$')
# plt.loglog(k[Kk_ZF>0], Kk_ZF[Kk_ZF>0], label = '$E_{kin,k,\mathrm{ZF}}$')
# plt.loglog(k, Kk_turb, label = '$E_{kin,k,\mathrm{turb}}$')
# plt.loglog(k, Kk[k1]*k**(-3), 'r--')
# plt.loglog(k, Kk[k1]*k**(-5), 'k--')
plt.axvline(x=1, color='k', linestyle='--', linewidth=2)
plt.axvline(x=k_f, color='k', linestyle=':', linewidth=2)
plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2)
ax = plt.gca()
ax.text(k_f, -0.025, r'$k_f$', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=xtick_fontsize)
ax.text(k_lin, -0.025, r'$k_{\mathrm{lin}}$', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=xtick_fontsize)
# plt.xlabel(r'$k$')
# plt.ylabel(r'$E_{kin,k}$')
# plt.legend()
# # annotate power laws
# # x_pos = k[int(0.6*len(k))]
# # plt.text(x_pos, Kk[k1]*x_pos**(-3)*1.2, '$k^{-3}$', color='r', fontsize=32)
# # plt.text(x_pos, Kk[k1]*x_pos**(-5)*0.8, '$k^{-5}$', color='k', fontsize=32)
# plt.tight_layout()
# if fname.endswith('out.h5'):
#     plt.savefig(datadir+'KE_spectrum.svg')
# else:
#     plt.savefig(datadir+"KE_spectrum_" + fname.split('/')[-1].split('out_')[-1].replace('.h5', '.svg'))
# plt.show()

#%% Plot generalized energy spectrum

plt.figure(figsize=figsize_single)
plt.loglog(k, Gk, label = r'$G_{k}$')
plt.loglog(k[Gk_ZF>0], Gk_ZF[Gk_ZF>0], label = r'$G_{k,\mathrm{ZF}}$')
plt.loglog(k, Gk_turb, label = r'$G_{k,\mathrm{turb}}$')
plt.loglog(k, Gk[k1]*k**(-3), 'r--')
plt.loglog(k, Gk[k1]*k**(-5), 'k--')
plt.axvline(x=1, color='k', linestyle='--', linewidth=2)
plt.axvline(x=k_f, color='k', linestyle=':', linewidth=2)
plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2)
ax = plt.gca()
ax.text(k_f, -0.025, r'$k_f$', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=xtick_fontsize)
ax.text(k_lin, -0.025, r'$k_{\mathrm{lin}}$', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=xtick_fontsize)
plt.xlabel(r'$k$')
plt.ylabel(r'$G_{k}$')
plt.legend()
 # annotate power-law lines as text
x_pos = k[int(0.6*len(k))]
plt.text(x_pos, Gk[k1]*x_pos**(-3)*1.2, '$k^{-3}$', color='r', fontsize=32)
plt.text(x_pos, Gk[k1]*x_pos**(-5)*0.8, '$k^{-5}$', color='k', fontsize=32)
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
if fname.endswith('out.h5'):
    plt.savefig(datadir+'G_spectrum.svg')
else:
    plt.savefig(datadir+"G_spectrum_" + fname.split('/')[-1].split('out_')[-1].replace('.h5', '.svg'))
plt.show()

#%% Plot generalized kinetic energy spectrum

# plt.figure(figsize=figsize_single)
# plt.loglog(k[1:-1], GKk[1:-1], label = r'$G_{kin,k}$')
# plt.loglog(k[GKk_ZF>0][1:-1], GKk_ZF[GKk_ZF>0][1:-1], label = r'$G_{kin,k,\mathrm{ZF}}$')
# plt.loglog(k[1:-1], GKk_turb[1:-1], label = r'$G_{kin,k,\mathrm{turb}}$')
# plt.loglog(k[1:-1], GKk[k1]*k[1:-1]**(-3), 'r--', label = r'$k^{-3}$')
# plt.loglog(k[1:-1], GKk[k1]*k[1:-1]**(-5), 'k--', label = r'$k^{-5}$')
plt.axvline(x=1, color='k', linestyle='--', linewidth=2)
plt.axvline(x=k_f, color='k', linestyle=':', linewidth=2)
plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2)
ax = plt.gca()
ax.text(k_f, -0.025, r'$k_f$', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=xtick_fontsize)
ax.text(k_lin, -0.025, r'$k_{\mathrm{lin}}$', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=xtick_fontsize)
# plt.xlabel(r'$k$')
# plt.ylabel(r'$G_{kin,k}$')
# plt.legend()
# plt.tight_layout()
# if fname.endswith('out.h5'):
#     plt.savefig(datadir+'KG_spectrum.svg')
# else:
#     plt.savefig(datadir+"KG_spectrum_" + fname.split('/')[-1].split('out_')[-1].replace('.h5', '.svg'))
# plt.show()

# %%