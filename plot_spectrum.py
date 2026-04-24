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
Npx=512
# Npx=1024
datadir=f'data/{Npx}/'
subdir='spectrum/'

# fname = 'out_kapt_0_4_D_0_1_H_3_6_em6.h5'
# fname = 'out_kapt_2_0_D_0_1_H_8_6_em6.h5'
fname = 'out_kapt_0_2_hyper_D_1_0_em5_H_8_0_em6.h5'
# fname = 'out_kapt_2_0_hyper_D_1_0_em5_H_1_1_em5.h5'

spectrum_file = datadir + subdir + fname.replace('out_', 'spectrum_')
with h5.File(spectrum_file, 'r') as fl:
    k         = fl['k'][:]
    Phi2k     = fl['Phi2k'][:];     Phi2k_ZF     = fl['Phi2k_ZF'][:];     Phi2k_turb     = fl['Phi2k_turb'][:]
    P2k       = fl['P2k'][:];       P2k_ZF       = fl['P2k_ZF'][:];       P2k_turb       = fl['P2k_turb'][:]
    Ek        = fl['Ek'][:];        Ek_ZF        = fl['Ek_ZF'][:];        Ek_turb        = fl['Ek_turb'][:]
    Kk        = fl['Kk'][:];        Kk_ZF        = fl['Kk_ZF'][:];        Kk_turb        = fl['Kk_turb'][:]
    Wk        = fl['Wk'][:];        Wk_ZF        = fl['Wk_ZF'][:];        Wk_turb        = fl['Wk_turb'][:]
    Gk        = fl['Gk'][:];        Gk_ZF        = fl['Gk_ZF'][:];        Gk_turb        = fl['Gk_turb'][:]
    GKk       = fl['GKk'][:];       GKk_ZF       = fl['GKk_ZF'][:];       GKk_turb       = fl['GKk_turb'][:]

flux_file = datadir + 'spectral_flux/' + fname.replace('out_', 'spectral_flux_')
with h5.File(flux_file, 'r') as fl:
    k_f_E     = float(fl['k_f_E'][()])
    k_f_P     = float(fl['k_f_P'][()])
    k_f_G     = float(fl['k_f_G'][()])
    k_D_E     = float(fl['k_D_E'][()])
    k_D_P     = float(fl['k_D_P'][()])
    k_D_G     = float(fl['k_D_G'][()])
    k_lin     = float(fl['k_lin'][()])

#%% Functions

savename = partial(_savename, datadir+subdir, fname)

def fit_slope(k, y, k_min, k_max, yoffset=3.0):
    mask = (k >= k_min) & (k <= k_max) & (y > 0)
    k_seg = k[mask]
    y_seg = y[mask]
    n, c = np.polyfit(np.log(k_seg), np.log(y_seg), 1)
    return k_seg, np.exp(c) * k_seg**n * yoffset, n

#%% Potential spectrum

plt.figure(figsize=figsize_single)
plt.loglog(k[1:-1], Phi2k[1:-1], label = r'$\left|\phi_{k}\right|^2$')
plt.loglog(k[1:-1][Phi2k_ZF[1:-1]>0], Phi2k_ZF[1:-1][Phi2k_ZF[1:-1]>0], label = r'$\left|\phi_{k,\mathrm{ZF}}\right|^2$')
plt.loglog(k[1:-1], Phi2k_turb[1:-1], label = r'$\left|\phi_{k,\mathrm{turb}}\right|^2$')
k_seg1, y_fit1, n1 = fit_slope(k, Phi2k, 0.8, 4.0)
plt.loglog(k_seg1, y_fit1, 'r--')
plt.axvline(x=k_f_E, color='k', linestyle=':', linewidth=2)
plt.axvline(x=k_D_E, color='k', linestyle=(0, (3, 1, 1, 1, 1, 1)), linewidth=2)
plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2)
ax = plt.gca()
ax.text(k_f_E, 1.01, r'$k_f$', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=xtick_fontsize)
ax.text(k_D_E, 1.01, r'$k_D$', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=xtick_fontsize)
ax.text(k_lin, 1.01, r'$k_{\mathrm{lin}}$', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=xtick_fontsize)
plt.xlabel('$k$')
plt.ylabel(r'$\left|\phi_{k}\right|^2$')
plt.legend()
plt.text(k_seg1[len(k_seg1)//2], y_fit1[len(k_seg1)//2] * 1.5, rf'$k^{{{n1:.2f}}}$', color='r', fontsize=32)
plt.tight_layout()
plt.savefig(savename('Phi2_spectrum'), bbox_inches='tight')
plt.show()

#%% Pressure variance spectrum

plt.figure(figsize=figsize_single)
plt.loglog(k[1:], P2k[1:], label = r'$\left|P_{k}\right|^2$')
plt.loglog(k[1:-1][P2k_ZF[1:-1]>0], P2k_ZF[1:-1][P2k_ZF[1:-1]>0], label = r'$\left|P_{k,\mathrm{ZF}}\right|^2$')
plt.loglog(k[1:], P2k_turb[1:], label = r'$\left|P_{k,\mathrm{turb}}\right|^2$')
k_seg1, y_fit1, n1 = fit_slope(k, P2k, 0.8, 4.0)
plt.loglog(k_seg1, y_fit1, 'r--')
k_seg2, y_fit2, n2 = fit_slope(k, P2k, 5, 15.0)
plt.loglog(k_seg2, y_fit2, 'r--')
plt.axvline(x=k_f_P, color='k', linestyle=':', linewidth=2)
plt.axvline(x=k_D_P, color='k', linestyle=(0, (3, 1, 1, 1, 1, 1)), linewidth=2)
plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2)
ax = plt.gca()
ax.text(k_f_P, 1.01, r'$k_f$', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=xtick_fontsize)
ax.text(k_D_P, 1.01, r'$k_D$', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=xtick_fontsize)
ax.text(k_lin, 1.01, r'$k_{\mathrm{lin}}$', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=xtick_fontsize)
plt.xlabel('$k$')
plt.ylabel(r'$\left|P_k\right|^2$')
plt.legend()
plt.text(k_seg1[len(k_seg1)//2], y_fit1[len(k_seg1)//2] * 1.5, rf'$k^{{{n1:.2f}}}$', color='r', fontsize=32)
plt.text(k_seg2[len(k_seg2)//2], y_fit2[len(k_seg2)//2] * 1.5, rf'$k^{{{n2:.2f}}}$', color='r', fontsize=32)    
plt.tight_layout()
plt.savefig(savename('EP_spectrum'), bbox_inches='tight')
plt.show()

#%% Energy spectrum

plt.figure(figsize=figsize_single)
plt.loglog(k[1:-1], Ek[1:-1], label = '$E_{k}$')
plt.loglog(k[1:-1][Ek_ZF[1:-1]>0], Ek_ZF[1:-1][Ek_ZF[1:-1]>0], label = r'$E_{k,\mathrm{ZF}}$')
plt.loglog(k[1:-1], Ek_turb[1:-1], label = r'$E_{k,\mathrm{turb}}$')
k_seg1, y_fit1, n1 = fit_slope(k, Ek, 0.8, 4.0)
plt.loglog(k_seg1, y_fit1, 'r--')
plt.axvline(x=k_f_E, color='k', linestyle=':', linewidth=2)
plt.axvline(x=k_D_E, color='k', linestyle=(0, (3, 1, 1, 1, 1, 1)), linewidth=2)
plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2)
ax = plt.gca()
ax.text(k_f_E, 1.01, r'$k_f$', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=xtick_fontsize)
ax.text(k_D_E, 1.01, r'$k_D$', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=xtick_fontsize)
ax.text(k_lin, 1.01, r'$k_{\mathrm{lin}}$', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=xtick_fontsize)
plt.xlabel('$k$')
plt.ylabel('$E_k$')
plt.legend()
plt.text(k_seg1[len(k_seg1)//2], y_fit1[len(k_seg1)//2] * 1.5, rf'$k^{{{n1:.2f}}}$', color='r', fontsize=32)
plt.tight_layout()
plt.savefig(savename('E_spectrum'), bbox_inches='tight')
plt.show()

#%% Kinetic energy spectrum

plt.figure(figsize=figsize_single)
plt.loglog(k[1:-1], Kk[1:-1], label = r'$E_{kin,k}$')
plt.loglog(k[1:-1][Kk_ZF[1:-1]>0], Kk_ZF[1:-1][Kk_ZF[1:-1]>0], label = r'$E_{kin,k,\mathrm{ZF}}$')
plt.loglog(k[1:-1], Kk_turb[1:-1], label = r'$E_{kin,k,\mathrm{turb}}$')
k_seg1, y_fit1, n1 = fit_slope(k, Kk, 0.8, 4.0)
plt.loglog(k_seg1, y_fit1, 'r--')
plt.axvline(x=k_f_E, color='k', linestyle=':', linewidth=2)
plt.axvline(x=k_D_E, color='k', linestyle=(0, (3, 1, 1, 1, 1, 1)), linewidth=2)
plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2)
ax = plt.gca()
ax.text(k_f_E, 1.01, r'$k_f$', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=xtick_fontsize)
ax.text(k_D_E, 1.01, r'$k_D$', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=xtick_fontsize)
ax.text(k_lin, 1.01, r'$k_{\mathrm{lin}}$', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=xtick_fontsize)
plt.xlabel('$k$')
plt.ylabel('$E_{kin,k}$')
plt.legend()
plt.text(k_seg1[len(k_seg1)//2], y_fit1[len(k_seg1)//2] * 1.5, rf'$k^{{{n1:.2f}}}$', color='r', fontsize=32)
plt.savefig(savename('KE_spectrum'), bbox_inches='tight')
plt.show()

#%% Enstrophy spectrum

# plt.figure(figsize=figsize_single)
# plt.loglog(k[1:-1], Wk[1:-1], label = r'$\mathcal{W}_{k}$')
# plt.loglog(k[1:-1][Wk_ZF[1:-1]>0], Wk_ZF[1:-1][Wk_ZF[1:-1]>0], label = r'$\mathcal{W}_{k,\mathrm{ZF}}$')
# plt.loglog(k[1:-1], Wk_turb[1:-1], label = r'$\mathcal{W}_{k,\mathrm{turb}}$')
# k_seg1, y_fit1, n1 = fit_slope(k, Wk, 0.8, 4.0)
# plt.loglog(k_seg1, y_fit1, 'r--')
# plt.axvline(x=1, color='k', linestyle='--', linewidth=2)
# plt.axvline(x=k_f, color='k', linestyle=':', linewidth=2)
# plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2)
# ax = plt.gca()
# ax.text(k_f, -0.025, r'$k_f$', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=xtick_fontsize)
# ax.text(k_lin, -0.025, r'$k_{\mathrm{lin}}$', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=xtick_fontsize)
# plt.xlabel('$k$')
# plt.ylabel(r'$\mathcal{W}_k$')
# plt.legend()
# # plt.text(k_seg1[-1], y_fit1[-1] * 1.5, rf'$k^{{{n1:.2f}}}$', color='r', fontsize=32)
# plt.savefig(savename('enstrophy_spectrum'), bbox_inches='tight')
# plt.show()

#%% Generalized energy spectrum

plt.figure(figsize=figsize_single)
plt.loglog(k[1:-1], Gk[1:-1], label = '$G_{k}$')
plt.loglog(k[1:-1][Gk_ZF[1:-1]>0], Gk_ZF[1:-1][Gk_ZF[1:-1]>0], label = r'$G_{k,\mathrm{ZF}}$')
plt.loglog(k[1:-1], Gk_turb[1:-1], label = r'$G_{k,\mathrm{turb}}$')
k_seg1, y_fit1, n1 = fit_slope(k, Gk, 0.8, 4.0)
plt.loglog(k_seg1, y_fit1, 'r--')
plt.axvline(x=k_f_G, color='k', linestyle=':', linewidth=2)
plt.axvline(x=k_D_G, color='k', linestyle=(0, (3, 1, 1, 1, 1, 1)), linewidth=2)
plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2)
ax = plt.gca()
ax.text(k_f_G, 1.01, r'$k_f$', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=xtick_fontsize)
ax.text(k_D_G, 1.01, r'$k_D$', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=xtick_fontsize)
ax.text(k_lin, 1.01, r'$k_{\mathrm{lin}}$', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=xtick_fontsize)
plt.xlabel('$k$')
plt.ylabel('$G_{k}$')
plt.legend()
plt.text(k_seg1[len(k_seg1)//2], y_fit1[len(k_seg1)//2] * 1.5, rf'$k^{{{n1:.2f}}}$', color='r', fontsize=32)
plt.tight_layout()
plt.savefig(savename('G_spectrum'), bbox_inches='tight')
plt.show()

#%% Generalized kinetic energy spectrum

# plt.figure(figsize=figsize_single)
# plt.loglog(k[1:-1], GKk[1:-1], label = '$G_{kin,k}$')
# plt.loglog(k[1:-1][GKk_ZF[1:-1]>0], GKk_ZF[1:-1][GKk_ZF[1:-1]>0], label = '$G_{kin,k,\mathrm{ZF}}$')
# plt.loglog(k[1:-1], GKk_turb[1:-1], label = '$G_{kin,k,\mathrm{turb}}$')
# k_seg1, y_fit1, n1 = fit_slope(k, GKk, 0.8, 4.0)
# plt.loglog(k_seg1, y_fit1, 'r--')
# plt.axvline(x=1, color='k', linestyle='--', linewidth=2)
# plt.axvline(x=k_f, color='k', linestyle=':', linewidth=2)
# plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2)
# ax = plt.gca()
# ax.text(k_f, -0.025, r'$k_f$', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=xtick_fontsize)
# ax.text(k_lin, -0.025, r'$k_{\mathrm{lin}}$', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=xtick_fontsize)
# plt.xlabel('$k$')
# plt.ylabel('$G_{kin,k}$')
# plt.legend()
# # plt.text(k_seg1[-1], y_fit1[-1] * 1.5, rf'$k^{{{n1:.2f}}}$', color='r', fontsize=32)
# plt.savefig(savename('KG_spectrum'), bbox_inches='tight')
# plt.show()

# %%
