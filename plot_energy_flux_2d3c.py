#%% Importing libraries
import numpy as np
import h5py as h5
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, skew, kurtosis

from modules.plot_basics import apply_style
apply_style()

#%% Load computed flux data

Npx=1024
datadir=f'data_2d3c/{Npx}/'
fname = datadir + 'out_2d3c_kapt_2_0_D_0_1_kz_0_1.h5'

flux_file = fname.replace('out_', 'energy_flux_')
with h5.File(flux_file, 'r') as fl:
    k         = fl['k'][:]
    k_f       = float(fl['k_f'][()])
    k_lin     = float(fl['k_lin'][()])
    Pik       = fl['Pik'][:]
    Pik_phi   = fl['Pik_phi'][:]
    Pik_d     = fl['Pik_d'][:]
    fk        = fl['fk'][:]
    dk        = fl['dk'][:]
    Pik_phi_t = fl['Pik_phi_t'][:]
    Pik_d_t   = fl['Pik_d_t'][:]
    PiGk      = fl['PiGk'][:]
    PiGk_P    = fl['PiGk_P'][:]
    PiGk_phi  = fl['PiGk_phi'][:]
    PiGk_d    = fl['PiGk_d'][:]
    fGk       = fl['fGk'][:]
    dGk       = fl['dGk'][:]
    PiGk_P_t  = fl['PiGk_P_t'][:]
    PiGk_phi_t= fl['PiGk_phi_t'][:]
    PiGk_d_t  = fl['PiGk_d_t'][:]

#%% Derived quantities for PDFs

# PDF of fluxes at k_f
idx_k_f = np.argmax(fk)
Pik_phi_series = Pik_phi_t[:, idx_k_f]
Pik_d_series   = Pik_d_t[:, idx_k_f]
Pik_series     = Pik_phi_series + Pik_d_series

PiGk_P_series   = PiGk_P_t[:, idx_k_f]
PiGk_phi_series = PiGk_phi_t[:, idx_k_f]
PiGk_d_series   = PiGk_d_t[:, idx_k_f]
PiGk_series     = PiGk_P_series + PiGk_phi_series + PiGk_d_series

PiGk_P_series_norm   = (PiGk_P_series - np.mean(PiGk_P_series)) / np.std(PiGk_P_series)
PiGk_phi_series_norm = (PiGk_phi_series - np.mean(PiGk_phi_series)) / np.std(PiGk_phi_series)
PiGk_d_series_norm   = (PiGk_d_series - np.mean(PiGk_d_series)) / np.std(PiGk_d_series)
PiGk_series_norm     = (PiGk_series - np.mean(PiGk_series)) / np.std(PiGk_series)

Pik_phi_series_norm = (Pik_phi_series - np.mean(Pik_phi_series)) / np.std(Pik_phi_series)
Pik_d_series_norm   = (Pik_d_series - np.mean(Pik_d_series)) / np.std(Pik_d_series)
Pik_series_norm     = (Pik_series - np.mean(Pik_series)) / np.std(Pik_series)

# PDF of fluxes at k=kymax
idx_k_lin = np.argmin(np.abs(k - k_lin))
Pik_phi_series_max = Pik_phi_t[:, idx_k_lin]
Pik_d_series_max   = Pik_d_t[:, idx_k_lin]
Pik_series_max     = Pik_phi_series_max + Pik_d_series_max

PiGk_P_series_max   = PiGk_P_t[:, idx_k_lin]
PiGk_phi_series_max = PiGk_phi_t[:, idx_k_lin]
PiGk_d_series_max   = PiGk_d_t[:, idx_k_lin]
PiGk_series_max     = PiGk_P_series_max + PiGk_phi_series_max + PiGk_d_series_max

PiGk_P_series_max_norm   = (PiGk_P_series_max - np.mean(PiGk_P_series_max)) / np.std(PiGk_P_series_max)
PiGk_phi_series_max_norm = (PiGk_phi_series_max - np.mean(PiGk_phi_series_max)) / np.std(PiGk_phi_series_max)
PiGk_d_series_max_norm   = (PiGk_d_series_max - np.mean(PiGk_d_series_max)) / np.std(PiGk_d_series_max)
PiGk_series_max_norm     = (PiGk_series_max - np.mean(PiGk_series_max)) / np.std(PiGk_series_max)

Pik_phi_series_max_norm = (Pik_phi_series_max - np.mean(Pik_phi_series_max)) / np.std(Pik_phi_series_max)
Pik_d_series_max_norm   = (Pik_d_series_max - np.mean(Pik_d_series_max)) / np.std(Pik_d_series_max)
Pik_series_max_norm     = (Pik_series_max - np.mean(Pik_series_max)) / np.std(Pik_series_max)

# PDF of fluxes at k=1
idx_k_1 = np.argmin(np.abs(k - 1))
Pik_phi_series_1 = Pik_phi_t[:, idx_k_1]
Pik_d_series_1   = Pik_d_t[:, idx_k_1]
Pik_series_1     = Pik_phi_series_1 + Pik_d_series_1

PiGk_P_series_1   = PiGk_P_t[:, idx_k_1]
PiGk_phi_series_1 = PiGk_phi_t[:, idx_k_1]
PiGk_d_series_1   = PiGk_d_t[:, idx_k_1]
PiGk_series_1     = PiGk_P_series_1 + PiGk_phi_series_1 + PiGk_d_series_1

PiGk_P_series_1_norm   = (PiGk_P_series_1 - np.mean(PiGk_P_series_1)) / np.std(PiGk_P_series_1)
PiGk_phi_series_1_norm = (PiGk_phi_series_1 - np.mean(PiGk_phi_series_1)) / np.std(PiGk_phi_series_1)
PiGk_d_series_1_norm   = (PiGk_d_series_1 - np.mean(PiGk_d_series_1)) / np.std(PiGk_d_series_1)
PiGk_series_1_norm     = (PiGk_series_1 - np.mean(PiGk_series_1)) / np.std(PiGk_series_1)

Pik_phi_series_1_norm = (Pik_phi_series_1 - np.mean(Pik_phi_series_1)) / np.std(Pik_phi_series_1)
Pik_d_series_1_norm   = (Pik_d_series_1 - np.mean(Pik_d_series_1)) / np.std(Pik_d_series_1)
Pik_series_1_norm     = (Pik_series_1 - np.mean(Pik_series_1)) / np.std(Pik_series_1)

#%% Plots

plt.figure(figsize=(16, 9))
plt.plot(k[1:-1], Pik[1:-1], label = r'$\Pi_{k}$')
plt.plot(k[1:-1], Pik_phi[1:-1], label = r'$\Pi_{k,\mathrm{\phi}}$')
plt.plot(k[1:-1], Pik_d[1:-1], label = r'$\Pi_{k,\mathrm{d}}$')
plt.axhline(0, color='k', linestyle='-', linewidth=1)
plt.axvline(x=k_f, color='k', linestyle=':', linewidth=2, label=f'$k_f={k_f:.2f}$')
plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2, label=f'$k_{{lin}}={k_lin:.2f}$')
plt.xscale('log')
plt.xlabel('$k$')
plt.ylabel(r'$\Pi_k$')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
if fname.endswith('out.h5'):
    plt.savefig(datadir+'E_flux.pdf', dpi=100)
else:
    plt.savefig(datadir+"E_flux_" + fname.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf'), dpi=100)
plt.show()

plt.figure(figsize=(16, 9))
plt.plot(k[1:-1], fk[1:-1], label = r'$f_{k,\mathrm{total}}$')
plt.axhline(0, color='k', linestyle='-', linewidth=1)
plt.axvline(x=k_f, color='k', linestyle=':', linewidth=2, label=f'$k_f={k_f:.2f}$')
plt.xscale('log')
plt.xlabel('$k$')
plt.ylabel(r'$f_k$')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
if fname.endswith('out.h5'):
    plt.savefig(datadir+'E_injection.pdf', dpi=100)
else:
    plt.savefig(datadir+"E_injection_" + fname.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf'), dpi=100)
plt.show()

plt.figure(figsize=(16, 9))
plt.plot(k[1:-1], dk[1:-1], label = r'$d_{k,\mathrm{total}}$')
plt.axhline(0, color='k', linestyle='-', linewidth=1)
plt.axvline(x=k_f, color='k', linestyle=':', linewidth=2, label=f'$k_f={k_f:.2f}$')
plt.xscale('log')
plt.xlabel('$k$')
plt.ylabel(r'$d_k$')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
if fname.endswith('out.h5'):
    plt.savefig(datadir+'E_dissipation.pdf', dpi=100)
else:
    plt.savefig(datadir+"E_dissipation_" + fname.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf'), dpi=100)
plt.show()

# PDF of fluxes at k_f
plt.figure(figsize=(16, 9))
xlim = max(np.percentile(np.abs(s), 99) for s in [Pik_series_norm, Pik_phi_series_norm, Pik_d_series_norm])
for series, label, color in zip([Pik_series_norm, Pik_phi_series_norm, Pik_d_series_norm],
                        [r'$\Pi_{k}$', r'$\Pi_{k,\mathrm{\phi}}$', r'$\Pi_{k,\mathrm{d}}$'],
                        ['C0', 'C1', 'C2']):
    s = skew(series)
    f = kurtosis(series, fisher=False)  # Gaussian = 3
    kde = gaussian_kde(series)
    x_range = np.linspace(-xlim, xlim, 200)
    plt.hist(series, bins=50, density=True, alpha=0.3, color=color, range=(-xlim, xlim))
    plt.plot(x_range, kde(x_range), label=rf'{label}  $S={s:.2f},\ F={f:.2f}$', color=color)
plt.xlabel(r'$\frac{\Pi_k-<\Pi_k>}{\sigma}$')
plt.ylabel('PDF')
plt.xlim(-xlim, xlim)
plt.gca().text(0.97, 0.97, rf'$k_f={k_f:.2f}$', transform=plt.gca().transAxes,
    fontsize=20, verticalalignment='top', horizontalalignment='right',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8))
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
if fname.endswith('out.h5'):
    plt.savefig(datadir+'E_flux_pdf_.pdf', dpi=100)
else:
    plt.savefig(datadir+"E_flux_pdf_" + fname.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf'), dpi=100)
plt.show()

# PDF of fluxes at k=kymax
plt.figure(figsize=(16, 9))
xlim = max(np.percentile(np.abs(s), 99) for s in [Pik_series_max_norm, Pik_phi_series_max_norm, Pik_d_series_max_norm])
for series, label, color in zip([Pik_series_max_norm, Pik_phi_series_max_norm, Pik_d_series_max_norm],
                        [r'$\Pi_{k}$', r'$\Pi_{k,\mathrm{\phi}}$', r'$\Pi_{k,\mathrm{d}}$'],
                        ['C0', 'C1', 'C2']):
    s = skew(series)
    f = kurtosis(series, fisher=False)  # Gaussian = 3
    kde = gaussian_kde(series)
    x_range = np.linspace(-xlim, xlim, 200)
    plt.hist(series, bins=50, density=True, alpha=0.3, color=color, range=(-xlim, xlim))
    plt.plot(x_range, kde(x_range), label=rf'{label}  $S={s:.2f},\ F={f:.2f}$', color=color)
plt.xlabel(r'$\frac{\Pi_k-<\Pi_k>}{\sigma}$')
plt.ylabel('PDF')
plt.xlim(-xlim, xlim)
plt.gca().text(0.97, 0.97, rf'$k_{{lin}}={k_lin:.2f}$', transform=plt.gca().transAxes,
    fontsize=20, verticalalignment='top', horizontalalignment='right',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8))
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
if fname.endswith('out.h5'):
    plt.savefig(datadir+'E_flux_kymax_pdf_.pdf', dpi=100)
else:
    plt.savefig(datadir+"E_flux_kymax_pdf_" + fname.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf'), dpi=100)
plt.show()

# PDF of fluxes at k=1
plt.figure(figsize=(16, 9))
xlim = max(np.percentile(np.abs(s), 99) for s in [Pik_series_1_norm, Pik_phi_series_1_norm, Pik_d_series_1_norm])
for series, label, color in zip([Pik_series_1_norm, Pik_phi_series_1_norm, Pik_d_series_1_norm],
                        [r'$\Pi_{k}$', r'$\Pi_{k,\mathrm{\phi}}$', r'$\Pi_{k,\mathrm{d}}$'],
                        ['C0', 'C1', 'C2']):
    s = skew(series)
    f = kurtosis(series, fisher=False)  # Gaussian = 3
    kde = gaussian_kde(series)
    x_range = np.linspace(-xlim, xlim, 200)
    plt.hist(series, bins=50, density=True, alpha=0.3, color=color, range=(-xlim, xlim))
    plt.plot(x_range, kde(x_range), label=rf'{label}  $S={s:.2f},\ F={f:.2f}$', color=color)
plt.xlabel(r'$\frac{\Pi_k-<\Pi_k>}{\sigma}$')
plt.ylabel('PDF')
plt.xlim(-xlim, xlim)
plt.gca().text(0.97, 0.97, r'$k=1$', transform=plt.gca().transAxes,
    fontsize=20, verticalalignment='top', horizontalalignment='right',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8))
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
if fname.endswith('out.h5'):
    plt.savefig(datadir+'E_flux_k1_pdf_.pdf', dpi=100)
else:
    plt.savefig(datadir+"E_flux_k1_pdf_" + fname.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf'), dpi=100)
plt.show()

# G-flux: PiGk components
plt.figure(figsize=(16, 9))
plt.plot(k[1:-1], PiGk[1:-1],     label=r'$\Pi_{G,k}$')
plt.plot(k[1:-1], PiGk_P[1:-1],   label=r'$\Pi_{G,k,P}$')
plt.plot(k[1:-1], PiGk_phi[1:-1], label=r'$\Pi_{G,k,\phi}$')
plt.plot(k[1:-1], PiGk_d[1:-1],   label=r'$\Pi_{G,k,d}$')
plt.axhline(0, color='k', linestyle='-', linewidth=1)
plt.axvline(x=k_f,   color='k', linestyle=':',  linewidth=2, label=f'$k_f={k_f:.2f}$')
plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2, label=f'$k_{{lin}}={k_lin:.2f}$')
plt.xscale('log')
plt.xlabel('$k$')
plt.ylabel(r'$\Pi_{G,k}$')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
if fname.endswith('out.h5'):
    plt.savefig(datadir+'G_flux.pdf', dpi=100)
else:
    plt.savefig(datadir+'G_flux_' + fname.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf'), dpi=100)
plt.show()

# G-flux: fGk
plt.figure(figsize=(16, 9))
plt.plot(k[1:-1], fGk[1:-1], label=r'$f_{G,k}$')
plt.axhline(0, color='k', linestyle='-', linewidth=1)
plt.axvline(x=k_f, color='k', linestyle=':', linewidth=2, label=f'$k_f={k_f:.2f}$')
plt.xscale('log')
plt.xlabel('$k$')
plt.ylabel(r'$f_{G,k}$')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
if fname.endswith('out.h5'):
    plt.savefig(datadir+'G_injection.pdf', dpi=100)
else:
    plt.savefig(datadir+'G_injection_' + fname.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf'), dpi=100)
plt.show()

# G-flux: dGk
plt.figure(figsize=(16, 9))
plt.plot(k[1:-1], dGk[1:-1], label=r'$d_{G,k}$')
plt.axhline(0, color='k', linestyle='-', linewidth=1)
plt.axvline(x=k_f, color='k', linestyle=':', linewidth=2, label=f'$k_f={k_f:.2f}$')
plt.xscale('log')
plt.xlabel('$k$')
plt.ylabel(r'$d_{G,k}$')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
if fname.endswith('out.h5'):
    plt.savefig(datadir+'G_dissipation.pdf', dpi=100)
else:
    plt.savefig(datadir+'G_dissipation_' + fname.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf'), dpi=100)
plt.show()

# G-flux PDF at k_f
plt.figure(figsize=(16, 9))
xlim = max(np.percentile(np.abs(s), 99) for s in [PiGk_series_norm, PiGk_P_series_norm, PiGk_phi_series_norm, PiGk_d_series_norm])
for series, label, color in zip(
        [PiGk_series_norm, PiGk_P_series_norm, PiGk_phi_series_norm, PiGk_d_series_norm],
        [r'$\Pi_{G,k}$', r'$\Pi_{G,k,P}$', r'$\Pi_{G,k,\phi}$', r'$\Pi_{G,k,d}$'],
        ['C0', 'C1', 'C2', 'C3']):
    s = skew(series)
    f = kurtosis(series, fisher=False)
    kde = gaussian_kde(series)
    x_range = np.linspace(-xlim, xlim, 200)
    plt.hist(series, bins=50, density=True, alpha=0.3, color=color, range=(-xlim, xlim))
    plt.plot(x_range, kde(x_range), label=rf'{label}  $S={s:.2f},\ F={f:.2f}$', color=color)
plt.xlabel(r'$\frac{\Pi_{G,k}-<\Pi_{G,k}>}{\sigma}$')
plt.ylabel('PDF')
plt.xlim(-xlim, xlim)
plt.gca().text(0.97, 0.97, rf'$k_f={k_f:.2f}$', transform=plt.gca().transAxes,
    fontsize=20, verticalalignment='top', horizontalalignment='right',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8))
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
if fname.endswith('out.h5'):
    plt.savefig(datadir+'G_flux_pdf_.pdf', dpi=100)
else:
    plt.savefig(datadir+'G_flux_pdf_' + fname.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf'), dpi=100)
plt.show()

# G-flux PDF at k_lin
plt.figure(figsize=(16, 9))
xlim = max(np.percentile(np.abs(s), 99) for s in [PiGk_series_max_norm, PiGk_P_series_max_norm, PiGk_phi_series_max_norm, PiGk_d_series_max_norm])
for series, label, color in zip(
        [PiGk_series_max_norm, PiGk_P_series_max_norm, PiGk_phi_series_max_norm, PiGk_d_series_max_norm],
        [r'$\Pi_{G,k}$', r'$\Pi_{G,k,P}$', r'$\Pi_{G,k,\phi}$', r'$\Pi_{G,k,d}$'],
        ['C0', 'C1', 'C2', 'C3']):
    s = skew(series)
    f = kurtosis(series, fisher=False)
    kde = gaussian_kde(series)
    x_range = np.linspace(-xlim, xlim, 200)
    plt.hist(series, bins=50, density=True, alpha=0.3, color=color, range=(-xlim, xlim))
    plt.plot(x_range, kde(x_range), label=rf'{label}  $S={s:.2f},\ F={f:.2f}$', color=color)
plt.xlabel(r'$\frac{\Pi_{G,k}-<\Pi_{G,k}>}{\sigma}$')
plt.ylabel('PDF')
plt.xlim(-xlim, xlim)
plt.gca().text(0.97, 0.97, rf'$k_{{lin}}={k_lin:.2f}$', transform=plt.gca().transAxes,
    fontsize=20, verticalalignment='top', horizontalalignment='right',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8))
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
if fname.endswith('out.h5'):
    plt.savefig(datadir+'G_flux_kymax_pdf_.pdf', dpi=100)
else:
    plt.savefig(datadir+'G_flux_kymax_pdf_' + fname.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf'), dpi=100)
plt.show()

# G-flux PDF at k=1
plt.figure(figsize=(16, 9))
xlim = max(np.percentile(np.abs(s), 99) for s in [PiGk_series_1_norm, PiGk_P_series_1_norm, PiGk_phi_series_1_norm, PiGk_d_series_1_norm])
for series, label, color in zip(
        [PiGk_series_1_norm, PiGk_P_series_1_norm, PiGk_phi_series_1_norm, PiGk_d_series_1_norm],
        [r'$\Pi_{G,k}$', r'$\Pi_{G,k,P}$', r'$\Pi_{G,k,\phi}$', r'$\Pi_{G,k,d}$'],
        ['C0', 'C1', 'C2', 'C3']):
    s = skew(series)
    f = kurtosis(series, fisher=False)
    kde = gaussian_kde(series)
    x_range = np.linspace(-xlim, xlim, 200)
    plt.hist(series, bins=50, density=True, alpha=0.3, color=color, range=(-xlim, xlim))
    plt.plot(x_range, kde(x_range), label=rf'{label}  $S={s:.2f},\ F={f:.2f}$', color=color)
plt.xlabel(r'$\frac{\Pi_{G,k}-<\Pi_{G,k}>}{\sigma}$')
plt.ylabel('PDF')
plt.xlim(-xlim, xlim)
plt.gca().text(0.97, 0.97, r'$k=1$', transform=plt.gca().transAxes,
    fontsize=20, verticalalignment='top', horizontalalignment='right',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8))
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
if fname.endswith('out.h5'):
    plt.savefig(datadir+'G_flux_k1_pdf_.pdf', dpi=100)
else:
    plt.savefig(datadir+'G_flux_k1_pdf_' + fname.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf'), dpi=100)
plt.show()
