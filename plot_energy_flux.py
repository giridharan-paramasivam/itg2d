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

# Npx=512
Npx=1024
datadir=f'data/{Npx}/'

# fname = datadir + 'out_kapt_0_4_D_0_1_H_3_6_em6.h5'
fname = datadir + 'out_kapt_2_0_D_0_1_H_8_6_em6.h5'
# fname = datadir + 'out_kapt_2_0_D_0_1_H_1_7_em5.h5'

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

#%% Derived quantities for PDFs

idx_k_f = np.argmin(np.abs(k - k_f))
Pik_phi_series = Pik_phi_t[:, idx_k_f]
Pik_d_series   = Pik_d_t[:, idx_k_f]
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

savename = partial(_savename, datadir, fname)

#%% E-flux

plt.figure(figsize=(16, 9))
plt.plot(k[1:-1], Pik[1:-1], label = r'$\Pi_{k}$')
plt.plot(k[1:-1], Pik_phi[1:-1], label = r'$\Pi_{k,\mathrm{\phi}}$')
plt.plot(k[1:-1], Pik_d[1:-1], label = r'$\Pi_{k,\mathrm{d}}$')
plt.axhline(0,color='k', linestyle='-', linewidth=1)
plt.axvline(x=k_f, color='k', linestyle=':', linewidth=2, label=f'$k_f={k_f:.2f}$')
plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2, label=f'$k_{{lin}}={k_lin:.2f}$')
plt.xscale('log')
plt.xlabel('$k$')
plt.ylabel(r'$\Pi_k$')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(savename('E_flux'), dpi=100)
plt.show()

#%% E-flux: injection

plt.figure(figsize=(16, 9))
plt.plot(k[1:-1], fk[1:-1], label = r'$f_{k,\mathrm{total}}$')
plt.axhline(0,color='k', linestyle='-', linewidth=1)
plt.axvline(x=k_f, color='k', linestyle=':', linewidth=2, label=f'$k_f={k_f:.2f}$')
plt.xscale('log')
plt.xlabel('$k$')
plt.ylabel(r'$f_k$')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(savename('E_injection'), dpi=100)
plt.show()

#%% E-flux: dissipation

plt.figure(figsize=(16, 9))
plt.plot(k[1:-1], dk[1:-1], label = r'$d_{k,\mathrm{total}}$')
plt.axhline(0,color='k', linestyle='-', linewidth=1)
plt.axvline(x=k_f, color='k', linestyle=':', linewidth=2, label=f'$k_f={k_f:.2f}$')
plt.xscale('log')
plt.xlabel('$k$')
plt.ylabel(r'$d_k$')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(savename('E_dissipation'), dpi=100)
plt.show()

#%% E-flux PDF at k_f

plt.figure(figsize=(16, 9))
xlim = max(np.percentile(np.abs(s), 95) for s in [Pik_series_norm, Pik_phi_series_norm, Pik_d_series_norm])
for series, label, color in zip([Pik_series_norm, Pik_phi_series_norm, Pik_d_series_norm],
                        [r'$\Pi_{k}$', r'$\Pi_{k,\mathrm{\phi}}$', r'$\Pi_{k,\mathrm{d}}$'],
                        ['C0', 'C1', 'C2']):
    s = skew(series)
    f = kurtosis(series, fisher=False)  # Gaussian = 3
    counts, edges = np.histogram(series, bins=200, density=True, range=(-xlim, xlim))
    centres = 0.5 * (edges[:-1] + edges[1:])
    mask = counts > 0
    plt.plot(centres[mask], counts[mask], '.-', color=color,
             label=rf'{label}  $S={s:.2f},\ F={f:.2f}$')
plt.xlabel(r'$\frac{\Pi_k-<\Pi_k>}{\sigma}$')
plt.ylabel('PDF')
plt.yscale('log')
plt.xlim(-xlim, xlim)
plt.gca().text(0.97, 0.97, rf'$k_f={k_f:.2f}$', transform=plt.gca().transAxes,
    fontsize=20, verticalalignment='top', horizontalalignment='right',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8))
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(savename('E_flux_pdf'), dpi=100)
plt.show()

#%% E-flux PDF at k=kymax

plt.figure(figsize=(16, 9))
xlim = max(np.percentile(np.abs(s), 95) for s in [Pik_series_max_norm, Pik_phi_series_max_norm, Pik_d_series_max_norm])
for series, label, color in zip([Pik_series_max_norm, Pik_phi_series_max_norm, Pik_d_series_max_norm],
                        [r'$\Pi_{k}$', r'$\Pi_{k,\mathrm{\phi}}$', r'$\Pi_{k,\mathrm{d}}$'],
                        ['C0', 'C1', 'C2']):
    s = skew(series)
    f = kurtosis(series, fisher=False)  # Gaussian = 3
    counts, edges = np.histogram(series, bins=200, density=True, range=(-xlim, xlim))
    centres = 0.5 * (edges[:-1] + edges[1:])
    mask = counts > 0
    plt.plot(centres[mask], counts[mask], '.-', color=color,
             label=rf'{label}  $S={s:.2f},\ F={f:.2f}$')
plt.xlabel(r'$\frac{\Pi_k-<\Pi_k>}{\sigma}$')
plt.ylabel('PDF')
plt.yscale('log')
plt.xlim(-xlim, xlim)
plt.gca().text(0.97, 0.97, rf'$k_{{lin}}={k_lin:.2f}$', transform=plt.gca().transAxes,
    fontsize=20, verticalalignment='top', horizontalalignment='right',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8))
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(savename('E_flux_kymax_pdf'), dpi=100)
plt.show()

#%% E-flux PDF at k=1

plt.figure(figsize=(16, 9))
xlim = max(np.percentile(np.abs(s), 95) for s in [Pik_series_1_norm, Pik_phi_series_1_norm, Pik_d_series_1_norm])
for series, label, color in zip([Pik_series_1_norm, Pik_phi_series_1_norm, Pik_d_series_1_norm],
                        [r'$\Pi_{k}$', r'$\Pi_{k,\mathrm{\phi}}$', r'$\Pi_{k,\mathrm{d}}$'],
                        ['C0', 'C1', 'C2']):
    s = skew(series)
    f = kurtosis(series, fisher=False)  # Gaussian = 3
    counts, edges = np.histogram(series, bins=200, density=True, range=(-xlim, xlim))
    centres = 0.5 * (edges[:-1] + edges[1:])
    mask = counts > 0
    plt.plot(centres[mask], counts[mask], '.-', color=color,
             label=rf'{label}  $S={s:.2f},\ F={f:.2f}$')
plt.xlabel(r'$\frac{\Pi_k-<\Pi_k>}{\sigma}$')
plt.ylabel('PDF')
plt.yscale('log')
plt.xlim(-xlim, xlim)
plt.gca().text(0.97, 0.97, r'$k=1$', transform=plt.gca().transAxes,
    fontsize=20, verticalalignment='top', horizontalalignment='right',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8))
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(savename('E_flux_k1_pdf'), dpi=100)
plt.show()

#%% E-flux CCDF at k_f

plt.figure(figsize=(16, 9))
for series, label, color in zip(
        [Pik_series_norm, Pik_phi_series_norm, Pik_d_series_norm],
        [r'$\Pi_{k}$', r'$\Pi_{k,\mathrm{\phi}}$', r'$\Pi_{k,\mathrm{d}}$'],
        ['C0', 'C1', 'C2']):
    sorted_s = np.sort(series)
    ccdf = 1 - np.arange(1, len(sorted_s) + 1) / len(sorted_s)
    plt.step(sorted_s, ccdf, where='post', color=color, label=label)
plt.xlabel(r'$X$')
plt.ylabel(r'$P\left(\frac{\Pi_k-\langle\Pi_k\rangle}{\sigma}>X\right)$')
plt.yscale('log')
plt.gca().text(0.97, 0.97, rf'$k_f={k_f:.2f}$', transform=plt.gca().transAxes,
    fontsize=20, verticalalignment='top', horizontalalignment='right',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8))
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(savename('E_flux_ccdf'), dpi=100)
plt.show()

#%% E-flux CCDF at k_lin

plt.figure(figsize=(16, 9))
for series, label, color in zip(
        [Pik_series_max_norm, Pik_phi_series_max_norm, Pik_d_series_max_norm],
        [r'$\Pi_{k}$', r'$\Pi_{k,\mathrm{\phi}}$', r'$\Pi_{k,\mathrm{d}}$'],
        ['C0', 'C1', 'C2']):
    sorted_s = np.sort(series)
    ccdf = 1 - np.arange(1, len(sorted_s) + 1) / len(sorted_s)
    plt.step(sorted_s, ccdf, where='post', color=color, label=label)
plt.xlabel(r'$X$')
plt.ylabel(r'$P\left(\frac{\Pi_k-\langle\Pi_k\rangle}{\sigma}>X\right)$')
plt.yscale('log')
plt.gca().text(0.97, 0.97, rf'$k_{{lin}}={k_lin:.2f}$', transform=plt.gca().transAxes,
    fontsize=20, verticalalignment='top', horizontalalignment='right',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8))
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(savename('E_flux_kymax_ccdf'), dpi=100)
plt.show()

#%% E-flux CCDF at k=1

plt.figure(figsize=(16, 9))
for series, label, color in zip(
        [Pik_series_1_norm, Pik_phi_series_1_norm, Pik_d_series_1_norm],
        [r'$\Pi_{k}$', r'$\Pi_{k,\mathrm{\phi}}$', r'$\Pi_{k,\mathrm{d}}$'],
        ['C0', 'C1', 'C2']):
    sorted_s = np.sort(series)
    ccdf = 1 - np.arange(1, len(sorted_s) + 1) / len(sorted_s)
    plt.step(sorted_s, ccdf, where='post', color=color, label=label)
plt.xlabel(r'$X$')
plt.ylabel(r'$P\left(\frac{\Pi_k-\langle\Pi_k\rangle}{\sigma}>X\right)$')
plt.yscale('log')
plt.gca().text(0.97, 0.97, r'$k=1$', transform=plt.gca().transAxes,
    fontsize=20, verticalalignment='top', horizontalalignment='right',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8))
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(savename('E_flux_k1_ccdf'), dpi=100)
plt.show()
