#%% Importing libraries
import numpy as np
import h5py as h5
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis # This line is correct
from functools import partial
from modules.plot_basics import apply_style, savename as _savename, figsize_single
apply_style()

#%% Load computed flux data

# Npx=512
Npx=1024
datadir=f'data/{Npx}/'
subdir='spectral_flux/'

fname = 'out_kapt_0_4_D_0_1_H_3_6_em6.h5'
# fname = 'out_kapt_2_0_D_0_1_H_8_6_em6.h5'
# fname = 'out_kapt_2_0_D_0_1_H_1_7_em5.h5'

flux_file = datadir + subdir + fname.replace('out_', 'spectral_flux_')
with h5.File(flux_file, 'r') as fl:
    k          = fl['k'][:]
    k_Gf       = float(fl['k_Gf'][()])
    k_lin      = float(fl['k_lin'][()])
    PiGk       = fl['PiGk'][:]
    PiGk_P     = fl['PiGk_P'][:]
    PiGk_phi   = fl['PiGk_phi'][:]
    PiGk_d     = fl['PiGk_d'][:]
    fGk        = fl['fGk'][:]
    dGk        = fl['dGk'][:]
    PiGk_P_t   = fl['PiGk_P_t'][:]
    PiGk_phi_t = fl['PiGk_phi_t'][:]
    PiGk_d_t   = fl['PiGk_d_t'][:]

#%% Functions

savename = partial(_savename, datadir+subdir, fname)

def smooth_pdf(series, bins=50, window=7, xlim=None):
    N = len(series)
    rng = (-xlim, xlim) if xlim is not None else None
    # bin the data into raw integer counts (not normalised)
    raw, edges = np.histogram(series, bins=bins, density=False, range=rng)
    bin_width = edges[1] - edges[0]
    centres = 0.5 * (edges[:-1] + edges[1:])
    # box-kernel running average: replace each bin count with the mean of
    # itself and (window-1)/2 neighbours on each side, removing spiky noise
    kernel = np.ones(window) / window
    smooth = np.convolve(raw, kernel, mode='same')
    # normalise smoothed counts to a probability density (area under curve = 1)
    density = smooth / (N * bin_width)
    # Poisson uncertainty: counts follow Poisson stats so std = sqrt(n)
    err = np.sqrt(smooth) / (N * bin_width)
    return centres, density, err

#%% Derived quantities for PDFs

idx_k_Gf = np.argmin(np.abs(k - k_Gf))
PiGk_P_series   = PiGk_P_t[:, idx_k_Gf]
PiGk_phi_series = PiGk_phi_t[:, idx_k_Gf]
PiGk_d_series   = PiGk_d_t[:, idx_k_Gf]
PiGk_series     = PiGk_P_series + PiGk_phi_series + PiGk_d_series

PiGk_P_series_norm   = (PiGk_P_series - np.mean(PiGk_P_series)) / np.std(PiGk_P_series)
PiGk_phi_series_norm = (PiGk_phi_series - np.mean(PiGk_phi_series)) / np.std(PiGk_phi_series)
PiGk_d_series_norm   = (PiGk_d_series - np.mean(PiGk_d_series)) / np.std(PiGk_d_series)
PiGk_series_norm     = (PiGk_series - np.mean(PiGk_series)) / np.std(PiGk_series)

idx_k_lin = np.argmin(np.abs(k - k_lin))
PiGk_P_series_max   = PiGk_P_t[:, idx_k_lin]
PiGk_phi_series_max = PiGk_phi_t[:, idx_k_lin]
PiGk_d_series_max   = PiGk_d_t[:, idx_k_lin]
PiGk_series_max     = PiGk_P_series_max + PiGk_phi_series_max + PiGk_d_series_max

PiGk_P_series_max_norm   = (PiGk_P_series_max - np.mean(PiGk_P_series_max)) / np.std(PiGk_P_series_max)
PiGk_phi_series_max_norm = (PiGk_phi_series_max - np.mean(PiGk_phi_series_max)) / np.std(PiGk_phi_series_max)
PiGk_d_series_max_norm   = (PiGk_d_series_max - np.mean(PiGk_d_series_max)) / np.std(PiGk_d_series_max)
PiGk_series_max_norm     = (PiGk_series_max - np.mean(PiGk_series_max)) / np.std(PiGk_series_max)

idx_k_1 = np.argmin(np.abs(k - 1))
PiGk_P_series_1   = PiGk_P_t[:, idx_k_1]
PiGk_phi_series_1 = PiGk_phi_t[:, idx_k_1]
PiGk_d_series_1   = PiGk_d_t[:, idx_k_1]
PiGk_series_1     = PiGk_P_series_1 + PiGk_phi_series_1 + PiGk_d_series_1

PiGk_P_series_1_norm   = (PiGk_P_series_1 - np.mean(PiGk_P_series_1)) / np.std(PiGk_P_series_1)
PiGk_phi_series_1_norm = (PiGk_phi_series_1 - np.mean(PiGk_phi_series_1)) / np.std(PiGk_phi_series_1)
PiGk_d_series_1_norm   = (PiGk_d_series_1 - np.mean(PiGk_d_series_1)) / np.std(PiGk_d_series_1)
PiGk_series_1_norm     = (PiGk_series_1 - np.mean(PiGk_series_1)) / np.std(PiGk_series_1)

#%% Gk-flux PDF smoothed at k_Gf

plt.figure(figsize=figsize_single)
xlim = max(np.percentile(np.abs(s), 95) for s in [PiGk_series_norm, PiGk_phi_series_norm, PiGk_d_series_norm, PiGk_P_series_norm])
for series, label, color in zip(
        [PiGk_series_norm, PiGk_phi_series_norm, PiGk_d_series_norm, PiGk_P_series_norm],
        [r'$\Pi_{G,k}$', r'$\Pi_{G,k}^{(\phi)}$', r'$\Pi_{G,k}^{(d)}$', r'$\Pi_{G,k}^{(P)}$'],
        ['C0', 'C1', 'C2', 'C3']):
    centres, density, err = smooth_pdf(series, bins=50, window=7, xlim=xlim)
    mask = density > 0
    lo = np.maximum(density[mask] - err[mask], density[mask] * 1e-6)
    plt.fill_between(centres[mask], lo, density[mask] + err[mask], alpha=0.2, color=color)
    plt.plot(centres[mask], density[mask], '-', color=color, label=rf'{label}')
plt.xlabel(r'$\frac{\Pi_{G,k}-<\Pi_{G,k}>}{\sigma}$')
plt.ylabel('PDF')
plt.xlim(-xlim, xlim)
plt.gca().text(0.97, 0.97, rf'$k_{{Gf}}={k_Gf:.2f}$', transform=plt.gca().transAxes,
    fontsize=20, verticalalignment='top', horizontalalignment='right',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8))
plt.legend()
plt.tight_layout()
plt.savefig(savename('Gk_flux_pdf'), bbox_inches='tight')
plt.show()

#%% Gk-flux PDF smoothed at k_lin

plt.figure(figsize=figsize_single)
xlim = max(np.percentile(np.abs(s), 95) for s in [PiGk_series_max_norm, PiGk_phi_series_max_norm, PiGk_d_series_max_norm, PiGk_P_series_max_norm])
for series, label, color in zip(
        [PiGk_series_max_norm, PiGk_phi_series_max_norm, PiGk_d_series_max_norm, PiGk_P_series_max_norm],
        [r'$\Pi_{G,k}$', r'$\Pi_{G,k}^{(\phi)}$', r'$\Pi_{G,k}^{(d)}$', r'$\Pi_{G,k}^{(P)}$'],
        ['C0', 'C1', 'C2', 'C3']):
    centres, density, err = smooth_pdf(series, bins=50, window=7, xlim=xlim)
    mask = density > 0
    lo = np.maximum(density[mask] - err[mask], density[mask] * 1e-6)
    plt.fill_between(centres[mask], lo, density[mask] + err[mask], alpha=0.2, color=color)
    plt.plot(centres[mask], density[mask], '-', color=color, label=rf'{label}')
plt.xlabel(r'$\frac{\Pi_{G,k}-<\Pi_{G,k}>}{\sigma}$')
plt.ylabel('PDF')
plt.xlim(-xlim, xlim)
plt.gca().text(0.97, 0.97, rf'$k_{{lin}}={k_lin:.2f}$', transform=plt.gca().transAxes,
    fontsize=20, verticalalignment='top', horizontalalignment='right',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8))
plt.legend()
plt.tight_layout()
plt.savefig(savename('Gk_flux_kymax_pdf'), bbox_inches='tight')
plt.show()

#%% Gk-flux PDF smoothed at k=1

plt.figure(figsize=figsize_single)
xlim = max(np.percentile(np.abs(s), 95) for s in [PiGk_series_1_norm, PiGk_phi_series_1_norm, PiGk_d_series_1_norm, PiGk_P_series_1_norm])
for series, label, color in zip(
        [PiGk_series_1_norm, PiGk_phi_series_1_norm, PiGk_d_series_1_norm, PiGk_P_series_1_norm],
        [r'$\Pi_{G,k}$', r'$\Pi_{G,k}^{(\phi)}$', r'$\Pi_{G,k}^{(d)}$', r'$\Pi_{G,k}^{(P)}$'],
        ['C0', 'C1', 'C2', 'C3']):
    centres, density, err = smooth_pdf(series, bins=50, window=7, xlim=xlim)
    mask = density > 0
    lo = np.maximum(density[mask] - err[mask], density[mask] * 1e-6)
    plt.fill_between(centres[mask], lo, density[mask] + err[mask], alpha=0.2, color=color)
    plt.plot(centres[mask], density[mask], '-', color=color, label=rf'{label}')
plt.xlabel(r'$\frac{\Pi_{G,k}-<\Pi_{G,k}>}{\sigma}$')
plt.ylabel('PDF')
plt.xlim(-xlim, xlim)
plt.gca().text(0.97, 0.97, r'$k=1$', transform=plt.gca().transAxes,
    fontsize=20, verticalalignment='top', horizontalalignment='right',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8))
plt.legend()
plt.tight_layout()
plt.savefig(savename('Gk_flux_k1_pdf'), bbox_inches='tight')
plt.show()