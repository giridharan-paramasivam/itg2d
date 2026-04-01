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
    k = fl['k'][:]
    k_Pf = float(fl['k_Pf'][()])
    k_lin = float(fl['k_lin'][()])
    PiPk = fl['PiPk'][:]
    fPk = fl['fPk'][:]
    dPk = fl['dPk'][:]
    PiPk_t = fl['PiPk_t'][:]

#%% Functions

savename = partial(_savename, datadir+subdir, fname)

def smooth_pdf(series, bins=100, window=7, xlim=None):
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

idx_k_Pf = np.argmin(np.abs(k - k_Pf))
PiPk_series = PiPk_t[:, idx_k_Pf]
PiPk_series_norm = (PiPk_series - np.mean(PiPk_series)) / np.std(PiPk_series)

idx_k_lin = np.argmin(np.abs(k - k_lin))
PiPk_series_max = PiPk_t[:, idx_k_lin]
PiPk_series_max_norm = (PiPk_series_max - np.mean(PiPk_series_max)) / np.std(PiPk_series_max)

idx_k_1 = np.argmin(np.abs(k - 1))
PiPk_series_1 = PiPk_t[:, idx_k_1]
PiPk_series_1_norm = (PiPk_series_1 - np.mean(PiPk_series_1)) / np.std(PiPk_series_1)

#%% EPk-flux PDF smoothed at k_Pf

plt.figure(figsize=figsize_single)
xlim = max(np.percentile(np.abs(s), 95) for s in [PiPk_series_norm])
for series, label, color in zip([PiPk_series_norm],
                        [r'$\Pi_{P,k}$'],
                        ['C0']):
    centres, density, err = smooth_pdf(series, bins=50, window=7, xlim=xlim)
    mask = density > 0
    lo = np.maximum(density[mask] - err[mask], density[mask] * 1e-6)
    plt.fill_between(centres[mask], lo, density[mask] + err[mask], alpha=0.2, color=color)
    plt.plot(centres[mask], density[mask], '-', color=color, label=rf'{label}')
plt.xlabel(r'$\frac{\Pi_{P,k}-<\Pi_{P,k}>}{\sigma}$')
plt.ylabel('PDF')
plt.xlim(-xlim, xlim)
plt.gca().text(0.97, 0.97, rf'$k_{{P,f}}={k_Pf:.2f}$', transform=plt.gca().transAxes,
    fontsize=20, verticalalignment='top', horizontalalignment='right',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8))
plt.tight_layout()
plt.savefig(savename('EPk_flux_pdf'), bbox_inches='tight')
plt.show()

#%% EPk-flux PDF smoothed at k_lin

plt.figure(figsize=figsize_single)
xlim = max(np.percentile(np.abs(s), 95) for s in [PiPk_series_max_norm])
for series, label, color in zip([PiPk_series_max_norm],
                        [r'$\Pi_{P,k}$'],
                        ['C0']):
    centres, density, err = smooth_pdf(series, bins=50, window=7, xlim=xlim)
    mask = density > 0
    lo = np.maximum(density[mask] - err[mask], density[mask] * 1e-6)
    plt.fill_between(centres[mask], lo, density[mask] + err[mask], alpha=0.2, color=color)
    plt.plot(centres[mask], density[mask], '-', color=color, label=rf'{label}')
plt.xlabel(r'$\frac{\Pi_{P,k}-<\Pi_{P,k}>}{\sigma}$')
plt.ylabel('PDF')
plt.xlim(-xlim, xlim)
plt.gca().text(0.97, 0.97, rf'$k_{{lin}}={k_lin:.2f}$', transform=plt.gca().transAxes,
    fontsize=20, verticalalignment='top', horizontalalignment='right',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8))
plt.tight_layout()
plt.savefig(savename('EPk_flux_kymax_pdf'), bbox_inches='tight')
plt.show()

#%% EPk-flux PDF smoothed at k=1

plt.figure(figsize=figsize_single)
xlim = max(np.percentile(np.abs(s), 95) for s in [PiPk_series_1_norm])
for series, label, color in zip([PiPk_series_1_norm],
                        [r'$\Pi_{P,k}$'],
                        ['C0']):
    centres, density, err = smooth_pdf(series, bins=50, window=7, xlim=xlim)
    mask = density > 0
    lo = np.maximum(density[mask] - err[mask], density[mask] * 1e-6)
    plt.fill_between(centres[mask], lo, density[mask] + err[mask], alpha=0.2, color=color)
    plt.plot(centres[mask], density[mask], '-', color=color, label=rf'{label}')
plt.xlabel(r'$\frac{\Pi_{P,k}-<\Pi_{P,k}>}{\sigma}$')
plt.ylabel('PDF')
plt.xlim(-xlim, xlim)
plt.gca().text(0.97, 0.97, r'$k=1$', transform=plt.gca().transAxes,
    fontsize=20, verticalalignment='top', horizontalalignment='right',
    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8))
plt.tight_layout()
plt.savefig(savename('EPk_flux_k1_pdf'), bbox_inches='tight')
plt.show()