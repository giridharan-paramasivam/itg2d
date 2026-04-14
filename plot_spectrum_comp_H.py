#%% Imports
import warnings
import h5py as h5
import numpy as np
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from modules.plot_basics import apply_style, figsize_single
apply_style()
xtick_fontsize = matplotlib.rcParams.get('xtick.labelsize', 32)

#%% Parameters

Npx = 512
datadir = f'data/{Npx}/'

fname1 = datadir + 'out_kapt_2_0_D_0_1_H_0_0_e0.h5'
fname2 = datadir + 'out_kapt_2_0_D_0_1_H_8_6_em6.h5'

#%% Helper functions

def to_latex_sci(val):
    s = f'{val:.3g}'
    if 'e' in s:
        mantissa, exp = s.split('e')
        return rf'{mantissa}\times 10^{{{int(exp)}}}'
    return s

def fit_slope(k, y, k_min, k_max, yoffset=3.0):
    mask = (k >= k_min) & (k <= k_max) & (y > 0)
    k_seg = k[mask]
    y_seg = y[mask]
    n, c = np.polyfit(np.log(k_seg), np.log(y_seg), 1)
    return k_seg, np.exp(c) * k_seg**n * yoffset, n

def find_peak_index(Qbox_t):
    """Find the final local peak index in Qbox_t."""
    # Find all local peaks in Qbox_t
    peaks, _ = find_peaks(Qbox_t)
    if len(peaks) > 0:
        # Use the final (rightmost) peak
        it_target = peaks[-1]
    else:
        # Fallback: use last index if no peaks found
        it_target = -1
    
    return it_target

#%% Find peak index

# Find peak in file2 and get index in spectrum arrays
evol_fname1 = datadir + 'evol/' + fname1.split('/')[-1].replace('out_', 'evol_')
evol_fname2 = datadir + 'evol/' + fname2.split('/')[-1].replace('out_', 'evol_')
with h5.File(evol_fname1, 'r') as fl:
    Qbox_t1 = fl['Qbox_t'][:]
with h5.File(evol_fname2, 'r') as fl:
    Qbox_t2 = fl['Qbox_t'][:]
    t = fl['t'][:]
nt = min(len(Qbox_t1), len(Qbox_t2))
it_peak = int(np.argmax(Qbox_t2[:nt]))

# Map evol index to spectrum index via idx_saved
spectrum_file2 = fname2.replace('out_', 'spectrum/spectrum_')
with h5.File(spectrum_file2, 'r') as fl:
    idx_saved2 = fl['idx_saved'][:]
peak_idx = int(np.argmin(np.abs(idx_saved2 - it_peak)))

print(f"Peak at t={t[it_peak]:.6e}, index={it_peak}, spectrum index={peak_idx}")

#%% Load spectra

# Load spectrum for file2 at peak
with h5.File(spectrum_file2, 'r') as fl:
    k2 = fl['k'][:]
    Ek_t2 = fl['Ek_t'][:]
    Ek_ZF_t2 = fl['Ek_ZF_t'][:]
    P2k_t2 = fl['P2k_t'][:]
    P2k_ZF_t2 = fl['P2k_ZF_t'][:]
    Gk_t2 = fl['Gk_t'][:]
    Gk_ZF_t2 = fl['Gk_ZF_t'][:]

flux_file2 = datadir + 'spectral_flux/' + fname2.split('/')[-1].replace('out_', 'spectral_flux_')
with h5.File(flux_file2, 'r') as fl:
    k_lin2 = float(fl['k_lin'][()])

Ek2 = Ek_t2[peak_idx]
Ek_ZF2 = Ek_ZF_t2[peak_idx]
Ek_turb2 = Ek2 - Ek_ZF2
P2k2 = P2k_t2[peak_idx]
P2k_ZF2 = P2k_ZF_t2[peak_idx]
Gk2 = Gk_t2[peak_idx]
Gk_ZF2 = Gk_ZF_t2[peak_idx]

# Load spectrum for file1 at same peak index
spectrum_file1 = fname1.replace('out_', 'spectrum/spectrum_')
with h5.File(spectrum_file1, 'r') as fl:
    k1 = fl['k'][:]
    Ek_t1 = fl['Ek_t'][:]
    Ek_ZF_t1 = fl['Ek_ZF_t'][:]
    P2k_t1 = fl['P2k_t'][:]
    P2k_ZF_t1 = fl['P2k_ZF_t'][:]
    Gk_t1 = fl['Gk_t'][:]
    Gk_ZF_t1 = fl['Gk_ZF_t'][:]

flux_file1 = datadir + 'spectral_flux/' + fname1.split('/')[-1].replace('out_', 'spectral_flux_')
with h5.File(flux_file1, 'r') as fl:
    k_lin1 = float(fl['k_lin'][()])

Ek1 = Ek_t1[peak_idx]
Ek_ZF1 = Ek_ZF_t1[peak_idx]
Ek_turb1 = Ek1 - Ek_ZF1
P2k1 = P2k_t1[peak_idx]
P2k_ZF1 = P2k_ZF_t1[peak_idx]
Gk1 = Gk_t1[peak_idx]
Gk_ZF1 = Gk_ZF_t1[peak_idx]

#%% Load H parameters for labels

with h5.File(fname1, 'r', swmr=True) as fl:
    H1 = fl['params/H'][()]
with h5.File(fname2, 'r', swmr=True) as fl:
    H2 = fl['params/H'][()]

#%% Plot: Energy spectrum comparison

fig, ax = plt.subplots(figsize=figsize_single)

ax.loglog(k1, Ek1, label=rf'$H = {to_latex_sci(H1)}$')
ax.loglog(k2, Ek2, label=rf'$H = {to_latex_sci(H2)}$')
ax.loglog(k1[Ek_ZF1 > 0], Ek_ZF1[Ek_ZF1 > 0], 'C0--', alpha=0.7, label=rf'$H = {to_latex_sci(H1)}$ (zonal)')
ax.loglog(k2[Ek_ZF2 > 0], Ek_ZF2[Ek_ZF2 > 0], 'C1--', alpha=0.7, label=rf'$H = {to_latex_sci(H2)}$ (zonal)')

ax.axvline(x=1, color='k', linestyle='--', linewidth=2)
ax.axvline(x=k_lin2, color='k', linestyle='-.', linewidth=2)
ax.text(k_lin2, -0.025, r'$k_{\mathrm{lin}}$',
        transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=xtick_fontsize)

ax.set_xlabel(r'$k$')
ax.set_ylabel(r'$E_k$')
ax.legend(fontsize=22)

plt.tight_layout()
plt.savefig(datadir + 'spectrum/E_spectrum_comp_H.svg', bbox_inches='tight')
plt.show()

#%% Plot: Pressure variance spectrum comparison

# fig, ax = plt.subplots(figsize=figsize_single)

# ax.loglog(k1, P2k1, label=rf'$H = {to_latex_sci(H1)}$')
# ax.loglog(k2, P2k2, label=rf'$H = {to_latex_sci(H2)}$')
# ax.loglog(k1[P2k_ZF1 > 0], P2k_ZF1[P2k_ZF1 > 0], 'C0--', alpha=0.7, label=rf'$H = {to_latex_sci(H1)}$ (zonal)')
# ax.loglog(k2[P2k_ZF2 > 0], P2k_ZF2[P2k_ZF2 > 0], 'C1--', alpha=0.7, label=rf'$H = {to_latex_sci(H2)}$ (zonal)')

# ax.axvline(x=1, color='k', linestyle='--', linewidth=2)
# ax.axvline(x=k_lin2, color='k', linestyle='-.', linewidth=2)
# ax.text(k_lin2, -0.025, r'$k_{\mathrm{lin}}$',
#         transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=xtick_fontsize)

# ax.set_xlabel(r'$k$')
# ax.set_ylabel(r'$\left|P_k\right|^2$')
# ax.legend(fontsize=22)

# plt.tight_layout()
# plt.savefig(datadir + 'spectrum/P2_spectrum_comp_H.svg', bbox_inches='tight')
# plt.show()

#%% Plot: Generalized energy spectrum comparison

# fig, ax = plt.subplots(figsize=figsize_single)

# ax.loglog(k1, Gk1, label=rf'$H = {to_latex_sci(H1)}$')
# ax.loglog(k2, Gk2, label=rf'$H = {to_latex_sci(H2)}$')
# ax.loglog(k1[Gk_ZF1 > 0], Gk_ZF1[Gk_ZF1 > 0], 'C0--', alpha=0.7, label=rf'$H = {to_latex_sci(H1)}$ (zonal)')
# ax.loglog(k2[Gk_ZF2 > 0], Gk_ZF2[Gk_ZF2 > 0], 'C1--', alpha=0.7, label=rf'$H = {to_latex_sci(H2)}$ (zonal)')

# ax.axvline(x=1, color='k', linestyle='--', linewidth=2)
# ax.axvline(x=k_lin2, color='k', linestyle='-.', linewidth=2)
# ax.text(k_lin2, -0.025, r'$k_{\mathrm{lin}}$',
#         transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=xtick_fontsize)

# ax.set_xlabel(r'$k$')
# ax.set_ylabel(r'$G_k$')
# ax.legend(fontsize=22)

# plt.tight_layout()
# plt.savefig(datadir + 'spectrum/G_spectrum_comp_H.svg', bbox_inches='tight')
# plt.show()
# %%
