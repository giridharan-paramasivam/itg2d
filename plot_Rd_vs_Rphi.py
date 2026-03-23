#%% Importing libraries
import h5py as h5
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from modules.gamma import gam_max

from modules.plot_basics import apply_style, savename as _savename
from functools import partial
apply_style()

#%% Load the HDF5 file

# Npx=512
Npx=1024
datadir=f'data/{Npx}/'

# fname = datadir + 'out_kapt_0_4_D_0_1_H_3_6_em6.h5'
fname = datadir + 'out_kapt_2_0_D_0_1_H_8_6_em6.h5'
# fname = datadir + 'out_kapt_2_0_D_0_1_H_1_7_em5.h5'

stride = 4

with h5.File(fname, 'r', swmr=True) as fl:
    Rphi_key = 'fluxes/Rphi' if 'fluxes/Rphi' in fl else 'fluxes/RPhi'
    Rd_key   = 'fluxes/Rd'   if 'fluxes/Rd'   in fl else 'fluxes/RP'
    Rphi_t   = fl[Rphi_key][::stride].astype(np.float64)
    Rd_t     = fl[Rd_key][::stride].astype(np.float64)
    vbar_t   = fl['zonal/vbar'][::stride].astype(np.float32)
    dxvbar_t = fl['zonal/Ombar'][::stride].astype(np.float32)
    Ombar_t  = fl['zonal/Ombar'][::stride].astype(np.float64)
    t        = fl['fluxes/t'][::stride].astype(np.float32)
    kx       = fl['data/kx'][:]
    ky       = fl['data/ky'][:]
    Lx       = fl['params/Lx'][()]
    Npx      = fl['params/Npx'][()]
    Npy      = fl['params/Npy'][()]
    kapn     = fl['params/kapn'][()]
    kapt     = fl['params/kapt'][()]
    kapb     = fl['params/kapb'][()]
    D        = fl['params/D'][()]
    if 'H' in fl['params']:
        H = fl['params/H'][()]
    elif 'HP' in fl['params']:
        HP = fl['params/HP'][()]
        H  = HP

gammax = gam_max(kx, ky, kapn, kapt, kapb, D, H)
t      = t * gammax
nt     = len(t)

#%% Find x-indices for the three spatial locations

vbar_mean  = np.mean(vbar_t[nt//2:], axis=0)

def most_central_peak(signal, npts):
    peaks, _ = find_peaks(signal)
    if len(peaks) == 0:
        return int(np.argmax(signal))
    return int(peaks[np.argmin(np.abs(peaks - npts // 2))])

ix_vmax  = most_central_peak(vbar_mean,         int(Npx))
ix_vmin  = most_central_peak(-vbar_mean,        int(Npx))
ix_vzero = most_central_peak(-np.abs(vbar_mean), int(Npx))

xl = np.arange(0, Lx, Lx / Npx)
print(f'ZF max   at x = {xl[ix_vmax]:.2f}  (ix={ix_vmax})')
print(f'ZF min   at x = {xl[ix_vmin]:.2f}  (ix={ix_vmin})')
print(f'ZF zero  at x = {xl[ix_vzero]:.2f}  (ix={ix_vzero})')

#%% MAD filter on Reynolds power (second half)

reynolds_power_t = np.mean((Rphi_t + Rd_t) * Ombar_t, axis=1)
rp_half   = reynolds_power_t[nt//2:]
rp_median = np.median(rp_half)
rp_mad    = np.median(np.abs(rp_half - rp_median))
mask      = np.abs(rp_half - rp_median) <= 24 * rp_mad
print(f'MAD filter: {np.sum(~mask)}/{len(mask)} second-half time steps excluded')

#%% Extract time series for each location

Rphi_vmax_t  = Rphi_t[nt//2:, ix_vmax][mask]
Rd_vmax_t    = Rd_t[nt//2:, ix_vmax][mask]

Rphi_vmin_t  = Rphi_t[nt//2:, ix_vmin][mask]
Rd_vmin_t    = Rd_t[nt//2:, ix_vmin][mask]

Rphi_vzero_t = Rphi_t[nt//2:, ix_vzero][mask]
Rd_vzero_t   = Rd_t[nt//2:, ix_vzero][mask]

Rphi_rms_t   = np.sqrt(np.mean(Rphi_t[nt//2:][mask]**2, axis=1))
Rd_rms_t     = np.sqrt(np.mean(Rd_t[nt//2:][mask]**2, axis=1))

#%% Plot

def scatter_with_fit(ax, x, y):
    ax.scatter(x, y, s=12, marker='.')
    ax.axhline(0, lw=1, color='gray')
    ax.axvline(0, lw=1, color='gray')
    m, b = np.polyfit(x, y, 1)
    xfit = np.linspace(x.min(), x.max(), 200)
    ax.plot(xfit, m * xfit + b, lw=2, color='C1')
    y_pred = m * x + b
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    R2 = 1.0 - ss_res / ss_tot
    ax.text(0.95, 0.88, rf'slope $= {m:.1f}$' + '\n' + rf'$R^2 = {R2:.3f}$',
            transform=ax.transAxes, fontsize=14, va='top', ha='right', linespacing=1.6)

savename = partial(_savename, datadir, fname)

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

scatter_with_fit(axs[0, 0], Rphi_vmax_t,  Rd_vmax_t)
axs[0, 0].set_xlabel(r'$R_\phi$')
axs[0, 0].set_ylabel(r'$R_d$')
axs[0, 0].set_title(rf'ZF max ($x={xl[ix_vmax]:.2f}$)')

scatter_with_fit(axs[0, 1], Rphi_vmin_t,  Rd_vmin_t)
axs[0, 1].set_xlabel(r'$R_\phi$')
axs[0, 1].set_ylabel(r'$R_d$')
axs[0, 1].set_title(rf'ZF min ($x={xl[ix_vmin]:.2f}$)')

scatter_with_fit(axs[1, 0], Rphi_vzero_t, Rd_vzero_t)
axs[1, 0].set_xlabel(r'$R_\phi$')
axs[1, 0].set_ylabel(r'$R_d$')
axs[1, 0].set_title(rf'ZF zero ($x={xl[ix_vzero]:.2f}$)')

scatter_with_fit(axs[1, 1], Rphi_rms_t,   Rd_rms_t)
axs[1, 1].set_xlabel(r'$R_\phi$ (RMS)')
axs[1, 1].set_ylabel(r'$R_d$ (RMS)')
axs[1, 1].set_title(r'RMS over $x$')

plt.tight_layout()
plt.savefig(savename('Rd_vs_Rphi'), dpi=100, bbox_inches='tight')
plt.show()
plt.close()
