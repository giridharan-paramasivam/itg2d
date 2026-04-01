#%% Importing libraries
import h5py as h5
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from modules.gamma import gam_max

from modules.plot_basics import apply_style, savename as _savename, figsize_single, figsize_double
from functools import partial
apply_style()

#%% Load the HDF5 file

# Npx=512
Npx=1024
datadir=f'data/{Npx}/'
subdir='reynolds-stress/'

fname = datadir + 'out_kapt_0_4_D_0_1_H_3_6_em6.h5'
# fname = datadir + 'out_kapt_2_0_D_0_1_H_8_6_em6.h5'

stride = 16
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

#%% Functions

def most_central_peak(signal, mode='max'):
    npts = len(signal)
    
    if mode == 'max':
        peaks, _ = find_peaks(signal)
        if len(peaks) == 0:
            return int(np.argmax(signal))
    elif mode == 'min':
        peaks, _ = find_peaks(-signal)
        if len(peaks) == 0:
            return int(np.argmin(signal))
    else:
        raise ValueError("mode must be 'max' or 'min'")
        
    # Find index of the peak closest to the array midpoint
    return int(peaks[np.argmin(np.abs(peaks - npts // 2))])

# def scatter_with_fit(ax, x, y):
#     ax.scatter(x, y, s=12, marker='.')
#     ax.axhline(0, lw=1, color='gray')
#     ax.axvline(0, lw=1, color='gray')
#     m, b = np.polyfit(x, y, 1)
#     xfit = np.linspace(x.min(), x.max(), 200)
#     ax.plot(xfit, m * xfit + b, lw=2, color='C1')
#     y_pred = m * x + b
#     ss_res = np.sum((y - y_pred)**2)
#     ss_tot = np.sum((y - np.mean(y))**2)
#     R2 = 1.0 - ss_res / ss_tot
#     ax.text(0.95, 0.88, rf'slope $= {m:.1f}$' + '\n' + rf'$R^2 = {R2:.3f}$',
#             transform=ax.transAxes, fontsize=14, va='top', ha='right', linespacing=1.6)

def scatter_with_fit(ax, x, y):
    # 1. Plot the raw data
    ax.scatter(x, y, s=12, marker='.', alpha=0.5) # alpha helps if points overlap
    ax.axhline(0, lw=1, color='gray', alpha=0.5)
    ax.axvline(0, lw=1, color='gray', alpha=0.5)

    # 2. Calculate RMA components
    std_x = np.std(x)
    std_y = np.std(y)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # Pearson correlation to get the sign
    correlation = np.corrcoef(x, y)[0, 1]
    
    # RMA Slope and Intercept
    m = np.sign(correlation) * (std_y / std_x)
    b = mean_y - m * mean_x

    # 3. Plot the RMA fit line
    xfit = np.linspace(x.min(), x.max(), 200)
    ax.plot(xfit, m * xfit + b, lw=2, color='C1', label='RMA Fit')

    # 4. Annotation
    # Note: R^2 in RMA is technically just (Pearson Correlation)^2
    r_squared = correlation**2
    
    text_str = (rf'm $_{{\mathrm{{RMA}}}}$ $= {m:.2f}$' + '\n' + 
                rf'$R^2 = {r_squared:.3f}$')
                
    ax.text(0.95, 0.88, text_str, transform=ax.transAxes, 
        fontsize=16, va='top', ha='right', linespacing=1.6,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

def corr_along_x(A, B):
    # Correlation along x axis (axis=1), for each t
    mu_A = np.mean(A, axis=1)
    mu_B = np.mean(B, axis=1)
    num = np.mean((A - mu_A[:, None]) * (B - mu_B[:, None]), axis=1)
    den = np.std(A, axis=1) * np.std(B, axis=1)
    return np.divide(num, den, out=np.zeros_like(num), where=den!=0)

savename = partial(_savename, datadir+subdir, fname)

#%% Computations

xl = np.arange(0, Lx, Lx / Npx)
vbar_mid = vbar_t[nt // 2]
ix_vmax  = most_central_peak(vbar_mid, mode='max')
ix_vmin  = most_central_peak(vbar_mid, mode='min')
ix_vzero = int(np.argmin(np.abs(vbar_mid)))

print(f'ZF max   at x = {xl[ix_vmax]:.2f}  (ix={ix_vmax})')
print(f'ZF min   at x = {xl[ix_vmin]:.2f}  (ix={ix_vmin})')
print(f'ZF zero  at x = {xl[ix_vzero]:.2f}  (ix={ix_vzero})')

Rphi_vmax_t  = Rphi_t[nt//2:, ix_vmax]
Rd_vmax_t    = Rd_t[nt//2:, ix_vmax]

Rphi_vmin_t  = Rphi_t[nt//2:, ix_vmin]
Rd_vmin_t    = Rd_t[nt//2:, ix_vmin]

Rphi_vzero_t = Rphi_t[nt//2:, ix_vzero]
Rd_vzero_t   = Rd_t[nt//2:, ix_vzero]

Rphi_rms_t   = np.sqrt(np.mean(Rphi_t[nt//2:]**2, axis=1))
Rd_rms_t     = np.sqrt(np.mean(Rd_t[nt//2:]**2, axis=1))

Rphi_all = Rphi_t[nt//2:].ravel()
Rd_all   = Rd_t[nt//2:].ravel()

#%% Plot: all (x, t) points vectorized

ax = plt.figure(figsize=figsize_single,dpi=200).gca()
scatter_with_fit(ax, Rphi_all[::2], Rd_all[::2])
ax.set_xlabel(r'$R_\phi$')
ax.set_ylabel(r'$R_d$')

plt.tight_layout()
plt.savefig(savename('Rd_vs_Rphi_all_xt'), bbox_inches='tight')
plt.show()
plt.close()

#%% Plot: t points; RMS over x

# ax=plt.figure(figsize=figsize_single).gca()
# scatter_with_fit(ax, Rphi_rms_t,   Rd_rms_t)
# ax.set_xlabel(r'$R_\phi$ (RMS)')
# ax.set_ylabel(r'$R_d$ (RMS)')

# plt.tight_layout()
# plt.savefig(savename('Rd_vs_Rphi_xrms'), bbox_inches='tight')
# plt.show()
# plt.close()

#%% Plot: t points; selected x locations

fig, axs = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

scatter_with_fit(axs[0], Rphi_vmax_t,  Rd_vmax_t)
axs[0].set_xlabel(r'$R_\phi$')
axs[0].set_ylabel(r'$R_d$')
axs[0].set_title(rf'ZF max ($x={xl[ix_vmax]:.2f}$)')

scatter_with_fit(axs[1], Rphi_vmin_t,  Rd_vmin_t)
axs[1].set_xlabel(r'$R_\phi$')
axs[1].set_title(rf'ZF min ($x={xl[ix_vmin]:.2f}$)')

scatter_with_fit(axs[2], Rphi_vzero_t, Rd_vzero_t)
axs[2].set_xlabel(r'$R_\phi$')
axs[2].set_title(rf'ZF zero ($x={xl[ix_vzero]:.2f}$)')

plt.tight_layout()
plt.savefig(savename('Rd_vs_Rphi_selected_x'), bbox_inches='tight')
plt.show()
plt.close()

#%% Plot: Rphi and Rd vs x; at selected time

plt.figure(figsize=figsize_single)
plt.plot(xl, Rphi_t[nt//2], label=r'electric')
plt.plot(xl, Rd_t[nt//2], label=r'diamagnetic')
# plt.plot(xl, Rphi_t[nt//2] + Rd_t[nt//2], label=r'total')
plt.plot(xl, vbar_t[nt//2]*0.5*np.max(np.abs(Rphi_t[nt//2] + Rd_t[nt//2]))/np.max(np.abs(vbar_t[nt//2])), label=r'$\bar{v}_y$', color='k', lw=1.5, ls='--')
plt.axhline(0, lw=1, color='black')
plt.xlabel('$x$')
plt.ylabel('$R$')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(savename('R_vs_x_selected_t'), bbox_inches='tight')
plt.show()
plt.close()

#%% Plot: Rphi and Rd vs x; averaged over time

plt.figure(figsize=figsize_single)
plt.plot(xl, np.mean(Rphi_t[nt//2:], axis=0), label=r'electric')
plt.plot(xl, np.mean(Rd_t[nt//2:], axis=0), label=r'diamagnetic')
# plt.plot(xl, np.mean(Rphi_t[nt//2:] + Rd_t[nt//2:], axis=0), label=r'total')
plt.plot(xl, np.mean(vbar_t[nt//2:], axis=0)*0.5*np.max(np.abs(np.mean(Rphi_t[nt//2:] + Rd_t[nt//2:], axis=0)))/np.max(np.abs(np.mean(vbar_t[nt//2:], axis=0))), label=r'$\langle\bar{v}_y\rangle_{T/2}$', color='k', lw=1.5, ls='--')
plt.axhline(0, lw=1, color='black')
plt.xlabel('$x$')
plt.ylabel(r'$\left< R \right>_{T/2}$')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(savename('R_vs_x_tavg'), bbox_inches='tight')
plt.show()
plt.close()

#%% Plot: correlation between Rphi and Rd vs t; averaged over x

corr_x = corr_along_x(Rphi_t, Rd_t)
corr_x_mean = np.mean(corr_x[nt//2:])
plt.figure(figsize=figsize_single)
plt.plot(t, corr_x,'.')
plt.axhline(corr_x_mean, color='k', lw=1.5, ls='--', label=rf'$\langle\mathrm{{corr}}_x\rangle_{{T/2}}={corr_x_mean:.2f}$')
plt.xlabel(r'$\gamma t$')
plt.ylabel(r'$\mathrm{corr}_x\left(R_\phi, R_d\right)$')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(savename('corr_vs_t_xavg'), bbox_inches='tight')
plt.show()
plt.close()

# %%
