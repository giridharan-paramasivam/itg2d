#%% Importing libraries
import h5py as h5
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from modules.basics import format_exp, mad_mean
from modules.plot_basics import apply_style, figsize_single
apply_style()

#%% Load the computed HDF5 files (produced by compute_evol.py)

Npx = 512
datadir = f'data/{Npx}/'
subdir = 'evol/'

fname1 = datadir + 'out_kapt_2_0_D_0_1_H_0_0_e0.h5'
fname2 = datadir + 'out_kapt_2_0_D_0_1_H_8_6_em6.h5'

evol_fname1 = datadir + subdir + fname1.split('/')[-1].replace('out_', 'evol_')
evol_fname2 = datadir + subdir + fname2.split('/')[-1].replace('out_', 'evol_')

with h5.File(fname1, 'r', swmr=True) as fl:
    kapt1 = fl['params/kapt'][()]
    H1 = fl['params/H'][()]
    D = fl['params/D'][()]

with h5.File(fname2, 'r', swmr=True) as fl:
    kapt2 = fl['params/kapt'][()]
    H2 = fl['params/H'][()]

with h5.File(evol_fname1, 'r', swmr=True) as fl:
    t1 = fl['t'][:]
    energy_t1 = fl['energy_t'][:]
    energy_ZF_t1 = fl['energy_ZF_t'][:]
    Qbox_t1 = fl['Qbox_t'][:]

with h5.File(evol_fname2, 'r', swmr=True) as fl:
    t2 = fl['t'][:]
    energy_t2 = fl['energy_t'][:]
    energy_ZF_t2 = fl['energy_ZF_t'][:]
    Qbox_t2 = fl['Qbox_t'][:]

nt = min(len(t1), len(t2))

#%% Find spectrum peak time
it_peak = np.argmax(Qbox_t2[:nt])
t_peak = t2[it_peak]
print(f"Peak at t={t_peak:.6e}, index={it_peak}")

#%% Calculate quantities

zonal_frac1 = energy_ZF_t1 / energy_t1
zonal_frac1_mean = np.mean(zonal_frac1[nt//2:])

zonal_frac2 = energy_ZF_t2 / energy_t2
zonal_frac2_mean = np.mean(zonal_frac2[nt//2:])

Qbox1_half = Qbox_t1[nt//2:]
Qbox2_half = Qbox_t2[nt//2:]
Qbox1_mean = mad_mean(Qbox1_half)
Qbox2_mean = mad_mean(Qbox2_half)

#%% Helper function for scientific notation in LaTeX
def to_latex_sci(val):
    s = f'{val:.3g}'
    if 'e' in s:
        mantissa, exp = s.split('e')
        return f'{mantissa}\\times 10^{{{int(exp)}}}'
    return s

#%% Plot: energy evolution comparison

plt.figure(figsize=figsize_single)
plt.semilogy(t1[:nt], energy_t1[:nt], label=rf'$H={to_latex_sci(H1)}$')
plt.semilogy(t2[:nt], energy_t2[:nt], label=rf'$H={to_latex_sci(H2)}$')
plt.axvline(x=t_peak, color='k', linestyle='--', linewidth=1.5)
plt.xlabel(r'$\gamma t$')
plt.ylabel(r'$E$')
plt.legend(fontsize=24)
plt.savefig(datadir+subdir+'energy_H_comp.svg', bbox_inches='tight')
plt.show()

# %% Plot: log(Qbox) vs time

fig, ax = plt.subplots(figsize=figsize_single)
ax.semilogy(t1[:nt], np.abs(Qbox_t1[:nt]), label=rf'$H={to_latex_sci(H1)}$')
ax.semilogy(t2[:nt], np.abs(Qbox_t2[:nt]), label=rf'$H={to_latex_sci(H2)}$')

# Shade the region where Qbox is negative using axvspan
neg_idx1 = np.where(Qbox_t1 < 0)[0]
for i in neg_idx1:
    ax.axvspan(t1[i], t1[min(i + 1, len(t1) - 1)], alpha=0.7, facecolor='C0', edgecolor='C0', linewidth=1.5)

neg_idx2 = np.where(Qbox_t2[:nt] < 0)[0]
for i in neg_idx2:
    ax.axvspan(t2[i], t2[min(i + 1, nt - 1)], alpha=0.7, facecolor='C1', edgecolor='C1', linewidth=1.5)

ax.axvline(x=t_peak, color='k', linestyle='--', linewidth=1.5)
ax.set_xlabel(r'$\gamma t$')
ax.set_ylabel(r'$|Q_{\mathrm{box}}|$')
ax.legend(fontsize=18)
plt.savefig(datadir+subdir+'Qbox_H_comp.svg', bbox_inches='tight')
plt.show()
