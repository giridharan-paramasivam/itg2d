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

Npx = 1024
datadir = f'data/{Npx}/'
subdir = 'evol/'

fname1 = datadir + 'out_kapt_0_4_D_0_1_H_3_6_em6.h5'
fname2 = datadir + 'out_kapt_2_0_D_0_1_H_8_6_em6.h5'

evol_fname1 = datadir + subdir + fname1.split('/')[-1].replace('out_', 'evol_')
evol_fname2 = datadir + subdir + fname2.split('/')[-1].replace('out_', 'evol_')

with h5.File(fname1, 'r', swmr=True) as fl:
    kapt1 = fl['params/kapt'][()]
    D = fl['params/D'][()]

with h5.File(fname2, 'r', swmr=True) as fl:
    kapt2 = fl['params/kapt'][()]

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

nt1 = len(t1)
nt2 = len(t2)

#%% Calculate quantities

zonal_frac1 = energy_ZF_t1 / energy_t1
zonal_frac1_mean = mad_mean(zonal_frac1[nt1//2:])

zonal_frac2 = energy_ZF_t2 / energy_t2
zonal_frac2_mean = mad_mean(zonal_frac2[nt2//2:])

Qbox1_half = Qbox_t1[nt1//2:]
Qbox2_half = Qbox_t2[nt2//2:]
Qbox1_mean = mad_mean(Qbox1_half)
Qbox2_mean = mad_mean(Qbox2_half)

#%% Plot: zonal energy fraction comparison

plt.figure(figsize=figsize_single)
plt.semilogy(t1, zonal_frac1, label=rf'$\kappa_T={kapt1}$')
plt.semilogy(t2, zonal_frac2, label=rf'$\kappa_T={kapt2}$')
plt.axhline(zonal_frac1_mean, color='C0', linestyle='--', linewidth=2.5, 
            label=rf'$\langle (E_{{\mathrm{{ZF}}}}/E)^{{<3\cdot\mathrm{{MAD}}}} \rangle_{{T/2}} = {zonal_frac1_mean:.3f}$')
plt.axhline(zonal_frac2_mean, color='C1', linestyle='--', linewidth=2.5, 
            label=rf'$\langle (E_{{\mathrm{{ZF}}}}/E)^{{<3\cdot\mathrm{{MAD}}}} \rangle_{{T/2}} = {zonal_frac2_mean:.3f}$')
plt.xlabel(r'$\gamma t$')
plt.ylabel(r'$E_{\mathrm{ZF}}/E$')
plt.legend(loc=(0.25, 0.1),fontsize=24)
plt.savefig(datadir+subdir+'zonal_energy_fraction_kapt_comp.svg', bbox_inches='tight')
plt.show()

# %% Plot: log(Qbox) vs time

fig, ax = plt.subplots(figsize=figsize_single)
ax.semilogy(t1, np.abs(Qbox_t1), label=rf'$\kappa_T={kapt1}$')
ax.semilogy(t2, np.abs(Qbox_t2), label=rf'$\kappa_T={kapt2}$')

# Shade the region where Qbox is negative using axvspan
neg_idx2 = np.where(Qbox_t2 < 0)[0]
for i in neg_idx2:
    ax.axvspan(t2[i], t2[min(i + 1, len(t2) - 1)], alpha=0.7, facecolor='grey', edgecolor='grey', linewidth=1.5)

ax.axhline(Qbox1_mean, color='C0', linestyle='--', linewidth=2.5,
           label=rf'$\langle Q_{{\mathrm{{box}}}}^{{<3\cdot\mathrm{{MAD}}}} \rangle_{{T/2}} = {Qbox1_mean:.3f}$')
ax.axhline(Qbox2_mean, color='C1', linestyle='--', linewidth=2.5,
           label=rf'$\langle Q_{{\mathrm{{box}}}}^{{<3\cdot\mathrm{{MAD}}}} \rangle_{{T/2}} = {Qbox2_mean:.3f}$')
ax.set_xlabel(r'$\gamma t$')
ax.set_ylabel(r'$|Q_{\mathrm{box}}|$')
ax.legend(loc=(0.11, 0.1), fontsize=24)
plt.savefig(datadir+subdir+'Qbox_kapt_comp.svg', bbox_inches='tight')
plt.show()
# %%
