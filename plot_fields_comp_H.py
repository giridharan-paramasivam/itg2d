#%% Importing libraries
import h5py as h5
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from modules.mlsarray import irft2np as original_irft2np
from modules.mlsarray import Slicelist
from functools import partial
import warnings
from modules.gamma import gam_max

from modules.plot_basics import apply_style, figsize_double
apply_style()
plt.rcParams['figure.autolayout'] = False


def to_latex_sci(val):
    s = f'{val:.3g}'
    if 'e' in s:
        mantissa, exp = s.split('e')
        return rf'{mantissa}\times 10^{{{int(exp)}}}'
    return s


def load_field_data(datadir, fname):
    evol_fname = datadir + 'evol/' + fname.replace('out_', 'evol_')
    try:
        with h5.File(evol_fname, 'r') as flevol:
            t_evol = flevol['t'][:]
            Qbox_t = flevol['Qbox_t'][:]
            it = int(np.argmax(Qbox_t))
            t_spike = t_evol[it]
    except Exception as e:
        warnings.warn(f"Could not load Qbox_t from {evol_fname}: {e}\nDefaulting to last time index.")
        it = -1
        t_spike = None

    with h5.File(datadir + fname, 'r', swmr=True) as fl:
        Omk = fl['fields/Omk'][it]
        kx = fl['data/kx'][:]
        ky = fl['data/ky'][:]
        Lx = fl['params/Lx'][()]
        Ly = fl['params/Ly'][()]
        Npx = fl['params/Npx'][()]
        Npy = fl['params/Npy'][()]
        kapt = fl['params/kapt'][()]
        kapn = fl['params/kapn'][()]
        kapb = fl['params/kapb'][()]
        D = fl['params/D'][()]
        if 'H' in fl['params']:
            H = fl['params/H'][()]
        else:
            H = fl['params/HP'][()]

        gammax = gam_max(kx, ky, kapn, kapt, kapb, D, H)
        if it >= 0 and t_spike is not None:
            print(
                f"{fname}: plotting at heat flux spike: "
                f"t = {t_spike:.3f}, gamma*t = {gammax * t_spike:.3f}, index = {it}"
            )

    Nx, Ny = 2 * Npx // 3, 2 * Npy // 3
    sl = Slicelist(Nx, Ny)
    irft2np = partial(original_irft2np, Npx=Npx, Npy=Npy, Nx=Nx, sl=sl)

    xl, yl = np.linspace(0, Lx, Npx), np.linspace(0, Ly, Npy)
    x, y = np.meshgrid(np.array(xl), np.array(yl), indexing='ij')
    Om = irft2np(Omk)

    return {
        'Om': Om,
        'x': x,
        'y': y,
        'H': H,
        'fname': fname,
    }

#%% Load the HDF5 files

Npx = 512
# Npx = 1024
datadir = f'data/{Npx}/'
subdir = 'fields/'

fname1 = 'out_kapt_2_0_D_0_1_H_0_0_e0.h5'
fname2 = 'out_kapt_2_0_D_0_1_H_8_6_em6.h5'

data1 = load_field_data(datadir, fname1)
data2 = load_field_data(datadir, fname2)

#%% Plot: Om comparison

Om_abs_max = np.max(np.abs(data2['Om']))

fig, axes = plt.subplots(1, 2, figsize=figsize_double, sharey=True)

for ax, data in zip(axes, [data1, data2]):
    c = ax.pcolormesh(
        data['x'],
        data['y'],
        data['Om'],
        cmap='seismic',
        vmin=-Om_abs_max,
        vmax=Om_abs_max,
        rasterized=True,
    )
    ax.set_xlabel('$x$')
    ax.set_title(rf'$H = {to_latex_sci(data["H"])}$')
axes[0].set_ylabel('$y$')

fig.subplots_adjust(bottom=0.35, wspace=0.15)
cax = fig.add_axes([0.15, 0.15, 0.7, 0.04])
cbar = fig.colorbar(c, cax=cax, orientation='horizontal')
cbar.set_label(r'$\Omega$')

plt.savefig(datadir + subdir + 'fields_Om_comp_H.svg', bbox_inches='tight')
plt.show()

# %%
