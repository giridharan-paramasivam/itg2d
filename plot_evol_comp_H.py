#%% Importing libraries
import h5py as h5
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from functools import partial
import warnings

from modules.basics import format_exp, mad_mean
from modules.gamma import gam_max
from modules.mlsarray import irft2np as original_irft2np
from modules.mlsarray import Slicelist
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


def load_om_field(datadir, fname, it, t_snapshot):
    with h5.File(fname, 'r', swmr=True) as fl:
        Omk = fl['fields/Omk'][it]
        kx = fl['data/kx'][:]
        ky = fl['data/ky'][:]
        Lx = fl['params/Lx'][()]
        Ly = fl['params/Ly'][()]
        Npx_loc = fl['params/Npx'][()]
        Npy_loc = fl['params/Npy'][()]
        kapt = fl['params/kapt'][()]
        kapn = fl['params/kapn'][()]
        kapb = fl['params/kapb'][()]
        D = fl['params/D'][()]
        if 'H' in fl['params']:
            H = fl['params/H'][()]
        else:
            H = fl['params/HP'][()]

        gammax = gam_max(kx, ky, kapn, kapt, kapb, D, H)
        print(
            f"{fname.split('/')[-1]}: plotting at shared snapshot time: "
            f"t = {t_snapshot:.3f}, gamma*t = {gammax * t_snapshot:.3f}, index = {it}"
        )

    Nx, Ny = 2 * Npx_loc // 3, 2 * Npy_loc // 3
    sl = Slicelist(Nx, Ny)
    irft2np = partial(original_irft2np, Npx=Npx_loc, Npy=Npy_loc, Nx=Nx, sl=sl)

    xl = np.linspace(0, Lx, Npx_loc)
    yl = np.linspace(0, Ly, Npy_loc)
    x, y = np.meshgrid(np.array(xl), np.array(yl), indexing='ij')

    return {
        'Om': irft2np(Omk),
        'x': x,
        'y': y,
        'H': H,
    }

#%% Plot: energy evolution comparison with Om inset

om_data1 = load_om_field(datadir, fname1, it_peak, t_peak)
om_data2 = load_om_field(datadir, fname2, it_peak, t_peak)
Om_abs_max = np.percentile(np.abs(om_data2['Om']), 99.5)

fig, ax = plt.subplots(figsize=figsize_single)
ax.semilogy(t1[:nt], energy_t1[:nt], label=rf'$H={to_latex_sci(H1)}$')
ax.semilogy(t2[:nt], energy_t2[:nt], label=rf'$H={to_latex_sci(H2)}$')
ax.axvline(x=t_peak, color='k', linestyle='--', linewidth=1.5, label=rf'$t_{{\mathrm{{peak}}}} = {t_peak:.3g}$')
ax.set_xlabel(r'$\gamma t$')
ax.set_ylabel(r'$E$')
ax.legend(fontsize=21, loc='upper left')

# inset axes for Om fields at peak time
inset_left = 0.35
inset_width = 0.25
inset_height = 0.24
cbar_bottom = 0.15
cbar_height = 0.03
axins1 = ax.inset_axes([inset_left, cbar_bottom+cbar_height+0.1, inset_width, inset_height])
axins2 = ax.inset_axes([inset_left+inset_width+0.02, cbar_bottom+cbar_height+0.1, inset_width, inset_height], sharey=axins1)
axbox = ax.inset_axes([inset_left-0.09, cbar_bottom-0.06, 2*inset_width+0.02+0.09+0.04, cbar_height+inset_height+0.15+0.06], zorder=2)
axbox.patch.set_alpha(0)
axbox.set_xticks([])
axbox.set_yticks([])
axbox.set_title(r'$\Omega$ at $t_{\mathrm{peak}}$', fontsize=18, pad=8)
for spine in axbox.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(2)
    spine.set_edgecolor('black')

peak_marker_y = np.sqrt(energy_t1[:nt].max() * energy_t2[:nt].max())
peak_connection = ConnectionPatch(
    xyA=(t_peak, peak_marker_y),
    coordsA=ax.transData,
    xyB=(0.52, 1.12),
    coordsB=axbox.transAxes,
    arrowstyle='->',
    linewidth=1.8,
    color='black',
    mutation_scale=18,
)
fig.add_artist(peak_connection)

for axins, data in zip([axins1, axins2], [om_data1, om_data2]):
    c = axins.pcolormesh(
        data['x'],
        data['y'],
        data['Om'],
        cmap='seismic',
        vmin=-Om_abs_max,
        vmax=Om_abs_max,
        rasterized=True,
    )
    axins.set_title(rf'$H = {to_latex_sci(data["H"])}$', fontsize=18, pad=4)
    axins.set_xlabel('$x$', fontsize=16, labelpad=2)
    axins.tick_params(axis='both', which='both', labelsize=12, length=3)

axins1.set_ylabel('$y$', fontsize=16, labelpad=2)
axins2.tick_params(labelleft=False)
axins2.xaxis.get_major_ticks()[0].label1.set_visible(False)

cax = ax.inset_axes([inset_left, cbar_bottom, 2*inset_width+0.02, cbar_height])
cbar = fig.colorbar(c, cax=cax, orientation='horizontal')
cbar.ax.tick_params(labelsize=12, length=3)

plt.savefig(datadir+subdir+'energy_vs_t_H_comp_w_field_inset.svg', bbox_inches='tight')
plt.show()

#%% Plot: energy evolution comparison

om_data1 = load_om_field(datadir, fname1, it_peak, t_peak)
om_data2 = load_om_field(datadir, fname2, it_peak, t_peak)
Om_abs_max = np.percentile(np.abs(om_data2['Om']), 99.5)

fig, ax = plt.subplots(figsize=figsize_single)
ax.semilogy(t1[:nt], energy_t1[:nt], label=rf'$H={to_latex_sci(H1)}$')
ax.semilogy(t2[:nt], energy_t2[:nt], label=rf'$H={to_latex_sci(H2)}$')
ax.axvline(x=t_peak, color='k', linestyle='--', linewidth=1.5, label=rf'$t_{{\mathrm{{peak}}}} = {t_peak:.3g}$')
ax.set_xlabel(r'$\gamma t$')
ax.set_ylabel(r'$E$')
ax.legend(fontsize=21, loc='upper left')

plt.savefig(datadir+subdir+'energy_vs_t_H_comp.svg', bbox_inches='tight')
plt.show()

# %% Plot: log(Qbox) vs time

# fig, ax = plt.subplots(figsize=figsize_single)
# ax.semilogy(t1[:nt], np.abs(Qbox_t1[:nt]), label=rf'$H={to_latex_sci(H1)}$')
# ax.semilogy(t2[:nt], np.abs(Qbox_t2[:nt]), label=rf'$H={to_latex_sci(H2)}$')

# # Shade the region where Qbox is negative using axvspan
# neg_idx1 = np.where(Qbox_t1 < 0)[0]
# for i in neg_idx1:
#     ax.axvspan(t1[i], t1[min(i + 1, len(t1) - 1)], alpha=0.7, facecolor='C0', edgecolor='C0', linewidth=1.5)

# neg_idx2 = np.where(Qbox_t2[:nt] < 0)[0]
# for i in neg_idx2:
#     ax.axvspan(t2[i], t2[min(i + 1, nt - 1)], alpha=0.7, facecolor='C1', edgecolor='C1', linewidth=1.5)

# ax.axvline(x=t_peak, color='k', linestyle='--', linewidth=1.5)
# ax.set_xlabel(r'$\gamma t$')
# ax.set_ylabel(r'$|Q_{\mathrm{box}}|$')
# ax.legend(fontsize=18)
# plt.tight_layout()
# plt.savefig(datadir+subdir+'Qbox_H_comp.svg', bbox_inches='tight')
# plt.show()
