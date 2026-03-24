import numpy as np
# from modules.mlsarray import slicelist
import matplotlib.pyplot as plt

def apply_style():
    plt.rcParams.update({
        'lines.linewidth': 4,
        'axes.linewidth': 3,
        'xtick.major.width': 3,
        'ytick.major.width': 3,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'xtick.minor.width': 1.5,
        'ytick.minor.width': 1.5,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'savefig.dpi': 300,
        'font.size': 22,
        'axes.titlesize': 22,
        'axes.labelsize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20,
        'legend.edgecolor': 'black'
    })

def savename(datadir, fname, prefix):
    if fname.endswith('out.h5'):
        return datadir + prefix + '_.pdf'
    return datadir + prefix + '_' + fname.split('/')[-1].split('out_')[-1].replace('.h5', '.pdf')

def symmetrize_y_axis(axes):
    y_max = np.abs(axes.get_ylim()).max()
    axes.set_ylim(ymin=-y_max, ymax=y_max)

def irft2_g(uk, Nx, Ny):
    Nxh = int(Nx/2)
    u = np.zeros((Npx, int(Npy/2)+1), dtype=complex)
    u[:Nxh,:Nxh] = uk[:Nxh,:Nxh]
    u[-Nxh+1:,:Nxh] = uk[-Nxh+1:,:Nxh]
    return np.fft.irfft2(u, norm='forward')

def rft2_g(u, Nx, Ny):
    Nxh = int(Nx/2)
    uk = np.zeros((Nx, int(Ny/2)+1), dtype=complex)
    yk = np.fft.rfft2(u, norm='forward')
    uk[:Nxh,:-1] = yk[:Nxh,:int(Ny/2)]
    uk[-1:-Nxh:-1,:-1] = yk[-1:-Nxh:-1,:int(Ny/2)]
    uk[0,0] = 0
    return uk

def irft_g(vk, Npx):
    Nxh = int(Npx/3)
    v = np.zeros(int(Npx/2)+1, dtype=complex)
    v[:Nxh] = vk[:Nxh]
    return np.fft.irfft(v, norm='forward')

def ubar(uk, Npx, Npy, Nx, Ny, sl):
    slbar=np.s_[int(Ny/2)-1:int(Ny/2)*int(Nx/2)-1:int(Nx/2)]
    Nxh = int(Npx/3)
    vk = np.zeros(int(Npx/2)+1, dtype=complex)
    vk[1:Nxh] = uk[slbar]
   
    return np.fft.irfft(vk, norm='forward')
    