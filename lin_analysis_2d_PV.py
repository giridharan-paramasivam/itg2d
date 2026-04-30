#%% Import modules

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from modules.plot_basics import apply_style, figsize_single
from modules.gamma_PV import Gamma
import torch

apply_style()

#%% Define functions

def init_kspace_grid(Nx,Ny,Lx,Ly):
    dkx=2*np.pi/Lx
    dky=2*np.pi/Ly
    kxl=np.r_[np.arange(0,Nx//2),np.arange(-Nx//2,0)]*dkx
    kyl=np.r_[np.arange(0,Ny//2+1)]*dky
    kx,ky=np.meshgrid(kxl,kyl,indexing='ij')
    return kx,ky

def one_over(x):
    out = np.zeros_like(x)
    return np.divide(1.0, x, out=out, where=x != 0)

def init_linmats(pars,kx,ky):
    kapn,kapt,kapb,D,H = [
        torch.tensor(pars[l]).cpu() for l in ['kapn','kapt','kapb','D','H']
    ]
    kpsq = kx**2 + ky**2
    kpsq = torch.where(kpsq==0, 1e-10, kpsq)
    sigk = ky>0
    Lk=sigk+kpsq
    lm=torch.zeros(kx.shape+(2,2),dtype=torch.complex64)
    lm[:,:,0,0]=-Gamma*kapb*ky-1j*D*kpsq-1j*sigk*H/kpsq**2
    lm[:,:,0,1]=(kapn+kapt)*ky-Gamma*(kapn+kapt)*ky*kpsq
    lm[:,:,1,0]=-kapb*ky/Lk
    lm[:,:,1,1]=(kapn*ky-(kapn+kapt)*ky*kpsq)/Lk-1j*D*kpsq-1j*sigk*H/kpsq**2
    return lm

def linfreq(pars, kx, ky):
    lm = init_linmats(pars, torch.from_numpy(kx), torch.from_numpy(ky)).cuda()
    w = torch.linalg.eigvals(lm)
    iw = torch.argsort(-w.imag, -1)
    lam = torch.gather(w, -1, iw).cpu().numpy()
    del lm, w, iw
    torch.cuda.empty_cache()
    return lam

#%% Initialize

Npx,Npy=4096,4096
Nx,Ny=2*int(Npx/3),2*int(Npy/3)
Lx,Ly=256*np.pi,256*np.pi
kx,ky=init_kspace_grid(Nx,Ny,Lx,Ly)
kapt=0.4
kapn=0.2
kapb=0.02
D=1e-3
H=0.0
base_pars={
    'kapn':kapn,
    'kapt':kapt,
    'kapb':kapb,
    'D':D,
    'H':H,
}

#%% Compute linear spectrum

om=linfreq(base_pars,kx,ky)
omr=om.real[:,:,0]
gam=om.imag[:,:,0]
Dturb=gam*one_over(kx**2+ky**2)
gammax=np.max(gam)

#%% Compute summary quantities

print('Gamma:',Gamma)
print('gammax:',gammax,'1/gammax:',1/gammax)
print('max index:', np.unravel_index(np.argmax(gam[:,:]), gam.shape))
print('max kx:', kx[np.unravel_index(np.argmax(gam[:,:]), gam.shape)])
print('max ky:', ky[np.unravel_index(np.argmax(gam[:,:]), gam.shape)])

ind_kxmax = np.argmax(gam, axis=0, keepdims=True)
gam_kxmax = np.take_along_axis(gam, ind_kxmax, axis=0).squeeze(axis=0)
gam_kx0 = gam[0,:]

ind_kxmax_Dturb = np.argmax(Dturb, axis=0, keepdims=True)
Dturb_kxmax = np.take_along_axis(Dturb, ind_kxmax_Dturb, axis=0).squeeze(axis=0)
Dturb_kx0 = Dturb[0,:]

kx_shifted = np.fft.fftshift(kx, axes=0)
ky_shifted = np.fft.fftshift(ky, axes=0)
gam_shifted = np.fft.fftshift(gam, axes=0)

#%% Plot gamma vs ky

plt.figure(figsize=figsize_single)
plt.plot(ky[0,:int(Ny/10)].T,gam_kxmax[:int(Ny/10)].T,'.-',label='$k_x= \\arg\\max_{k_x} \\left(\\gamma\\right)$')
plt.plot(ky[0,:int(Ny/10)].T,gam_kx0[:int(Ny/10)].T,'.-',label='$k_x=0$')
plt.axhline(0,color='k', linestyle='-', linewidth=1)
plt.plot(ky[0,:int(Ny/10)],-D*ky[0,:int(Ny/10)]**2,'k--',label='$-Dk_y^2$')
plt.legend()
plt.xlabel('$k_y$')
plt.ylabel('$\\gamma(k_y)$')
plt.tight_layout()
# plt.savefig(f'data_linear/gam_vs_ky_kapt_{str(kapt).replace(".", "_")}_itg2d_PV.svg', bbox_inches='tight')
plt.show()

#%% Plot Dturb vs ky

plt.figure(figsize=figsize_single)
plt.plot(ky[0,1:int(Ny/10)].T,Dturb_kxmax[1:int(Ny/10)].T,'.-',label='$k_x= \\arg\\max_{k_x} \\left(\\frac{\\gamma}{k_\\perp^2}\\right)$')
plt.plot(ky[0,1:int(Ny/10)].T,Dturb_kx0[1:int(Ny/10)].T,'.-',label='$k_x=0$')
plt.axhline(0,color='k', linestyle='-', linewidth=1)
plt.plot(ky[0,1:int(Ny/10)],-D*np.ones_like(ky[0,1:int(Ny/10)]),'k--',label='$-D$')
plt.legend()
plt.xlabel('$k_y$')
plt.ylabel('$\\left(\\frac{\\gamma}{k_\\perp^2}\\right)$')
plt.tight_layout()
# plt.savefig(f'data_linear/Dturb_vs_ky_kapt_{str(kapt).replace(".", "_")}_itg2d_PV.svg', bbox_inches='tight')
plt.show()

#%% Plot gamma colormesh

kx_central = kx_shifted[int(3*Nx/8):int(5*Nx/8), :int(Ny/8)]
ky_central = ky_shifted[int(3*Nx/8):int(5*Nx/8), :int(Ny/8)]
gam_central = gam_shifted[int(3*Nx/8):int(5*Nx/8), :int(Ny/8)]

plt.figure(figsize=figsize_single)
plt.pcolormesh(kx_central, ky_central, gam_central,vmax=gammax,vmin=-gammax,cmap='seismic', rasterized=True)
plt.xlabel('$k_x$')
plt.ylabel('$k_y$')
plt.colorbar()
plt.tight_layout()
# plt.savefig(f'data_linear/gamkxky_kapt_{str(kapt).replace(".", "_")}_itg2d_PV.svg', bbox_inches='tight')
plt.show()
