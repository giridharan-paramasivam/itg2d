#%% Import modules

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from modules.plot_basics import apply_style, figsize_single
import torch

apply_style()

#%% Define Functions

def init_kspace_grid(Nx,Ny,Lx,Ly):
    dkx=2*np.pi/Lx
    dky=2*np.pi/Ly
    kxl=np.r_[np.arange(0,Nx//2),np.arange(-Nx//2,0)]*dkx
    kyl=np.r_[np.arange(0,Ny//2+1)]*dky
    kx,ky=np.meshgrid(kxl,kyl,indexing='ij')
    return kx,ky

def init_linmats(pars,kx,ky,FLR=1.0):    
    # Initializing the linear matrices
    kapn,kapt,kapb,tau,D,H = [
        torch.tensor(pars[l]).cpu() for l in ['kapn','kapt','kapb','tau','D','H']
    ]
    kpsq = kx**2 + ky**2
    kpsq = torch.where(kpsq==0, 1e-10, kpsq)
        
    sigk = ky>0
    Lk=tau*sigk+kpsq
    lm=torch.zeros(kx.shape+(2,2),dtype=torch.complex64)
    lm[:,:,0,0]=-1j*sigk*D*kpsq-1j*sigk*H/kpsq**2
    lm[:,:,0,1]=(kapn+kapt)*ky
    lm[:,:,1,0]=-kapb*ky/Lk
    lm[:,:,1,1]=(kapn*ky-FLR*(kapn+kapt)*ky*kpsq)/Lk-1j*sigk*D*kpsq-1j*sigk*H/kpsq**2

    return lm

def linfreq(pars, kx, ky, FLR=1.0):
    lm = init_linmats(pars, torch.from_numpy(kx), torch.from_numpy(ky), FLR=FLR).cuda()
    # print(lm.device)
    w = torch.linalg.eigvals(lm)
    iw = torch.argsort(-w.imag, -1)
    lam = torch.gather(w, -1, iw).cpu().numpy()
    # vi = torch.gather(v, -1, iw.unsqueeze(-2).expand_as(v)).cpu().numpy()
    del lm, w, iw
    torch.cuda.empty_cache()
    return lam

def one_over(x):
    out = np.zeros_like(x)
    return np.divide(1.0, x, out=out, where=x != 0)

#%% Initialize

Npx,Npy=4096,4096
Nx,Ny=2*int(Npx/3),2*int(Npy/3)
# Lx,Ly=32*np.pi,32*np.pi #sim for 512x512
Lx,Ly=256*np.pi,256*np.pi 
kx,ky=init_kspace_grid(Nx,Ny,Lx,Ly)
kapt=0.4 #rho_i/L_T >0.2
kapn=0.2 #rho_i/L_n
kapb=0.02 #2*rho_i/L_B
D=1e-3 #0.1
H= 0*1e-06 # 4.2581e-06 for kapt=0.5
base_pars={'kapn':kapn,
      'kapt':kapt,
      'kapb':kapb,
      'tau':1.,#Ti/Te
      'D':D,
      'H':H}

#%% Compute om

om=linfreq(base_pars,kx,ky,FLR=1.0)
omr=om.real[:,:,0]
gam=om.imag[:,:,0]
gammax=np.max(gam)

om_wo_FLR=linfreq(base_pars,kx,ky,FLR=0.0)
omr_wo_FLR=om_wo_FLR.real[:,:,0]
gam_wo_FLR=om_wo_FLR.imag[:,:,0]
gammax_wo_FLR=np.max(gam_wo_FLR)

#%% Compute quantities

print('gammax:',gammax,'1/gammax:',1/gammax)
print('max index:', np.unravel_index(np.argmax(gam[:,:]), gam.shape))
print('max kx:', kx[np.unravel_index(np.argmax(gam[:,:]), gam.shape)])
print('max ky:', ky[np.unravel_index(np.argmax(gam[:,:]), gam.shape)])

ind_kxmax = np.argmax(gam, axis=0, keepdims=True) 
gam_kxmax = np.take_along_axis(gam, ind_kxmax, axis=0).squeeze(axis=0)  #select gam at the max kx index
omr_kxmax = np.take_along_axis(omr, ind_kxmax, axis=0).squeeze(axis=0) 
kxmax_ky= np.take_along_axis(kx, ind_kxmax, axis=0).squeeze(axis=0) #select kx at the max kx index
gam_kx0 = gam[0,:]
omr_kx0 = omr[0,:]

ind_kxmax_wo_FLR = np.argmax(gam_wo_FLR, axis=0, keepdims=True)
gam_wo_FLR_kxmax = np.take_along_axis(gam_wo_FLR, ind_kxmax_wo_FLR, axis=0).squeeze(axis=0)
omr_wo_FLR_kxmax = np.take_along_axis(omr_wo_FLR, ind_kxmax_wo_FLR, axis=0).squeeze(axis=0)
kxmax_ky_wo_FLR = np.take_along_axis(kx, ind_kxmax_wo_FLR, axis=0).squeeze(axis=0)
gam_wo_FLR_kx0 = gam_wo_FLR[0,:]
omr_wo_FLR_kx0 = omr_wo_FLR[0,:]

kx_shifted = np.fft.fftshift(kx, axes=0)
ky_shifted = np.fft.fftshift(ky, axes=0)
gam_shifted = np.fft.fftshift(gam, axes=0)
gam_wo_FLR_shifted = np.fft.fftshift(gam_wo_FLR, axes=0)

#%% Plot: gam vs ky finite kx and FLR comparison

def plot_gam_vs_ky_FLR_comp(all_legends=False):
    slky = slice(0, int(Ny/10))
    plt.figure(figsize=figsize_single)
    l1 = plt.plot(ky[0,slky].T,gam_kxmax[slky].T,'.-',label=r'$\displaystyle k_x= \arg\max_{k_x} \left(\gamma\right)$')
    l2 = plt.plot(ky[0,slky].T,gam_kx0[slky].T,'.-',label=r'$\displaystyle k_x=0$')
    l3 = plt.plot(ky[0,slky].T,gam_wo_FLR_kxmax[slky].T,'.-',label=r'$\displaystyle k_x= \arg\max_{k_x} \left(\gamma\right)$, no FLR')
    l4 = plt.plot(ky[0,slky].T,gam_wo_FLR_kx0[slky].T,'.-',label=r'$\displaystyle k_x=0$, no FLR')
    plt.axhline(0,color='k', linestyle='-', linewidth=1)
    l5 = plt.plot(ky[0,slky],-D*ky[0,slky]**2,'k--',label=r'$-Dk_y^2$')
    if all_legends:
        leg1 = plt.legend(handles=[l1[0], l2[0], l3[0], l4[0]], loc='best', fontsize=18)
        plt.gca().add_artist(leg1)
    plt.legend([l5[0]], [r'$-Dk_y^2$'], loc='best', fontsize=24)
    plt.xlabel('$k_y$')
    plt.ylabel('$\\gamma(k_y)$')
    plt.tight_layout()
    plt.savefig(f'data_linear/gam_vs_ky_FLR_comp_kapt_{str(kapt).replace(".", "_")}_itg2d.svg', bbox_inches='tight')
    plt.show()

plot_gam_vs_ky_FLR_comp(all_legends=True)  
# %%
