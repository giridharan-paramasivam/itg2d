#%% Import modules
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from modules.plot_basics import FIGSIZE_DOUBLE
import torch 

plt.rcParams['lines.linewidth'] = 4
plt.rcParams['axes.linewidth'] = 3  
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['ytick.major.width'] = 3
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['xtick.minor.width'] = 1.5 
plt.rcParams['ytick.minor.width'] = 1.5 
plt.rcParams['savefig.dpi'] = 100
plt.rcParams.update({
    "font.size": 22,          # default text
    "axes.titlesize": 30,     # figure title
    "axes.labelsize": 26,     # x/y labels
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 22
})

#%% Define Functions

def init_kspace_grid(Nx,Ny,Lx,Ly):
    dkx=2*np.pi/Lx
    dky=2*np.pi/Ly
    kxl=np.r_[np.arange(0,Nx//2),np.arange(-Nx//2,0)]*dkx
    kyl=np.r_[np.arange(0,Ny//2+1)]*dky
    kx,ky=np.meshgrid(kxl,kyl,indexing='ij')
    return kx,ky

def init_linmats(pars,kx,ky):    
    # Initializing the linear matrices
    kapn,kapt,kapb,tau,D,H = [
        torch.tensor(pars[l]).cpu() for l in ['kapn','kapt','kapb','tau','D','H']
    ]
    kpsq = kx**2 + ky**2
    kpsq = torch.where(kpsq==0, 1e-10, kpsq)
        
    sigk = ky>0
    Wk=tau*sigk+kpsq
    lm=torch.zeros(kx.shape+(2,2),dtype=torch.complex64)
    lm[:,:,0,0]=-1j*sigk*D*kpsq-1j*sigk*H/kpsq**2
    lm[:,:,0,1]=(kapn+kapt)*ky
    lm[:,:,1,0]=-kapb*ky/Wk
    lm[:,:,1,1]=(kapn*ky-(kapn+kapt)*ky*kpsq)/Wk-1j*sigk*D*kpsq-1j*sigk*H/kpsq**2

    return lm

def linfreq(pars, kx, ky):
    lm = init_linmats(pars, torch.from_numpy(kx), torch.from_numpy(ky)).cuda()
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
base_pars={'kapn':kapn, 'kapt':kapt, 'kapb':kapb, 'tau':1.}

cases = [
    {'D': 0.0,  'H': 0.0,  'label': r'$D=0,\ H=0$'},
    {'D': 0.1,  'H': 0.0,  'label': r'$D=0.1,\ H=0$'},
    {'D': 0.0,  'H': 1e-6, 'label': r'$D=0,\ H=10^{-6}$'},
    {'D': 0.1,  'H': 1e-6, 'label': r'$D=0.1,\ H=10^{-6}$'},
]

#%% Compute om for all cases

slky = slice(1, int(Ny/10))

for c in cases:
    pars = {**base_pars, 'D': c['D'], 'H': c['H']}
    om = linfreq(pars, kx, ky)
    gam = om.imag[:,:,0]
    Dturb = gam * one_over(kx**2 + ky**2)

    ind_kxmax = np.argmax(gam, axis=0, keepdims=True)
    c['gam_kxmax'] = np.take_along_axis(gam, ind_kxmax, axis=0).squeeze(axis=0)
    c['gam_kx0']   = gam[0,:]

    ind_kxmax_D = np.argmax(Dturb, axis=0, keepdims=True)
    c['Dturb_kxmax'] = np.take_along_axis(Dturb, ind_kxmax_D, axis=0).squeeze(axis=0)
    c['Dturb_kx0']   = Dturb[0,:]

#%% gam vs ky

plt.figure(figsize=FIGSIZE_DOUBLE)
for c in cases:
    plt.plot(ky[0,slky], c['gam_kxmax'][slky], '.-', label=c['label'])
plt.axhline(0, color='k', linestyle='-', linewidth=1)
plt.legend()
plt.grid(which='major', linestyle='--', linewidth=0.5)
plt.xlabel('$k_y$')
plt.ylabel(r'$\gamma(k_y)$')
plt.tight_layout()
plt.savefig(f'data_linear/gam_vs_ky_kapt_{str(kapt).replace(".", "_")}_itg2d_comp.pdf', dpi=100)
plt.show()

#%% Dturb vs ky

plt.figure(figsize=FIGSIZE_DOUBLE)
for c in cases:
    plt.plot(ky[0,slky], c['Dturb_kxmax'][slky], '.-', label=c['label'])
plt.axhline(0, color='k', linestyle='-', linewidth=1)
plt.ylim(-0.15, 0.55)
plt.legend()
plt.grid(which='major', linestyle='--', linewidth=0.5)
plt.xlabel('$k_y$')
plt.ylabel(r'$\max_{k_x}(\gamma / k^2)$')
plt.tight_layout()
plt.savefig(f'data_linear/Dturb_vs_ky_kapt_{str(kapt).replace(".", "_")}_itg2d_comp.pdf', dpi=100)
plt.show()
# %%
