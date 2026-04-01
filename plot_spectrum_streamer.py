#%% Importing libraries
import h5py as h5
import numpy as np
import cupy as cp
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from modules.mlsarray import MLSarray,Slicelist,irft2np,rft2np,irftnp,rftnp
from modules.gamma import gam_max
from functools import partial
import warnings
from modules.plot_basics import apply_style, savename as _savename, figsize_single
apply_style()
xtick_fontsize = matplotlib.rcParams.get('xtick.labelsize', 32)

#%% Load the HDF5 file

# Npx=512
Npx=1024
datadir=f'data/{Npx}/'
subdir = 'spectrum/'

# fname = datadir + 'out_kapt_0_4_D_0_1_H_3_6_em6.h5'
fname = datadir + 'out_kapt_2_0_D_0_1_H_8_6_em6.h5'
# fname = datadir + 'out_kapt_2_0_D_0_1_H_1_7_em5.h5'

with h5.File(fname, 'r', swmr=True) as fl:
    t = fl['fields/t'][:]
    nt = len(t)
    kx = fl['data/kx'][:]
    ky = fl['data/ky'][:]
    Lx = fl['params/Lx'][()]
    Ly = fl['params/Ly'][()]
    Npx= fl['params/Npx'][()]
    Npy= fl['params/Npy'][()]
    kapn = fl['params/kapn'][()]
    kapt = fl['params/kapt'][()]
    kapb = fl['params/kapb'][()]
    if 'D' in fl['params']:
        D = fl['params/D'][()]
    elif 'chi' in fl['params']:
        chi = fl['params/chi'][()]
        D = chi
    if 'H' in fl['params']:
        H = fl['params/H'][()]
    elif 'HP' in fl['params']:
        HP = fl['params/HP'][()]
        H = HP

Nx,Ny=2*Npx//3,2*Npy//3  
sl=Slicelist(Nx,Ny)
slbar=np.s_[int(Ny/2)-1:int(Ny/2)*int(Nx/2)-1:int(Nx/2)]
gammax=gam_max(kx,ky,kapn,kapt,kapb,D,H)
t=t*gammax
q = np.sqrt(np.abs(kx)**2 + np.abs(ky)**2)

flux_file = datadir + 'spectral_flux/' + fname.split('/')[-1].replace('out_', 'spectral_flux_')
with h5.File(flux_file, 'r') as fl:
    k_f = float(fl['k_f'][()])
    k_lin = float(fl['k_lin'][()])

nt = len(t)
print(f"nt: {nt}")

#%% Functions for energy and enstrophy

def Phi2S(phik, q, k, dk):
    ''' Returns the var(phi) spectrum'''
    phi2k = np.abs(phik)**2
    
    Phi2k = np.zeros(len(k))
    for i in range(len(k)):
        Phi2k[i] = np.sum(phi2k[np.where(np.logical_and(q>=k[i]-dk/2, q<k[i]+dk/2))])*dk
    return Phi2k

def Phi2S_ZF(phik, q, k, dk, slbar):
    ''' Returns the zonal var(phi) spectrum'''   
    phi2k_ZF = np.abs(phik[slbar])**2
    
    Phi2k_ZF = np.zeros(len(k))
    for i in range(len(k)):
        Phi2k_ZF[i] = np.sum(phi2k_ZF[np.where(np.logical_and(q[slbar]>=k[i]-dk/2, q[slbar]<k[i]+dk/2))])*dk
    return Phi2k_ZF

def P2S(pk, q, k, dk):
    ''' Returns the var(P) spectrum'''
    p2k = np.abs(pk)**2
    
    P2k = np.zeros(len(k))
    for i in range(len(k)):
        P2k[i] = np.sum(p2k[np.where(np.logical_and(q>=k[i]-dk/2, q<k[i]+dk/2))])*dk
    return P2k

def P2S_ZF(pk, q, k, dk, slbar):
    ''' Returns the zonal var(P) spectrum'''   
    pk_ZF = np.abs(pk[slbar])**2
    
    P2k_ZF = np.zeros(len(k))
    for i in range(len(k)):
        P2k_ZF[i] = np.sum(pk_ZF[np.where(np.logical_and(q[slbar]>=k[i]-dk/2, q[slbar]<k[i]+dk/2))])*dk
    return P2k_ZF

def ES(omk, q, k, dk):
    ''' Returns the total energy spectrum'''
    sigk=np.sign(ky)
    Wk = sigk+q**2
    ek = Wk*np.abs(omk)**2/q**4

    Ek = np.zeros(len(k))
    for i in range(len(k)):
        Ek[i] = np.sum(ek[np.where(np.logical_and(q>k[i]-dk/2,q<k[i]+dk/2))])*dk
    return Ek

def ES_ZF(omk, q, k, dk, slbar):
    ''' Returns the zonal total energy spectrum'''  
    sigk=np.sign(ky[slbar])
    Wk = sigk+q[slbar]**2 
    ek_ZF = Wk*np.abs(omk[slbar])**2/q[slbar]**4
    
    Ek_ZF = np.zeros(len(k))
    for i in range(len(k)):
        Ek_ZF[i] = np.sum(ek_ZF[np.where(np.logical_and(q[slbar]>=k[i]-dk/2, q[slbar]<k[i]+dk/2))])*dk
    return Ek_ZF

def KS(omk, q, k, dk):
    ''' Returns the kinetic energy spectrum'''
    ek = np.abs(omk)**2/q**2

    Ek = np.zeros(len(k))
    for i in range(len(k)):
        Ek[i] = np.sum(ek[np.where(np.logical_and(q>k[i]-dk/2,q<k[i]+dk/2))])*dk
    return Ek

def KS_ZF(omk, q, k, dk, slbar):
    ''' Returns the zonal kinetic energy spectrum'''  
    ek_ZF = np.abs(omk[slbar])**2/q[slbar]**2
    
    Ek_ZF = np.zeros(len(k))
    for i in range(len(k)):
        Ek_ZF[i] = np.sum(ek_ZF[np.where(np.logical_and(q[slbar]>=k[i]-dk/2, q[slbar]<k[i]+dk/2))])*dk
    return Ek_ZF

def GS(omk, pk, q, k, dk):
    ''' Returns the generalized energy spectrum'''
    sigk=np.sign(ky)
    phik=omk/q**2
    ek = np.abs(sigk*phik+pk)**2+q**2*np.abs(phik+pk)**2

    Ek = np.zeros(len(k))
    for i in range(len(k)):
        Ek[i] = np.sum(ek[np.where(np.logical_and(q>k[i]-dk/2,q<k[i]+dk/2))])*dk
    return Ek

def GS_ZF(omk, pk, q, k, dk, slbar):
    ''' Returns the zonal generalized energy spectrum'''  
    sigk=np.sign(ky)
    phik=omk/q**2
    ek_ZF = np.abs(sigk[slbar]*phik[slbar]+pk[slbar])**2+q[slbar]**2*np.abs(phik[slbar]+pk[slbar])**2
    
    Ek_ZF = np.zeros(len(k))
    for i in range(len(k)):
        Ek_ZF[i] = np.sum(ek_ZF[np.where(np.logical_and(q[slbar]>=k[i]-dk/2, q[slbar]<k[i]+dk/2))])*dk
    return Ek_ZF

def GKS(omk, pk, q, k, dk):
    ''' Returns the generalized kinetic energy spectrum'''
    phik=omk/q**2
    ek = q**2*np.abs(phik+pk)**2

    Ek = np.zeros(len(k))
    for i in range(len(k)):
        Ek[i] = np.sum(ek[np.where(np.logical_and(q>k[i]-dk/2,q<k[i]+dk/2))])*dk
    return Ek

def GKS_ZF(omk, pk, q, k, dk, slbar):
    ''' Returns the zonal generalized kinetic energy spectrum'''  
    phik=omk/q**2
    ek_ZF = q[slbar]**2*np.abs(phik[slbar]+pk[slbar])**2
    
    Ek_ZF = np.zeros(len(k))
    for i in range(len(k)):
        Ek_ZF[i] = np.sum(ek_ZF[np.where(np.logical_and(q[slbar]>=k[i]-dk/2, q[slbar]<k[i]+dk/2))])*dk
    return Ek_ZF

savename = partial(_savename, datadir+'spectrum/', fname)

#%% compute quantities

dk = ky[1]-ky[0]
k = np.arange(dk, np.max(ky), dk)

evol_fname = datadir + 'evol/' + fname.split('/')[-1].replace('out_', 'evol_')
try:
    with h5.File(evol_fname, 'r') as flevol:
        t_evol = flevol['t'][:]
        Qbox_t = flevol['Qbox_t'][:]
        it = int(np.argmax(Qbox_t))
        t_spike = t_evol[it]
except Exception as e:
    warnings.warn(f"Could not load Qbox_t from {evol_fname}: {e}\nDefaulting to last time index.")
    it = -1

with h5.File(fname, 'r', swmr=True) as fl:
    Omk = fl['fields/Omk'][it]
    Pk = fl['fields/Pk'][it]
    kpsq_loc  = kx**2 + ky**2
    Phik_loc  = -Omk / kpsq_loc
    Om_loc    = irft2np(Omk, Npx=Npx, Npy=Npy, Nx=Nx, sl=sl)
    vx_loc    = irft2np(-1j*ky*Phik_loc, Npx=Npx, Npy=Npy, Nx=Nx, sl=sl)
    vy_loc    = irft2np(1j*kx*Phik_loc, Npx=Npx, Npy=Npy, Nx=Nx, sl=sl)
    wx_loc    = irft2np(-1j*ky*Pk, Npx=Npx, Npy=Npy, Nx=Nx, sl=sl)
    Ombar_loc = np.mean(Om_loc, axis=1)
    RPhi_loc  = np.mean(vy_loc*vx_loc, axis=1)
    RP_loc    = np.mean(vy_loc*wx_loc, axis=1)
    reynolds_power = np.mean((RPhi_loc + RP_loc) * Ombar_loc)
    Phi2k = Phi2S(Pk, q, k, dk)
    Phi2k_ZF = Phi2S_ZF(Pk, q, k, dk, slbar)
    P2k = P2S(Pk, q, k, dk)
    P2k_ZF = P2S_ZF(Pk, q, k, dk, slbar)
    Ek = ES(Omk, q, k, dk)
    Ek_ZF = ES_ZF(Omk, q, k, dk, slbar)
    Kk = KS(Omk, q, k, dk)
    Kk_ZF = KS_ZF(Omk, q, k, dk, slbar)
    Gk = GS(Omk, Pk, q, k, dk)
    Gk_ZF = GS_ZF(Omk, Pk, q, k, dk, slbar)
    GKk = GKS(Omk, Pk, q, k, dk)
    GKk_ZF = GKS_ZF(Omk, Pk, q, k, dk, slbar)


print(f"Computing spectra at heat flux spike: t = {t_spike:.3f}, gamma*t = {gammax * t_spike:.3f}, index = {it}")

# Compute turbulent (non-zonal) spectra
Phi2k_turb = Phi2k - Phi2k_ZF
P2k_turb = P2k - P2k_ZF
Ek_turb = Ek - Ek_ZF
Kk_turb = Kk - Kk_ZF
Gk_turb = Gk - Gk_ZF
GKk_turb = GKk - GKk_ZF

k1 = np.argmin(np.abs(k - 1))

#%% Plot: Pressure variance spectrum

plt.figure(figsize=figsize_single)
plt.loglog(k, P2k, label = r'$\left|P_{k}\right|^2$')
plt.loglog(k[P2k_ZF>0], P2k_ZF[P2k_ZF>0], label = r'$\left|P_{k,\mathrm{ZF}}\right|^2$')
plt.loglog(k, P2k_turb, label = r'$\left|P_{k,\mathrm{turb}}\right|^2$')
plt.loglog(k, P2k[k1]*k**(-6), 'r--')
plt.loglog(k, P2k[k1]*k**(-8), 'k--')
plt.axvline(x=1, color='k', linestyle='--', linewidth=2)
plt.axvline(x=k_f, color='k', linestyle=':', linewidth=2)
plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2)
ax = plt.gca()
ax.text(k_f, -0.025, r'$k_f$', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=xtick_fontsize)
ax.text(k_lin, -0.025, r'$k_{\mathrm{lin}}$', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=xtick_fontsize)
plt.xlabel('$k$')
plt.ylabel(r'$\left|P_k\right|^2$')
plt.legend()
x_pos = k[int(0.6*len(k))]
plt.text(x_pos, P2k[k1]*x_pos**(-6)*1.2, '$k^{-6}$', color='r', fontsize=32)
plt.text(x_pos, P2k[k1]*x_pos**(-8)*0.8, '$k^{-8}$', color='k', fontsize=32)
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(savename('EP_spectrum_streamer'), bbox_inches='tight')
plt.show()

#%% Plot: Energy spectrum

plt.figure(figsize=figsize_single)
plt.loglog(k, Ek, label = '$E_{k}$')
plt.loglog(k[Ek_ZF>0], Ek_ZF[Ek_ZF>0], label = r'$E_{k,\mathrm{ZF}}$')
plt.loglog(k, Ek_turb, label = r'$E_{k,\mathrm{turb}}$')
plt.loglog(k, Ek[k1]*k**(-3), 'r--')
plt.loglog(k, Ek[k1]*k**(-5), 'k--')
plt.axvline(x=1, color='k', linestyle='--', linewidth=2)
plt.axvline(x=k_f, color='k', linestyle=':', linewidth=2)
plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2)
ax = plt.gca()
ax.text(k_f, -0.025, r'$k_f$', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=xtick_fontsize)
ax.text(k_lin, -0.025, r'$k_{\mathrm{lin}}$', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=xtick_fontsize)
plt.xlabel('$k$')
plt.ylabel('$E_k$')
plt.legend()
x_pos = k[int(0.6*len(k))]
plt.text(x_pos, Ek[k1]*x_pos**(-3)*1.2, '$k^{-3}$', color='r', fontsize=32)
plt.text(x_pos, Ek[k1]*x_pos**(-5)*0.8, '$k^{-5}$', color='k', fontsize=32)
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(savename('E_spectrum_streamer'), bbox_inches='tight')
plt.show()

#%% Plot: Generalized energy spectrum

plt.figure(figsize=figsize_single)
plt.loglog(k, Gk, label = '$G_{k}$')
plt.loglog(k[Gk_ZF>0], Gk_ZF[Gk_ZF>0], label = r'$G_{k,\mathrm{ZF}}$')
plt.loglog(k, Gk_turb, label = r'$G_{k,\mathrm{turb}}$')
plt.loglog(k, Gk[k1]*k**(-3), 'r--')
plt.loglog(k, Gk[k1]*k**(-5), 'k--')
plt.axvline(x=1, color='k', linestyle='--', linewidth=2)
plt.axvline(x=k_f, color='k', linestyle=':', linewidth=2)
plt.axvline(x=k_lin, color='k', linestyle='-.', linewidth=2)
ax = plt.gca()
ax.text(k_f, -0.025, r'$k_f$', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=xtick_fontsize)
ax.text(k_lin, -0.025, r'$k_{\mathrm{lin}}$', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=xtick_fontsize)
plt.xlabel('$k$')
plt.ylabel('$G_k$')
plt.legend()
x_pos = k[int(0.6*len(k))]
plt.text(x_pos, Gk[k1]*x_pos**(-3)*1.2, '$k^{-3}$', color='r', fontsize=32)
plt.text(x_pos, Gk[k1]*x_pos**(-5)*0.8, '$k^{-5}$', color='k', fontsize=32)
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(savename('G_spectrum_streamer'), bbox_inches='tight')
plt.show()
# %%
