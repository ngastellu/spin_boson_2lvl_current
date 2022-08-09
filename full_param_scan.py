#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson


def lor(e,gamma,e0):
    return gamma / ( (e-e0)*(e-e0) + gamma*(gamma/4)  )

def bose_einstein(e,beta):
    return 1.0 / (np.exp(beta*e) - 1)

def fermi_dirac(e,mu,beta):
    return 1.0 / ( np.exp(beta*(e-mu)) + 1 )

def transfer_rate(e,mu,gamma_i,gamma_f,e_i,e_f,beta,kappa,w):
    lor_i = lor(e,gamma_i,e_i) 
    print(lor_i.shape)
    lor_f = lor(e+w,gamma_f,e_f)
    print(lor_f.shape)


    fd_i = fermi_dirac(e,mu,beta)
    print(fd_i.shape)
    cfd_f = 1 - fermi_dirac(e+w,-mu,beta)
    print(cfd_f.shape)
    print(kappa.shape)

    prefactor = kappa*kappa
    print(prefactor.shape)
    integral = simpson(lor_i*lor_f*fd_i*cfd_f, e, axis=-1)
    print(integral.shape)
    out = prefactor[:,None,None]*integral

    print(out.shape)

    return out


# ****** MAIN ******
if __name__ == '__main__':

    # Define static params

    kB = 8.617e-5 # eV/K

    e_d = -0.2 # LUMO energy [eV]
    e_a = 0.4 # LUMO+1 energy [eV]

    gamL = 0.2 #eV
    gamR = 0.2 #eV
    gam_phonon = 0.1

    kappa_grid = np.linspace(0.1,1.0,10)
    w0_grid = np.linspace(0.1,1.0,10)
    #muL_grid = np.linspace(-1,0.25,20) #muR = -muL always
    muL_grid = np.ones(20)*0.4
    temp_grid = np.linspace(40,4000,200)

    beta_grid = 1.0 / (kB * temp_grid)
    e_grid = np.linspace(-5,5,20000)

    mm, bb, ee = np.meshgrid(muL_grid,beta_grid,e_grid,indexing='ij',sparse=True)

    output_shape = (w0_grid.shape[0], kappa_grid.shape[0], muL_grid.shape[0], beta_grid.shape[0])


    k_LR_01 = np.zeros(output_shape)
    k_RL_01 = np.zeros(output_shape)
    k_LR_10 = np.zeros(output_shape)
    k_RL_10 = np.zeros(output_shape)


    for j, ww in enumerate(w0_grid):
        print(j)

        k_LR_01[j,:,:,:] = transfer_rate(ee,mm,gamL,gamR,e_d,e_a,bb,kappa_grid,ww)
        k_RL_01[j,:,:,:] = transfer_rate(ee,-mm,gamR,gamL,e_a,e_d,bb,kappa_grid,ww)

        k_LR_10[j,:,:,:] = transfer_rate(ee,mm,gamL,gamR,e_d,e_a,bb,kappa_grid,-ww)
        k_RL_10[j,:,:,:] = transfer_rate(ee,-mm,gamR,gamL,e_a,e_d,bb,kappa_grid,-ww)

    plt.plot(temp_grid,k_RL_01[0,0,0,:],'r-',lw=0.8)
    plt.plot(temp_grid,k_LR_01[0,0,0,:],'b-',lw=0.8)
    plt.plot(temp_grid,k_RL_10[0,0,0,:],'r--',lw=0.8)
    plt.plot(temp_grid,k_LR_10[0,0,0,:],'b--',lw=0.8)
    plt.xlabel('T [K]')
    plt.ylabel('Electron transfer rate [Hz]')
    plt.show()

    np.save('full_kLR01.npy',k_LR_01)
    np.save('full_kLR10.npy',k_LR_10)
    np.save('full_kRL01.npy',k_RL_01)
    np.save('full_kRL10.npy',k_RL_10)

    k_01 = k_LR_01 + k_RL_01
    k_10 = k_LR_10 + k_RL_10

    # non-dissipative mode
    p1 = k_01 / (k_01 + k_10)
    p0 = 1 - p1

    current = p1 * (k_LR_10 - k_RL_10) + p0 * (k_LR_01 - k_RL_01)

    np.save('full_current_non-dis.npy', current)

    plt.plot(temp_grid,current[0,9,0,:],'b-',lw=0.8)
    plt.plot(temp_grid,current[9,9,0,:],'r-',lw=0.8)
    plt.plot(temp_grid,current[0,0,0,:],'g-',lw=0.8)
    plt.xlabel('T [K]')
    plt.ylabel('Current [Hz]')
    plt.show()
    
    #dissipative mode


    ww, bb = np.meshgrid(w0_grid, beta_grid, indexing='ij')
    nph = bose_einstein(ww, bb)
    print(nph.shape)
    p1 = (k_01 + gam_phonon * nph[:, None, None, :]) / (k_01 + k_10 + gam_phonon * (2* nph[: None, None,:] + 1))
    p0 = 1 - p1

    current = p1 * (k_LR_10 - k_RL_10) + p0 * (k_LR_01 - k_RL_01)
    np.save('full_current_dis.npy', current)

    plt.plot(temp_grid,current[0,9,0,:],'b-',lw=0.8)
    plt.plot(temp_grid,current[9,9,0,:],'r-',lw=0.8)
    plt.plot(temp_grid,current[0,0,0,:],'g-',lw=0.8)
    plt.xlabel('T [K]')
    plt.ylabel('Current [Hz]')
    plt.show()
