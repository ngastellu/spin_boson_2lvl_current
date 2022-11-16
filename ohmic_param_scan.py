#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from time import perf_counter
from os import path
from param_parser import ParameterParser


def lor(e,gamma,e0):
    return gamma / ( (e-e0)*(e-e0) + gamma*(gamma/4)  )

def bose_einstein(e,beta):
    return 1.0 / (np.exp(beta*e) - 1)

def fermi_dirac(e,mu,beta):
    return 1.0 / ( np.exp(beta*(e-mu)) + 1 )

def transfer_rate(e,mu,gamma_i,gamma_f,e_i,e_f,beta,kappa,w):
    #print('Evaluating Lorentzians...')
    lor_start = perf_counter()
    lor_i = lor(e,gamma_i,e_i) 
    lor_f = lor(e+w,gamma_f,e_f)
    lor_end = perf_counter()
    #print('Done with Lorentzians. Computation time [seconds] = ', lor_end - lor_start)


    #print('Evaluating Fermi Dirac distributions...')
    fd_start = perf_counter()
    fd_i = fermi_dirac(e,mu,beta)
    cfd_f = 1 - fermi_dirac(e+w,-mu,beta)
    fd_end = perf_counter()
    #print('Done with Fermi Dirac distributions. Computation time [seconds] = ', fd_end - fd_start)

    prefactor = kappa*kappa
    #print(prefactor.shape)
    #print('Evaluating integral...')
    int_start = perf_counter()
    integral = simpson(lor_i*lor_f*fd_i*cfd_f, e, axis=-1)
    int_end = perf_counter()
    #print('Done with integral. Computation time [seconds] = ', int_end - int_start)
    out = prefactor[:,None]*integral

    #print(out.shape)

    return out / (2.0*np.pi)


# ****** MAIN ******
if __name__ == '__main__':

    kB = 8.617e-5 # eV/K

    outdir = 'ohmic_dmu0.05_aligned_focused'

    param_file = 'aligned_focused_max_dmu0.02_fine_grids.json'
    pp = ParameterParser(param_file)

    e_d, e_a, gamL, gamR, gam_phonon = pp.load_intrinsic()
    
    kappa_grid, w0_grid, temp_grid, e_grid = \
        pp.load_grids(plist=['kappa_grid', 'frequency_grid',
            'temperature_grid', 'energy_grid'])
    
    muL = pp.load_specific(['ohmic_dmu'])[0] / 2

    beta_grid = 1.0/(kB * temp_grid)

    bb, ee = np.meshgrid(beta_grid,e_grid,indexing='ij',sparse=True)

    output_shape = (w0_grid.shape[0], kappa_grid.shape[0], beta_grid.shape[0])

    k_LR_01 = np.zeros(output_shape)
    k_RL_01 = np.zeros(output_shape)
    k_LR_10 = np.zeros(output_shape)
    k_RL_10 = np.zeros(output_shape)

    shift_lvls = False


    for j, ww in enumerate(w0_grid):
        print('\n')
        print(j)

        if shift_lvls:
            k_LR_01[j,:,:] = transfer_rate(ee,muL,gamL,gamR,e_d+1*muL,e_a-1*muL,bb,kappa_grid,-ww)
            k_RL_01[j,:,:] = transfer_rate(ee,-muL,gamR,gamL,e_a-1*muL,e_d+1*muL,bb,kappa_grid,-ww)

            k_LR_10[j,:,:] = transfer_rate(ee,muL,gamL,gamR,e_d+1*muL,e_a-1*muL,bb,kappa_grid,ww)
            k_RL_10[j,:,:] = transfer_rate(ee,-muL,gamR,gamL,e_a-1*muL,e_d+1*muL,bb,kappa_grid,ww)

        else:
            k_LR_01[j,:,:] = transfer_rate(ee,muL,gamL,gamR,e_d,e_a,bb,kappa_grid,-ww)
            k_RL_01[j,:,:] = transfer_rate(ee,-muL,gamR,gamL,e_a,e_d,bb,kappa_grid,-ww)

            k_LR_10[j,:,:] = transfer_rate(ee,muL,gamL,gamR,e_d,e_a,bb,kappa_grid,ww)
            k_RL_10[j,:,:] = transfer_rate(ee,-muL,gamR,gamL,e_a,e_d,bb,kappa_grid,ww)

    plt.plot(temp_grid,k_RL_01[0,0,:],'b-',lw=0.8,label='RL 01')
    plt.plot(temp_grid,k_LR_01[0,0,:],'r-',lw=0.8,label='LR 01')
    plt.plot(temp_grid,k_RL_10[0,0,:],'b--',lw=0.8,label='RL 10')
    plt.plot(temp_grid,k_LR_10[0,0,:],'r--',lw=0.8, label='RL 10')
    plt.xlabel('T [K]')
    plt.ylabel('Electron transfer rate [Hz]')
    plt.legend()
    plt.show()

    np.save('MAC_kLR01_dmu0.005_aligned.npy',k_LR_01)
    np.save('MAC_kLR10_dmu0.005_aligned.npy',k_LR_10)
    np.save('MAC_kRL01_dmu0.005_aligned.npy',k_RL_01)
    np.save('MAC_kRL10_dmu0.005_aligned.npy',k_RL_10)

    k_01 = k_LR_01 + k_RL_01
    k_10 = k_LR_10 + k_RL_10

    # non-dissipative mode
    p1 = k_01 / (k_01 + k_10)
    p0 = 1 - p1

    current = p1 * (k_LR_10 - k_RL_10) + p0 * (k_LR_01 - k_RL_01)

    np.save('MAC_current_non-dis_dmu0.005_aligned.npy', current)

    plt.plot(temp_grid,current[0,-1,:],'b-',lw=0.8)
    plt.plot(temp_grid,current[-1,-1,:],'r-',lw=0.8)
    plt.plot(temp_grid,current[0,0,:],'g-',lw=0.8)
    plt.xlabel('T [K]')
    plt.ylabel('Current [Hz]')
    plt.show()
    
    #dissipative mode

    ww, bb = np.meshgrid(w0_grid, beta_grid, indexing='ij',sparse=True)
    nph = bose_einstein(ww, bb)
    print(nph.shape)
    p1 = (k_01 + gam_phonon * nph[:, None, :]) / (k_01 + k_10 + gam_phonon * (2* nph[:, None, :] + 1))
    p0 = 1 - p1

    current = p1 * (k_LR_10 - k_RL_10) + p0 * (k_LR_01 - k_RL_01)
    np.save('MAC_current_dis_dmu0.005_aligned.npy', current)

    plt.plot(temp_grid,current[0,-1,:],'b-',lw=0.8)
    plt.plot(temp_grid,current[-1,-1,:],'r-',lw=0.8)
    plt.plot(temp_grid,current[0,0,:],'g-',lw=0.8)
    plt.xlabel('T [K]')
    plt.ylabel('Current [Hz]')
    plt.show()
