#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.integrate import simpson


def lor(e,gamma,e0):
    return gamma / ( (e-e0)*(e-e0) + gamma*(gamma/4)  )

def bose_einstein(e,beta):
    return 1.0 / (np.exp(beta*e) - 1)

def fermi_dirac(e,mu,beta):
    return 1.0 / ( np.exp(beta*(e-mu)) + 1 )

def transfer_rate(e,mu,gamma_i,gamma_f,e_i,e_f,beta,kappa,w):
    lor_i = lor(e,gamma_i,e_i) 
    lor_f = lor(e+w,gamma_f,e_f)


    fd_i = fermi_dirac(e,mu,beta)
    cfd_f = 1 - fermi_dirac(e+w,-mu,beta)
    #cfd_f = 1 - fermi_dirac(e+w,0,beta)

    prefactor = kappa*kappa
    integral = simpson(lor_i*lor_f*fd_i*cfd_f, e, axis=-1)
    out = prefactor[:,None,None]*integral


    return out


# ****** MAIN ******
if __name__ == '__main__':

    
    rcParams['text.usetex'] = True
    rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}'

    # Define static params

    kB = 8.617e-5 # eV/K

    e_d = -0.2 # LUMO energy [eV]
    #e_a = 0.4 # LUMO+1 energy [eV]
    e_a = 0.4 # LUMO+1 energy [eV]

    gamL = 0.2 #eV
    gamR = 0.2 #eV

    kappa_grid = np.ones(4) * 0.1
    w0_grid = np.ones(2) * 0.05
    #muL_grid = np.linspace(-1,0.25,20) #muR = -muL always
    muL_grid = np.linspace(-0.5,1,200)
    muL_grid[np.argmin(np.abs(muL_grid))] = 0
    dmu = 2*muL_grid
    #temp_grid = np.linspace(40,4000,200)

    beta_grid = np.ones(3) * 200.0
    e_grid = np.linspace(-5,5,20000)

    mm, bb, ee = np.meshgrid(muL_grid,beta_grid,e_grid,indexing='ij',sparse=True)

    output_shape = (w0_grid.shape[0], kappa_grid.shape[0], muL_grid.shape[0], beta_grid.shape[0])


    k_LR_01 = np.zeros(output_shape)
    k_RL_01 = np.zeros(output_shape)
    k_LR_10 = np.zeros(output_shape)
    k_RL_10 = np.zeros(output_shape)


    for j, ww in enumerate(w0_grid):
        print(j)

        k_LR_01[j,:,:,:] = transfer_rate(ee,mm,gamL,gamR,e_d+1*mm,e_a-1*mm,bb,kappa_grid,-ww)
        k_RL_01[j,:,:,:] = transfer_rate(ee,-mm,gamR,gamL,e_a-1*mm,e_d+1*mm,bb,kappa_grid,-ww)
        #k_RL_01[j,:,:,:] = transfer_rate(ee,mm,gamR,gamL,e_a,e_d,bb,kappa_grid,ww)

        k_LR_10[j,:,:,:] = transfer_rate(ee,mm,gamL,gamR,e_d+1*mm,e_a-1*mm,bb,kappa_grid,ww)
        k_RL_10[j,:,:,:] = transfer_rate(ee,-mm,gamR,gamL,e_a-1*mm,e_d+1*mm,bb,kappa_grid,ww)
        #k_RL_10[j,:,:,:] = transfer_rate(ee,mm,gamR,gamL,e_a,e_d,bb,kappa_grid,-ww)



    print(k_LR_01.shape)
    k_01 = k_LR_01 + k_RL_01
    k_10 = k_LR_10 + k_RL_10

    # non-disspative mode

    p1 = k_01 / (k_01 + k_10)
    p0 = 1 - p1

    dmu = 2*muL_grid

    plt.plot(dmu,p1[0,0,:,0],'b-',lw=1.0,label='$p_1$')
    plt.plot(dmu,p0[0,0,:,0],'r-',lw=1.0,label='$p_0$')
    plt.xlabel('$\Delta\mu$ [eV]')
    plt.ylabel('population')
    plt.legend()
    plt.show()


 
    #dissipative mode

    gam_phonon = np.array([0, 0.001, 0.4])
    beta_ph = np.ones(3) * 40
    cs = ['k','r','b']
    lstyles = ['--', '-','-.']
    ww, bb = np.meshgrid(w0_grid, beta_ph, indexing='ij',sparse=True)
    nph = bose_einstein(ww, bb)
    print(nph.shape)

    
    print(k_01.shape)
    print(k_10.shape)
    
    for g, clr, l in zip(gam_phonon,cs,lstyles):

        p1 = (k_01 + g * nph[:, None, None, :]) / (k_01 + k_10 + g * (2* nph[:, None, None, :] + 1))
        p0 = 1 - p1
        beta_eff = np.log(p0/p1)/w0_grid[0]

        plt.plot(dmu,beta_eff[0,0,:,0],c=clr,ls=l,lw=0.8, label='$\Gamma_{ph} = %4.3f\,$eV'%g)
        plt.xlabel('$\Delta\mu$ [eV]')
        plt.ylabel('population')
    plt.legend()
    plt.show()

    np.save('full_beta_eff.npy',beta_eff)
