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

def funcc(mu,beta,kappa,w):
    prefactor = kappa*kappa
    return np.sqrt(mu**2 + (beta/200)**2)*prefactor[:,None,None]*np.exp(-w/2)



# ****** MAIN ******
if __name__ == '__main__':

    
    rcParams['text.usetex'] = True
    rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}', r'\usepackage{amssymb}']

    # Define static params

    kB = 8.617e-5 # eV/K

    e_d = -0.2 # LUMO energy [eV]
    e_a = 0.4 # LUMO+1 energy [eV]

    gamL = 0.2 #eV
    gamR = 0.2 #eV

    kappa_grid = np.ones(4) * 0.5
    w0_grid = np.ones(2) * 0.05
    #muL_grid = np.linspace(-1,0.25,20) #muR = -muL always
    #muL_grid = np.linspace(-1,1,200)
    muL_grid = np.ones(5)*10
    dmu = 2*muL_grid
    #temp_grid = np.linspace(40,4000,200)

    beta_grid = np.ones(3) * 200.0

    mm, bb = np.meshgrid(muL_grid,beta_grid,indexing='ij',sparse=True)

    output_shape = (w0_grid.shape[0], kappa_grid.shape[0], muL_grid.shape[0], beta_grid.shape[0])


    k_LR_01 = np.zeros(output_shape)


    for j, ww in enumerate(w0_grid):
        print(j)
        k_LR_01[j,:,:,:] = funcc(mm,bb,kappa_grid,ww)



    print(k_LR_01.shape)
    plt.plot(dmu, k_LR_01[0,0,:,0],'-',label='ye')
    plt.plot(dmu, k_LR_01[1,0,:,0],'--',label='yi')
    plt.plot(dmu, k_LR_01[1,0,:,1],'-.',label='yo')    
    plt.plot(dmu, k_LR_01[0,0,:,1],':',label='ya')
    plt.legend()
    plt.show()


 
    #dissipative mode

    ww, bb = np.meshgrid(w0_grid, beta_grid, indexing='ij',sparse=True)
    nph = bose_einstein(ww, bb)
    print(nph.shape)

     
    p1 = (k_LR_01 +  nph[:, None, None, :]) / (k_LR_01  + (2* nph[:, None, None, :] + 1))
    p0 = 1 - p1

    print(p1[0,1,:,0])

    plt.plot(dmu,p1[0,1,:,0],'b-',lw=0.8,label='$p_1$')
    plt.plot(dmu,p0[0,1,:,0],'r-',lw=0.8,label='$p_0$')
    plt.xlabel('$\Delta\mu$ [eV]')
    plt.ylabel('population')
    plt.legend()
    plt.show()

    print(p1[0,0,0,0])
    print(np.all(p1 == p1[0,0,0,0]))
