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

def c_fermi_dirac(e,mu,beta):
    '''Complementary FD distribution; 1-FD(e,mu,beta);
        DO NOT USE: Numerically unstable.'''
    exp_term = np.exp(beta*(e-mu))
    #print(np.isinf(exp_term).sum())
    return exp_term / ( exp_term + 1)

def transfer_rate(e_grid,mu_i,mu_f,gamma_i,gamma_f,e_i,e_f,beta,kappa,w,return_min=False):
    if return_min: mins=np.zeros(4,float)
    lor_i = lor(e_grid,gamma_i,e_i) 
    #if np.any(np.isnan(lor_i)):
    #    print('lor_i is fucked')
    lor_f = lor(e_grid+w,gamma_f,e_f)
    #if np.any(np.isnan(lor_f)):
    #    print('lor_f is fucked')

    fd_i = fermi_dirac(e_grid,mu_i,beta)
    #if np.any(np.isnan(fd_i)):
    #    print('fd_i is fucked')
    #cfd_f = c_fermi_dirac(e_grid+w,mu_f,beta)
    cfd_f = 1 - fermi_dirac(e_grid+w,mu_f,beta)
    if return_min:
        mins[0] = np.min(lor_i)
        mins[1] = np.min(lor_f)
        mins[2] = np.min(fd_i)
        mins[3] = np.min(cfd_f)

        return kappa*kappa*simpson(lor_i*lor_f*fd_i*cfd_f, e_grid), mins
    #if np.any(np.isnan(cfd_f)):
    #    print('cfd_f is fucked')

    else:
        return kappa*kappa*simpson(lor_i*lor_f*fd_i*cfd_f, e_grid)


# ****** MAIN ******
if __name__ == '__main__':

    # Define static params

    kB = 8.617e-5 # eV/K

    e_d = -0.2 # LUMO energy [eV]
    e_a = 0.4 # LUMO+1 energy [eV]

    gamL = 0.2 #e
    gamR = 0.2 #eV
    gam_phonon = 0.1

    muL_grid = np.linspace(-0.5,1,200)

    kappa = 0.1
    w0_grid = np.ones(2) * 0.05

    beta_grid = np.ones(3) * 200

    #kk, ww, bb = np.meshgrid(kappa_grid, w0_grid, beta_grid)

    e_grid = np.linspace(-5,5,20000)

    k_LR_01 = np.zeros((muL_grid.shape[0], w0_grid.shape[0], beta_grid.shape[0]))
    k_RL_01 = np.zeros((muL_grid.shape[0], w0_grid.shape[0], beta_grid.shape[0]))
    k_LR_10 = np.zeros((muL_grid.shape[0], w0_grid.shape[0], beta_grid.shape[0]))
    k_RL_10 = np.zeros((muL_grid.shape[0], w0_grid.shape[0], beta_grid.shape[0]))
    mins = np.zeros(4)
    #mins = np.zeros((kappa_grid.shape[0], w0_grid.shape[0], beta_grid.shape[0],4))
    kk = kappa

    for i, mm in enumerate(muL_grid):
        print(i)
        for j, ww in enumerate(w0_grid):
            for k, bb in enumerate(beta_grid):

                tm = np.zeros(4)

                k_LR_01[i,j,k], tm[:] = transfer_rate(e_grid,mm,-mm,gamL,gamR,e_d+2*mm,e_a-2*mm,bb,kk,-ww,1)
                k_RL_01[i,j,k] = transfer_rate(e_grid,-mm,mm,gamR,gamL,e_a-2*mm,e_d+2*mm,bb,kk,-ww)


                k_LR_10[i,j,k] = transfer_rate(e_grid,mm,-mm,gamL,gamR,e_d+2*mm,e_a-2*mm,bb,kk,ww)
                #k_RL_10[i,j,k], mins[i,j,k,:] = transfer_rate(e_grid,muR,muL,e_a,e_d,gamR,gamL,bb,kk,-ww,1)
                k_RL_10[i,j,k] = transfer_rate(e_grid,-mm,mm,gamR,gamL,e_a-2*mm,e_d+2*mm,bb,kk,ww)
                #k_RL_10[i,j,k], tm[:] = transfer_rate(e_grid,muR,muL,e_a,e_d,gamR,gamL,bb,kk,-ww,1)
                diffs = (tm - mins < 0)
                mins[diffs] = tm[diffs]

    k_01 = k_LR_01 + k_RL_01
    k_10 = k_LR_10 + k_RL_10

    # non-dissipative mode
    p1 = k_01 / (k_01 + k_10)
    p0 = 1 - p1

    dmu = 2*muL_grid

    plt.plot(dmu,p1[:,0,0],'b-',lw=1.0,label='$p_1$')
    plt.plot(dmu,p0[:,0,0],'r-',lw=1.0,label='$p_0$')
    plt.xlabel('$\Delta\mu$ [eV]')
    plt.ylabel('population')
    plt.legend()
    plt.show()

    np.save('p1_non-dis.npy',p1)
 
    #dissipative mode

    gam_phonon = np.array([0.001,0.1])
    lws = np.array([0.8,1.7])
    ww, bb = np.meshgrid(w0_grid, beta_grid, indexing='ij',sparse=True)
    nph = bose_einstein(ww, bb)
    print(nph.shape)

    
    print(k_01.shape)
    print(k_10.shape)
    
    for g, l in zip(gam_phonon,lws):
        p1 = (k_01 + g * nph[None, :, :]) / (k_01 + k_10 + g * (2* nph[None,:,:] + 1))
        p0 = 1 - p1


        plt.plot(dmu,p1[:,0,0],'b-',lw=l,label='$\Gamma_{ph} = %4.3f\,$eV'%g)
        plt.plot(dmu,p0[:,0,0],'r-',lw=l)
        plt.xlabel('$\Delta\mu$ [eV]')
        plt.ylabel('population')
    plt.legend()
    plt.show()

    np.save('p1_dis.npy',p1)
