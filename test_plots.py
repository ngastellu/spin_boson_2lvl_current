#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, rcParams
from param_scan import *

def get_cm(vals, cmap, max_val=0.7):
    if not isinstance(vals, np.ndarray):
        vals = np.array(vals)
    sorted_vals = np.sort(vals)
    delta = sorted_vals[-1] - sorted_vals[0]
    x = max_val * (vals - sorted_vals[0]) / delta
    return cmap(x)



rcParams['text.usetex'] = True

kB = 8.617e-5 # eV/K

e_d = -0.2 # LUMO energy [eV]
e_a = 0.4 # LUMO+1 energy [eV]

gamL = 0.2 #e
gamR = 0.2 #eV
gam_phonon = 0.1

muL = 0.4
muR = -muL

kappa_grid = np.linspace(0.01,1.0,10)
w0_grid = np.linspace(0.01,1.0,10)
temp_grid = np.linspace(40,400,200)

beta = 1.0 / (kB * temp_grid)

#kk, ww, bb = np.meshgrid(kappa_grid, w0_grid, beta_grid)

e_grid = np.linspace(-4,4,10000)
#w_grid =  np.linspace(0.01,1.0,10000)

lor_L = lor(e_grid, gamL, e_d)
lor_R0 = lor(e_grid-w0_grid[0], gamR, e_a)
lor_R9 = lor(e_grid-w0_grid[9], gamR, e_a)

fe_L0 = fermi_dirac(e_grid,muL,beta[0])
fe_L9 = fermi_dirac(e_grid,muL,beta[9])

fe_R00 = fermi_dirac(e_grid-w0_grid[0],muR,beta[0])
fe_R09 = fermi_dirac(e_grid-w0_grid[0],muR,beta[9])
fe_R90 = fermi_dirac(e_grid-w0_grid[9],muR,beta[0])
fe_R99 = fermi_dirac(e_grid-w0_grid[9],muR,beta[9])

integrand00 = lor_L*lor_R0*fe_L0*fe_R00
integrand09 = lor_L*lor_R0*fe_L9*fe_R09
integrand90 = lor_L*lor_R9*fe_L0*fe_R90
integrand99 = lor_L*lor_R9*fe_L9*fe_R99

clrs = get_cm(temp_grid[[0,9]],cm.hot,max_val=0.6)

temp = temp_grid

plt.plot(e_grid,integrand00,c=clrs[0],ls='-',label='$T = %4.1f$ K, $\omega_0 = %2.1f$ eV'%(temp[0],w0_grid[0]))
plt.plot(e_grid,integrand09,c=clrs[1],ls='-',label='$T = %4.1f$ K, $\omega_0 = %2.1f$ eV'%(temp[9],w0_grid[0]))
plt.plot(e_grid,integrand90,c=clrs[0],ls='--',label='$T = %4.1f$ K, $\omega_0 = %2.1f$ eV'%(temp[0],w0_grid[9]))
plt.plot(e_grid,integrand99,c=clrs[1],ls='--',label='$T = %4.1f$ K, $\omega_0 = %2.1f$ eV'%(temp[9],w0_grid[9]))

plt.legend()
plt.show()
