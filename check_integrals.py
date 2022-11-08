#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import plt_utils
from param_parser import ParameterParser
from full_param_scan import lor
from scipy.integrate import simpson


kB = 8.617e-5 # eV/K
param_file = 'aligned_focused_egrid.json'
outdir = 'MAC_aligned_focused_egrid'

pp = ParameterParser(param_file)

e_d, e_a, gamL, gamR, gam_phonon = pp.load_intrinsic()

print('Donor energy = {:5.6f} eV'.format(e_d))
print('Acceptor energy = {:5.6f} eV'.format(e_a))
print('gamL = {:5.6f} eV'.format(gamL))
print('gamR = {:5.6f} eV'.format(gamR))

kappa_grid, w0_grid, muL_grid, temp_grid, e_grid = \
    pp.load_grids(plist=['kappa_grid', 'frequency_grid','muL_grid',\
        'temperature_grid', 'energy_grid'])

beta_grid = 1.0 / (kB * temp_grid)

lorL = lor(e_grid,gamL,e_d)
lorR = lor(e_grid,gamR,e_a)

plt.plot(e_grid,lorL*lorR,'k-',lw=0.8,label='$J_L\cdot J_R$')
plt.plot(e_grid,lorL,'r--',lw=0.8,label='$J_L$')
plt.plot(e_grid,lorR,'b--',lw=0.8,label='$J_R$')
plt.legend()
plt.show()

lpeak_ind = np.argmax(lorL)
print(lorL[lpeak_ind])
print(lorR[lpeak_ind])
print(lorL[lpeak_ind]*lorR[lpeak_ind])

integrand = lorL[:,None]*lor(e_grid[:,None]-w0_grid,gamR,e_a)
integral = simpson(integrand,e_grid,axis=0)
print(integral.shape)
plt.plot(w0_grid,integral)
plt.show()

print(integral.argmax())

plt.plot(e_grid,lorL*lor(e_grid-w0_grid[integral.argmax()],gamR,e_d),'r-',label='$J_(\epsilon)\cdot J_R(\epsilon-\omega_0)$')
plt.plot(e_grid,lorL*lorR,'b--',label='$J_(\epsilon)\cdot J_R(\epsilon)$',lw=0.8)
plt.axvline(x=e_d,c='k',ls='--',lw=0.6)
plt.axvline(x=e_a-w0_grid[integral.argmax()],c='k',ls='--',lw=0.6)
plt.axvline(x=e_d,c='tab:gray',ls='--',lw=0.6)
plt.legend()
plt.show()