#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import plt_utils


w0_grid = np.linspace(0.001,0.1,41)
mott_params, rch_params = [np.load('fit_params_%s.npy'%s) for s in ['mott', 'rch']]


plt_utils.setup_tex()

# Plot r^2 values for both fits as a function of w0
plt.plot(w0_grid,mott_params[2,:]**2,lw=0.8,label='Mott')
plt.plot(w0_grid,rch_params[2,:]**2,lw=0.8,label='RCH')
plt.xlabel('$\omega_0$ [eV]')
plt.ylabel('$r^2$')
plt.legend()
plt.show()

#Plot values of the linear fit slopes
plt.plot(w0_grid,mott_params[0,:]**2,lw=0.8,label='Mott')
plt.plot(w0_grid,rch_params[0,:]**2,lw=0.8,label='RCH')
plt.xlabel('$\omega_0$ [eV]')
plt.ylabel('$Slope of linear fit$')
plt.legend()
plt.show()


