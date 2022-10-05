#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import plt_utils
from matplotlib import rcParams

def tensor_linregress(x,y,yaxis=-1):
    xavg = np.mean(x)
    yavg = np.mean(y,axis=yaxis)
    vary = np.var(y,axis=yaxis)

    if yaxis == -1:
        xy_avg = np.mean(x*y,axis=-1)
        varx = np.var(x)
        #xshape = eval('(' +','.join(['None'*y.ndim-1]) + ',' + str(x.shape[0]) ')') 
        cov_xy = xy_avg - xavg*yavg
        slope = cov_xy / varx
        intercept = yavg - slope* xavg
        r = cov_xy / np.sqrt(varx * vary)
        return slope, intercept, r

    else:
        print('[tensor_linregress] ERROR: yaxis != -1 not yet implemented! Returning 0.')
        return 0


# ********** MAIN ***********

rcParams['font.size'] = 25
rcParams['image.aspect'] = 'auto'

kB = 8.617e-5
#temp_grid = np.linspace(40,4000,200)
temp_grid = np.linspace(40,400,400)
beta_grid = 1.0 / (kB * temp_grid)
dmu = 0.08
kappa_grid = np.linspace(0.001,0.1,21)
w0_grid = np.linspace(0.001,0.1,41)


Id = np.load('MAC_current_dis_dmu0.08.npy')
Gd = Id / dmu

logGd = np.log(Gd)

#logI = np.log(np.abs(I))

plt_utils.setup_tex()


#beta_arr = np.power(beta_grid[None,:], 1.0/np.arange(1,5)[:,None])

rtol = 0.95

for d in [1]:
    x = beta_grid**(1/(d+1))
    a1, b1, r1 = tensor_linregress(x,logGd)

    plt.imshow((r1**2).T,origin='lower',extent=[*w0_grid[[0,-1]], *kappa_grid[[0,-1]]])
    plt.xlabel('$\omega_0$ [eV]')
    plt.ylabel('$\kappa$ [eV]')
    plt.suptitle('Quality of exponential fit of $G_{nd}$ vs. $\\beta^{1/%d}$ assuming $\Delta\mu = %f$ eV.'%(d+1, dmu))
    plt.colorbar()
    plt.show()

    plt.imshow(((r1**2) > rtol).T,origin='lower',extent=[*w0_grid[[0,-1]], *kappa_grid[[0,-1]]])
    plt.xlabel('$\omega_0$ [eV]')
    plt.ylabel('$\kappa$ [eV]')
    plt.suptitle('Quality of exponential fit of $G_{nd}$ vs. $\\beta^{1/%d}$ assuming $\Delta\mu = %f$ eV\nShowing only fits with $r^2 > %f$.'%(d+1, dmu,rtol))
    plt.colorbar()
    plt.show()

print(np.max(np.abs(r1 - r1[:,0][:,None])))

plt.plot(w0_grid, a1, 'r-', lw=0.8)
plt.xlabel('$\omega_0$ [eV]')
plt.ylabel('Slope of linear fit')
plt.show()

np.save('fit_params_mott.npy',np.vstack((a1[:,0],b1[:,0],r1[:,0])))