#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import plt_utils

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


kB = 8.617e-5
#temp_grid = np.linspace(40,4000,200)
temp_grid = np.linspace(40,400,200)
beta_grid = 1.0 / (kB * temp_grid)
dmu = 2.0*np.linspace(-0.5,1,51)
kappa_grid = np.linspace(0.1,1.0,11)
w0_grid = np.linspace(0.01,1.0,21)

#Ind = np.load('40-4000K/full_current_non-dis.npy')
Ind = np.load('full_current_non-dis.npy')
Gnd = Ind / dmu[None,None,:,None]

#Id = np.load('40-4000K/full_current_dis.npy')
Id = np.load('full_current_dis.npy')
Gd = Id / dmu[None,None,:,None]

print(np.all(Id == Ind))

logGnd = np.log(Gnd)
logGd = np.log(Gd)

#logI = np.log(np.abs(I))

plt_utils.setup_tex()

for d in range(4): # number of spatial dimensions (Mott's law)
    plt.plot(beta_grid**(1/(d+1)),logGnd[-1,-1,0,:],label='non-dissipative')
    plt.plot(beta_grid**(1/(d+1)),logGd[-1,-1,0,:],label='dissipative')
    plt.xlabel('$\\beta^{1/%d}$'%(d+1))
    plt.ylabel('ln$\,G$')
    plt.suptitle('Current vs temperature plot assuming $d=%d$ dimensions.'%d)
    plt.legend()
    plt.show()

#beta_arr = np.power(beta_grid[None,:], 1.0/np.arange(1,5)[:,None])

dmu_ind1 = 0
dmu_ind2 = 33
kappa_ind1 = 0
kappa_ind2 = -1
rtol = 0.95

for d in range(1,4):
    x = beta_grid**(1/(d+1))
    a1, b1, r1 = tensor_linregress(x,logGd[:,:,dmu_ind1,:])
    a2, b2, r2 = tensor_linregress(x,logGd[:,:,dmu_ind2,:])
    a3, b3, r3 = tensor_linregress(x,logGd[:,kappa_ind1,:,:])
    a4, b4, r4 = tensor_linregress(x,logGd[:,kappa_ind2,:,:])

    plt.imshow((r1**2).T,origin='lower',extent=[*w0_grid[[0,-1]], *kappa_grid[[0,-1]]])
    plt.xlabel('$\omega_0$ [eV]')
    plt.ylabel('$\kappa$ [eV]')
    plt.suptitle('Quality of exponential fit of $G_{nd}$ vs. $\\beta^{1/%d}$ assuming $\Delta\mu = %f$ eV.'%(d+1, dmu[dmu_ind1]))
    plt.colorbar()
    plt.show()

    plt.imshow(((r1**2) > rtol).T,origin='lower',extent=[*w0_grid[[0,-1]], *kappa_grid[[0,-1]]])
    plt.xlabel('$\omega_0$ [eV]')
    plt.ylabel('$\kappa$ [eV]')
    plt.suptitle('Quality of exponential fit of $G_{nd}$ vs. $\\beta^{1/%d}$ assuming $\Delta\mu = %f$ eV\nShowing only fits with $r^2 > %f$.'%(d+1, dmu[dmu_ind1],rtol))
    plt.colorbar()
    plt.show()

    plt.imshow((r2**2).T,origin='lower',extent=[*w0_grid[[0,-1]], *kappa_grid[[0,-1]]])
    plt.xlabel('$\omega_0$ [eV]')
    plt.ylabel('$\kappa$ [eV]')
    plt.suptitle('Quality of exponential fit of $G_{nd}$ vs. $\\beta^{1/%d}$ assuming $\Delta\mu = %f$ eV.'%(d+1, dmu[dmu_ind2]))
    plt.colorbar()
    plt.show()

    plt.imshow(((r2**2) > rtol).T,origin='lower',extent=[*w0_grid[[0,-1]], *kappa_grid[[0,-1]]])
    plt.xlabel('$\omega_0$ [eV]')
    plt.ylabel('$\kappa$ [eV]')
    plt.suptitle('Quality of exponential fit of $G_{nd}$ vs. $\\beta^{1/%d}$ assuming $\Delta\mu = %f$ eV.\nShowing only fits with $r^2 > %f$.'%(d+1, dmu[dmu_ind2],rtol))
    plt.colorbar()
    plt.show()

    plt.imshow((r3**2).T,origin='lower',extent=[*w0_grid[[0,-1]], *dmu[[0,-1]]])
    plt.xlabel('$\omega_0$ [eV]')
    plt.ylabel('$\Delta\mu$ [eV]')
    plt.suptitle('Quality of exponential fit of $G_{nd}$ vs. $\\beta^{1/%d}$ assuming $\kappa = %f$ eV.'%(d+1, kappa_grid[kappa_ind1]))
    plt.colorbar()
    plt.show()

    plt.imshow(((r3**2) > rtol).T,origin='lower',extent=[*w0_grid[[0,-1]], *dmu[[0,-1]]])
    plt.xlabel('$\omega_0$ [eV]')
    plt.ylabel('$\Delta\mu$ [eV]')
    plt.suptitle('Quality of exponential fit of $G_{nd}$ vs. $\\beta^{1/%d}$ assuming $\kappa = %f$ eV.\nShowing only fits with $r^2 > %f$.'%(d+1, kappa_grid[kappa_ind1], rtol))
    plt.colorbar()
    plt.show()

    plt.imshow((r4**2).T,origin='lower',extent=[*w0_grid[[0,-1]], *dmu[[0,-1]]])
    plt.xlabel('$\omega_0$ [eV]')
    plt.ylabel('$\Delta\mu$ [eV]')
    plt.suptitle('Quality of exponential fit of $G_{nd}$ vs. $\\beta^{1/%d}$ assuming $\kappa = %f$ eV.'%(d+1, kappa_grid[kappa_ind2]))
    plt.colorbar()
    plt.show()

    plt.imshow(((r4**2) > rtol).T,origin='lower',extent=[*w0_grid[[0,-1]], *dmu[[0,-1]]])
    plt.xlabel('$\omega_0$ [eV]')
    plt.ylabel('$\Delta\mu$ [eV]')
    plt.suptitle('Quality of exponential fit of $G_{nd}$ vs. $\\beta^{1/%d}$ assuming $\kappa = %f$ eV.\nShowing only fits with $r^2 > %f$.'%(d+1, kappa_grid[kappa_ind2],rtol))
    plt.colorbar()
    plt.show()
