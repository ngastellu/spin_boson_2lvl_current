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
temp_grid = np.linspace(40,400,200)
beta_grid = 1.0 / (kB * temp_grid)
dmu = 2.0*np.linspace(-0.5,1,51)
kappa_grid = np.linspace(0.01,0.1,11)
w0_grid = np.linspace(0.01,1.0,21)

#Ind = np.load('40-4000K/full_current_non-dis.npy')
Ind = np.load('MAC_current_non-dis.npy')
Gnd = Ind / dmu[None,None,:,None]

#Id = np.load('40-4000K/full_current_dis.npy')
Id = np.load('MAC_current_dis.npy')
Gd = Id / dmu[None,None,:,None]

print(np.all(Id == Ind))

logGnd = np.log(Gnd)
logGd = np.log(Gd)

#logI = np.log(np.abs(I))

plt_utils.setup_tex()

plt.plot(np.log(beta_grid),logGnd[-1,-1,0,:],label='non-dissipative')
plt.plot(np.log(beta_grid),logGd[-1,-1,0,:],label='dissipative')
plt.xlabel('ln$\,\\beta$')
plt.ylabel('ln$\,G$')
plt.suptitle('Conductance vs temperature plot.')
plt.legend()
plt.show()

#beta_arr = np.power(beta_grid[None,:], 1.0/np.arange(1,5)[:,None])

dmu_ind1 = 0
dmu_ind2 = 33
kappa_ind1 = 0
kappa_ind2 = -1
rtol = 0.95

x = np.log(beta_grid)
a1, b1, r1 = tensor_linregress(x,logGd[:,:,dmu_ind1,:])
a2, b2, r2 = tensor_linregress(x,logGd[:,:,dmu_ind2,:])
a3, b3, r3 = tensor_linregress(x,logGd[:,kappa_ind1,:,:])
a4, b4, r4 = tensor_linregress(x,logGd[:,kappa_ind2,:,:])

plt.imshow((r1**2).T,origin='lower',extent=[*w0_grid[[0,-1]], *kappa_grid[[0,-1]]])
plt.xlabel('$\omega_0$ [eV]')
plt.ylabel('$\kappa$ [eV]')
plt.suptitle('Quality of power law fit of $G_{nd}$ vs. $\\beta$ assuming $\Delta\mu = %f$ eV.'%(dmu[dmu_ind1]))
plt.colorbar()
plt.show()

plt.imshow(((r1**2) > rtol).T,origin='lower',extent=[*w0_grid[[0,-1]], *kappa_grid[[0,-1]]])
plt.xlabel('$\omega_0$ [eV]')
plt.ylabel('$\kappa$ [eV]')
plt.suptitle('Quality of power law fit of $G_{nd}$ vs. $\\beta$ assuming $\Delta\mu = %f$ eV\nShowing only fits with $r^2 > %f$.'%(dmu[dmu_ind1],rtol))
plt.colorbar()
plt.show()

plt.imshow((r2**2).T,origin='lower',extent=[*w0_grid[[0,-1]], *kappa_grid[[0,-1]]])
plt.xlabel('$\omega_0$ [eV]')
plt.ylabel('$\kappa$ [eV]')
plt.suptitle('Quality of power law fit of $G_{nd}$ vs. $\\beta$ assuming $\Delta\mu = %f$ eV.'%(dmu[dmu_ind2]))
plt.colorbar()
plt.show()

plt.imshow(((r2**2) > rtol).T,origin='lower',extent=[*w0_grid[[0,-1]], *kappa_grid[[0,-1]]])
plt.xlabel('$\omega_0$ [eV]')
plt.ylabel('$\kappa$ [eV]')
plt.suptitle('Quality of power law fit of $G_{nd}$ vs. $\\beta$ assuming $\Delta\mu = %f$ eV.\nShowing only fits with $r^2 > %f$.'%(dmu[dmu_ind2],rtol))
plt.colorbar()
plt.show()

plt.imshow((r3**2).T,origin='lower',extent=[*w0_grid[[0,-1]], *dmu[[0,-1]]])
plt.xlabel('$\omega_0$ [eV]')
plt.ylabel('$\Delta\mu$ [eV]')
plt.suptitle('Quality of power law fit of $G_{nd}$ vs. $\\beta$ assuming $\kappa = %f$ eV.'%(kappa_grid[kappa_ind1]))
plt.colorbar()
plt.show()

plt.imshow(((r3**2) > rtol).T,origin='lower',extent=[*w0_grid[[0,-1]], *dmu[[0,-1]]])
plt.xlabel('$\omega_0$ [eV]')
plt.ylabel('$\Delta\mu$ [eV]')
plt.suptitle('Quality of power law fit of $G_{nd}$ vs. $\\beta$ assuming $\kappa = %f$ eV.\nShowing only fits with $r^2 > %f$.'%(kappa_grid[kappa_ind1], rtol))
plt.colorbar()
plt.show()

plt.imshow((r4**2).T,origin='lower',extent=[*w0_grid[[0,-1]], *dmu[[0,-1]]])
plt.xlabel('$\omega_0$ [eV]')
plt.ylabel('$\Delta\mu$ [eV]')
plt.suptitle('Quality of power law fit of $G_{nd}$ vs. $\\beta$ assuming $\kappa = %f$ eV.'%(kappa_grid[kappa_ind2]))
plt.colorbar()
plt.show()

plt.imshow(((r4**2) > rtol).T,origin='lower',extent=[*w0_grid[[0,-1]], *dmu[[0,-1]]])
plt.xlabel('$\omega_0$ [eV]')
plt.ylabel('$\Delta\mu$ [eV]')
plt.suptitle('Quality of power law fit of $G_{nd}$ vs. $\\beta$ assuming $\kappa = %f$ eV.\nShowing only fits with $r^2 > %f$.'%(kappa_grid[kappa_ind2],rtol))
plt.colorbar()
plt.show()


for r in [r1,r2,r3,r4]:
    print(np.sum(r**2    > rtol))
