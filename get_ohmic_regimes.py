#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
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


# ************* MAIN *************

if __name__ == '__main__':

    plt_utils.setup_tex()
    rcParams['image.aspect'] = 'auto'

    kB = 8.617e-5
    #temp_grid = np.linspace(40,4000,200)
    temp_grid = np.linspace(40,400,200)
    beta_grid = 1.0 / (kB * temp_grid)
    dmu_grid = 2.0*np.linspace(-0.5,1,51)
    kappa_grid = np.linspace(0.01,0.1,11)
    w0_grid = np.linspace(0.01,1.0,21)

    I = np.moveaxis(np.load('MAC_current_dis_dmu0.08_aligned.npy'),2,-1) #move dmu axis to last position to use tensor_linregress painlessly
    # axes of I are now: (w0, kappa, beta, dmu)

    *_, rvals = tensor_linregress(dmu_grid, I)

    # ohm_bools will be True where I ~ V for all values of V for specific values of (w0,kappa,beta)
    ohm_bools = (rvals**2 > 0.985)
    print('Ohmic for all V for %d of %d possible parameter combinations.'%(np.sum(ohm_bools), ohm_bools.size))

    # Obtain (w0, kappa) values for which ohmic regime is maintained at all temperatures
    ohm_bools_allT = np.all(ohm_bools, axis=2)
    print('Ohmic for all V for %d of %d possible parameter combinations.'%(np.sum(ohm_bools_allT), ohm_bools_allT.size))
    plt.imshow(ohm_bools_allT.T,origin='lower', extent=[*w0_grid[[0,-1]], *kappa_grid[[0,-1]]])
    plt.xlabel('$\omega_0$ [eV]')
    plt.ylabel('$\kappa$ [eV]')
    plt.show()

    # Now we do this for real; for which values of V do we have an ohmic regime for all values of all other parameters?
    # We focus only on postive bias (ie V > 0).

    pos_inds = (dmu_grid > 0).nonzero()[0]
    dmu_grid = dmu_grid[pos_inds]
    I = I[:,:,:,pos_inds]

    all_ohmic = True
    dmu_ind = 2

    while all_ohmic and (dmu_ind < dmu_grid.size):

        *_, rvals = tensor_linregress(dmu_grid[:dmu_ind],I[:,:,:,:dmu_ind])
        ohm_bools = (rvals >= 0.99)
        print('Number of ohmic realisations for dmu <= %5.3f = '%dmu_grid[dmu_ind-1], np.sum(ohm_bools))

        all_ohmic = np.all(ohm_bools)
        dmu_ind += 1

    print(dmu_ind)
    print('Max dmu value = ', dmu_grid[dmu_ind-2])

    print(dmu_grid[:3])
