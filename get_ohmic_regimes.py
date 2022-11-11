#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tensor_linregress import tensor_linregress
from param_parser import ParameterParser
import plt_utils


# ************* MAIN *************

if __name__ == '__main__':

    plt_utils.setup_tex()
    rcParams['image.aspect'] = 'auto'

    kB = 8.617e-5
    #temp_grid = np.linspace(40,4000,200)

    #npydir = 'MAC_full_misaligned'
    #param_file = 'original_params.json'

    npydir = 'MAC_aligned_focused_smaller_dmu_no_shift'
    param_file = 'aligned_focused_smaller_dmu.json'

    pp = ParameterParser(param_file)

    kappa_grid, w0_grid, muL_grid, temp_grid, e_grid = \
    pp.load_grids(plist=['kappa_grid', 'frequency_grid','muL_grid',\
        'temperature_grid', 'energy_grid'])

    dmu_grid = muL_grid * 2
    print(dmu_grid[0])

    I = np.moveaxis(np.load('%s/current_dis.npy'%npydir),2,-1) #move dmu axis to last position to use tensor_linregress painlessly
    # axes of I are now: (w0, kappa, beta, dmu)
    print(I.shape)

    print(np.min(I[:,:,:,0]))
    print(np.max(I[:,:,:,0]))

    I_zero_bias = I[:,:,:,0]
    plt_utils.histogram(I_zero_bias, nbins=200,xlabel='$I(\Delta\mu = 0)$')
    print('Mean 0 bias current = ', np.mean(I_zero_bias))

    


    *_, rvals = tensor_linregress(dmu_grid, I)
    print(np.max(rvals**2))

    most_ohmic_inds = np.unravel_index((rvals**2).argmax(),rvals.shape)
    #print(most_ohmic_inds)

    plt.imshow(I_zero_bias[0].T,origin='lower')
    plt.suptitle('Zero bias current for $\omega_0 = %5.3f\,$eV'%w0_grid[0])
    plt.xlabel('$\kappa$ indices')
    plt.ylabel('$T$ indices')
    plt.colorbar()
    plt.show()

    plt.imshow(I_zero_bias[:,-1,:].T,origin='lower')
    plt.suptitle('Zero bias current for $\kappa = %4.2f\,$eV'%kappa_grid[-1])
    plt.xlabel('$\omega_0$ indices')
    plt.ylabel('$T$ indices')
    plt.colorbar()
    plt.show()

    plt.imshow(I_zero_bias[:,:,-1].T,origin='lower')
    plt.suptitle('Zero bias current for $T = %5.1f\,$K'%temp_grid[-1])
    plt.ylabel('$\kappa$ indices')
    plt.xlabel('$\omega_0$ indices')
    plt.colorbar()
    plt.show()

    plt.imshow(I_zero_bias[:,:,0].T,origin='lower')
    plt.suptitle('Zero bias current for $T = %5.1f\,$K'%temp_grid[0])
    plt.ylabel('$\kappa$ indices')
    plt.xlabel('$\omega_0$ indices')
    plt.colorbar()
    plt.show()

    I_finite_dmu = I[:,:,:,1:]
    plt_utils.histogram(I_finite_dmu,nbins=200,xlabel='Current',show=False,normalised=True,\
        plt_kwargs={'alpha':0.6, 'color': 'b','label':'$\Delta\mu > 0\,$eV'})
    plt_utils.histogram(I_zero_bias,nbins=200,xlabel='Current',show=False,normalised=True,\
        plt_kwargs={'alpha':0.6, 'color': 'r','label':'$\Delta\mu = 0\,$eV'})
    plt.legend()
    plt.show()


    plt.plot(dmu_grid,I[most_ohmic_inds],'r-',lw=0.8)
    plt.show()

    #print(dmu_grid)

    # ohm_bools will be True where I ~ V for all values of V for specific values of (w0,kappa,beta)
    ohm_bools = (rvals**2 > 0.985)
    #print('Ohmic for all V for %d of %d possible parameter combinations.'%(np.sum(ohm_bools), ohm_bools.size))

    # Obtain (w0, kappa) values for which ohmic regime is maintained at all temperatures
    ohm_bools_allT = np.all(ohm_bools, axis=2)
    print('Ohmic for all V for %d of %d possible parameter combinations.'%(np.sum(ohm_bools_allT), ohm_bools_allT.size))
    plt.imshow(ohm_bools_allT.T,origin='lower', extent=[*w0_grid[[0,-1]], *kappa_grid[[0,-1]]])
    plt.xlabel('$\omega_0$ [eV]')
    plt.ylabel('$\kappa$ [eV]')
    plt.show()


    # ******** IMPORTANT PART STARTS HERE ********

    # Now we do this for real; for which values of V do we have an ohmic regime for all values of all other parameters?
    # We focus only on postive bias (ie V > 0).

    pos_inds = (dmu_grid > 0).nonzero()[0]
    dmu_grid = dmu_grid[pos_inds]
    I = I[:,:,:,pos_inds]

    all_ohmic = True
    dmu_ind = 2

    while all_ohmic and (dmu_ind < dmu_grid.size):
        print('\n')
        *_, rvals = tensor_linregress(dmu_grid[:dmu_ind],I[:,:,:,:dmu_ind])
        ohm_bools = (rvals**2 >= 0.99)
        print('Worst linear fit rval = ', np.min(rvals))
        minds = np.unravel_index(np.argmin(rvals), rvals.shape)
        print('Worst fit inds: ', minds)
        maxinds = np.unravel_index(np.argmax(rvals), rvals.shape)
        print('Best linear fit rval = ', np.max(rvals))
        print('Best fit inds: ', maxinds)
        print('Number of ohmic realisations for dmu <= %5.3f = '%dmu_grid[dmu_ind-1], np.sum(ohm_bools))

        all_ohmic = np.all(ohm_bools)
        dmu_ind += 1

    print(dmu_ind)
    print('Max dmu value = ', dmu_grid[dmu_ind-2])

    print(dmu_grid[:dmu_ind-1])

# ************************************** #
# Now, we conduct a different test for the ohmic regime.
# Going off of the assumption that the 0-bias voltage is null, the conductance is simply:
# I()

G_values = I_finite_dmu/dmu[1:]
deviations = 

