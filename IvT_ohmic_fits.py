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


if __name__ == '__main__':

    w0_grid1 = np.linspace(0.001,0.1,41)
    w0_grid2 = np.linspace(0.1,0.3,79)
    w0_grid = np.hstack((w0_grid1,w0_grid2))
    temp_grid = np.linspace(40,400,400)

    dmu = 0.08
    I1 = np.load('MAC_current_dis_dmu0.08.npy')
    I2 = np.load('MAC_current_dis_dmu0.08_hi-w0.npy')
    print(I1.shape)
    print(I2.shape)
    #I = np.hstack((I1,I2))
    G = I1/dmu
    
    # Do fits; first corresponds to 1D Mott law, second fit corresponds to RCH
    xs = [np.sqrt(1/temp_grid), np.log(temp_grid)]
    params = np.array([tensor_linregress(x, np.log(G)) for x in xs])

    print(params.shape)

    r2 = params[:,2,:,0]**2
    
    plt_utils.setup_tex()
    rcParams['font.size'] = 20

    plt.plot(w0_grid1, r2[0],lw=0.8,label='Mott')
    plt.plot(w0_grid1,r2[1],lw=0.8,label='RCH')
    plt.xlabel('$\omega_0$ [eV]')
    plt.ylabel('$r^2$')
    plt.legend()
    plt.show()

    # Plot exponent of power law fit vs omega_0
    plt.plot(w0_grid1,params[1][0,:],lw=0.8)
    plt.xlabel('$\omega_0$ [eV]')
    plt.ylabel('RCH power law exponent')
    plt.show()

    min_w0_inds = np.argmin(r2,axis=1) #both are equal to 19

    fig, axs = plt.subplots(1,2,sharey=True)

    for ax, x, i, s in zip(axs, xs, min_w0_inds, ['$\sqrt{1/T}$', 'ln($T$)']):
        ax.plot(x, np.log(G[i,0,:]), 'r-', lw=0.8)
        ax.set_xlabel(s)
        ax.set_ylabel('ln($G$)')
    
    plt.suptitle('Conductance v. temperature plots for $\omega_0 = %3.4f\,$eV. These should be linear for Mott (left) and RCH (right).'%w0_grid1[min_w0_inds[0]])
    plt.show()



    

