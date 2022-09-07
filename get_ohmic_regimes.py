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


plt_utils.setup_tex()
rcParams['image.aspect'] = 'auto'

kB = 8.617e-5
#temp_grid = np.linspace(40,4000,200)
temp_grid = np.linspace(40,400,200)
beta_grid = 1.0 / (kB * temp_grid)
dmu = 2.0*np.linspace(-0.5,1,51)
kappa_grid = np.linspace(0.01,0.1,11)
w0_grid = np.linspace(0.01,1.0,21)

I = np.moveaxis(np.load('MAC_current_dis.npy'),2,-1) #move dmu axis to last position to use tensor_linregress painlessly

*_, rvals = tensor_linregress(dmu_grid, I)
