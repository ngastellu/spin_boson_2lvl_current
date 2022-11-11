#!/usr/bin/env python

import numpy as np


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