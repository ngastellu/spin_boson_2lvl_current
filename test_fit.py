#!/usr/bin/env python

from time import perf_counter
import numpy as np
from numpy.random import default_rng
from scipy.optimize import curve_fit
from scipy.stats import linregress
import matplotlib.pyplot as plt
import plt_utils

def gen_data(x, func, func_params,noise=0, num_outliers=0,seed=None):

    rng = default_rng(seed)

    y = func(x, *func_params)

    if x.ndim == 1: #only one set of parameters to try (i.e. no meshgrids)
        error = noise * rng.standard_normal(x.size)
        outlier_inds = rng.integers(x.size, size=num_outliers)
        error[outlier_inds] *= 10

    else:
        error = 0

    return y + error


def cool_f(x,a,b,c):
    return np.exp(-a*x) * b + c

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


# ******* MAIN *******


plt_utils.setup_tex()

x = np.linspace(-1,1,200)
func_params = [2,4,3]

y = gen_data(x,cool_f, func_params, 0.1, num_outliers = 10)

print(y.shape)

popt, pcov = curve_fit(cool_f, x, y)

a_fit, b_fit, c_fit = popt

logy = np.log(y)

a_lfit, b_lfit, rval, *_ = linregress(x,logy)

print('r^2 of linear regression = ', rval**2)

plt.plot(x,y,'ro',ms=2.0)
plt.plot(x,cool_f(x,a_fit,b_fit,c_fit),'b-',lw=0.8)
plt.show()

plt.plot(x,logy,'ro',ms=2.0)
plt.plot(x,a_lfit*x + b_lfit,'b-',lw=0.8)
plt.show()

# *** Try multidimensional fit ***

a_grid = np.arange(2,5)
b_grid = np.arange(1,5)
#c_grid = np.arange(10)
c_grid = np.zeros(10)

clrs = plt_utils.get_cm(b_grid,'plasma')

aa, bb, cc, xx = np.meshgrid(a_grid, b_grid, c_grid, x, indexing='ij', sparse=True)
y = gen_data(xx,cool_f,[aa,bb,cc]) 
logy = np.log(y)
print(logy.shape)

for k, cval in enumerate(b_grid):
    plt.plot(x, logy[0,k,0,:],ls='-',c=clrs[k],label=str(cval))
plt.legend()
plt.show()

a_lfit, b_lfit, rval = tensor_linregress(x,logy)

print('r^2 of linear regression = ', rval**2)

for k, bval in enumerate(b_grid):
    plt.plot(x, logy[0,k,0,:],'o',c=clrs[k],label=str(cval),ms=2.0)
    plt.plot(x, a_lfit[0,k,0]*x + b_lfit[0,k,0], ls='-',c=clrs[k],label=str(cval))
plt.legend()
plt.show()

plt.imshow(rval[0,:,:]**2,origin='lower')
plt.colorbar()
plt.show()
