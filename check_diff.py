#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


handles = ['kLR01','kLR10','kRL01','kRL10','current_non-dis','current_dis']

print('\n--------STARTING COMPARISON--------')

for h in handles:
    print('\n\n******* Checking %s... *******'%h)
    control_swapped = np.load('%s.npy'%h)
    control = np.zeros_like(control_swapped)
    for i in range(control.shape[0]):
        control[i,:,:] = control_swapped[:,i,:]
    test = np.load('full_%s.npy'%h)[:,:,0,:]
    diff = np.abs(test-control)
    print('Avg of test = ', np.mean(test.flatten()))
    print('Avg of control = ', np.mean(control.flatten()))
    print('Avg. |contol - test| = ', np.mean(diff.flatten()))
    #atols = np.logspace(-4,-10,7)
    print('Nb of exact agreements = ', np.sum(diff == 0))
    print('Nb of elements = ', diff.flatten().shape[0])
    #agreements = control[diff==0]
    #ag_inds = np.vstack((diff == 0).nonzero()).T
    #print(ag_inds)
    #print(np.unique(ag_inds,axis=0).shape)
    #print(np.unique(ag_inds[:,:-1],axis=0).shape)
    #print(np.unique(ag_inds[:,:-1],axis=0))





   # atols = np.logspace(-4,-5,2)
   # diff_test = (diff > atols[:,None,None,None])
   # errsums = np.sum(diff_test.reshape(atols.shape[0], -1),axis=1) # nb. of elements which differ by atol for atol in atols
#    for n, atol, dt in zip(errsums,atols, diff_test):
#        print('\n%d of %d elements differ by more than %1.0e'%(n,diff.flatten().shape[0],atol))
#
#        # In this next part, we want to see if there's a pattern for which elements of the control and test arrays actually
#        # do agree.
#        for d, arr in zip(dt.shape, (~dt).nonzero()): # loop over the axes of the diff_test associated with a given tolerance atol
#            good_inds = np.unique(arr)
#            n = good_inds.shape[0]
#            if n == d:
#                #print('No missing inds.')
#                pass
#            else:
#                delta = good_inds[1:] - good_inds[:-1]
#                diff_inds = (delta > 0).nonzero()[0] - 1
#                #print(diff_inds)
#                pass


