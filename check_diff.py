#!/usr/bin/env python


from full_param_scan import *
import numpy as np
import matplotlib.pyplot as plt



print('\n--------STARTING COMPARISON--------')

headers = ['kLR01','kLR10','kRL01', 'kRL10', 'current_non-dis', 'current_dis']
#headers = ['current_dis']
#headers = ['p1_non-dis','p1_dis','beta_eff','current_dis']

for h in headers:
    print('\n\n******* Checking %s... *******'%h)
    control = np.load('MAC_%s.npy'%h)
    test = np.load('MAC_%sb.npy'%h)
    diff = np.abs(test-control)
    print('Avg of test = ', np.mean(test.flatten()))
    print('Avg of control = ', np.mean(control.flatten()))
    print('Avg. |contol - test| = ', np.mean(diff.flatten()))
    #atols = np.logspace(-4,-10,7)
    print('Nb of exact agreements = ', np.sum(diff == 0))
    print('Nb of elements = ', diff.flatten().shape[0])

#agreement_inds = np.vstack((diff==0).nonzero()).T
#oo_inds = np.all(agreement_inds[:,:2] == 0, axis=1).nonzero()[0]
##oo_inds = agreement_inds[oo_inds,2]
#print(oo_inds.shape)
#
##nn_inds = ((agreement_inds[:,0] == agreement_inds[:,1])*(agreement_inds[:,0] > 0)).nonzero()[0]
#for n in range(control.shape[0]): 
#    nn_inds = ((agreement_inds[:,0] == agreement_inds[:,1])*(agreement_inds[:,0] == n)).nonzero()[0]
#    print('Nb.of perfect agreements for elements [%d,%d] = %d'%(n,n,agreement_inds[nn_inds,2].shape[0]))
#
#    nm_inds = ((agreement_inds[:,0] > agreement_inds[:,1])*(agreement_inds[:,0] == n)).nonzero()[0]
#    mn_inds = ((agreement_inds[:,0] < agreement_inds[:,1])*(agreement_inds[:,0] == n)).nonzero()[0]
#    print('Nb.of perfect agreements for elements [%d,m] = %d'%(n,agreement_inds[nm_inds,2].shape[0]))
#    print('Nb.of perfect agreements for elements [%d,n] = %d\n'%(n,agreement_inds[mn_inds,2].shape[0]))
