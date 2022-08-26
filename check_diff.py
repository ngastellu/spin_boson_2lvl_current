#!/usr/bin/env python


from full_param_scan import *
import numpy as np
import matplotlib.pyplot as plt



print('\n--------STARTING COMPARISON--------')

#headers = ['kLR01','kLR10','kRL01', 'kRL10', 'current_non-dis', 'current_dis']
#headers = ['current_dis']
headers = ['p1_non-dis','p1_dis','beta_eff','current_dis']

for h in headers:
    print('\n\n******* Checking %s... *******'%h)
    #control = np.load('%s.npy'%h)
    control_swapped = np.load('%s.npy'%h)
    control = np.zeros((control_swapped.shape[1],control_swapped.shape[0],control_swapped.shape[2]))
    for i in range(control.shape[0]):
        control[i,:,:] = control_swapped[:,i,:]
    #test = np.load('full_%s.npy'%h)[:,:,0,:]
    test = np.load('full_%s.npy'%h)[:,0,:,:]
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


#w0_grid = np.linspace(0.1,1.0,10)
#temp_grid = np.linspace(40,4000,200)
#kB = 8.617e-5
#beta_grid = 1.0 / (kB * temp_grid)
#
#ww, bb = np.meshgrid(w0_grid, beta_grid, indexing='ij', sparse=True)
#
#plt.plot(beta_grid, control[0,1,:],'r-',label='control',lw=0.8)
#plt.plot(beta_grid, test[0,1,:],'b-',label='test',lw=0.8)
#plt.legend()
#plt.show()
#
#plt.plot(beta_grid, control[0,1,:] - test[0,1,:],'r-',lw=0.8,label='control - test')
#plt.legend()
#plt.show()
#
#plt.plot(beta_grid,bose_einstein(ww,bb)[0,:]/(control[0,1,:] - test[0,1,:]))
#plt.show()
