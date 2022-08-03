import numpy as np
import matplotlib.pyplot as plt


handles = ['kLR01','kLR10','kRL01','kRL10','current_non-dis','current_dis']

for h in handles:
    print('Checking %s...'%h)
    control = np.load('%s.npy'%h)
    test = np.load('full_%s.npy'%h)[:,:,0,:]
    diff = np.abs(test-control)
    print('Avg. |contol - test| = ', np.mean(diff.flatten()))
    atols = np.logspace(-4,-10,7)
    diff_test = (diff > atols[:,None,None,None])
    errsums = np.sum(diff_test.reshape(atols.shape[0], -1),axis=1) # nb. of elements which differ by atol for atol in atols
    for n, atol in zip(errsums,atols):
        print('%d elements differ by more than %1.0e'%(n,atol))
