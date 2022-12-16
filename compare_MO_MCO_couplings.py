#!/usr/bin/env pythonw

from os import path
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, cm
from ao_hamiltonian import read_energies, read_MO_file, AO_hamiltonian,\
        AO_gammas, MCOs, plot_MO


Ha2eV = 27.2114
gamma = 0.1

rcParams['text.usetex'] = True
rcParams['font.size'] = 16
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

datadir = '../../simulation_outputs/qcffpi_data/20x40_MAC_ensemble/localisation_coupling_MCO'
npyfiles = glob(path.join(datadir, '*npy'))

Nsamples = len(npyfiles)

datalist = [np.load(f) for f in npyfiles]
lens = np.array([arr.shape[1] for arr in datalist])
print(lens)
inds = lens.cumsum() #inds corresponding to the first MCO of the next structure in the ensemble
energies_MCO, couplings, iprs_MCO, rgyrs_MCO, gamL_MCO, gamR_MCO = np.hstack(datalist)



couplings = np.abs(couplings)

#coupling_metric = (2*gamL*gamR)/(gamL + gamR)
gam_sum_MCO = gamL_MCO + gamR_MCO
plt.scatter(couplings, gam_sum_MCO,s=1.0,c=np.abs(energies_MCO))
#plt.plot(X,X,'-',lw=0.8,label='$y=x$',c='#0080FF')
plt.xlabel('$\gamma$ [eV]')
plt.ylabel('$\Gamma_L + \Gamma_R$ [eV]')
plt.show()


datadir = '../../simulation_outputs/qcffpi_data/20x40_MAC_ensemble/localisation_coupling_MO'
npyfiles = glob(path.join(datadir, '*npy'))
datalist = [np.load(f) for f in npyfiles]

energies_MO, iprs_MO, rgyrs_MO, gamL_MO, gamR_MO = np.hstack(datalist)

plt.scatter(energies_MO, energies_MCO, s=1.0,c=np.abs(couplings))
plt.xlabel('$\\varepsilon_{MO}$ [eV]')
plt.ylabel('$\\varepsilon_{MCO}$ [eV]')
plt.show()

gam_sum_MO = gamL_MO + gamR_MO

plt.scatter(gam_sum_MO,gam_sum_MCO,s=1.0)
plt.xlabel('Gamma sum MO')
plt.ylabel('Gamma sum MCO')
plt.show()


plt.scatter(energies_MO, gam_sum_MO,s=1.0)
plt.xlabel('$\\varepsilon_{MO}$ [eV]')
plt.ylabel('Gamma sum MO')
plt.show()
