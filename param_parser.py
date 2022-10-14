#!/usr/bin/env python

import json
import numpy as np

class ParameterParser:

    def __init__(self, filepath):
        self.path = filepath
        with open(filepath) as fo:
            self.pdict = json.load(fo)


    def load_intrinsic(self, plist=None):
        idict = self.pdict['intrinsic']
        if plist:
            out = [0] * len(plist)
            for k, p in enumerate(plist):
                out[k] = idict[p]
        else:
            e_d = idict['e_donor']
            e_a = idict['e_acceptor']
            g_p = idict['gamma_phonon']
            return e_d, e_a, g_p
    
    def load_specific(self,plist=None):
        sdict = self.pdict['specific']
        if plist: #returns NumPy array
            out = [0] * len(plist)
            for k, p in enumerate(param_list):
                out[k] = sdict[p]
        else: #returns dictionary
            out = sdict

        return out


    def load_grids(self,plist=None):
        gdict = self.pdict['grids']
        if plist: #creates a list
            out = [None] * len(plist) #creates a fixed length list; avoids appending
            for k, p in plist:
                out[k] = np.linspace(*gdict[p])
        else:  
            out = gdict.fromkeys(gdict.keys())
            for key in out:
                out[key] = np.linspace(*gdict[key])
        
        return out


    