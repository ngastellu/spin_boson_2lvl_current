#!/usr/bin/env python

import json
from os import path
import numpy as np

class ParameterParser:

    def __init__(self, filepath):
        self.path = filepath
        with open(filepath) as fo:
            self.pdict = json.load(fo)


    def load_intrinsic(self):
        idict = self.pdict['intrinsic']

        dpath = path.expanduser(idict['path_to_data'])
        LUMO_data = np.load(path.join(dpath, idict['LUMO_data_file']))
        LUMOp1_data = np.load(path.join(dpath, idict['LUMO+1_data_file']))

        e_d = np.mean(LUMO_data[1,:])
        e_a = np.mean(LUMOp1_data[1,:])
        gamL = np.mean(LUMO_data[2,:])
        gamR = np.mean(LUMOp1_data[3,:])

        gam_ph = idict['gamma_phonon']
        shift = idict['e_shift']

        e_d -= shift
        e_a -= shift

        return e_d, e_a, gamL, gamR, gam_ph

    def load_specific(self,plist=None):
        sdict = self.pdict['specific']
        if plist: #returns NumPy array
            out = [0] * len(plist)
            for k, p in enumerate(plist):
                out[k] = sdict[p]
        else: #returns dictionary
            out = sdict

        return out


    def load_grids(self,plist=None):
        gdict = self.pdict['grids']
        if plist: #creates a list
            out = [None] * len(plist) #creates a fixed length list; avoids appending
            for k, p in enumerate(plist):
                out[k] = np.linspace(*gdict[p])
        else:  
            out = gdict.fromkeys(gdict.keys())
            for key in out:
                out[key] = np.linspace(*gdict[key])
        
        return out


    