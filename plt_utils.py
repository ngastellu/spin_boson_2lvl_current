import numpy as np
from matplotlib import rcParams, cm


def get_cm(vals, cmap_str, max_val=0.7):
    '''Creates a list of colours from an array of numbers. Can be used to colour-code curves corresponding to
        different values of a given paramter.'''
    if not isinstance(vals, np.ndarray):
        vals = np.array(vals)
    sorted_vals = np.sort(vals)
    delta = sorted_vals[-1] - sorted_vals[0]
    x = max_val * (vals - sorted_vals[0]) / delta
    if isinstance(cmap_str, str):
        if cmap_str[:3] == 'cm.':
            cmap = eval(cmap_str)
        else:
            cmap = eval('cm.' + cmap_str)
    else:
        print('[get_cm] ERROR: The colour map must be specified as a string (e.g. "plasma" --> cm.plasma).\nSetting the colour map to viridis.')
        cmap = cm.viridis
    return cmap(x)


def setup_tex(preamble_str=None):
    rcParams['text.usetex'] = True
    if preamble_str:
        rcParams['text.latex.preamble'] = preamble_str
    else:
        rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}  \usepackage{bm}'

    
