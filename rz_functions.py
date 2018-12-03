# import statements

import platform
print("python version:",platform.python_version()) # prints python version

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import datetime
import pandas as pd
import numpy as np
import scipy
from scipy import stats
from scipy import sparse
import copy

import statsmodels.sandbox.stats.multicomp



# Functions

def oset(a_list):
    """given a list/1d-array, returns an ordered set (list)"""
    seen = set()
    seen_add = seen.add
    return [x for x in a_list if not (x in seen or seen_add(x))]


#by AV for saving and loading pandas dataframes
def save_df(obj, filename):
    np.savez_compressed(filename, data=obj.values, index=obj.index.values, columns=obj.columns.values)
    
def load_df(filename,encoding=u'ASCII'):
    """you may want to specify encoding='latin1'
    when loading python 2 pickle with python 3.
    https://stackoverflow.com/questions/28218466/unpickling-a-python-2-object-with-python-3
    """
    with np.load(filename,encoding=encoding) as f:
        obj = pd.DataFrame(**f)
    return obj

# for reading barcode and gene list (single column)
def read_col(path):
    l = []
    with open(path,'r') as f:
        for line in f:
            line = line.strip()
            if line!='':
                l.append(line)
            
    return l


def startfig(w=4,h=2,rows=1,columns=1,wrs=None,hrs=None,frameon=True,return_first_ax=True):

    '''
    for initiating figures, w and h in centimeters
    example of use:
    a,fig,gs = startfig(w=10,h=2.2,rows=1,columns=3,wr=[4,50,1],hrs=None,frameon=True)
    hrs - height ratios
    wrs - width ratios
    frameon - whether first axes with frame
    
    returns:
    if return_first_ax=True
    a,fig,gs
    else
    fig,gs
    '''
    
    ratio = 0.393701 #1 cm in inch
    myfigsize = (w*ratio,h*ratio)
    fig = plt.figure(figsize = (myfigsize))
    gs = mpl.gridspec.GridSpec(rows, columns ,width_ratios=wrs,height_ratios=hrs)
    if return_first_ax==True:
        a = fig.add_subplot(gs[0,0],frameon=frameon)
        return a,fig,gs
    else:
        return fig,gs  