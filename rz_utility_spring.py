import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
from scipy import stats
from scipy import sparse
import copy
import sys,os

def start_spring_params(adata,subplotname):
    
    """
    utility function to create a dictionary with SPRING parameters to use
    within adata.uns. This should help keep track of parameters used.
    
    adata = AnnData object
    subplotname = name of subplot to make
    
    return: nothing, motifies adata, adds a dictionary with spring parameters,
    adata.uns['spring_params'][subplotname] = {'k'=..., ...}
    
    k - # neighbors for kNN graph, default 5
    cell_mask - boolean mask (np.array) for selecting a desired subset of cells, default all cells in adata.X
    min_counts and min_cells - for a gene to be retained, at least min_cells have to express the gene at min_counts.
    
    base_ix - index of cells to use as reference for selecting variable genes and calcuting eigenvalues
              default - all cells in adata.X
              
    num_pc - number of principle components to use
              
    
    """
    
    d = {}
    d['k'] = 5
    d['cell_mask'] = np.repeat(True,adata.X.shape[0])
    d['min_counts'] = 3
    d['min_cells'] = 3
    d['base_ix'] = np.arange(adata.X.shape[0])
    d['num_pc'] = 20
    d['plot_name'] = subplotname
    
    
    if 'spring_params' not in adata.uns:
        adata.uns['spring_params'] = {}
    adata.uns['spring_params'][subplotname] = d


###################################################################################################

def filter_abund_genes(
                        E,
                        min_counts,
                        min_cells,
                        ):
    
    """Get boolean mask for selecting genes expressed at at least min_counts in at least min_cells.
    Input:
        E - sparse matrix (scipy.sparse)
        min_counts = counts at least
        min_cells = cells at least
        
    Return: boolean mask
    """

    gmask = np.array((E>=min_counts).sum(axis=0))[0]>=min_cells
    print(sum(gmask),"genes passing abundance filter")
    return gmask


###################################################################################################

def get_vscores_sparse(E, min_mean=0, nBins=50, fit_percentile=0.1, error_wt=1):
    """copied from https://github.com/AllonKleinLab/SPRING_dev/blob/master/data_prep/helper_functions.py on 2018 12 05
    For calculating variability scores as described in Klein et al., Cell 2015.
    See equations S4 and S13 in the manuscript for more details.
    """
    
    ncell = E.shape[0]

    mu_gene = E.mean(axis=0).A.squeeze()
    gene_ix = np.nonzero(mu_gene > min_mean)[0]
    mu_gene = mu_gene[gene_ix]

    tmp = E[:,gene_ix]
    tmp.data **= 2
    var_gene = tmp.mean(axis=0).A.squeeze() - mu_gene ** 2
    del tmp
    FF_gene = var_gene / mu_gene

    data_x = np.log(mu_gene)
    data_y = np.log(FF_gene / mu_gene)

    x, y = runningquantile(data_x, data_y, fit_percentile, nBins)
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]

    gLog = lambda input: np.log(input[1] * np.exp(-input[0]) + input[2])
    h,b = np.histogram(np.log(FF_gene[mu_gene>0]), bins=200)
    b = b[:-1] + np.diff(b)/2
    max_ix = np.argmax(h)
    c = np.max((np.exp(b[max_ix]), 1))
    errFun = lambda b2: np.sum(abs(gLog([x,c,b2])-y) ** error_wt)
    b0 = 0.1
    b = scipy.optimize.fmin(func = errFun, x0=[b0], disp=False)
    a = c / (1+b) - 1


    v_scores = FF_gene / ((1+a)*(1+b) + b * mu_gene);
    CV_eff = np.sqrt((1+a)*(1+b) - 1);
    CV_input = np.sqrt(b);

    return v_scores, CV_eff, CV_input, gene_ix, mu_gene, FF_gene, a, b

###################################################################################################


def runningquantile(x, y, p, nBins):

    """copied from
    https://github.com/AllonKleinLab/SPRING_dev/blob/master/data_prep/helper_functions.py
    on 2018 12 05"""

    ind = np.argsort(x)
    x = x[ind]
    y = y[ind]


    dx = (x[-1] - x[0]) / nBins
    xOut = np.linspace(x[0]+dx/2, x[-1]-dx/2, nBins)

    yOut = np.zeros(xOut.shape)

    for i in range(len(xOut)):
        ind = np.nonzero((x >= xOut[i]-dx/2) & (x < xOut[i]+dx/2))[0]
        if len(ind) > 0:
            yOut[i] = np.percentile(y[ind], p)
        else:
            if i > 0:
                yOut[i] = yOut[i-1]
            else:
                yOut[i] = np.nan

    return xOut, yOut
    
    
###################################################################################################



def vscores(
                E,
                base_ix,
               ):
    
    """
    Calculate gene variability scores as described in Klein et al., Cell 2015.
    See equations S4 and S13 in the manuscript for more details.
    Mostly wrapper around get_vscores_sparse from SPRING helper_functions but also gives
    vscores above mode as "var_gene_mask"
    
    Input:
        E - scipy.sparse expression matrix
        base_ix - index for selecting cells to use as reference
    Returs:
        dictionary, with keys:
            v_scores - v scores
            mu_gene - average expression for gene
            ff_gene - fano factor for gene
            var_gene_mask - variable gene mask
            a - what is called "CV_M" in Klein et al,  variability in total UMIs detected.
            b - what is called "CV_<1/N>, variability in total number
                of mRNA transcripts in originally present cells (N) (cell size correction)
        
    """
    
    # calculate v scores
    v_scores, CV_eff, CV_input, gene_ix, mu_gene, FF_gene, a, b = get_vscores_sparse(E[base_ix, :])
    
    # get v_scores above mode
    f,x,_=plt.hist(np.log10(v_scores),bins=100);
    mode_S = x[np.argmax(f==max(f))]
    thresh_S = 10**mode_S
    plt.close()

    var_gene_mask = v_scores>=thresh_S
    
    return {
           'v_scores':v_scores,
           'mu_gene':mu_gene,
           'ff_gene':FF_gene,
           'var_gene_mask':var_gene_mask,
           'a':a,
           'b':b
           }

###################################################################################################
