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
import json
import time
from pandas.api.types import is_categorical
from sklearn.decomposition import PCA, TruncatedSVD



######################################
# depend only on standard libraries, #
# e.g. numpy, pandas, scanpy         #
######################################

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
    
def sparse_multiply(E, a):
    
    """
    Copied from:
    https://github.com/AllonKleinLab/SPRING_dev/blob/master/data_prep/helper_functions.py
    2018 12 06
    """
    
    nrow = E.shape[0]
    w = scipy.sparse.lil_matrix((nrow, nrow))
    w.setdiag(a)
    return w * E

###################################################################################################

def sparse_var(E, axis=0):
    
    """
    Copied from:
    https://github.com/AllonKleinLab/SPRING_dev/blob/master/data_prep/helper_functions.py
    2018 12 06
    """
    
    mean_gene = E.mean(axis=axis).A.squeeze()
    tmp = E.copy()
    tmp.data **= 2
    return tmp.mean(axis=axis).A.squeeze() - mean_gene ** 2 
 
###################################################################################################   
    
def sparse_corrcoef(A, B=None):
    
    """from https://stackoverflow.com/questions/19231268/correlation-coefficients-for-sparse-matrix-in-python?rq=1"""
    
    t0 = time.time()
    if B is not None:
        A = scipy.sparse.vstack((A, B), format='csr')

    A = A.astype(np.float64)
    n = A.shape[1]

    # Compute the covariance matrix
    rowsum = A.sum(1)
    centering = rowsum.dot(rowsum.T.conjugate()) / n
    C = (A.dot(A.T.conjugate()) - centering) / (n - 1)

    # The correlation coefficients are given by
    # C_{i,j} / sqrt(C_{i} * C_{j})
    d = np.diag(C)
    coeffs = C / np.sqrt(np.outer(d, d))
    t1 = time.time()
    print("%.2f min."%((t1-t0)/60.))
    return coeffs
    
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

def shuffle_rows(Z,seed=None,sparse=False,nonzeros=None):
    
    """
    input:
            Z - np.array, each row will be shuffled separately
            seed - seed for randomization
            sparse - boolean, default False, specify True if sparse matrix
            nonzeros - np.array with number of nonzero values per row, optional
            
    returns:
            copy of Z with shuffled columns
            
    Note: the sparse version was not particularly fast on with my toy datasets around 2000 x 3000 in size.
    Will try to improve only when faced with a very large dataset. My intuition is that it should be memory-efficient
    at least.
    """
    
    nrows = Z.shape[0]
    ncols = Z.shape[1]
    
    if sparse:
        
        # get the number of non-zero values per row
        if nonzeros is None:
            nonzeros = np.array((Z>0).sum(axis=1).T)[0]

        np.random.seed(seed)
        c = [np.random.choice(np.arange(ncols),nrix,replace=False)+ncols*rowid\
            for rowid, nrix in enumerate(nonzeros)]
        
        # flatten list of lists
        c = [item for sublist in c for item in sublist]
        
        Zshuf = Z.reshape(1,nrows*ncols).tocsr()
        
        Zshuf.indices = np.array(c)
        Zshuf = Zshuf.reshape((nrows,ncols))
        

    else:
        b = range(ncols)
        np.random.seed(seed) #to be able to recreate the exact same results
        c = np.array([np.random.permutation(b) for i in range(nrows)])
        d = np.arange(nrows)*ncols
        e = (c.T+d).T
        Zshuf = Z.flatten()[e.flatten()].reshape(Z.shape)
        
    return Zshuf
    
###################################################################################################
  
def export_spring_plot(
        adata, project_dir,
        subplot_name,
        E = None,
        gene_list = None,
        cell_ix = None,
        embedding_method = 'draw_graph',
        cell_groupings=None,
        custom_color_tracks=None,
        ):
    
    from scanpy.exporting import (write_color_tracks,
                                 get_color_stats_genes,
                                 get_color_stats_custom,
                                 write_color_stats, 
                                 build_categ_colors,
                                 write_cell_groupings,
                                 get_edges,
                                 write_graph,
                                 write_edges)
    
    import logging as logg
    
    """
    Modified from:
    https://scanpy.readthedocs.io/en/stable/api/scanpy.api.export_to.spring_project.html#scanpy.api.export_to.spring_project
    
    This function will not save the expression data in the project_directory, it assumes it's done separately
    
    ----------
    adata : :class:`~anndata.AnnData`
        Annotated data matrix: `adata.uns['neighbors']` needs to
        be present.
    project_dir : `str`
        Path to directory for exported SPRING files.
    subplot_name: `str`
        Name of subplot folder to be created at `project_dir+"/"+subplot_name`
    E: `scipy.sparse.csr`. Expression data save in project_dir and containing all gene expression values.
        This is used to calculate saturation levels in the subplot (saturated at 99.6th percentile).
        If None, will use adata.raw.
    gene_list: `np.array` or None
        If None, will look for gene list in adata.raw or adata.X
    cell_ix: `np.array` or None
        Positional index of cell in adata relatively to expression data saved in project_dir.
        If None, will assume that adata.X contains all cells saved in project_dir
    embedding_method: `str`
        Name of a 2-D embedding in `adata.obsm`
    cell_groupings : `str`, `list` of `str`, optional (default: `None`)
        Instead of importing all categorical annotations when `None`,
        pass a list of keys for `adata.obs`.
    custom_color_tracks : `str`, `list` of `str`, optional (default: `None`)
        Specify `adata.obs` keys for continuous coloring.

    """

    # get the helper functions:
    
    
    # need to get nearest neighbors first
    if 'neighbors' not in adata.uns:
        raise ValueError('Run `sc.pp.neighbors` first.')

    # check that requested 2-D embedding has been generated
    if embedding_method not in adata.obsm_keys():
        if 'X_' + embedding_method in adata.obsm_keys():
            embedding_method = 'X_' + embedding_method
        else:
            if embedding_method in adata.uns:
                embedding_method = 'X_' + embedding_method + '_' + adata.uns[embedding_method]['params']['layout']
            else:
                raise ValueError('Run the specified embedding method `%s` first.' %embedding_method)

    coords = adata.obsm[embedding_method]

    # Make subplot directory (subplot has same name as project)
    project_dir = project_dir.rstrip('/') + '/'
    subplot_dir = project_dir + subplot_name + '/'
    if not os.path.exists(subplot_dir):
        os.makedirs(subplot_dir)
    print("Writing subplot to %s" %subplot_dir)
    
    
    # check if E and gene_list specified
    if (E is None) & (gene_list is None):
        # Ideally, all genes will be written from adata.raw
        if adata.raw is not None:
            E = adata.raw.X.tocsc()
            gene_list = list(adata.raw.var_names)
        else:
            E = adata.X.tocsc()
            gene_list = list(adata.var_names)
    else:
        gene_list = list(gene_list)

    
    # check if cell_ix given
    if cell_ix is None:
        cell_ix = np.arange(adata.X.shape[0])
    

    # Get categorical and continuous metadata
    categorical_extras = {}
    continuous_extras = {}
    if cell_groupings is None:
        for obs_name in adata.obs:
            if is_categorical(adata.obs[obs_name]):
                categorical_extras[obs_name] = [str(x) for x in adata.obs[obs_name]]
    else:
        if isinstance(cell_groupings, str):
            cell_groupings = [cell_groupings]
        for obs_name in cell_groupings:
            if obs_name not in adata.obs:
                logg.warning('Cell grouping "%s" is not in adata.obs' %obs_name)
            elif is_categorical(adata.obs[obs_name]):
                categorical_extras[obs_name] = [str(x) for x in adata.obs[obs_name]]
            else:
                logg.warning('Cell grouping "%s" may be not a categorical variable' %obs_name)
                categorical_extras[obs_name] = [str(x) for x in adata.obs[obs_name]]

                
    if custom_color_tracks is None:
        for obs_name in adata.obs:
            if not is_categorical(adata.obs[obs_name]):
                continuous_extras[obs_name] = np.array(adata.obs[obs_name])
    else:
        if isinstance(custom_color_tracks, str):
            custom_color_tracks = [custom_color_tracks]
        for obs_name in custom_color_tracks:
            if obs_name not in adata.obs:
                logg.warning('Custom color track "%s" is not in adata.obs' %obs_name)
            elif not is_categorical(adata.obs[obs_name]):
                continuous_extras[obs_name] = np.array(adata.obs[obs_name])
            else:
                logg.warning('Custom color track "%s" is not a continuous variable' %obs_name)


    # Write continuous colors
    continuous_extras['Uniform'] = np.zeros(adata.X.shape[0])
    write_color_tracks(continuous_extras, subplot_dir+'color_data_gene_sets.csv')

    # Create and write a dictionary of color profiles to be used by the visualizer
    color_stats = {}
    color_stats = get_color_stats_genes(color_stats, E[cell_ix,:], gene_list)
    color_stats = get_color_stats_custom(color_stats, continuous_extras)
    write_color_stats(subplot_dir + 'color_stats.json', color_stats)

    # Write categorical data
    categorical_coloring_data = {}
    categorical_coloring_data = build_categ_colors(categorical_coloring_data, categorical_extras)
    write_cell_groupings(subplot_dir+'categorical_coloring_data.json', categorical_coloring_data)

    # Write graph in two formats for backwards compatibility
    edges = get_edges(adata)
    write_graph(subplot_dir + 'graph_data.json', len(cell_ix), edges)
    write_edges(subplot_dir + 'edges.csv', edges)


    # Write cell filter; for now, subplots must be generated from within SPRING,
    # so cell filter includes all cells.
    np.savetxt(subplot_dir + 'cell_filter.txt', cell_ix, fmt='%i')
    np.save(subplot_dir + 'cell_filter.npy', cell_ix)

    # Write 2-D coordinates, after adjusting to roughly match SPRING's default d3js force layout parameters
    coords = coords - coords.min(0)[None,:]
    coords = coords * (np.array([1000, 1000]) / coords.ptp(0))[None,:] + np.array([200,-200])[None,:]
    np.savetxt(subplot_dir + 'coordinates.txt',
               np.hstack((np.arange(len(cell_ix))[:,None], coords)),
               fmt='%i,%.6f,%.6f')


    return None

######################################
# dependend on standard libraries	 #
# AND custom functions defined above #
######################################

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

def zscore_sparse(E,base_ix=[],var_stab_only=False):
    """zscores along the columns. If specified, only base_ix rows with be used
    to calculate mean and st. dev. Inspired by "get_pca" function from
    https://github.com/AllonKleinLab/SPRING_dev/blob/master/data_prep/helper_functions.py
    2018 12 07
    
    Input:
        E - sp.sparse matrix, cells x genes
        base_ix - np.array with positional index
        var_stab_only - variance stabilize only, bool
    returns:
        Zscores, if var_stab_only==True,
        then sparse matrix, else and np.array
    
    
    
    """
    
    if len(base_ix) == 0:
        base_ix = np.arange(E.shape[0])
    
    zstd = np.sqrt(sparse_var(E[base_ix,:]))
    
    if var_stab_only:
        Z = sparse_multiply(E.T, 1/zstd).T
        return Z
        
    else:
        zmean = E[base_ix,:].mean(0)
        Z = sparse_multiply((E - zmean).T, 1/zstd).T
        return np.array(Z)

###################################################################################################

def find_num_pc(Z,n=10,start_pc = 200,sparse=False,svd_solver='randomized'):
    
    """Find the number of non-random principle components.
    Steps:
        get observed eigenvalues for matrix Z
        for n rounds:
            shuffle each column of Z separately the get Zshuffled
            get the eigenvalues of Zshuffled
            record i: the number observed eigenvalues larger than the largest random eigenvalue.
            
        Consider the 5th percentile across i1,i2 ... in the number of non-random principle components
        with 95% confidence.
        
    input:
        Z - mean centered variance stabilized matrix (along the columns)
        n - number of shuffling rounds
        start_pc - guest the max number of PCs to consider, this an attempt to make the code more efficient
        sparse - boolean, default False, if True, will use truncateSVD
        svd_solver - ‘arpack’ or ‘randomized’. If sparse=False, ‘auto’ or ‘full’ are also supported.
            Check the documentation for sklearn.decomposition.PCA and sklearn.decomposition.TruncatedSVD
            for more details.
            On a quick test randomized seem faster than arpack
        
        
    returns:
        a dictionary with:
        'list_non_rand' - list with numbers of observed eigenv. larger than the largest random eigenv.
        'pca' - pca object from sklearn.decomposition after fitting the observed Zscores
        'num_pc' - integer, number of non-random PCs
        
        """
    
    start=time.time()
    
    nonzeros = None
    if sparse:
        nonzeros = np.array((Z>0).sum(axis=0))[0]
        

    # Get observed eigenvalues:
    insuff_pcs = True
    maxpc = 0
    
   
    
    l = []
    counter=0
    while insuff_pcs:
        # a loop to calculate more observed PCs in case maxpc (e.g. 200) not enough
        # this is an attempt to make the code more efficient
        maxpc+=int(start_pc)
        numpc = min(maxpc,Z.shape[0]-1,Z.shape[1]-1)
        
        # choose PCA method
        if sparse:
            pca = TruncatedSVD(n_components=numpc,algorithm=svd_solver)
            pca_shuff = TruncatedSVD(n_components=numpc,algorithm=svd_solver)
        else:
            pca = PCA(n_components=numpc,svd_solver=svd_solver)
            pca_shuff = PCA(n_components=numpc,svd_solver=svd_solver)
            

        # get observed eigenvalues
        print("calculating the first %d observed eigenvalues..."%numpc)
        pca.fit(Z)
        ev_obs = np.msort(pca.explained_variance_)[::-1] #make sure to sort eigenvalues
        
        # shuffle once
        counter+=1
        print("calculating the random eigenvalues for %d rounds of shuffling..."%n)
        Zshuff = shuffle_rows(Z.T,seed=counter,sparse=sparse,nonzeros=nonzeros).T
        pca_shuff.fit(Zshuff)
        ev_rnd = max(pca_shuff.explained_variance_)
        l.append(ev_rnd)
        print(counter,'\t',sum(ev_obs>ev_rnd),'\t','%.2f min.'%((time.time()-start)/60.))

        insuff_pcs = (ev_obs<(ev_rnd*0.9)).sum()==0 #this sum is 0 if too few observed PCs calculated

    #iterate more
    while n>counter:
        
        # choose PCA method again
        if sparse:
            pca_shuff = TruncatedSVD(n_components=numpc,algorithm=svd_solver)
        else:
            pca_shuff = PCA(n_components=numpc,svd_solver=svd_solver)
        
        
        counter+=1
        Zshuff = shuffle_rows(Z.T,seed=counter,sparse=sparse,nonzeros=nonzeros).T
        pca_shuff.fit(Zshuff)
        ev_rnd = max(pca_shuff.explained_variance_)
        l.append(ev_rnd)
        print(counter,'\t',sum(ev_obs>ev_rnd),'\t','%.2f min.'%((time.time()-start)/60.))

    #for each round of shuffling genes, what is the number of obs eigenvalues larger than the largest random eigenvalue
    nrlarger = [(ev_obs>i).sum() for i in l]
    num_pc = int(np.percentile(np.array(nrlarger),0.05))
    
    return {'list_non_rand':nrlarger,
           'pca':pca,
           'num_pc':num_pc}






