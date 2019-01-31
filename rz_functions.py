# Functions

from rz_import_statements import *


##################################################
# Function dependent on standard libraries only, #
# e.g. pandas, numpy, scanpy					 #
##################################################

def oset(a_list):
    """given a list/1d-array, returns an ordered set (list)"""
    seen = set()
    seen_add = seen.add
    return [x for x in a_list if not (x in seen or seen_add(x))]


# From Adrian Veres for saving and loading pandas dataframes (modified)
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
        
def get_centroids(meta,E,colname,gene_list):
    
    """
    input:
        meta - pd.DataFrame containing per-cell infomation (cells x features)
        E - sparse expression matrix (cells x genes)
        colname - column of meta to get centroids for
        gene_list - gene list, len(gene_list) must be equal E.shape[1]
        
    return:
        centroids, labels x genes, pd.DataFrame
    """
    
    centroids = {}
    uq = sorted(meta[colname].unique())
    for label in uq:
        msk = (meta[colname] == label).values #use "values" to turn pd.Series into row-label-less np.array,
                                             #sometimes row labels mess up the order

        centroids[label] = np.array(E[msk,:].mean(axis=0))[0]
    centroids=pd.DataFrame(centroids)[uq].T
    centroids.columns = gene_list
    print('Check also a more AnnData-friendly alternative function "centroids"')
    return centroids


def text_to_sparse_in_chunks(
    path,
    sep = ',',
    chunksize = 100,
    skiprows = 1,
    skipcols = 1,
    compressed = True,
    save_skipped_rows = True,
    save_skipped_columns = True,
    comment = '#',
    verbose = True,
    ):
    
    """ for reading and simultaneously sparsifying giant csv/tsv of count data.
    input:
        path - path to a counts table.
        sep - separator
        chunksize - how many lines to load at a time before sparsifying
        skiprows - nr of rows to skip
        skipcols - nr of columns to skip
        compressed - whether gzip or not
        save_skipped_rows - if True, will return skipped rows as a dictionary
                            of the form {rows_number: line_as_string}
        
        save_skipped_columns - if True, will return skipped columns as a dictionary
                            of the form {rows_number: [col_1,col_2...col_skipcols]}
                            only for those rows that were not skipped
        
    
    """
    
    if compressed:
        
        f = io.TextIOWrapper(io.BufferedReader(gzip.open(path)))
        
    else:
        f = open(path,"r")
    
    skipped_rows = {}
    skipped_columns = {}
    
    
    counter = 0
    chunks = []
    frame = []

    for line in f:
        counter += 1
        if (counter <= skiprows)|(line.startswith(comment)):
            if verbose:
                print("skipping row starting with:",line[:25])
            if save_skipped_rows:
                #add line to dictionary
                skipped_rows[counter-1] = line
            continue

        l = line.strip('\n').split(sep)

        # save skipped columns, but only for rows that are not skipped.
        skipped_columns[counter-1] = l[:skipcols]

        frame.append(l[skipcols:])
        if float(counter/chunksize) == int(counter/chunksize):
            if verbose:
                print(counter)
            frame = np.array(frame).astype(np.float)
            frame = scipy.sparse.csc_matrix(frame)
            chunks.append(frame)

            # reset frame
            del frame
            frame = []
    
    # in case the total number of lines is a multiple of 
    if not (float(counter/chunksize) == int(counter/chunksize)):
        print(counter)
        frame = np.array(frame).astype(np.float)
        frame = scipy.sparse.csc_matrix(frame)
        chunks.append(frame)
        
        # reset frame
        del frame
        frame = []
    
    f.close()
    
    print("concatenating chunks...")
    E = scipy.sparse.vstack(chunks)
    print("turning into a csc matrix...")
    E = E.tocsc()
    print("done")

    return {'E':E,'skipped_rows':skipped_rows,'skipped_columns':skipped_columns}



def bayesian_classifier(op,cp):
    '''
    op - observed gene expression profile, genes x samples
    cp - class profiles, genes x samples, same genes as op
    returns log10(P(E|type)), the max value is the closes cell type
    '''
    
    #we assume that each cell type has a well define centroid, let's represent this expression vector
    #as the fractions of all mRNAs for each genes (i.e. normalized the expression such that the expression of
    #all genes sums to 1)
    
    cp = cp/cp.sum()
    
    #we assume that the exact expression pattern we observe (E) is multinomially distributed around the centroid.
    #Bayes' formula: P(type|E) = P(E|type)*P(type)/P(E)
    #our classifier is naive, so each E is equally likely (this is how I interpret "naive", although it
    #may have more to do with the assumption that genes are uncorrelated)
    
    ptes = pd.DataFrame({cell:(np.log10(cp.T.values)*op[cell].values).sum(axis=1) for cell in op.columns})
    ptes.index = cp.columns
    return ptes


def custom_colormap(colors,positions=[],cmap_name = 'my_cmap',register=False):
    """
    example of use:
            my_cmap = custom_colormap(['#000000','#f800f8',''#748c08'],[-100,2,50])
    
    input:
        colors: list of colors as hex codes
        positions: list of floats, same lengths as colors, indicating at what position
                    to place the pure color, if empty list, will space colors evenly.
                    The range can be all real numbers, will rescale to go from 0 to 1.
        
        register: if True, will "register" the colormap name, which can be useful for some applications
            
    output:
        colormap object.
    
    Would be nice to add:
    option to provide for color names, e.g. 'magenta', not just hex codes
    
    More info about making colormaps:
    https://matplotlib.org/examples/pylab_examples/custom_cmap.html
    
    """
    
    # make position range from 0 to 1:
    if len(positions)==len(colors):
        positions = np.array(positions).astype(float)
        positions = (positions-positions.min())
        positions = positions/positions.max()
    else:
        positions = np.linspace(0,1,len(colors))

    rgbs = []

    #turn hex into rgb,scale to max 1, append position
    for h,pos in zip(colors,positions):
        h = h.strip('#')
        rgb = np.array([int(h[i:i+2], 16) for i in (0, 2 ,4)])
        rgb = rgb/255.
        rgbs.append(list(rgb)+[pos])

    reds = []
    greens = []
    blues = []
    ll = []
    
    # prepare the color dictionary as described in
    # https://matplotlib.org/examples/pylab_examples/custom_cmap.html
    for nr,[r,g,b,pos] in enumerate(rgbs):
        for ll,col in zip([reds,greens,blues],[r,g,b]): #ll - list of lists
            #append the left position, the starting red value
            ll.append([pos,col,col])

    cdict = {}
    for key,value in zip(['red','green','blue'],[reds,greens,blues]):
        cdict[key] = tuple([tuple(i) for i in value])

    # make colormap
    cm = LinearSegmentedColormap(cmap_name, cdict)
    
    #register cmap:
    plt.register_cmap(cmap=cm)
    return cm


def value_to_color(value,vmin,vmax,cmap=mpl.cm.get_cmap('RdBu_r'),string_color='#FFFFFF'):
    
    """takes a value (float or int) and turns a hex code (string).
    input:
        - value: float or integer
        - vmin: min value in full range
        - vmax: max value in full range
        - cmap: colormap
        - string_color: what color to return is string given as "value",
          default is white (#FFFFFF)
        
        
    output:
        - hex code (string)
    """
    if type(value)==str:
        return string_color
    
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    rgb = cmap(norm(float(value)))
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]*255),int(rgb[1]*255),int(rgb[2]*255))


def flatten_list_of_lists(list_of_lists):
    '''one line, but hard to memorize:
    [item for sublist in list_of_lists for item in sublist]
    Therefore a function
    flat_list = [item for sublist in list_of_lists for item in sublist]
    '''
    return [item for sublist in list_of_lists for item in sublist]
    
    
def pearsonr(a,b):
    
    """ calculated pairwise Pearson correlation between two arrays. Along the rows.
    The length of rows has to match between a and b.
    
    Input:
        a and b - np.array
    Returns:
        np.array() with observations corresponding to rows of a as rows and observations corresponding
        to rows of b as columns.
    """
    
    # check the shape of a
    if len(a.shape)==2:
        # nothing to change
        pass
    elif len(a.shape)==1:
        # make 2D
        a = a.reshape(1,len(a))
    else:
        print("a has to be either 1D or 2D")
        return
        
    
    # check the shape of b
    if len(b.shape)==2:
        # nothing to change
        pass
    elif len(b.shape)==1:
        # make 2D
        b = b.reshape(1,len(b))
    else:
        print("b has to be either 1D or 2D")
        return
        
    # check that row lengths match:
    if a.shape[1]!=b.shape[1]:
        print("Row length mismatch: %d for a and %d for b"%(a.shape[1],b.shape[1]))
        return
    
    # mean center and variance stabilize
    a = (a.T - a.mean(axis=1))/a.std(axis=1)
    b = (b.T - b.mean(axis=1))/b.std(axis=1)
    
    # get the correlation coefficients
    return a.T.dot(np.array(b))/len(a)
    
#######################################################################################################
    
def now():
    """spring current date and time as filename-friendly string"""
    return datetime.datetime.now().strftime('%y%m%d_%Hh%M')
    
#######################################################################################################

def centroids(label,adata,E=None,gene_list=None):
    
    """
    Calculate average gene expression level per cell label (e.g. cluster).
    input:
        - label: name of column that stores the label of interest in adata.obs
        - adata: AnnData object OR a cell x feature pandas dataframe with label as one of the columns
        - E and gene_list: optional and only used when adata is not an AnnData object. In that case
        the cells x genes sparse expression matrix E and the gene_list must be specified
        
    returns:
        pandas dataframe, centroids x genes
        
    """
    
    if isinstance(adata,AnnData):
        E = adata.X
        gene_list = adata.var_names
        meta = adata.obs
    else:
        meta = adata
        
        
    labs = meta[label].unique()
    centroids = {}
    for lab in labs:
        msk = (meta[label] == lab).values #use "values" to turn pd.Series into row-label-less np.array,
                                             #sometimes row labels mess up the order

        centroids[lab] = np.array(E[msk,:].mean(axis=0))[0]
    centroids=pd.DataFrame(centroids).T
    centroids.columns = gene_list
    return centroids

#######################################################################################################

def max_to_2nd_max(avector,pseudo=0):
    s = sorted(avector)
    themax = s[-1]
    the2ndmax = s[-2]
    return (themax+pseudo)/(the2ndmax+pseudo)

#######################################################################################################

def mwu(cg1,cg2,genes,print_progression=True):
    """perform MWU test for each gene comparing
    cells group 1 (cg1) and cell group 2 (cg2).
    Input:
        - cg1 and cg2: expression matrixes to compared, np.array, cells x genes
        - genes: gene list
        - if print_progression, will print a message every 1000 genes
    returns:
        pd.DataFrame with results, includes FDR calculation
    
    """
    
    # calculate the average per group compared to add to results.
    m1 = cg1.mean(axis=0)
    m2 = cg2.mean(axis=0)
    
    res = []
    counter = 0
    for i in range(cg1.shape[1]):
        counter+=1
        if print_progression:
            if int(counter/1000)==counter/1000.:
                print(counter)
                
        res.append(scipy.stats.mannwhitneyu(cg1[:,i],cg2[:,i],
                                            alternative='two-sided'))

    us = [i[0] for i in res]
    ps = [i[1] for i in res]
    import statsmodels
    cps = statsmodels.sandbox.stats.multicomp.multipletests(ps,method = 'fdr_bh')[1]

    return pd.DataFrame([us,ps,list(cps),list(m1),list(m2)],
                        columns=list(genes),index = ['U_statistic','p','fdr','mean1','mean2']).T

#######################################################################################################

#for saving dictionaries
def save_stuff(stuff,path):
    u"""for saving dictionaries, but probably works with lists and other pickleable objects"""
    import pickle
    with open(path+u'.pickle', u'wb') as handle:
        pickle.dump(stuff, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
#######################################################################################################
        
def load_stuff(path):
    """for loading object saved using 'save_stuff'"""
    import pickle
    with open(path, u'rb') as handle:
        return pickle.load(handle)    
    
#######################################################################################################

def yticks_fancy(a,totick,labels_all,emptychar = '',fontsize=5):
    
    """
    utility function originally made for ticking only a subset of selected genes in a genes x observations heatmap.
    example of use: yticks_fancy(a,['Csf1r','Ccr2','','','Arg1','S100a9'],genes_by_cells.index)
    input:
        a - axis with heatmap
        totick - list of yticklabels to display. Use the string defined by
        emptychar to add spacing between groups of genes.
        labels_all - all yticklabels.
        emptychar - string that will be treated as white space
        
    returns: nothing
    
    """


    a.set_yticks([])
    leftshift = 0
    totick = np.array(totick)
    nr_slots = len(totick)
    tickmask = np.array([i!=emptychar for i in totick])
    totick = totick[tickmask]
    y_right = np.array([pd.Index(labels_all).get_loc(i) for i in totick])
    
    #if genes were not typed in in the correct order, account for that to avoid lines crossing
    tickorder = np.argsort(y_right)
    y_right = y_right[tickorder]
    totick = totick[tickorder]
    y_left = np.linspace(0,len(labels_all),nr_slots)[tickmask]
    for l,r,gene in zip(y_left,y_right,totick):
        a.plot((-0.8-leftshift,-0.5-leftshift),(r,r),lw=0.5,color='0.2')
        a.plot((-1.2-leftshift,-0.8-leftshift),(l,r),lw=0.5,color='0.2')
        a.plot((-1.5-leftshift,-1.2-leftshift),(l,l),lw=0.5,color='0.2')
        a.text(-1.6-(leftshift*1.6),l,gene,ha='right',va='center',fontsize=fontsize)
        
#######################################################################################################    

def hier_cluster(datatable,hier_clust_rows=True,hier_clust_cols=True,method='ward',metric='sqrt_correlation'):
    
    """
    assumes that the data table is a pandas dataframe, should also work on an numpy array.
    My favorite combinations is sqrt_correlation distance (proportional to euclidean on zscored data) with
    Ward linkage"""
    
    data = datatable.copy()
    row_link=np.nan
    col_link=np.nan
    if hier_clust_rows:
        #hierarchically cluster:
        if metric=='sqrt_correlation':
            pdist = scipy.spatial.distance.pdist(data,metric='correlation')**0.5
        else:
            pdist = scipy.spatial.distance.pdist(data,metric=metric)
        row_link = fastcluster.linkage(pdist, method=method)
        row_order = scipy.cluster.hierarchy.leaves_list(row_link)
        try:
            #pandas-style indexing
            data = data.iloc[row_order,:]
        except:
            #numpy-style indexing
            data = data[row_order,:]
        
    if hier_clust_cols:
        #hierarchically cluster:
        if metric=='sqrt_correlation':
            pdist = scipy.spatial.distance.pdist(data.T,metric='correlation')**0.5
        else:
            pdist = scipy.spatial.distance.pdist(data.T,metric=metric)
        col_link = fastcluster.linkage(pdist, method=method)
        col_order = scipy.cluster.hierarchy.leaves_list(col_link)
        try:
            data = data.iloc[:,col_order]
        except:
            data = data[:,col_order]
        
    return {'data':data,'row_link':row_link,'col_link':col_link}



#######################################################################################################    

def showspines(an_axes,top=False,right=False,bottom=False,left=False):
    """
    for specifying which spines to make visible in a plot.
    input: 
        an_axes - matplotlib axes object
    returns: nothing

    """
    #after reinstalling conda, top and left switches places...
    [i for i in an_axes.spines.items()][3][1].set_visible(top) #top line
    [i for i in an_axes.spines.items()][1][1].set_visible(right) #right line
    [i for i in an_axes.spines.items()][2][1].set_visible(bottom) #bottom line
    [i for i in an_axes.spines.items()][0][1].set_visible(left) #left line
    an_axes.tick_params(bottom=bottom,right=right,left=left,top=top)
    
    
#######################################################################################################    
    
    
################################################
# Function dependent on standard libraries AND #
# custom functions defined above  	           #
################################################


def color_dataframe_cells(
    frame,
    cmap = mpl.cm.get_cmap('RdBu_r'),
    vmin = None,
    vmax = None,
    ):
    
        
    """colors cells of dataframe by their values
    input:
        - frame: pandas dataframe
        - cmap: colormap to use, e.g mpl.cm.get_cmap('RdBu_r')
        - vmin: min value to saturate colormap at
        - vmax: max value to saturate colormap at
        
    output: pandas "styler" object with cells colored. This "styler" object  does not have the
    full functionality of a pandas dataframe.
    
    Example of use (including saving to excel):
    color_dataframe_cells(my_dataframe).to_excel('table.xlsx')
    
    Depends on custom function value_to_color
    
    """
    
    if vmin is None:
        vmin = frame.min().min()
    if vmax is None:
        vmax = frame.max().max()
        
    return frame.style.applymap(lambda x: 'background-color: %s'%value_to_color(x,vmin,vmax,cmap=cmap))

