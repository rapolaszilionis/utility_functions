# Functions

from rz_import_statements import *

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
    
    """takes a value (float or int) and turns  a hex code (string).
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


def color_dataframe_cells(
    frame,
    cmap = custom_colormap(['#FFFFFF','#808080']),
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
    
    """
    
    if vmin is None:
        vmin = frame.min().min()
    if vmax is None:
        vmax = frame.max().max()
        
    return frame.style.applymap(lambda x: 'background-color: %s'%value_to_color(x,vmin,vmax,cmap=cmap))


def flatten_list_of_lists(list_of_lists):
    '''one line, but hard to memorize,
    so here is a function.
    flat_list = [item for sublist in list_of_lists for item in sublist]
    '''
    return [item for sublist in list_of_lists for item in sublist]
    