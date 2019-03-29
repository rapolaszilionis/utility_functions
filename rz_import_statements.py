# import statements

import platform
print("python version:",platform.python_version()) # prints python version

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

import time
import datetime
import pandas as pd
import numpy as np
import scipy
from scipy import stats
from scipy import sparse
import copy
import sys,os
import json
import sklearn.cluster
from sklearn.cluster import SpectralClustering
import fastcluster

import statsmodels.sandbox.stats.multicomp
from sklearn.decomposition import PCA, TruncatedSVD

from anndata import AnnData
import itertools
from collections import Counter
from collections import OrderedDict
import glob

import io
import gzip
