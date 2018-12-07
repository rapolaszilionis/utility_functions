Utility functions I routinely use, mostly for scRNAseq data analysis.

rz_import_statements.py - my typical import statements\
rz_fig_params.py - my global plotting parameters for the entire notebook\
rz_functions.py - custom utility functions\
rz_utility_spring.py - custom functions mostly related visualizing data using SPRING (https://github.com/AllonKleinLab/SPRING). Some of them use the AnnData format which is the core object of scanpy (https://github.com/theislab/scanpy).
	
Suggestion for how to import:\
from rz_import_statements import *\
from rz_fig_params import *\
import rz_functions as rz\
import rz_utility_spring as srz
