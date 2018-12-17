# my global plotting parameters for the entire notebook

import matplotlib as mpl
import matplotlib.pyplot as plt

mylines = 0.15*2.82 # the number 2.82 is the difference
					# between Illustrator 1 pt and python 1 pt.
mpl.rcParams['axes.linewidth'] = mylines # default 1
mpl.rcParams['ytick.direction'] = 'out' # default 'in'
mpl.rcParams['xtick.direction'] = 'out' # default 'in'
mpl.rcParams['xtick.major.size'] = 2 # default 4
mpl.rcParams['ytick.major.size'] = 2 # default 4
mpl.rcParams['xtick.major.width'] = mylines # default 0.5
mpl.rcParams['ytick.major.width'] = mylines # default 0.5
mpl.rcParams['grid.linewidth'] = mylines/1.5 # default 0.5
mpl.rcParams['grid.color'] = '0.8' # default 'k'
mpl.rcParams['grid.linestyle'] = 'solid'# default ':'
mpl.rcParams['legend.frameon'] = False # default True

# the one below is to increase the resolution of inline images
mpl.rcParams['figure.dpi']= 150
mpl.rc("savefig", dpi=150)

# the two lines below will make fonts in pdfs readable
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# set fontsize and style:
plt.rc('font', family = 'Myriad Pro',size = 6)

# it seems like all but the following 2 lines get imported, to figure out at some point
# mpl.rcParams['figure.dpi']= 150
# plt.rc('font', family = 'Myriad Pro',size = 6)
