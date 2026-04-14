import numpy as np
# from modules.mlsarray import slicelist
import matplotlib.pyplot as plt

# custom figsize for subplots
figsize_single = (10, 8)   # half-width panel in two panel layout
figsize_double = (16, 9)   # full-width single panel

def apply_style():
    plt.rcParams.update({
        # Line and Frame thicknesses
        'lines.linewidth': 4,
        'axes.linewidth': 3,
        'lines.markersize': 12,
        
        # Ticks: Direction and Size
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.width': 3,
        'ytick.major.width': 3,
        'xtick.major.size': 8,    
        'ytick.major.size': 8,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'xtick.minor.width': 1.5,
        'ytick.minor.width': 1.5,
        'xtick.minor.size': 5,      # Increased to clear the 3pt frame
        
        # Font and LaTeX
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        
        # Font Sizes
        'font.size': 36,
        'axes.titlesize': 36,
        'axes.labelsize': 36,
        'xtick.labelsize': 32,
        'ytick.labelsize': 32, 
        'legend.fontsize': 30,      
        
        # Spacing and Legend
        'axes.labelpad': 20,        # Increased for more space between axis and label
        'xtick.major.pad': 12,      # Add extra padding between x-axis and xtick labels
        'ytick.major.pad': 8,       # Add extra padding for y-axis ticks (optional)
        'legend.edgecolor': 'black',
        'legend.framealpha': 1.0,
        'legend.fancybox': False,   # Sharp corners look more professional
        'figure.autolayout': True,
        'savefig.dpi': 300,
    })

def savename(datadir, fname, prefix):
    if fname.endswith('out.h5'):
        return datadir + prefix + '_.svg'
    return datadir + prefix + '_' + fname.split('/')[-1].split('out_')[-1].replace('.h5', '.svg')

def symmetrize_y_axis(axes):
    y_max = np.abs(axes.get_ylim()).max()
    axes.set_ylim(ymin=-y_max, ymax=y_max)
    