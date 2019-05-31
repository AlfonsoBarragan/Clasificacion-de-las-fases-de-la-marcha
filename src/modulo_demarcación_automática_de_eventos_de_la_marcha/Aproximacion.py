#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 16:46:26 2019

@author: alfonso
"""

from numpy import corrcoef, transpose, arange
from pylab import pcolor, show, colorbar, xticks, yticks
import numpy as np
import pandas as pd

import scipy

from matplotlib.pylab import hist


import seaborn as sns
import matplotlib.pyplot as plt

# 0. Load Data
df = pd.read_csv("../modulo_de_etiquetado/data/insoleR_dataset.csv")
df2 = pd.read_csv("../modulo_de_etiquetado/data/insoleL_dataset.csv")

df = df.drop (columns=['Timestamp_init', 'Timestamp_end', 'Date_init','Date_end','Source'])
df2 = df2.drop (columns=['Timestamp_init', 'Timestamp_end', 'Date_init','Date_end','Source'])


# plotting the correlation matrix
#http://glowingpython.blogspot.com.es/2012/10/visualizing-correlation-matrices.html
R = corrcoef(transpose(df))
pcolor(R)
colorbar()
yticks(arange(0,33),range(0,33))
xticks(arange(0,33),range(0,33))
show()

R2 = corrcoef(transpose(df2))
pcolor(R2)
colorbar()
yticks(arange(0,33),range(0,33))
xticks(arange(0,33),range(0,33))
show()


# http://stanford.edu/~mwaskom/software/seaborn/examples/many_pairwise_correlations.html
# Generate a mask for the upper triangle
sns.set(style="white")
mask = np.zeros_like(R, dtype=np.bool)
mask2 = np.zeros_like(R2, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True
mask2[np.triu_indices_from(mask2)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(200, 10, as_cmap=True)
cmap2 = sns.diverging_palette(200, 10, as_cmap=True)


# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(R, mask=mask, cmap=cmap, vmax=.8,
            square=True, xticklabels=2, yticklabels=2,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

sns.heatmap(R2, mask=mask2, cmap=cmap2, vmax=.8,
            square=True, xticklabels=2, yticklabels=2,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)


