#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 16:18:11 2019

@author: alfonso
"""



from collections import defaultdict

import numpy
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette
from fastcluster import linkage
import seaborn as sns
from matplotlib.colors import rgb2hex, colorConverter

sns.set_palette('Set1', 10, 0.65)
palette = sns.color_palette()
set_link_color_palette(map(rgb2hex, palette))
sns.set_style('white')

numpy.random.seed(25)

x, y = 3, 10
df = pd.DataFrame(np.random.randn(x, y),
                  index=['sample_{}'.format(i) for i in range(1, x + 1)],
                  columns=['gene_{}'.format(i) for i in range(1, y + 1)])

link = linkage(df, metric='correlation', method='ward')

figsize(8, 3)
den = dendrogram(link, labels=df.index, abv_threshold_color='#AAAAAA')
plt.xticks(rotation=90)
no_spine = {'left': True, 'bottom': True, 'right': True, 'top': True}
sns.despine(**no_spine);

plt.tight_layout()
plt.savefig('tree1.png');

den