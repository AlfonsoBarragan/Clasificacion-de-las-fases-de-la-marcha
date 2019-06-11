#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 16:46:26 2019

@author: alfonso
"""

from modulo_de_funciones_de_soporte import utils
from numpy import corrcoef, transpose, arange
from pylab import pcolor, show, colorbar, xticks, yticks
import numpy as np
import pandas as pd
import cv2

import scipy
from scipy import cluster

from matplotlib.colors import rgb2hex, colorConverter
from collections import defaultdict

from matplotlib.pylab import hist
from sklearn import preprocessing


import seaborn as sns
import matplotlib.pyplot as plt

import sklearn.neighbors
import sklearn.cluster
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydot
import pickle

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

def calculatePCA(dataframe, path):

    estimator = PCA (n_components = 2)
    X_pca = estimator.fit_transform(dataframe)

    #Print
    print(estimator.explained_variance_ratio_)
    pd.DataFrame(np.matrix.transpose(estimator.components_),
    columns=['PC-1', 'PC-2'], index=dataframe.columns)

    #Print
    fig, ax = plt.subplots()

    for i in range(len(X_pca)):
        plt.text(X_pca[i][0], X_pca[i][1], ".")
    
    plt.xlim(-2, 4)
    plt.ylim(-2, 4)
    ax.grid(True)
    fig.tight_layout()
    
    plt.savefig(path+"pca.png")
    return X_pca

def clustering(data_norm, X_pca):

    # 1.0 Clustering execution
    k = 3 # one attack for each no attack

    # 1.1 k-means ++
    centroidsplus, labelsplus, zplus =  sklearn.cluster.k_means(data_norm, k, init="k-means++" )
    plot_pca(X_pca, labelsplus, "kmeans++", "./pca_clustering.png")

    # Characterization
    n_clusters_ = len(set(labelsplus)) #- (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d' % n_clusters_)
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(data_norm, labelsplus))
    df['group'] = labelsplus
    df.groupby(('group')).mean()
    
    db = DBSCAN(eps=0.2, min_samples=10, metric='euclidean')
    y_db = db.fit_predict(X_pca)
    
    #3. Validation/Evaluation
    # Only using silhouette coefficient
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X_pca, y_db))
          
         
    # 4. Plot results
    
    plt.scatter(X_pca[y_db==0,0], X_pca[y_db==0,1], c='lightblue', marker='o', s=40, label='cluster 1')
    plt.scatter(X_pca[y_db==1,0], X_pca[y_db==1,1], c='red', marker='s', s=40, label='cluster 2')
    plt.legend()
    plt.show()
    
    return centroidsplus

def hierarchical_clustering(datanorm):
    dist = sklearn.neighbors.DistanceMetric.get_metric('euclidean')
    matsim = dist.pairwise(datanorm)
    avSim = np.average(matsim)
    print ("%s\t%6.2f" % ('Distancia Media', avSim))
    
    clusters = cluster.hierarchy.linkage(matsim, method = 'complete')
    # http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.hierarchy.dendrogram.html
    cluster.hierarchy.dendrogram(clusters, color_threshold=15, orientation='left')
    
    cut = 190 # !!!! ad-hoc
    labels = cluster.hierarchy.fcluster(clusters, cut , criterion = 'distance')
    print ('Number of clusters %d' % (len(set(labels))))
    plt.show()

    return labels


def plot_pca(X_pca, labels, type_clus, save_path):

    colors      = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors      = np.hstack([colors] * 20)
    fig, ax     = plt.subplots()
    
    for i in range(len(X_pca)):
        plt.text(X_pca[i][0], X_pca[i][1], '.', color=colors[labels[i]])
    
    plt.xlim(-2, 4)
    plt.ylim(-2, 4)
    ax.grid(True)
    fig.tight_layout()

    plt.savefig(save_path)

def random_forest(train, test):
  
    features  = train.columns[:32]
    x_train   = train[features]
    y_train   = train['groups']
    
    
    x_test    = test[features]
    y_test    = test['groups']
    
    X, y      = x_train, y_train
    
    clf_rf = RandomForestClassifier(n_estimators=3, random_state = 0)
    """
    param_dist = {"max_depth": [None],
              "max_features": sp_randint(1, 13),
              "min_samples_split": sp_randint(2, 95),
              "min_samples_leaf": sp_randint(1, 95),
              "bootstrap": [True, False], 'class_weight':['balanced'],
              "criterion": ["gini", "entropy"]}
    
    random_search = RandomizedSearchCV(clf_rf, scoring= 'f1_micro', 
                                   param_distributions=param_dist, 
                                   n_iter= 80)  
    random_search.fit(X, y)
    """
    clf_rf.fit(X, y) # Construcción del modelo
    
    preds_rf = clf_rf.predict(x_test) # Test del modelo
    
#    report(random_search.cv_results_)    
    
    print("Random Forest: \n" 
          +classification_report(y_true=y_test, y_pred=preds_rf))
    
    # Matriz de confusión
    
    print("Matriz de confusión:\n")
    matriz = pd.crosstab(test['groups'], preds_rf, rownames=['actual'], colnames=['preds'])
    print(matriz)
    
    # Variables relevantes
    
    print("Relevancia de variables:\n")
    print(pd.DataFrame({'Indicador': features ,
                  'Relevancia': clf_rf.feature_importances_}),"\n")
    print("Máxima relevancia RF :" , max(clf_rf.feature_importances_), "\n")

    for birch in range(len(clf_rf.estimators_)):
      print(clf_rf.estimators_[birch])
      export_graphviz(clf_rf.estimators_[birch], out_file='tree_from_forest.dot',
                      feature_names=list(test.drop(['groups'], axis=1)),
                      filled=True, rounded=True,
                      special_characters=True, class_names = ['Swing','Ground', 'Movement'])
      (graph,) = pydot.graph_from_dot_file('tree_from_forest.dot')
      graph.write_png('trees/tree_from_forest_birch_{}.png'.format(birch))
    
    return clf_rf

def load_data():
    df_s1_e2 = pd.read_csv("data/sujeto_1/2/data/samples_l_with_frames.csv")
    df_s1_e2.drop(df_s1_e2.columns[[0]], axis=1, inplace=True)

    
    df_s2_e1 = pd.read_csv("data/sujeto_2/1/data/samples_l_with_frames.csv")
    df_s2_e2 = pd.read_csv("data/sujeto_2/2/data/dataset_combined_frames_samples_insoleL.csv")
    df_s2_e1.drop(df_s2_e1.columns[[0]], axis=1, inplace=True)
    df_s2_e2.drop(df_s2_e2.columns[[0]], axis=1, inplace=True)

    df_s3_e1 = pd.read_csv("data/sujeto_3/1/data/samples_l_with_frames.csv")
    df_s3_e2 = pd.read_csv("data/sujeto_3/2/data/samples_l_with_frames.csv")
    df_s3_e1.drop(df_s3_e1.columns[[0]], axis=1, inplace=True)
    df_s3_e2.drop(df_s3_e2.columns[[0]], axis=1, inplace=True)
    
    df_combined = pd.concat([df_s1_e2, df_s2_e1, df_s2_e2, df_s3_e1, df_s3_e2], sort=False)
    
    df_divided_1, df_divided_2 = utils.divide_datasets(df_combined, 0.5)

    df_combined_pressure = df_divided_2.drop(columns=['Timestamp_init', 'Timestamp_end', 'Date_init','Date_end','Source','Subject','experiment','Frame'])
    
    df_combined_pressure = normalize_data(df_combined_pressure)
    
    labels = hierarchical_clustering(df_combined_pressure)
    
def flip_horizontal_frames(directory, new_directory):
    # Load all the images names
    images_names_list = utils.ls(directory)
    
    utils.printProgressBar(0, len(images_names_list), prefix = 'Flipping better than Enrique Domingo Pérez Vergara:', suffix = 'Complete', length = 50)
    
    for image in images_names_list:

        img_aux = cv2.imread("{}/{}".format(directory, image))
        img_aux = cv2.flip(img_aux, 1)
        
        cv2.imwrite("{}/{}".format(new_directory, image), img_aux)
        
        utils.printProgressBar(images_names_list.index(image), len(images_names_list), prefix = 'Flipping better than Enrique Domingo Pérez Vergara:', suffix = 'Complete', length = 50)
        
# Limpiar los outlayers antes de hacer nada
    
# GRUPOS
    # 1 - Giro
    # 2 - 
    # 3 - Giro
    # 4 - Levantamiento de puntera