#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 19:24:17 2019

@author: alfonso
"""

from modulo_de_funciones_de_soporte import utils
import Squire

import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

def test_classifiers(test_data, models_list):
    
    test_data_aux = test_data[test_data.groups > 1]
    test_data_norm = utils.normalize_data(test_data_aux.drop(columns=['Timestamp_init', 'Timestamp_end', 'Date_init','Date_end','Source','Subject','experiment','Frame', 'groups']))
    
    acc_cross = []
    for models_trained in models_list:
        aux = []
        
        for model in models_trained:
            
            aux.append(cross_validations_list_models(test_data_norm, 10, True, [model]))
            
        print(aux)
        acc_cross.append(aux)

    return acc_cross, test_data_aux

def cross_validations_list_models(dataframe, splits, shuffle_bool, model_list):
    cv = KFold(n_splits = splits, shuffle=shuffle_bool)
    acc_by_model = []
    conf_matrix_by_model = []
    max_attb = len(list(dataframe))
    depth_range = range(1, max_attb * 2)
    
    features    = dataframe.columns[:32]
        
    for model in model_list:
        
        for depth in depth_range:
          fold_accuracy         = []
          confusion_matrix_acc  = []
          
          for train_fold, test_fold in cv.split(dataframe):
            x_train     = dataframe.iloc[train_fold][features]
            y_train     = dataframe.iloc[train_fold]['Gait_event']
            
            model.fit(x_train, y_train)
            f_test = dataframe.iloc[test_fold]
        
            test_acc = model.score(X = f_test.drop(['Gait_event'], axis=1), 
                                      y = f_test['Gait_event'])
            fold_accuracy.append(test_acc)
            confusion_matrix_acc.append(Squire.confusion_matrix(f_test, model.predict(X = f_test.drop(['Gait_event'], axis=1)), False))
        
        average = sum(fold_accuracy)/len(fold_accuracy)        
        acc_by_model.append([average])
        conf_matrix_by_model.append([confusion_matrix_acc])
        
    return acc_by_model, conf_matrix_by_model

def cross_validations_list_models_without_learn(dataframe, splits, shuffle_bool, model_list):
    cv = KFold(n_splits = splits, shuffle=shuffle_bool)
    acc_by_model = []
    conf_matrix_by_model = []
    max_attb = len(list(dataframe))
    depth_range = range(1, max_attb * 2)
            
    for model in model_list:
        
        for depth in depth_range:
          fold_accuracy         = []
          confusion_matrix_acc  = []
          
          for train_fold, test_fold in cv.split(dataframe):
              
            f_test = dataframe.iloc[test_fold]
        
            test_acc = model.score(X = f_test.drop(['Gait_event'], axis=1), 
                                      y = f_test['Gait_event'])
            fold_accuracy.append(test_acc)
            confusion_matrix_acc.append(Squire.confusion_matrix(f_test, model.predict(X = f_test.drop(['Gait_event'], axis=1)), False))
        
        average = sum(fold_accuracy)/len(fold_accuracy)        
        acc_by_model.append([average])
        conf_matrix_by_model.append([confusion_matrix_acc])
        
    return acc_by_model, conf_matrix_by_model

def resume_results(x_test, y_pred, y_test, model, dataset, confusion_matrix):
    results_series = pd.DataFrame(data=[[x_test.shape[0], (y_test == y_pred).sum(), (y_test != y_pred).sum(), model, dataset]],
                               columns=['Total_test_samples', 'Success_classified', 'Failed_classified', 'Model', 'Dataset'])
    
    dict_conf_matrix = {'model':model, 'dataset': dataset, 'conf_matrix':confusion_matrix}
    
    return (results_series, dict_conf_matrix)

def plot_hist_acc_classifiers(results_dataframe, results_conf_matrix, route):
    # Important repo https://github.com/wcipriano/pretty-print-confusion-matrix
    g = sns.factorplot("Dataset", "Accuracy_cv", "Model", data=results_dataframe, kind="bar", 
                       size=6, aspect=2, palette="deep", legend=True)
    plt.savefig("{}/AccuracyHistByDatasetAndModel.png".format(route))

    for dictionary in results_conf_matrix:
        utils.pretty_plot_confusion_matrix(dictionary['conf_matrix'], save_route="{}/{}{}.jpg".format(route, dictionary['dataset'],
                                                                                                       dictionary['model']))
            
    plt.close(g.fig)