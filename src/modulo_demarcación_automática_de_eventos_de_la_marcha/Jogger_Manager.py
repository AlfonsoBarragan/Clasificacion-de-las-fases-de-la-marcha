#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 09:53:35 2019

@author: alfonso
"""
from modulo_de_funciones_de_soporte import utils, routes
import Squire
import Jogger_Court

import numpy as np
import pandas as pd
import random
import os

import matplotlib.pyplot as plt

from scipy.stats import randint as sp_randint

from sklearn import metrics, neighbors
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.tree import export_graphviz

import pydot
import math

# A parte, también nos vamos a quitar todas las muestras que tenga asignado el frame 0 porque lo mismo 
# empezaron un poco antes las plantillas que la grabación y de momento no se pueden demostrar como 
# acierto o fallo del sistema

def execution_routine():
    data_list = Squire.load_data_from_exp(routes.subject_list, routes.experiment_list)
    Squire.clean_data(data_list)
    Squire.calculate_labels(data_list)
    
    data_test = data_list[5]['data'].copy()
    data_list.pop(5)
    
    Squire.get_heel_strikes_and_toe_offs(data_list)
    full_data_combined = Squire.concat_all_the_datasets(data_list)

    gait_events_inter = full_data_combined[(full_data_combined.Gait_event == 0)]
    gait_events_to_hs = full_data_combined[full_data_combined.Gait_event > 0]
    
    gait_events_to_hs_norm = utils.normalize_data(gait_events_to_hs.drop(columns=['Timestamp_init', 'Timestamp_end','Date_init','Date_end','Source','Subject','experiment','Frame','groups', 'Gait_event']))
    gait_events_to_hs_norm['Gait_event'] = gait_events_to_hs.Gait_event.values

    gait_events_to_hs_aux = gait_events_to_hs.drop(columns=['Timestamp_init', 'Timestamp_end','Date_init','Date_end','Source','Subject','experiment','Frame','groups', 'Gait_event'])
    gait_events_to_hs_aux['Gait_event'] = gait_events_to_hs.Gait_event.values

    train, _ = utils.divide_datasets(gait_events_to_hs_aux, 1)

    return train


def preprocess_data_random_sample_balance(gait_events_inter, gait_events_to_hs, seed = 0):
    
    # Equilibrado de registros mediante aleatoriedad en muestreo.
    np.random.RandomState(seed)
    
    sample_percent = 1 - (len(gait_events_to_hs) / len(gait_events_inter))
    
    _, sampled_dataframe = utils.divide_datasets(gait_events_inter, percentage=sample_percent)
    
    df_sampled = pd.concat([sampled_dataframe, gait_events_to_hs], sort=False)
    
    train, test = utils.divide_datasets(df_sampled, 0.8)
    
    train_norm = utils.normalize_data(train.drop(columns=['Timestamp_init', 'Timestamp_end', 'Date_init','Date_end','Source','Subject','experiment','Frame', 'groups', 'Gait_event']))
    test_norm = utils.normalize_data(test.drop(columns=['Timestamp_init', 'Timestamp_end', 'Date_init','Date_end','Source','Subject','experiment','Frame', 'groups', 'Gait_event']))
    
    train_norm['Gait_event']    = train.Gait_event.values
    test_norm['Gait_event']     = test.Gait_event.values
    
    
    return train_norm, test_norm

def preprocess_data_clustering_balance(gait_events_inter, gait_events_to_hs):
    
    # Equilibrado de registros mediante la ejecución de un algoritmo de 
    # clustering y el uso de los centroides a modo de representantes
    
    gait_events_inter_norm = utils.normalize_data(gait_events_inter.drop(columns=['Timestamp_init', 'Timestamp_end', 'Date_init','Date_end','Source','Subject','experiment','Frame', 'groups', 'Gait_event']))
    X_pca = Squire.calculatePCA(gait_events_inter_norm)
    
    gait_events_to_hs_norm = utils.normalize_data(gait_events_to_hs.drop(columns=['Timestamp_init', 'Timestamp_end', 'Date_init','Date_end','Source','Subject','experiment','Frame', 'groups', 'Gait_event']))
    gait_events_to_hs_norm['Gait_event'] = gait_events_to_hs.Gait_event.values
    
    centroids, centroids_plus = Squire.clustering(gait_events_inter_norm, X_pca, 396)
    
    df_centroids = pd.DataFrame(data=centroids, columns=gait_events_inter_norm.columns)
    df_centroids_plus = pd.DataFrame(data=centroids_plus, columns=gait_events_inter_norm.columns)
    
    df_centroids['Gait_event']      = 0
    df_centroids_plus['Gait_event'] = 0
    
    df_train_centroids      = pd.concat([df_centroids, gait_events_to_hs_norm])
    df_train_centroids_plus = pd.concat([df_centroids_plus, gait_events_to_hs_norm])
    
    # clustering_dbscan(gait_events_inter_norm, X_pca)
    train_centroids, test_centroids = utils.divide_datasets(df_train_centroids, 0.8)
    train_centroids_plus, test_centroids_plus = utils.divide_datasets(df_train_centroids_plus, 0.8)
    
    return train_centroids, test_centroids, train_centroids_plus, test_centroids_plus

def preprocess_data_math_agrupation_balance(gait_events_inter, gait_events_to_hs):
    
    # Equilibrado de registros mediante la reducción de muestras no deseadas
    # usando diversas funciones de agrupación matemática (Medias, medianas)
    
    compression_index = int(len(gait_events_inter) / (len(gait_events_to_hs)))
    
    gait_events_inter_norm = utils.normalize_data(gait_events_inter.drop(columns=['Timestamp_init', 'Timestamp_end', 'Date_init','Date_end','Source','Subject','experiment','Frame', 'groups', 'Gait_event']))
    
    gait_events_to_hs_norm = utils.normalize_data(gait_events_to_hs.drop(columns=['Timestamp_init', 'Timestamp_end', 'Date_init','Date_end','Source','Subject','experiment','Frame', 'groups', 'Gait_event']))
    gait_events_to_hs_norm['Gait_event'] = gait_events_to_hs.Gait_event.values
    
    for i in range(0, len(gait_events_inter_norm), compression_index):
        if i == 0:
            sample_accumulated = gait_events_inter_norm.iloc[i]
            df_compressed = pd.DataFrame(data=sample_accumulated, columns=gait_events_to_hs.columns)
            df_compressed = df_compressed.drop(columns=['Timestamp_init', 'Timestamp_end', 'Date_init','Date_end','Source','Subject','experiment','Frame', 'groups', 'Gait_event'])
            
        elif i % compression_index != 0:
            sample_accumulated += gait_events_inter_norm.iloc[i]
        else:
            df_compressed = df_compressed.append(sample_accumulated.copy().apply(lambda x: x / compression_index))
            sample_accumulated = gait_events_inter_norm.iloc[i].copy()
    
    df_compressed['Gait_event'] = 0
    
    df_combined = pd.concat([df_compressed, gait_events_to_hs_norm], sort=False)    
    
    train, _    = utils.divide_datasets(df_combined, 0.8)
    _, test     = preprocess_data_random_sample_balance(gait_events_inter, gait_events_to_hs) 
        
    return train, test 
            
def preprocess_data_nothing(gait_events_inter, gait_events_to_hs):

    gait_events_inter_norm = utils.normalize_data(gait_events_inter.drop(columns=['Timestamp_init', 'Timestamp_end', 'Date_init','Date_end','Source','Subject','experiment','Frame','groups', 'Gait_event']))
    gait_events_to_hs_norm = utils.normalize_data(gait_events_to_hs.drop(columns=['Timestamp_init', 'Timestamp_end','Date_init','Date_end','Source','Subject','experiment','Frame','groups', 'Gait_event']))

    gait_events_inter_norm['Gait_event'] = 0
    gait_events_to_hs_norm['Gait_event'] = gait_events_to_hs.Gait_event.values

    combined_dataframe = pd.concat([gait_events_inter_norm, gait_events_to_hs_norm], sort=False)
    train, test = utils.divide_datasets(combined_dataframe, 0.8)
    
    return train, test
    

def train_classifiers(gait_events_inter, gait_events_to_hs):
    # Dataframes generated in order to balance the data
    random_sampled_train, random_sampled_test = preprocess_data_random_sample_balance(gait_events_inter.copy(), gait_events_to_hs.copy())
    centroids_train, centroids_test, centroids_plus_train, centroids_plus_test = preprocess_data_clustering_balance(gait_events_inter.copy(), gait_events_to_hs.copy())
    math_balance_train, math_balance_test = preprocess_data_math_agrupation_balance(gait_events_inter.copy(), gait_events_to_hs.copy())
    nothing_train, nothing_test = preprocess_data_nothing(gait_events_inter.copy(), gait_events_to_hs.copy())
    
    # Full data test
    data_test = full_data_combined[full_data_combined.groups > 1]
    data_test_norm = utils.normalize_data(data_test.drop(columns=['Timestamp_init', 'Timestamp_end', 'Date_init','Date_end','Source','Subject','experiment','Frame','groups', 'Gait_event']))
    data_test_norm['Gait_event'] = data_test.Gait_event.values
    
    # Dataframe to store results from the classifiers
    df_results = pd.DataFrame(columns=['Total_test_samples', 'Success_classified', 'Failed_classified', 
                                       'Model', 'Dataset', 'Accuracy_Mean'])
    
    # Dictionary for models
    # 'model', 'results', 'dataset', 'model_name'
    models_list = []
    
    # Training clasiffiers with the variants of datasets
    # Random sampled
    rf_rs_class, results_rf_rs                              = random_forest(random_sampled_train, data_test_norm, 'random_sampled')
    dt_rs_class, results_dt_rs                              = decision_tree(random_sampled_train, data_test_norm, 'random_sampled')
    kbg_rs, kbb_rs, results_kbg_rs, results_kbb_rs          = knaive_bayes(random_sampled_train, data_test_norm, 'random_sampled')
    u_knn_rs, d_knn_rs, results_uknn_rs, results_dknn_rs    = k_nearest_neightbours(random_sampled_train, data_test_norm, 'random_sampled')
    
    # Adding the models
    models_list.append({'model_name': 'Random Forest', 'dataset': 'Random Sampled', 'model': rf_rs_class, 'results': results_rf_rs })
    models_list.append({'model_name': 'Decision Tree', 'dataset': 'Random Sampled', 'model': dt_rs_class, 'results': results_dt_rs })
    models_list.append({'model_name': 'Naive Bayes (Gaussian)', 'dataset': 'Random Sampled', 'model': kbg_rs , 'results': results_kbg_rs })
    models_list.append({'model_name': 'Naive Bayes (Complement)', 'dataset': 'Random Sampled', 'model': kbb_rs, 'results': results_kbb_rs})
    models_list.append({'model_name': 'Uniform KNN', 'dataset': 'Random Sampled', 'model': u_knn_rs, 'results': results_uknn_rs})
    models_list.append({'model_name': 'Distance KNN', 'dataset': 'Random Sampled', 'model': d_knn_rs, 'results': results_dknn_rs})
    
    # Clustering balanced
    # K-means
    rf_ct_class, results_rf_ct                              = random_forest(centroids_train, data_test_norm, 'centroids')
    dt_ct_class, results_dt_ct                              = decision_tree(centroids_train, data_test_norm, 'centroids')
    kbg_ct, kbb_ct, results_kbg_ct, results_kbb_ct          = knaive_bayes(centroids_train, data_test_norm, 'centroids')
    u_knn_ct, d_knn_ct, results_uknn_ct, results_dknn_ct    = k_nearest_neightbours(centroids_train, data_test_norm, 'centroids')
    
    # Adding the models
    models_list.append({'model_name': 'Random Forest', 'dataset': 'Centroids', 'model': rf_ct_class, 'results': results_rf_ct})
    models_list.append({'model_name': 'Decision Tree', 'dataset': 'Centroids', 'model': dt_ct_class, 'results': results_dt_ct })
    models_list.append({'model_name': 'Naive Bayes (Gaussian)', 'dataset': 'Centroids', 'model': kbg_ct , 'results': results_kbg_ct})
    models_list.append({'model_name': 'Naive Bayes (Complement)', 'dataset': 'Centroids', 'model': kbb_ct, 'results': results_kbb_ct})
    models_list.append({'model_name': 'Uniform KNN', 'dataset': 'Centroids', 'model': u_knn_ct, 'results': results_uknn_ct})
    models_list.append({'model_name': 'Distance KNN', 'dataset': 'Centroids', 'model': d_knn_ct, 'results': results_dknn_ct})
    
    # K-means++
    rf_ctp_class, results_rf_ctp                                = random_forest(centroids_plus_train, data_test_norm, 'centroids++')
    dt_ctp_class, results_dt_ctp                                = decision_tree(random_sampled_train, data_test_norm, 'centroids++')    
    kbg_ctp, kbb_ctp, results_kbg_ctp, results_kbb_ctp          = knaive_bayes(centroids_plus_train, data_test_norm, 'centroids++')
    u_knn_ctp, d_knn_ctp, results_uknn_ctp, results_dknn_ctp    = k_nearest_neightbours(centroids_plus_train, data_test_norm, 'centroids++')
    
    # Adding the models
    models_list.append({'model_name': 'Random Forest', 'dataset': 'Centroids++', 'model': rf_ctp_class, 'results': results_rf_ctp})
    models_list.append({'model_name': 'Decision Tree', 'dataset': 'Centroids++', 'model': dt_ctp_class, 'results': results_dt_ctp })    
    models_list.append({'model_name': 'Naive Bayes (Gaussian)', 'dataset': 'Centroids++', 'model': kbg_ctp , 'results': results_kbg_ctp})
    models_list.append({'model_name': 'Naive Bayes (Complement)', 'dataset': 'Centroids++', 'model': kbb_ctp, 'results': results_kbb_ctp})
    models_list.append({'model_name': 'Uniform KNN', 'dataset': 'Centroids++', 'model': u_knn_ctp, 'results': results_uknn_ctp})
    models_list.append({'model_name': 'Distance KNN', 'dataset': 'Centroids++', 'model': d_knn_ctp, 'results': results_dknn_ctp})
    
    # Math function compression
    rf_mf_class, results_rf_mf                              = random_forest(math_balance_train, data_test_norm, 'math_function')
    dt_mf_class, results_dt_mf                              = decision_tree(math_balance_train, data_test_norm, 'math_function')    
    kbg_mf, kbb_mf, results_kbg_mf, results_kbb_mf          = knaive_bayes(math_balance_train, data_test_norm, 'math_function')
    u_knn_mf, d_knn_mf, results_uknn_mf, results_dknn_mf    = k_nearest_neightbours(math_balance_train, data_test_norm, 'math_function')
    
    # Adding the models
    models_list.append({'model_name': 'Random Forest', 'dataset': 'Math Function', 'model': rf_mf_class, 'results': results_rf_mf})
    models_list.append({'model_name': 'Decision Tree', 'dataset': 'Math Function', 'model': dt_mf_class, 'results': results_dt_mf})
    models_list.append({'model_name': 'Naive Bayes (Gaussian)', 'dataset': 'Math Function', 'model': kbg_mf , 'results': results_kbg_mf})
    models_list.append({'model_name': 'Naive Bayes (Complement)', 'dataset': 'Math Function', 'model': kbb_mf, 'results': results_kbb_mf})
    models_list.append({'model_name': 'Uniform KNN', 'dataset': 'Math Function', 'model': u_knn_mf, 'results': results_uknn_mf})
    models_list.append({'model_name': 'Distance KNN', 'dataset': 'Math Function', 'model': d_knn_mf, 'results': results_dknn_mf})
    
    # Nothing
    rf_no_class, results_rf_no                              = random_forest(nothing_train, data_test_norm, 'no_balanced')
    dt_no_class, results_dt_no                              = decision_tree(nothing_train, data_test_norm, 'no_balanced')    
    kbg_no, kbb_no, results_kbg_no, results_kbb_no          = knaive_bayes(nothing_train, data_test_norm, 'no_balanced')
    u_knn_no, d_knn_no, results_uknn_no, results_dknn_no    = k_nearest_neightbours(nothing_train, data_test_norm, 'no_balanced')
    
    # Adding the models
    models_list.append({'model_name': 'Random Forest', 'dataset': 'Nothing', 'model': rf_no_class, 'results': results_rf_no})
    models_list.append({'model_name': 'Decision Tree', 'dataset': 'Nothing', 'model': dt_no_class, 'results': results_dt_no})
    models_list.append({'model_name': 'Naive Bayes (Gaussian)', 'dataset': 'Nothing', 'model': kbg_no, 'results': results_kbg_no})
    models_list.append({'model_name': 'Naive Bayes (Complement)', 'dataset': 'Nothing', 'model': kbb_no, 'results': results_kbb_no})
    models_list.append({'model_name': 'Uniform KNN', 'dataset': 'Nothing', 'model': u_knn_no, 'results': results_uknn_no})
    models_list.append({'model_name': 'DistanceKNN', 'dataset': 'Nothing', 'model': d_knn_no, 'results': results_dknn_no})

  # Cross validation full-data
    acc_cross_validation_by_model_rs, conf_matrix_cross_validation_by_model_rs    = Jogger_Court.cross_validations_list_models_without_learn(data_test_norm, 5, True, [rf_rs_class, dt_rs_class, kbg_rs, kbb_rs, u_knn_rs, d_knn_rs])
    acc_cross_validation_by_model_ct, conf_matrix_cross_validation_by_model_ct    = Jogger_Court.cross_validations_list_models_without_learn(data_test_norm, 5, True, [rf_ct_class, dt_ct_class, kbg_ct, kbb_ct, u_knn_ct, d_knn_ct])
    acc_cross_validation_by_model_ctp, conf_matrix_cross_validation_by_model_ctp   = Jogger_Court.cross_validations_list_models_without_learn(data_test_norm, 5, True, [rf_ctp_class, dt_ctp_class, kbg_ctp, kbb_ctp, u_knn_ctp, d_knn_ctp])
    acc_cross_validation_by_model_mf, conf_matrix_cross_validation_by_model_mf    = Jogger_Court.cross_validations_list_models_without_learn(data_test_norm, 5, True, [rf_mf_class, dt_mf_class, kbg_mf, kbb_mf, u_knn_mf, d_knn_mf])
    acc_cross_validation_by_model_no, conf_matrix_cross_validation_by_model_no    = Jogger_Court.cross_validations_list_models_without_learn(data_test_norm, 5, True, [rf_no_class, dt_no_class, kbg_no, kbb_no, u_knn_no, d_knn_no])
    
    acc_by_dataset                      = [acc_cross_validation_by_model_rs, acc_cross_validation_by_model_ct, acc_cross_validation_by_model_ctp,
                                           acc_cross_validation_by_model_mf, acc_cross_validation_by_model_no]
    
    conf_matrix_by_dataset              = [conf_matrix_cross_validation_by_model_rs, conf_matrix_cross_validation_by_model_ct, conf_matrix_cross_validation_by_model_ctp,
                                           conf_matrix_cross_validation_by_model_no]

    df_results = pd.concat([results_rf_rs[0], results_dt_rs[0], results_kbg_rs[0], results_kbb_rs[0], results_uknn_rs[0], results_dknn_rs[0],
                            results_rf_ct[0], results_dt_ct[0], results_kbg_ct[0], results_kbb_ct[0], results_uknn_ct[0], results_dknn_ct[0], 
                            results_rf_ctp[0], results_dt_ctp[0], results_kbg_ctp[0], results_kbb_ctp[0], results_uknn_ctp[0], results_dknn_ctp[0],
                            results_rf_mf[0], results_dt_mf[0], results_kbg_mf[0], results_kbb_mf[0], results_uknn_mf[0], results_dknn_mf[0], 
                            results_rf_no[0], results_dt_no[0], results_kbg_no[0], results_kbb_no[0], results_uknn_no[0], results_dknn_no[0]], sort=False)
    
    list_conf_matrix = [results_rf_rs[1], results_dt_rs[1], results_kbg_rs[1], results_kbb_rs[1], results_uknn_rs[1], results_dknn_rs[1],
                          results_rf_ct[1], results_dt_ct[1], results_kbg_ct[1], results_kbb_ct[1], results_uknn_ct[1], results_dknn_ct[1], 
                        results_rf_ctp[1], results_dt_ctp[1], results_kbg_ctp[1], results_kbb_ctp[1], results_uknn_ctp[1], results_dknn_ctp[1],
                        results_rf_mf[1], results_dt_mf[1], results_kbg_mf[1], results_kbb_mf[1], results_uknn_mf[1], results_dknn_mf[1], 
                        results_rf_no[1], results_dt_no[1], results_kbg_no[1], results_kbb_no[1], results_uknn_no[1], results_dknn_no[1]]
    
    df_results = df_results.reset_index(drop=True)
    df_results['Accuracy_cv'] = 0
    last_index = 0
    
    for list_acc in acc_by_dataset:
        for element in list_acc:
            
            df_results.loc[last_index, 'Accuracy_cv'] = element[0]
            
            last_index += 1
            
    Squire.plot_hist_acc_classifiers(df_results, list_conf_matrix, '{}/ConfusionMatrixTraining'.format(routes.path_plot))
    


def save_classifiers(models_list, test_data, feature_to_predict):
    for model in models_list:
        Squire.save_model(model['model'], test_data, "{}_{}".format(model['model_name'], model['dataset']), routes.models_route, feature_to_predict)
        

def random_forest(train, test, dataset):
  
    features  = train.columns[:32]
    x_train   = train[features]
    y_train   = train['Gait_event']
    
    
    x_test    = test[features]
    y_test    = test['Gait_event']
    
    
    clf_rf = RandomForestClassifier(n_estimators=random.randint(1,20), random_state = 0)
    param_dist = {"max_depth": [None],
              "max_features": sp_randint(1, 32),
              "min_samples_split": sp_randint(2, 95),
              "min_samples_leaf": sp_randint(1, 95),
              "bootstrap": [True, False], 'class_weight':['balanced'],
              "criterion": ["gini", "entropy"]}
    
    random_search = RandomizedSearchCV(clf_rf, scoring= 'f1_micro', 
                                   param_distributions=param_dist, 
                                   n_iter= 80)  
    random_search.fit(x_train, y_train)
    clf_rf = random_search.best_estimator_

#    clf_rf.fit(x_train, y_train) # Construcción del modelo
    
    preds_rf = clf_rf.predict(x_test) # Test del modelo
    
#    report(random_search.cv_results_)    
    
    print("Random Forest: \n" 
          +classification_report(y_true=y_test, y_pred=preds_rf))
    
    # Matriz de confusión
    
    rf_confusion_matrix = Squire.confusion_matrix(test, preds_rf)
 
#    print("Matriz de confusión:\n")
#    matriz = pd.crosstab(test['Gait_event'], preds_rf, rownames=['actual'], colnames=['preds'])
#    print(matriz)
 
    # Variables relevantes
    
    print("Relevancia de variables:\n")
    print(pd.DataFrame({'Indicador': features ,
                  'Relevancia': clf_rf.feature_importances_}),"\n")
    print("Máxima relevancia RF :" , max(clf_rf.feature_importances_), "\n")

    os.system('mkdir Graphics/trees/{}'.format(dataset))
    
    for birch in range(len(clf_rf.estimators_)):
      print(clf_rf.estimators_[birch])
      export_graphviz(clf_rf.estimators_[birch], out_file='tree_from_forest.dot',
                      feature_names=list(test.drop(['Gait_event'], axis=1)),
                      filled=True, rounded=True,
                      special_characters=True, class_names = ['Intermediate_event','Heel Strike', 'Toe off'])
      (graph,) = pydot.graph_from_dot_file('tree_from_forest.dot')
      graph.write_png('Graphics/trees/{}/tree_from_forest_birch_{}.png'.format(dataset, birch))
    
    results_rf = Jogger_Court.resume_results(x_test, preds_rf, y_test, 'RandomForest', dataset, rf_confusion_matrix)
    
    return clf_rf, results_rf

def decision_tree(train, test, dataset):
  
    features  = train.columns[:32]
    x_train   = train[features]
    y_train   = train['Gait_event']
    
    
    x_test    = test[features]
    y_test    = test['Gait_event']
    
    
    clf_rf = DecisionTreeClassifier()
    clf_rf.fit(x_train, y_train) # Construcción del modelo
    
    preds_rf = clf_rf.predict(x_test) # Test del modelo
    
#    report(random_search.cv_results_)    
    
    print("Random Forest: \n" 
          +classification_report(y_true=y_test, y_pred=preds_rf))
    
    # Matriz de confusión
    
    rf_confusion_matrix = Squire.confusion_matrix(test, preds_rf)
 
#    print("Matriz de confusión:\n")
#    matriz = pd.crosstab(test['Gait_event'], preds_rf, rownames=['actual'], colnames=['preds'])
#    print(matriz)
 
    # Variables relevantes
    
    print("Relevancia de variables:\n")
    print(pd.DataFrame({'Indicador': features ,
                  'Relevancia': clf_rf.feature_importances_}),"\n")
    print("Máxima relevancia RF :" , max(clf_rf.feature_importances_), "\n")

    os.system('mkdir Graphics/trees/{}'.format(dataset))
    
    export_graphviz(clf_rf, out_file='tree_from_forest.dot',
                      feature_names=list(test.drop(['Gait_event'], axis=1)),
                      filled=True, rounded=True,
                      special_characters=True, class_names = ['Intermediate_event','Heel Strike', 'Toe off'])
    (graph,) = pydot.graph_from_dot_file('tree_from_forest.dot')
    graph.write_png('Graphics/trees/{}/decision_tree.png'.format(dataset))
    
    results_rf = Jogger_Court.resume_results(x_test, preds_rf, y_test, 'DecisionTree', dataset, rf_confusion_matrix)
    
    return clf_rf, results_rf

def knaive_bayes(train, test, dataset):

    features    = train.columns[:32]
    x_train     = train[features]
    y_train     = train['Gait_event']
        
    x_test      = test[features]
    y_test      = test['Gait_event']
    
    gnb = GaussianNB()
    gnb = gnb.fit(x_train, y_train)
    
    y_pred = gnb.predict(x_test)
    
    gnb_confusion_matrix = Squire.confusion_matrix(test, y_pred)
    print("[Gaussian] Number of mislabeled points out of a total %d points : %d" % (x_test.shape[0],(y_test != y_pred).sum()))  
    
    results_kbg = Jogger_Court.resume_results(x_test, y_pred, y_test, 'GaussianKB', dataset, gnb_confusion_matrix)
    
    cnb = ComplementNB()
    cnb = cnb.fit(x_train, y_train)
    
    y_pred_cnb = cnb.predict(x_test)

    cnb_confusion_matrix = Squire.confusion_matrix(test, y_pred_cnb)
    print("[Complement] Number of mislabeled points out of a total %d points : %d" % (x_test.shape[0],(y_test != y_pred_cnb).sum()))
    
    results_kbc = Jogger_Court.resume_results(x_test, y_pred_cnb, y_test, 'ComplementKB', dataset, cnb_confusion_matrix)
    
    
    return cnb, gnb, results_kbg, results_kbc
    
def k_nearest_neightbours(train, test, dataset):
    
    n_min = 1
    n_max = int(math.sqrt(len(train) + len(test)))
    
    features    = train.columns[:32]
    x_train     = train[features]
    y_train     = train['Gait_event']
    
    x_test      = test[features]
    y_test      = test['Gait_event']
    
    scores = {}
    scores_list_uniform = []
    scores_list_distance = []
        
    for i, weights in enumerate(['uniform', 'distance']):

        for n_neighbors in range(n_min, n_max):
            if weights == 'uniform':
                knn = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
                knn.fit(x_train,y_train)
                y_pred = knn.predict(x_test)
                scores[n_neighbors] = metrics.accuracy_score(y_test, y_pred)
                scores_list_uniform.append(metrics.accuracy_score(y_test, y_pred))
            else:
                knn = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
                knn.fit(x_train,y_train)
                y_pred = knn.predict(x_test)
                scores[n_neighbors] = metrics.accuracy_score(y_test, y_pred)
                scores_list_distance.append(metrics.accuracy_score(y_test, y_pred))
    
    plt.subplot(2, 1, 1)              
    plt.plot(range(n_min, n_max), scores_list_uniform, c='b', label='uniform')
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Testing Accuracy')
    plt.title("KNeighborsClassifier Estimation Accuracy Uniform")
    
    
    plt.subplot(2, 1, 2)
    plt.plot(range(n_min, n_max), scores_list_distance, c='r', label='distance')
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Testing Accuracy')
    plt.title("KNeighborsClassifier Estimation Accuracy Distance")
          
    plt.savefig("Graphics/KNNs/bestKForKNN{}".format(dataset))
    
    n_neighbors_uniform     = scores_list_uniform.index(max(scores_list_uniform)) + 1 # BEST PARAMETER
    n_neighbors_distance    = scores_list_distance.index(max(scores_list_distance)) + 1 # BEST PARAMETER

    for i, weights in enumerate(['uniform', 'distance']):
        
        if weights == 'uniform':
            u_knn = neighbors.KNeighborsClassifier(n_neighbors_uniform, weights=weights)
            u_knn.fit(x_train,y_train)
            uniform_pred = u_knn.predict(x_test)
            
            u_knn_confusion_matrix = Squire.confusion_matrix(test, uniform_pred)
            print("Number of mislabeled points out of a total %d points : %d" % (x_test.shape[0],(y_test != uniform_pred).sum()))    
            
            results_u_knn = Jogger_Court.resume_results(x_test, uniform_pred, y_test, 'UniformKNN', dataset, u_knn_confusion_matrix)
            
        else:
            d_knn = neighbors.KNeighborsClassifier(n_neighbors_distance, weights=weights)
            d_knn.fit(x_train,y_train)
            distance_pred = d_knn.predict(x_test)
            
            d_knn_confusion_matrix = Squire.confusion_matrix(test, distance_pred)
            print("Number of mislabeled points out of a total %d points : %d" % (x_test.shape[0],(y_test != distance_pred).sum()))    
            
            results_d_knn = resume_results(x_test, distance_pred, y_test, 'DistanceKNN', dataset, d_knn_confusion_matrix)
            
    return u_knn, d_knn, results_u_knn, results_d_knn

            