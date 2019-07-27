#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from modulo_de_funciones_de_soporte import utils, routes
import Squire

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

from pylab import rcParams
import statsmodels.api as sm

import math
import warnings

warnings.filterwarnings("ignore") # specify to ignore warning messages
# Agrupar sensores en base a la organización de los
# musculos plantares
#   -> Se pueden agrupar en tres grupos
#       * Musculos de la eminencia plantar medial
#       * Musculos de la eminencia plantar mediana
#       * Musculos de la eminencia plantar lateral
#   Al ser musculos de la misma naturaleza (flexores)
#   La relación entre los musculos y la presión
#   captada por las plantillas, puede ser interesante

# Equivalencia de sensores a musculos:
#   -> Eminencia plantar medial
#       * M. Flexor hallucis brevis     = Sensores 17, 18, 21 , 22 * 0.3, 23, 25 * 0.75, 26, 27, 29, 30, 32
#       * Mm. interossei                = Sensores 19, 22 * 0.75, 20, 25 * 0.25, 24, 28 * 0.5, 31 * 0.25
#       * M. digiti Minimi              = Sensores 15, 14, 13, 12, 11, 10, 9, 8, 7 * 0.75
#       * M. Flexor digitorum brevis    = Sensores 1, 2, 3, 4, 5, 6, 7 * 0.25

# Hare la agregación esa, y después con el articulo de J.Moreno, aplicaremos los algoritmos
# para generar las Time fuzzy chains a lo largo de toda la pisada

# Posteriormente mediante estas Time fuzzy chains, compararemos el ritmo entre las dos piernas

# Finalmente mediante los análisis anteriores buscaremos daremos el indice de simetria a nivel de pie
# y a nivel de zacanda. Respondiendo a la pregunta de: ¿es asimetrica la marcha del sujeto?

def execution_routine():
    data_list = Squire.load_data_from_exp(routes.subject_list, routes.experiment_list)
    data_list_mirror = group_sensors_by_muscles(data_list)
    
    best_estimator_hs_to = utils.load_model("models/Distance KNN_Centroids++", [], 'Gait_event')
    best_estimator_air_ground = utils.load_model("clasificador_swing_ground_move.pkl", [], 'groups')
    
    dataframe_step_segmentated = step_segmentation(data_list_mirror[0]['data'], best_estimator_hs_to, best_estimator_air_ground)
    df_resume_steps = step_tokenization(dataframe_step_segmentated)
    
    ###############################################
    
    dataset_left = pd.read_csv("../modulo_de_etiquetado/data/sujeto_1/2/data/insoleL_dataset.csv")
    dataset_right = pd.read_csv("../modulo_de_etiquetado/data/sujeto_1/2/data/insoleR_dataset.csv")
    
    dataset_left_normalized = utils.normalize_data(dataset_left.drop(columns=['Timestamp_init', 'Timestamp_end','Date_init','Date_end',
                                                  'Source']))
    
    dataset_right_normalized = utils.normalize_data(dataset_right.drop(columns=['Timestamp_init', 'Timestamp_end','Date_init','Date_end',
                                                  'Source']))
    
    dataset_left_normalized['Timestamp_init'] = dataset_left.Timestamp_init.values
    dataset_left_normalized['Timestamp_end'] = dataset_left.Timestamp_end.values
    dataset_left_normalized['Date_init'] = dataset_left.Date_init.values
    dataset_left_normalized['Date_end'] = dataset_left.Date_end.values
    dataset_left_normalized['Source'] = dataset_left.Date_end.values

    dataset_right_normalized['Timestamp_init'] = dataset_right.Timestamp_init.values
    dataset_right_normalized['Timestamp_end'] = dataset_right.Timestamp_end.values
    dataset_right_normalized['Date_init'] = dataset_right.Date_init.values
    dataset_right_normalized['Date_end'] = dataset_right.Date_end.values
    dataset_right_normalized['Source'] = dataset_right.Date_end.values
    
    best_estimator_hs_to = utils.load_model("models/Distance KNN_Centroids++", [], 'Gait_event')
    best_estimator_air_ground = utils.load_model("clasificador_swing_ground_move.pkl", [], 'groups')
    
    dictionary_subject_1 = [{'subject': 'sujeto_1', 'experiment': 2, 'data':dataset_left_normalized}]
    dictionary_subject_1.append({'subject': 'sujeto_1', 'experiment': 2, 'data':dataset_right_normalized})
    
    data_list_mirror = group_sensors_by_muscles(dictionary_subject_1)
    
    dataframe_steps_l_segmentated = step_segmentation(data_list_mirror[0]['data'], best_estimator_hs_to, best_estimator_air_ground)    
    dataframe_steps_r_segmentated = step_segmentation(data_list_mirror[1]['data'], best_estimator_hs_to, best_estimator_air_ground)
    
    df_resume_steps_l = step_tokenization(dataframe_steps_l_segmentated)
    df_resume_steps_r = step_tokenization(dataframe_steps_r_segmentated)
    
    lexical_statistical_analysis_gait(dataframe_steps_l_segmentated)
    lexical_statistical_analysis_gait(dataframe_steps_r_segmentated)
    
    stomp_index_compute(None, df_resume_steps_l)
    stomp_index_compute(None, df_resume_steps_r)
    
    sym_secuencial_analysis(df_resume_steps_l, df_resume_steps_r)

def group_sensors_by_muscles(data_list):
    # Equivalencia de sensores a musculos:
#   -> Eminencia plantar medial
#       * M. Flexor hallucis brevis     = Sensores 17, 18, 21 , 22 * 0.3, 23, 25 * 0.75, 26, 27, 29, 30, 32
#       * Mm. interossei                = Sensores 19, 22 * 0.75, 20, 25 * 0.25, 24, 28 * 0.5, 31 * 0.25
#       * M. digiti Minimi              = Sensores 15, 14, 13, 12, 11, 10, 9, 8, 7 * 0.75
#       * M. Flexor digitorum brevis    = Sensores 1, 2, 3, 4, 5, 6, 7 * 0.25
#   Haremos una agregación de la lectura de cada sensor, y luego calculamos la media. Algunos sensores se les
#   ha aplicado un coeficiente debido a la superficie del sensor que contacta con el musculo
    
    data_list_mirror = []
    for data in data_list:
        
        data_aux = data['data'].copy()
        
        data_aux['MFHB'] = ((data_aux['Sensor_17'] + data_aux['Sensor_18'] + data_aux['Sensor_21'] +
                            data_aux['Sensor_22'] * 0.25 + data_aux['Sensor_23'] + data_aux['Sensor_25'] * 0.75 +
                            data_aux['Sensor_26'] + data_aux['Sensor_27'] + data_aux['Sensor_29'] + data_aux['Sensor_30'] +
                            data_aux['Sensor_32']) / 11)
        
        data_aux['MINO'] = ((data_aux['Sensor_19'] + data_aux['Sensor_22'] * 0.75 + data_aux['Sensor_20'] +
                            data_aux['Sensor_25'] * 0.25 + data_aux['Sensor_24'] + data_aux['Sensor_28'] * 0.5 +
                            data_aux['Sensor_31'] * 0.25) / 7)
        
        data_aux['MDIM'] = ((data_aux['Sensor_15'] + data_aux['Sensor_14'] + data_aux['Sensor_13'] +
                            data_aux['Sensor_12'] + data_aux['Sensor_12'] + data_aux['Sensor_11'] +
                            data_aux['Sensor_10'] + data_aux['Sensor_9'] + data_aux['Sensor_8'] +
                            data_aux['Sensor_16'] + data_aux['Sensor_7'] * 0.75) / 11)
        
        data_aux['MFDB'] = ((data_aux['Sensor_1'] + data_aux['Sensor_2'] + data_aux['Sensor_3'] +
                            data_aux['Sensor_4'] + data_aux['Sensor_5'] + data_aux['Sensor_6'] +
                            data_aux['Sensor_7'] * 0.25) / 7)
        
        data_list_mirror.append({'subject':data['subject'], 'experiment':data['experiment'], 'data':data_aux})
        
    return data_list_mirror

def step_segmentation(dataframe, best_estimator_hs_to, best_estimator_air_ground):
    dataframe_original  = dataframe.copy()
#    dataframe_original  = dataframe_original.drop(columns=['groups', 'Gait_event'])
    
    dataframe_to_group  = dataframe.drop(columns=['Timestamp_init', 'Timestamp_end','Date_init','Date_end',
                                                  'Source','MFHB','MINO',
                                                  'MDIM','MFDB'])
    
#    dataframe_to_group  = utils.normalize_data(dataframe_to_group)
    
    labels_groups       = best_estimator_air_ground.predict(dataframe_to_group)
    dataframe_to_group['groups'] = labels_groups
    dataframe_original['groups'] = labels_groups
    
    dataframe_to_segmentate = dataframe_to_group[dataframe_to_group.groups > 1]
    dataframe_to_segmentate = dataframe_to_segmentate.drop(columns=['groups'])
    
    labels_hs_to = best_estimator_hs_to.predict(dataframe_to_segmentate)
#    
    dataframe_tokens_step = dataframe_original[dataframe_original.groups > 1].copy()
    dataframe_tokens_step['Gait_event'] = labels_hs_to
        
    dataframes_step_list    = []
    
    dataframes_step_list = automata_segmentativo(dataframe_tokens_step)
    
    return dataframes_step_list

def automata_segmentativo(dataframe):
    states      = ["HEEL_STRIKE", "EVENT_INTERMEDIATE", "TOE_OFF"]
    trans       = [0, 2, 1]
    
    current_state   = "HEEL_STRIKE"
    timestamp_init  = dataframe.iloc[0].Timestamp_init

    dataframes_stride_list    = []
    stride_counter            = 1
    
    for i in range(len(dataframe)):
        if dataframe.iloc[i].Gait_event == trans[0] and current_state == states[0]:
            current_state = states[1]
        
        elif dataframe.iloc[i].Gait_event == trans[1] and current_state == states[1]:
            current_state = states[2]
        
        elif dataframe.iloc[i].Gait_event == trans[0] and current_state == states[2]:
            current_state = states[1]
                
        elif dataframe.iloc[i].Gait_event == trans[2] and current_state == states[1]:
            current_state = states[0]
            timestamp_end = dataframe.iloc[i - 1].Timestamp_end
            
            step_dataframe = dataframe[(dataframe.Timestamp_init >= timestamp_init) & 
                                       (dataframe.Timestamp_end <= timestamp_end)].copy()
            step_dataframe['step'] = stride_counter
            
            dataframes_stride_list.append(step_dataframe)
            
            timestamp_init  = dataframe.iloc[i].Timestamp_init                
            stride_counter    += 1
            
    return dataframes_stride_list
    
def step_tokenization(dataframes_step_list):
    
    dataframe_resume_steps = pd.DataFrame(columns=['Timestamp_init', 'Timestamp_end', 'Stride', 'Mean_MFHB',
                                                   'Mean_MINO', 'Mean_MDIM', 'Mean_MFDB', 'Mean_MFHB_inter',
                                                   'Mean_MINO_inter', 'Mean_MDIM_inter', 'Mean_MFDB_inter','Mean_MFHB_hs',
                                                   'Mean_MINO_hs', 'Mean_MDIM_hs', 'Mean_MFDB_hs','Mean_MFHB_to',
                                                   'Mean_MINO_to', 'Mean_MDIM_to', 'Mean_MFDB_to', 'Median_MFHB',
                                                   'Median_MINO', 'Median_MDIM', 'Median_MFDB', 'Max_MFHB',
                                                   'Max_MINO', 'Max_MDIM', 'Max_MFDB', 'Min_MFHB',
                                                   'Min_MINO', 'Min_MDIM', 'Min_MFDB', 'Std_MFHB',
                                                   'Std_MINO', 'Std_MDIM', 'Std_MFDB', 'Std_MFHB_inter',
                                                   'Std_MINO_inter', 'Std_MDIM_inter', 'Std_MFDB_inter', 'Std_MFHB_hs',
                                                   'Std_MINO_hs', 'Std_MDIM_hs', 'Std_MFDB_hs', 'Std_MFHB_to',
                                                   'Std_MINO_to', 'Std_MDIM_to', 'Std_MFDB_to'])
    counter_index = 0
    
    for dataframe in dataframes_step_list:
        timestamp_init  = dataframe.iloc[0].Timestamp_init
        timestamp_end   = dataframe.iloc[len(dataframe) - 1].Timestamp_end
        step            = dataframe.step.unique()[0]

        mean_MFHB = dataframe['MFHB'].mean()
        mean_MINO = dataframe['MINO'].mean()
        mean_MDIM = dataframe['MDIM'].mean()
        mean_MFDB = dataframe['MFDB'].mean()
        
        mean_MFHB_inter = dataframe[dataframe.Gait_event == 0]['MFHB'].mean()
        mean_MINO_inter = dataframe[dataframe.Gait_event == 0]['MINO'].mean()
        mean_MDIM_inter = dataframe[dataframe.Gait_event == 0]['MDIM'].mean()
        mean_MFDB_inter = dataframe[dataframe.Gait_event == 0]['MFDB'].mean()
    
        mean_MFHB_hs = dataframe[dataframe.Gait_event == 1]['MFHB'].mean()
        mean_MINO_hs = dataframe[dataframe.Gait_event == 1]['MINO'].mean()
        mean_MDIM_hs = dataframe[dataframe.Gait_event == 1]['MDIM'].mean()
        mean_MFDB_hs = dataframe[dataframe.Gait_event == 1]['MFDB'].mean()
        
        mean_MFHB_to = dataframe[dataframe.Gait_event == 2]['MFHB'].mean()
        mean_MINO_to = dataframe[dataframe.Gait_event == 2]['MINO'].mean()
        mean_MDIM_to = dataframe[dataframe.Gait_event == 2]['MDIM'].mean()
        mean_MFDB_to = dataframe[dataframe.Gait_event == 2]['MFDB'].mean()
        
        median_MFHB = dataframe['MFHB'].median()
        median_MINO = dataframe['MINO'].median()
        median_MDIM = dataframe['MDIM'].median()
        median_MFDB = dataframe['MFDB'].median()
        
        max_MFHB = dataframe['MFHB'].max()
        max_MINO = dataframe['MINO'].max()
        max_MDIM = dataframe['MDIM'].max()
        max_MFDB = dataframe['MFDB'].max()
        
        min_MFHB = dataframe['MFHB'].min()
        min_MINO = dataframe['MINO'].min()
        min_MDIM = dataframe['MDIM'].min()
        min_MFDB = dataframe['MFDB'].min()
        
        std_MFHB = dataframe['MFHB'].std()
        std_MINO = dataframe['MINO'].std()
        std_MDIM = dataframe['MDIM'].std()
        std_MFDB = dataframe['MFDB'].std()
        
        std_MFHB_inter = dataframe[dataframe.Gait_event == 0]['MFHB'].std()
        std_MINO_inter = dataframe[dataframe.Gait_event == 0]['MINO'].std()
        std_MDIM_inter = dataframe[dataframe.Gait_event == 0]['MDIM'].std()
        std_MFDB_inter = dataframe[dataframe.Gait_event == 0]['MFDB'].std()
    
        std_MFHB_hs = dataframe[dataframe.Gait_event == 1]['MFHB'].std()
        std_MINO_hs = dataframe[dataframe.Gait_event == 1]['MINO'].std()
        std_MDIM_hs = dataframe[dataframe.Gait_event == 1]['MDIM'].std()
        std_MFDB_hs = dataframe[dataframe.Gait_event == 1]['MFDB'].std()
        
        std_MFHB_to = dataframe[dataframe.Gait_event == 2]['MFHB'].std()
        std_MINO_to = dataframe[dataframe.Gait_event == 2]['MINO'].std()
        std_MDIM_to = dataframe[dataframe.Gait_event == 2]['MDIM'].std()
        std_MFDB_to = dataframe[dataframe.Gait_event == 2]['MFDB'].std()
               
        dataframe_resume_steps.loc[counter_index] = [timestamp_init, timestamp_end, step, mean_MFHB, mean_MINO, mean_MDIM,
                                                     mean_MFDB, mean_MFHB_inter,
                                                     mean_MINO_inter, mean_MDIM_inter, mean_MFDB_inter, mean_MFHB_hs,
                                                     mean_MINO_hs, mean_MDIM_hs, mean_MFDB_hs, mean_MFHB_to,
                                                     mean_MINO_to, mean_MDIM_to, mean_MFDB_to, median_MFHB, 
                                                     median_MINO, median_MDIM, median_MFDB,
                                                     max_MFHB, max_MINO, max_MDIM, max_MFDB, min_MFHB, min_MINO,
                                                     min_MDIM, min_MFDB, std_MFHB, std_MINO, std_MDIM, std_MFDB,
                                                     std_MFHB_inter, std_MINO_inter, std_MDIM_inter, std_MFDB_inter, 
                                                     std_MFHB_hs, std_MINO_hs, std_MDIM_hs, std_MFDB_hs, 
                                                     std_MFHB_to, std_MINO_to, std_MDIM_to, std_MFDB_to]
        
        counter_index += 1
        
    dataframe_resume_steps = dataframe_resume_steps.dropna() 
    
    return dataframe_resume_steps
    
def lexical_statistical_analysis_gait(dataframes_step_list):
    dataframe = pd.concat(dataframes_step_list, sort=False)
    
    mean_MFHB_inter = dataframe[dataframe.Gait_event == 0]['MFHB'].mean()
    mean_MINO_inter = dataframe[dataframe.Gait_event == 0]['MINO'].mean()
    mean_MDIM_inter = dataframe[dataframe.Gait_event == 0]['MDIM'].mean()
    mean_MFDB_inter = dataframe[dataframe.Gait_event == 0]['MFDB'].mean()
    
    std_MFHB_inter = dataframe[dataframe.Gait_event == 0]['MFHB'].std()/math.sqrt(len(dataframe[dataframe.Gait_event == 0]['MFHB']))
    std_MINO_inter = dataframe[dataframe.Gait_event == 0]['MINO'].std()/math.sqrt(len(dataframe[dataframe.Gait_event == 0]['MINO']))
    std_MDIM_inter = dataframe[dataframe.Gait_event == 0]['MDIM'].std()/math.sqrt(len(dataframe[dataframe.Gait_event == 0]['MDIM']))
    std_MFDB_inter = dataframe[dataframe.Gait_event == 0]['MFDB'].std()/math.sqrt(len(dataframe[dataframe.Gait_event == 0]['MFDB']))
    
    conf_interval_MFHB_inter = [mean_MFHB_inter - std_MFHB_inter * 1.96 , mean_MFHB_inter + std_MFHB_inter * 1.96]
    conf_interval_MINO_inter = [mean_MINO_inter - std_MINO_inter * 1.96 , mean_MINO_inter + std_MINO_inter * 1.96]
    conf_interval_MDIM_inter = [mean_MDIM_inter - std_MDIM_inter * 1.96 , mean_MDIM_inter + std_MDIM_inter * 1.96]
    conf_interval_MFDB_inter = [mean_MFDB_inter - std_MFDB_inter * 1.96 , mean_MFDB_inter + std_MFDB_inter * 1.96]
    
    print ("Confidence interval (MFHB Intermediate gait events):", conf_interval_MFHB_inter)
    print ("Confidence interval (MINO Intermediate gait events):", conf_interval_MINO_inter)
    print ("Confidence interval (MDIM Intermediate gait events):", conf_interval_MDIM_inter)
    print ("Confidence interval (MFDB Intermediate gait events):", conf_interval_MFDB_inter)
    
    mean_MFHB_hs = dataframe[dataframe.Gait_event == 1]['MFHB'].mean()
    mean_MINO_hs = dataframe[dataframe.Gait_event == 1]['MINO'].mean()
    mean_MDIM_hs = dataframe[dataframe.Gait_event == 1]['MDIM'].mean()
    mean_MFDB_hs = dataframe[dataframe.Gait_event == 1]['MFDB'].mean()
    
    std_MFHB_hs = dataframe[dataframe.Gait_event == 1]['MFHB'].std()/math.sqrt(len(dataframe[dataframe.Gait_event == 1]['MFHB']))
    std_MINO_hs = dataframe[dataframe.Gait_event == 1]['MINO'].std()/math.sqrt(len(dataframe[dataframe.Gait_event == 1]['MINO']))
    std_MDIM_hs = dataframe[dataframe.Gait_event == 1]['MDIM'].std()/math.sqrt(len(dataframe[dataframe.Gait_event == 1]['MDIM']))
    std_MFDB_hs = dataframe[dataframe.Gait_event == 1]['MFDB'].std()/math.sqrt(len(dataframe[dataframe.Gait_event == 1]['MFDB']))
    
    conf_interval_MFHB_hs = [mean_MFHB_hs - std_MFHB_hs * 1.96 , mean_MFHB_hs + std_MFHB_hs * 1.96]
    conf_interval_MINO_hs = [mean_MINO_hs - std_MINO_hs * 1.96 , mean_MINO_hs + std_MINO_hs * 1.96]
    conf_interval_MDIM_hs = [mean_MDIM_hs - std_MDIM_hs * 1.96 , mean_MDIM_hs + std_MDIM_hs * 1.96]
    conf_interval_MFDB_hs = [mean_MFDB_hs - std_MFDB_hs * 1.96 , mean_MFDB_hs + std_MFDB_hs * 1.96]
    
    print ("Confidence interval (MFHB Heel strike event):", conf_interval_MFHB_hs)
    print ("Confidence interval (MINO Heel strike event):", conf_interval_MINO_hs)
    print ("Confidence interval (MDIM Heel strike event):", conf_interval_MDIM_hs)
    print ("Confidence interval (MFDB Heel strike event):", conf_interval_MFDB_hs)
    
    mean_MFHB_to = dataframe[dataframe.Gait_event == 2]['MFHB'].mean()
    mean_MINO_to = dataframe[dataframe.Gait_event == 2]['MINO'].mean()
    mean_MDIM_to = dataframe[dataframe.Gait_event == 2]['MDIM'].mean()
    mean_MFDB_to = dataframe[dataframe.Gait_event == 2]['MFDB'].mean()
    
    std_MFHB_to = dataframe[dataframe.Gait_event == 2]['MFHB'].std()/math.sqrt(len(dataframe[dataframe.Gait_event == 2]['MFHB']))
    std_MINO_to = dataframe[dataframe.Gait_event == 2]['MINO'].std()/math.sqrt(len(dataframe[dataframe.Gait_event == 2]['MINO']))
    std_MDIM_to = dataframe[dataframe.Gait_event == 2]['MDIM'].std()/math.sqrt(len(dataframe[dataframe.Gait_event == 2]['MDIM']))
    std_MFDB_to = dataframe[dataframe.Gait_event == 2]['MFDB'].std()/math.sqrt(len(dataframe[dataframe.Gait_event == 2]['MFDB']))
    
    conf_interval_MFHB_to = [mean_MFHB_to - std_MFHB_to * 1.96 , mean_MFHB_to + std_MFHB_to * 1.96]
    conf_interval_MINO_to = [mean_MINO_to - std_MINO_to * 1.96 , mean_MINO_to + std_MINO_to * 1.96]
    conf_interval_MDIM_to = [mean_MDIM_to - std_MDIM_to * 1.96 , mean_MDIM_to + std_MDIM_to * 1.96]
    conf_interval_MFDB_to = [mean_MFDB_to - std_MFDB_to * 1.96 , mean_MFDB_to + std_MFDB_to * 1.96]
    
    print ("Confidence interval (MFHB Toe off event):", conf_interval_MFHB_to)
    print ("Confidence interval (MINO Toe off event):", conf_interval_MINO_to)
    print ("Confidence interval (MDIM Toe off event):", conf_interval_MDIM_to)
    print ("Confidence interval (MFDB Toe off event):", conf_interval_MFDB_to)
    
    assimetric_samples_inter = dataframe[dataframe.Gait_event == 0]
    assimetric_samples_inter = assimetric_samples_inter[(assimetric_samples_inter.MFHB + assimetric_samples_inter['MFHB'].std() < conf_interval_MFHB_inter[0]) &
                                                        (assimetric_samples_inter.MFHB - assimetric_samples_inter['MFHB'].std() < conf_interval_MFHB_inter[0]) &
                                                        (assimetric_samples_inter.MFHB + assimetric_samples_inter['MFHB'].std() > conf_interval_MFHB_inter[1]) &
                                                        (assimetric_samples_inter.MFHB - assimetric_samples_inter['MFHB'].std() > conf_interval_MFHB_inter[1]) &
                                                        (assimetric_samples_inter.MINO + assimetric_samples_inter['MINO'].std() < conf_interval_MINO_inter[0]) &
                                                        (assimetric_samples_inter.MINO - assimetric_samples_inter['MINO'].std() < conf_interval_MINO_inter[0]) &
                                                        (assimetric_samples_inter.MINO + assimetric_samples_inter['MINO'].std() > conf_interval_MINO_inter[1]) &
                                                        (assimetric_samples_inter.MINO - assimetric_samples_inter['MINO'].std() > conf_interval_MINO_inter[1]) &
                                                        (assimetric_samples_inter.MDIM + assimetric_samples_inter['MDIM'].std() < conf_interval_MDIM_inter[0]) &
                                                        (assimetric_samples_inter.MDIM - assimetric_samples_inter['MDIM'].std() < conf_interval_MDIM_inter[0]) &
                                                        (assimetric_samples_inter.MDIM + assimetric_samples_inter['MDIM'].std() > conf_interval_MDIM_inter[1]) &
                                                        (assimetric_samples_inter.MDIM - assimetric_samples_inter['MDIM'].std() > conf_interval_MDIM_inter[1]) &
                                                        (assimetric_samples_inter.MFDB + assimetric_samples_inter['MFDB'].std() < conf_interval_MFDB_inter[0]) &
                                                        (assimetric_samples_inter.MFDB - assimetric_samples_inter['MFDB'].std() < conf_interval_MFDB_inter[0]) &
                                                        (assimetric_samples_inter.MFDB + assimetric_samples_inter['MFDB'].std() > conf_interval_MFDB_inter[1]) &
                                                        (assimetric_samples_inter.MFDB - assimetric_samples_inter['MFDB'].std() > conf_interval_MFDB_inter[1])  ]
    
    assimetric_samples_hs = dataframe[dataframe.Gait_event == 1]
    assimetric_samples_hs = assimetric_samples_hs[(assimetric_samples_hs.MFHB + assimetric_samples_hs['MFHB'].std() < conf_interval_MFHB_hs[0]) &
                                                  (assimetric_samples_hs.MFHB - assimetric_samples_hs['MFHB'].std() < conf_interval_MFHB_hs[0]) &
                                                  (assimetric_samples_hs.MFHB + assimetric_samples_hs['MFHB'].std() > conf_interval_MFHB_hs[1]) &
                                                  (assimetric_samples_hs.MFHB - assimetric_samples_hs['MFHB'].std() > conf_interval_MFHB_hs[1]) &
                                                  (assimetric_samples_hs.MINO + assimetric_samples_hs['MINO'].std() < conf_interval_MINO_hs[0]) &
                                                  (assimetric_samples_hs.MINO - assimetric_samples_hs['MINO'].std() < conf_interval_MINO_hs[0]) &
                                                  (assimetric_samples_hs.MINO + assimetric_samples_hs['MINO'].std() > conf_interval_MINO_hs[1]) &
                                                  (assimetric_samples_hs.MINO - assimetric_samples_hs['MINO'].std() > conf_interval_MINO_hs[1]) &
                                                  (assimetric_samples_hs.MDIM + assimetric_samples_hs['MDIM'].std() < conf_interval_MDIM_hs[0]) &
                                                  (assimetric_samples_hs.MDIM - assimetric_samples_hs['MDIM'].std() < conf_interval_MDIM_hs[0]) &
                                                  (assimetric_samples_hs.MDIM + assimetric_samples_hs['MDIM'].std() > conf_interval_MDIM_hs[1]) &
                                                  (assimetric_samples_hs.MDIM - assimetric_samples_hs['MDIM'].std() > conf_interval_MDIM_hs[1]) &
                                                  (assimetric_samples_hs.MFDB + assimetric_samples_hs['MFDB'].std() < conf_interval_MFDB_hs[0]) &
                                                  (assimetric_samples_hs.MFDB - assimetric_samples_hs['MFDB'].std() < conf_interval_MFDB_hs[0]) &
                                                  (assimetric_samples_hs.MFDB + assimetric_samples_hs['MFDB'].std() > conf_interval_MFDB_hs[1]) &
                                                  (assimetric_samples_hs.MFDB - assimetric_samples_hs['MFDB'].std() > conf_interval_MFDB_hs[1])  ]
    
    assimetric_samples_to = dataframe[dataframe.Gait_event == 2]
    assimetric_samples_to = assimetric_samples_to[(assimetric_samples_to.MFHB + assimetric_samples_to['MFHB'].std() < conf_interval_MFHB_to[0]) &
                                                  (assimetric_samples_to.MFHB - assimetric_samples_to['MFHB'].std() < conf_interval_MFHB_to[0]) &
                                                  (assimetric_samples_to.MFHB + assimetric_samples_to['MFHB'].std() > conf_interval_MFHB_to[1]) &
                                                  (assimetric_samples_to.MFHB - assimetric_samples_to['MFHB'].std() > conf_interval_MFHB_to[1]) &
                                                  (assimetric_samples_to.MINO + assimetric_samples_to['MINO'].std() < conf_interval_MINO_to[0]) &
                                                  (assimetric_samples_to.MINO - assimetric_samples_to['MINO'].std() < conf_interval_MINO_to[0]) &
                                                  (assimetric_samples_to.MINO + assimetric_samples_to['MINO'].std() > conf_interval_MINO_to[1]) &
                                                  (assimetric_samples_to.MINO - assimetric_samples_to['MINO'].std() > conf_interval_MINO_to[1]) &
                                                  (assimetric_samples_to.MDIM + assimetric_samples_to['MDIM'].std() < conf_interval_MDIM_to[0]) &
                                                  (assimetric_samples_to.MDIM - assimetric_samples_to['MDIM'].std() < conf_interval_MDIM_to[0]) &
                                                  (assimetric_samples_to.MDIM + assimetric_samples_to['MDIM'].std() > conf_interval_MDIM_to[1]) &
                                                  (assimetric_samples_to.MDIM - assimetric_samples_to['MDIM'].std() > conf_interval_MDIM_to[1]) &
                                                  (assimetric_samples_to.MFDB + assimetric_samples_to['MFDB'].std() < conf_interval_MFDB_to[0]) &
                                                  (assimetric_samples_to.MFDB - assimetric_samples_to['MFDB'].std() < conf_interval_MFDB_to[0]) &
                                                  (assimetric_samples_to.MFDB + assimetric_samples_to['MFDB'].std() > conf_interval_MFDB_to[1]) &
                                                  (assimetric_samples_to.MFDB - assimetric_samples_to['MFDB'].std() > conf_interval_MFDB_to[1])  ]
    
    assimetric_samples = pd.concat([assimetric_samples_inter, assimetric_samples_hs, assimetric_samples_to], sort=False)
    
    return assimetric_samples

def stomp_index_compute(dataframes_assymetry, dataframe_resume_steps):
    
    accumulated = 0
    
    if dataframes_assymetry == None:
        
        accumulated += (dataframe_resume_steps['Mean_MFHB'].mean() + dataframe_resume_steps['Mean_MINO'].mean() + dataframe_resume_steps['Mean_MDIM'].mean() + dataframe_resume_steps['Mean_MFDB'].mean())
    
        return accumulated / 4
    else:        
        for sample in dataframes_assymetry:
            mean_MFHB = dataframe_resume_steps.iloc[sample.step - 1].Mean_MFHB
            mean_MINO = dataframe_resume_steps.iloc[sample.step - 1].Mean_MINO
            mean_MDIM = dataframe_resume_steps.iloc[sample.step - 1].Mean_MDIM
            mean_MFDB = dataframe_resume_steps.iloc[sample.step - 1].Mean_MFDB
            
            accumulated += (mean_MFHB - sample.MFHB + mean_MINO - sample.MINO + mean_MDIM - sample.MDIM + mean_MFDB - sample.MFDB)
            
            
        return accumulated / len(dataframes_assymetry)



def sym_secuencial_analysis(dataframe_steps_l, dataframe_steps_r):
    
    if dataframe_steps_l['Stride'].max() > dataframe_steps_r['Stride'].max():
        steps = dataframe_steps_r.Stride.values
    else:
        steps = dataframe_steps_l.Stride.values
    
    sym_ind = []
    
    for step in range(len(steps)):
        
        dataframe_aux_l = dataframe_steps_l.iloc[step]
        dataframe_aux_r = dataframe_steps_r.iloc[step]
        
        acc = (dataframe_aux_l.Mean_MFHB / dataframe_aux_r.Mean_MFHB) + (dataframe_aux_l.Mean_MINO / dataframe_aux_r.Mean_MINO) + (dataframe_aux_l.Mean_MDIM / dataframe_aux_r.Mean_MDIM) + (dataframe_aux_l.Mean_MFDB / dataframe_aux_r.Mean_MFDB)
        acc = acc / 4
        
        sym_ind.append(acc)
        
    return sum(sym_ind) / len(sym_ind)


def symmetry_index(index_stom, index_secuencial, mean_value):
    
    return math.sqrt((mean_value - index_stom * index_secuencial)**2)
