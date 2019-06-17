#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 20:59:50 2019

@author: alfonso
"""
from modulo_de_funciones_de_soporte import utils, routes

import pandas as pd

def load_data_from_exp(subject_list, exp_per_subject):
    
    dataframes_list = []    

    for subject in subject_list:
        for exp in exp_per_subject[subject_list.index(subject)]:            
            path = "../modulo_de_etiquetado/data/{}/{}/{}/{}".format(subject, exp, routes.data_directory, routes.samples_l_with_frames)
            
            df_auxiliar = pd.read_csv(path)
            df_auxiliar.drop(df_auxiliar.columns[[0]], axis=1, inplace=True)
            
            heel_strike_list, toe_off_list = load_frames_labeled_as_list(subject, exp)
            
            dataframes_list.append({'subject':subject, 'experiment': exp, 'data':df_auxiliar, 
                                    'heel_strike_list': heel_strike_list, 'toe_off_list': toe_off_list})             
    
    return dataframes_list

def load_frames_labeled_as_list(subject, experiment):
    path = "../modulo_de_etiquetado/data/{}/{}/{}/{}".format(subject, experiment, routes.data_directory, routes.labeled_frames_route)
    
    frames_labeled_file = open(path, 'r')
    
    line = frames_labeled_file.readline()
    while line:
        line_aux = line.split('->')
        
        if line_aux[0] == 'HS':
            heel_strike_list    = converse_frames_to_list(line_aux[1])
            
        elif line_aux[0] == 'TO':
            toe_off_list        = converse_frames_to_list(line_aux[1])
            
        line = frames_labeled_file.readline()
        
    return heel_strike_list, toe_off_list

def converse_frames_to_list(line_from_frames_labeled_file):
    
    frame_list = []
    for frame in line_from_frames_labeled_file.split(','):
        frame_list.append(int(frame))
        
    return frame_list

def clean_data(data_list):
    
    for data in data_list:
        # Se eliminan los etiquetados con frame 0 y 1 para eliminar muestras no demostrables (de momento)
        data['data'] = data['data'][(data['data'].Frame > 1)]
        
        # Se eliminan los que poseen alguna lectura erronea producto de fallos en la comunicacion bluetooth
        data['data'] = data['data'][(data['data'].Sensor_1 <= 4096)  & (data['data'].Sensor_2 <= 4096)  &
                                    (data['data'].Sensor_3 <= 4096)  & (data['data'].Sensor_4 <= 4096)  &
                                    (data['data'].Sensor_5 <= 4096)  & (data['data'].Sensor_6 <= 4096)  &                                    
                                    (data['data'].Sensor_7 <= 4096)  & (data['data'].Sensor_8 <= 4096)  &
                                    (data['data'].Sensor_9 <= 4096)  & (data['data'].Sensor_10 <= 4096) &                                    
                                    (data['data'].Sensor_11 <= 4096) & (data['data'].Sensor_12 <= 4096) &
                                    (data['data'].Sensor_13 <= 4096) & (data['data'].Sensor_14 <= 4096) &
                                    (data['data'].Sensor_15 <= 4096) & (data['data'].Sensor_16 <= 4096) &
                                    (data['data'].Sensor_17 <= 4096) & (data['data'].Sensor_18 <= 4096) &
                                    (data['data'].Sensor_19 <= 4096) & (data['data'].Sensor_20 <= 4096) &
                                    (data['data'].Sensor_21 <= 4096) & (data['data'].Sensor_22 <= 4096) &
                                    (data['data'].Sensor_23 <= 4096) & (data['data'].Sensor_24 <= 4096) &
                                    (data['data'].Sensor_25 <= 4096) & (data['data'].Sensor_26 <= 4096) &
                                    (data['data'].Sensor_27 <= 4096) & (data['data'].Sensor_28 <= 4096) &
                                    (data['data'].Sensor_29 <= 4096) & (data['data'].Sensor_30 <= 4096) &
                                    (data['data'].Sensor_31 <= 4096) & (data['data'].Sensor_32 <= 4096)]
        
def concat_all_the_datasets(data_list):
    for data_index in range(len(data_list)):
        if data_index == 0:
            df_combined = data_list[data_index]['data'].copy()
        else:
            df_combined = pd.concat([df_combined, data_list[data_index]['data'].copy()], sort=False)
            
    return df_combined

def generate_dataset_to_label(dataframe):
    df_aux = dataframe.drop(columns=['Timestamp_init', 'Timestamp_end', 'Date_init','Date_end','Source','Subject','experiment','Frame'])
    norm_data = utils.normalize_data(df_aux)
    
    return norm_data

def calculate_labels(data_list):
    random_forest = utils.load_model('clasificador_swing_ground_move.pkl', [], 'groups')
    
    # Hacer que el random forest prediga las etiquetas, recortando el dataset para quedarte solo con los
    # valores de presion plantar
    
    for data in data_list:
        df_norm_to_label    = generate_dataset_to_label(data['data'])
        
        # Los significados de las etiquetas serian:
        # -> 1 ---- Pie en el aire
        # -> 2 ---- Pie en pleno midstage
        # -> 1 ---- Pie en contacto con el suelo
        labels = random_forest.predict(df_norm_to_label)
        
        data['data']['groups'] = labels
        
def get_heel_strikes_and_toe_offs(data_list):
    
    for data in data_list:
        data['data']['Gait_event'] = 0
    
        # Los significados de los identificadores de eventos de la marcha serian:
        # -> 1 ---- Heel Strike (golpe de talon)
        # -> 2 ---- Toe off (levantamiento de puntera)
        get_gait_event(data['heel_strike_list'], data['data'], 1)
        get_gait_event(data['toe_off_list'], data['data'], 2)

def get_gait_event(list_gait_event, df_samples, gait_event_identifier):
    
    for frame in list_gait_event:
        df_frames_and_samples = df_samples[(df_samples.groups == 3)]
        df_frames_and_samples['Diff'] = df_frames_and_samples['Frame'].apply(lambda x: abs(x-frame))
        
        sample_selected = df_frames_and_samples[(df_frames_and_samples.Diff == df_frames_and_samples['Diff'].min())]
        
        for i in sample_selected.index:
            df_samples.loc[i, 'Gait_event'] = gait_event_identifier
            
def extract_to_hs_from_data_list(data_list):
    
    full_data_combined = concat_all_the_datasets(data_list)
    
    gait_events_inter = full_data_combined[(full_data_combined.Gait_event == 0) & (full_data_combined.groups == 3)]
    gait_events_to_hs = full_data_combined[full_data_combined.Gait_event > 0]
    
    # Debido a que tenemos muchos más eventos de la marcha que no son HS ni TO
    # deberemos aplicar un preproceso, para reducir el desequilibrio entre las
    # muestras. Se han planteado varias opciones:
    # -> Muestreo aleatorio directo de los eventos intermedios (Gait_event == 0)
    # -> Hacer clustering (K-Means) de entre esas muestras y usar los centroides
    #    como muestras para entrenar los clasificadores en vez de las muestras en si
    # -> Reducir el numero de muestras mediante la agrupación de las mismas a través
    #    de diversas funciones matemáticas (Media, mediana, ...)
    # -> Generando registros sintéticos mediante una red generativa
    # -> No hacer nada, y a ver que pasa
    
    return gait_events_inter, gait_events_to_hs