#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 09:53:35 2019

@author: alfonso
"""
from modulo_de_funciones_de_soporte import utils, routes

from numpy import corrcoef, transpose, arange
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import sklearn.neighbors
import sklearn.cluster
from sklearn import metrics, neighbors
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.tree import export_graphviz
from sklearn.model_selection import cross_val_score

import keras
from keras.layers import Dense, Dropout, Input
from keras.models import Model,Sequential
from keras.datasets import mnist
from tqdm import tqdm
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import adam

import pydot

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

# A parte, también nos vamos a quitar todas las muestras que tenga asignado el frame 0 porque lo mismo 
# empezaron un poco antes las plantillas que la grabación y de momento no se pueden demostrar como 
# acierto o fallo del sistema

subject_list    = ['sujeto_1', 'sujeto_2', 'sujeto_3']
experiment_list = [['2'],['1', '2'],['1', '2']]

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
    
    # Procesar datos en bruto
    # os.system("python3 ../modulo_de_etiquetado/Stomper.py -c -p {}".format(path))
    # os.system("python3 ../modulo_de_etiquetado/Mixer.py -m -p {}".format(path))
    
    # Cargar datasets de cada experimento por separado
    
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
    random_forest = utils.load_model('clasificador_swing_ground_move.pkl', [])
    
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
    
    
    
def preprocess_data_random_sample_balance(gait_events_inter, gait_events_to_hs, seed = 0):
    
    # Equilibrado de registros mediante aleatoriedad en muestreo.
    np.random.RandomState(seed)
    
    sample_percent = 1 - (len(gait_events_to_hs) / len(gait_events_inter))
    
    _, sampled_dataframe = utils.divide_datasets(gait_events_inter, percentage=sample_percent)
    
    df_sampled = pd.concat([sampled_dataframe, gait_events_to_hs], sort=False)
    
    train, test = utils.divide_datasets(df_sampled, 0.8)
    
    return train, test

def preprocess_data_clustering_balance(gait_events_inter, gait_events_to_hs):
    
    # Equilibrado de registros mediante la ejecución de un algoritmo de 
    # clustering y el uso de los centroides a modo de representantes
    
    gait_events_inter_norm = utils.normalize_data(gait_events_inter.drop(columns=['Timestamp_init', 'Timestamp_end', 'Date_init','Date_end','Source','Subject','experiment','Frame', 'groups', 'Gait_event']))
    X_pca = calculatePCA(gait_events_inter_norm, './Preprocess_')
    
    gait_events_to_hs_norm = utils.normalize_data(gait_events_to_hs.drop(columns=['Timestamp_init', 'Timestamp_end', 'Date_init','Date_end','Source','Subject','experiment','Frame', 'groups', 'Gait_event']))
    gait_events_to_hs_norm['Gait_event'] = gait_events_to_hs.Gait_event.values
    
    centroids, centroids_plus = clustering(gait_events_inter_norm, X_pca, 396)
    
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

def calculatePCA(dataframe):
    #file = pd.read_csv(path, low_memory=False)
    estimator = PCA (n_components = 2)
    X_pca = estimator.fit_transform(dataframe)

    #Print
    print(estimator.explained_variance_ratio_)
    pd.DataFrame(numpy.matrix.transpose(estimator.components_),
    columns=['PC-1', 'PC-2'], index=dataframe.columns)

    #Print
    fig, ax = plt.subplots()

    for i in range(len(X_pca)):
        plt.text(X_pca[i][0], X_pca[i][1], ".")

    plt.xlim(-1, 1.5)
    plt.ylim(-1, 1.5)
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(path_plot+"pca.png")
    
    return X_pca

def clustering(data_norm, X_pca, k):
    
    # 2.1 random inicialization
    centroids, labels, z =  sklearn.cluster.k_means(data_norm, k, init="random" )
    plot_pca(X_pca,labels, "kmeans", 'plots')

    # 2.2 k-means ++
    centroidsplus, labelsplus, zplus =  sklearn.cluster.k_means(data_norm, k, init="k-means++" )
    plot_pca(X_pca, labelsplus, "kmeans++", 'plots')

    # 6. characterization
    n_clusters_ = len(set(labels)) #- (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d' % n_clusters_)
    
    print("Silhouette Coefficient (kmeans): %0.3f"
          % metrics.silhouette_score(data_norm, labels))
    print("Silhouette Coefficient (kmeans++): %0.3f"
      % metrics.silhouette_score(data_norm, labelsplus))

    return centroids, centroidsplus

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

def preprocess_data_math_agrupation_balance(gait_events_inter, gait_events_to_hs):
    
    # Equilibrado de registros mediante la reducción de muestras no deseadas
    # usando diversas funciones de agrupación matemática (Medias, medianas)
    
    compression_index = int(len(gait_events_inter) / len(gait_events_to_hs))
    
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
    
    train, test = utils.divide_datasets(df_combined, 0.8)
    
    return train, test 
            

def preprocess_data_generative_balance(gait_events_inter, gait_events_to_hs, epochs, batch_size):
    
    # Equilibrado de registros mediante datos sinteticos generados por una 
    # red neuronal de tipo GAN
    
    gen, disc = training(train, test, epochs, batch_size)

    pass

def create_generator():
    generator=Sequential()
    generator.add(Dense(units=256,input_dim=100))
    generator.add(LeakyReLU(0.2))
    
    generator.add(Dense(units=512))
    generator.add(LeakyReLU(0.2))
    
    generator.add(Dense(units=1024))
    generator.add(LeakyReLU(0.2))
    
    generator.add(Dense(units=33, activation='tanh'))
    
    generator.compile(loss='binary_crossentropy', optimizer=(adam(lr=0.0002, beta_1=0.5)))
    
    return generator

def create_discriminator():
    discriminator=Sequential()
    discriminator.add(Dense(units=1024,input_dim=33))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
       
    
    discriminator.add(Dense(units=512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
       
    discriminator.add(Dense(units=256))
    discriminator.add(LeakyReLU(0.2))
    
    discriminator.add(Dense(units=1, activation='sigmoid'))
    
    discriminator.compile(loss='binary_crossentropy', optimizer=(adam(lr=0.0002, beta_1=0.5)))
    return discriminator

def create_gan(discriminator, generator):
    discriminator.trainable=False
    gan_input = Input(shape=(100,))
    x = generator(gan_input)
    gan_output= discriminator(x)
    gan= Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan

def training(train, test, epochs=1, batch_size=128):
    
    #Loading the data
    (X_train, y_train, X_test, y_test) = load_data_to_gan(train, test)
    batch_count = X_train.shape[0] / batch_size
    
    # Creating GAN
    generator= create_generator()
    discriminator= create_discriminator()
    gan = create_gan(discriminator, generator)
    
    for e in range(1,epochs+1 ):
        print("Epoch %d" %e)
        for _ in tqdm(range(batch_size)):
        #generate  random noise as an input  to  initialize the  generator
            noise= np.random.normal(0,1, [batch_size, 100])
            
            # Generate fake MNIST images from noised input
            generated_images = generator.predict(noise)
            
            # Get a random set of  real images
            image_batch =X_train[np.random.randint(low=0,high=X_train.shape[0],size=batch_size)]
            
            #Construct different batches of  real and fake data 
            X= np.concatenate([image_batch, generated_images])
            
            # Labels for generated and real data
            y_dis=np.zeros(2*batch_size)
            y_dis[:batch_size]=0.9
            
            #Pre train discriminator on  fake and real data  before starting the gan. 
            discriminator.trainable=True
            discriminator.train_on_batch(X, y_dis)
            
            #Tricking the noised input of the Generator as real data
            noise= np.random.normal(0,1, [batch_size, 100])
            y_gen = np.ones(batch_size)
            
            # During the training of gan, 
            # the weights of discriminator should be fixed. 
            #We can enforce that by setting the trainable flag
            discriminator.trainable=False
            
            #training  the GAN by alternating the training of the Discriminator 
            #and training the chained GAN model with Discriminator’s weights freezed.
            gan.train_on_batch(noise, y_gen)
            
        if e == 1 or e % 20 == 0:
           
            print("HOLA")
            
    return generator, discriminator
            
def load_data_to_gan(train, test):
    features  = train.columns[:32]
    x_train   = train[features]
    y_train   = train['Gait_event']
    
    
    x_test    = test[features]
    y_test    = test['Gait_event']
    
    # convert shape of x_train from (60000, 28, 28) to (60000, 784) 
    # 784 columns per row
    x_train = x_train.values.reshape(len(train), 32)
    return (x_train, y_train.values, x_test.values, y_test.values)


def train_classifiers(gait_events_inter, gait_events_to_hs):
    # Dataframes generated in order to balance the data
    random_sampled_train, random_sampled_test = preprocess_data_random_sample_balance(gait_events_inter, gait_events_to_hs)
    centroids_train, centroids_test, centroids_plus_train, centroids_plus_test = preprocess_data_clustering_balance(gait_events_inter, gait_events_to_hs)
    math_balance_train, math_balance_test = preprocess_data_math_agrupation_balance(gait_events_inter, gait_events_to_hs)
    
    # Training clasiffiers with the variants of datasets
    
    # Random sampled
    random_forest(random_sampled_train, random_sampled_test)
    knaive_bayes(random_sampled_train, random_sampled_test)
    k_nearest_neightbours(random_sampled_train, random_sampled_test)

    # Clustering balanced
    # K-means
    random_forest(centroids_train, centroids_test)
    knaive_bayes(centroids_train, centroids_test)
    k_nearest_neightbours(centroids_train, centroids_test)
    
    # K-means++
    random_forest(centroids_plus_train, centroids_plus_test)
    knaive_bayes(centroids_plus_train, centroids_plus_test)
    k_nearest_neightbours(centroids_plus_train, centroids_plus_test)
    
    # Math function compression
    random_forest(math_balance_train, math_balance_test)
    knaive_bayes(math_balance_train, math_balance_test)
    k_nearest_neightbours(math_balance_train, math_balance_test)

def random_forest(train, test):
  
    features  = train.columns[:32]
    x_train   = train[features]
    y_train   = train['Gait_event']
    
    
    x_test    = test[features]
    y_test    = test['Gait_event']
    
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

def knaive_bayes(train, test):

    features    = train.columns[:32]
    x_train     = train[features]
    y_train     = train['Gait_event']
    
    x_test      = test[features]
    y_test      = test['Gait_event']
    
    gnb = GaussianNB()
    gnb = gnb.fit(x_train, y_train)
    
    y_pred = gnb.predict(x_test)
    print("Number of mislabeled points out of a total %d points : %d" % (x_test.shape[0],(y_test != y_pred).sum()))    

    bnb = BernoulliNB()
    bnb = bnb.fit(x_train, y_train)
    
    y_pred_bnb = bnb.predict(x_test)
    print("Number of mislabeled points out of a total %d points : %d" % (x_test.shape[0],(y_test != y_pred_bnb).sum()))
    
    return bnb
    
def k_nearest_neightbours(train, test, n_neighbors, n_min=1, n_max=30):
    
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
          
    plt.show()
    
    #1. Build the model
    n_neighbors = scores_list_uniform.index(max(scores_list_uniform)) # BEST PARAMETER
    for i, weights in enumerate(['uniform', 'distance']):
        knn = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
        
        if weights == 'uniform':
            uniform_pred = knn.fit(x_train,y_train).predict(x_test)
             
            print("Number of mislabeled points out of a total %d points : %d" % (x_test.shape[0],(y_test != uniform_pred).sum()))    

        else:
            distance_pred = knn.fit(x_train,y_train).predict(x_test)
            print("Number of mislabeled points out of a total %d points : %d" % (x_test.shape[0],(y_test != distance_pred).sum()))    
        