# -*- coding: utf-8 -*-

import numpy as np
import datetime as dt

from sklearn.metrics import classification_report, confusion_matrix


class neural_network_G:    
    from keras.models import Sequential
    from keras.utils import np_utils
    from keras.layers.core import Dense, Activation, Dropout
    
    from sklearn.metrics import classification_report, confusion_matrix
    
    import pandas as pd
    import numpy as np
        
    import utils as data_utilities
    import routes 
    import random

    def __init__(self, n_layers, type_of_model, list_types_per_layer, 
                 list_neurons_per_layer, input_size, n_classes, 
                 input_parameters, list_activation_function, 
                 list_threshold_function, epochs, performance,
                 compile_loss_function, compile_optimizer_function):

        born_moment = dt.datetime.now()
        
        self.n_layers                       = n_layers
        self.type_of_model                  = type_of_model
        self.list_types_per_layer           = list_types_per_layer
        self.list_neurons_per_layer         = list_neurons_per_layer
        self.input_size                     = input_size
        self.n_classes                      = n_classes
        
        self.input_parameters               = input_parameters
        
        self.list_activation_function       = list_activation_function
        self.list_threshold_function        = list_threshold_function
        
        self.epochs                         = epochs
        self.performance                    = performance
        
        self.compile_loss_function          = compile_loss_function
        self.compile_optimizer_function     = compile_optimizer_function
        
        self.model_id = hash("{}_{}_{}_{}_{}_{}_{}_{}".format(
                self.n_layers, self.type_of_model, self.n_classes,
                born_moment.hour, born_moment.minute, born_moment.second,
                born_moment.day, born_moment.month)) 
        
    def assign_data_randomize(self):
        
        full_data_train = self.data_utilities.read_dataset(self.routes.full_data, ";")
        
        data_train_to_fit, data_train_to_test = self.data_utilities.divide_datasets(
                                                full_data_train, percentage = 0.87)
        self.data_train                       = data_train_to_fit
        self.data_test                        = data_train_to_test
        
    def compose_model(self):    
        
        self.neural_model = eval("{}()".format(self.type_of_model))
        
        for i in range(self.n_layers):
            if i == 0:
                eval("self.neural_model.add({}({}, input_dim={}))".format(
                        self.list_types_per_layer[i], 
                        self.list_neurons_per_layer[i],
                        self.input_size))
                
                eval("self.neural_model.add(Activation('{}'))".format(
                        self.list_activation_function[i]))
                
                eval("self.neural_model.add(Dropout({}))".format(
                        self.list_threshold_function[i]))
                
            elif i == self.n_layers - 1:
                eval("self.neural_model.add({}({}))".format(
                        self.list_types_per_layer[i],
                        self.n_classes))
                
                eval("self.neural_model.add(Activation('{}'))".format(
                        self.list_activation_function[i]))
                
            else:
                eval("self.neural_model.add({}({}))".format(
                        self.list_types_per_layer[i], 
                        self.list_neurons_per_layer[i]))
                
                eval("self.neural_model.add(Activation('{}'))".format(
                        self.list_activation_function[i]))
                
                eval("self.neural_model.add(Dropout({}))".format(
                        self.list_threshold_function[i]))
                
            eval("self.neural_model.compile(loss='{}', optimizer='{}')".format(
                        self.compile_loss_function,
                        self.compile_optimizer_function))

    def prepare_data_for_fit(self, exclude):
        data_norm = self.data_utilities.clean_and_normalize_data(self.data_train, exclude)
        
        data_train, data_test = self.data_utilities.divide_datasets(data_norm)
        
        X_train = data_train.loc[:, data_train.columns.difference(exclude)]
        y_train = data_train[exclude]
        
        X_test = data_test.loc[:, data_test.columns.difference(exclude)]
        y_test = data_test[exclude]
        
        # convert list of labels to binary class matrix
        y_train = self.np_utils.to_categorical(y_train) 
        
        return X_train, y_train, X_test, y_test

    def fit_model(self, exclude):
        
        X_train, y_train, X_test, y_test = self.prepare_data_for_fit(exclude)
        
        if self.epochs <= 17:
            self.neural_model.fit(X_train, y_train, nb_epoch=17, batch_size=16, validation_split=0.1, verbose=2)
        
        else:
            new_learn_epochs = self.random.randint(10)
            self.neural_model.fit(X_train, y_train, nb_epoch=new_learn_epochs, batch_size=16, validation_split=0.1, verbose=2)
            
            self.epochs += new_learn_epochs
            
        self.generate_report(X_test, y_test)
        
        if self.epochs % 50 == 0:
            self.neural_model.save("keras_model_{}_epoch_{}".format(
                    self.model_id, self.epochs))
        
    def test_model(self):
        pass

    def init_and_test_class(self, exclude):
        print("\---- Asignando particion de los datos al modelo ... ----/")
        self.assign_data_randomize()
        
        print("\---- Componiendo modelo de red neuronal evolutiva ... ----/")
        self.compose_model()
        
        print("\---- Iniciando el entrenamiento del modelo ... ----/")
        self.fit_model(exclude)
        
    def generate_report(self, X_test, y_test):
        
        y_pred = self.neural_model.predict(X_test)
        
        print("Matriz de confusión I:\n")
        
        matrix = confusion_matrix(y_test, y_pred.argmax(axis=1))
        print(matrix)
        
        
        print("Matriz de confusión II:\n")
       
        matriz = self.pd.crosstab(y_test['quality'], y_pred.argmax(axis=1), rownames=['actual'], colnames=['preds'])
        print(matriz)
    
    def proccess_execution_report():
        pass
    
    def genetic_cross():
        # SELECTION #
        # Cruce
        # Generacion de nuevos individuos
        # Agragacion a la poblacion
        pass

