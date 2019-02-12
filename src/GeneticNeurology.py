# -*- coding: utf-8 -*-

import numpy as np
import datetime as dt

class neural_network_G:    
    from keras.models import Sequential
    from keras.utils import np_utils
    from keras.layers.core import Dense, Activation, Dropout
    
    import pandas as pd
    import numpy as np
        
    import utils as data_utilities
    import routes 
    import random

    def __init__(self, n_layers, type_of_model, list_types_per_layer, 
                 list_neurons_per_layer, input_size, n_classes, 
                 list_activation_function, list_threshold_function, epochs, 
                 performance, compile_loss_function, compile_optimizer_function):

        born_moment = dt.datetime.now()
        
        self.n_layers                       = n_layers
        self.type_of_model                  = type_of_model
        self.list_types_per_layer           = list_types_per_layer
        self.list_neurons_per_layer         = list_neurons_per_layer
        self.input_size                     = input_size
        self.n_classes                      = n_classes
        
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
        
        full_data_train = self.data_utilities.read_dataset(self.routes.data_train)
        
        data_train_to_fit, data_train_to_test = self.data_utilities.divide_datasets(
                                                full_data_train, percentage = 0.67)
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

    def prepare_data_for_fit(self):
        data_train, data_test = self.data_utilities.divide_datasets(self.data_train)
        
        labels      = data_train.ix[:,11].values.astype('int32')
        X_train     = (data_train.ix[:,0:].values).astype('float32')
        X_test      = (self.pd.read_csv('data/test.csv').values).astype('float32')
        
        # convert list of labels to binary class matrix
        y_train = self.np_utils.to_categorical(labels) 
        
        # pre-processing: divide by max and substract mean
        scale = np.max(X_train)
        X_train /= scale
        X_test /= scale
        
        mean = np.std(X_train)
        X_train -= mean
        X_test -= mean
        
        return X_train, y_train, X_test

    def fit_model(self):
        
        X_train, y_train, X_test = self.prepare_data_for_fit()
        
        if self.epochs <= 17:
            self.neural_model.fit(X_train, y_train, nb_epoch=17, batch_size=16, validation_split=0.1, verbose=2)
        
        else:
            new_learn_epochs = self.random.randint(10)
            self.neural_model.fit(X_train, y_train, nb_epoch=new_learn_epochs, batch_size=16, validation_split=0.1, verbose=2)
            
            self.epochs += new_learn_epochs
            
        if self.epochs % 50 == 0:
            self.neural_model.save("keras_model_{}_epoch_{}".format(
                    self.model_id, self.epochs))
        
    def test_model(self):
        pass

    def init_and_test_class(self):
        print("\---- Asignando particion de los datos al modelo ... ----/")
        self.assign_data_randomize()
        
        print("\---- Componiendo modelo de red neuronal evolutiva ... ----/")
        self.compose_model()
        
        print("\---- Iniciando el entrenamiento del modelo ... ----/")
        self.fit_model()
        
        
    
def proccess_execution_report():
    pass

def genetic_cross():
    # SELECTION #
    # Cruce
    # Generacion de nuevos individuos
    # Agragacion a la poblacion

    pass

if __name__ == '__main__':
    neural_birkin = neural_network_G(3, 'Sequential', ['Dense', 'Dense', 'Dense'], 
                    [128, 128], 12, 9, 12, ['relu', 'relu', 'softmax'], 
                    [0.15, 0.15], 17, 0, 'categorical_crossentropy', 'rmsprop')
    
    neural_birkin.init_and_test_class()