# -*- coding: utf-8 -*-

from sklearn.metrics import confusion_matrix, precision_score
from sklearn.metrics import recall_score, f1_score, cohen_kappa_score

from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout

import random
import os

import datetime as dt
import numpy as np
import pandas as pd

import utils as data_utilities
import routes



class Neural_Network_G:

    def __init__(self, n_layers, type_of_model, list_types_per_layer,
                 list_neurons_per_layer, input_size, n_classes,
                 input_parameters, list_activation_function,
                 list_threshold_function, epochs, performance,
                 compile_loss_function, compile_opt_function, batch_size):

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
        self.compile_optimizer_function     = compile_opt_function
        
        self.batch_size                     = batch_size
        
        self.model_id = hash("{}_{}_{}_{}_{}_{}_{}_{}".format(
                self.n_layers, self.type_of_model, self.n_classes,
                born_moment.hour, born_moment.minute, born_moment.second,
                born_moment.day, born_moment.month)) 
        
    def assign_data_randomize(self):
        
        full_data_train = data_utilities.read_dataset(routes.full_data)
        
        data_train_to_fit, data_train_to_test = data_utilities.divide_datasets(
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
        data_norm = data_utilities.clean_and_normalize_data(self.data_train, exclude)
        
        data_train, data_test = data_utilities.divide_datasets(data_norm)
        
        X_train = data_train.loc[:, data_train.columns.difference(exclude)]
        y_train = data_train[exclude]
        
        X_test = data_test.loc[:, data_test.columns.difference(exclude)]
        y_test = data_test[exclude]
        
        # convert list of labels to binary class matrix
        y_train = np_utils.to_categorical(y_train) 
        
        return X_train, y_train, X_test, y_test

    def fit_model(self, exclude):
        
        X_train, y_train, X_test, y_test = self.prepare_data_for_fit(exclude)
        
        if self.epochs <= 17:
            self.neural_model.fit(X_train, y_train, epochs=18, 
                                  batch_size=self.batch_size, 
                                  validation_split=0.1, verbose=2)
            self.epochs += 1
            
        else:
            new_learn_epochs = random.randint(1, 100)
            self.neural_model.fit(X_train, y_train, nb_epoch=new_learn_epochs, batch_size=16, validation_split=0.1, verbose=2)
            
            self.epochs += new_learn_epochs
            
        self.generate_report(X_test, y_test)
        
        if self.epochs % 50 > 1:
            self.neural_model.save("models/keras_model_{}_epoch_{}".format(
                    self.model_id, self.epochs))
        
    def test_model(self):
        pass

    def init_and_test_class(self, exclude):
        contador = 0
        
        print("\---- Asignando particion de los datos al modelo ... ----/")
        self.assign_data_randomize()
        
        print("\---- Componiendo modelo de red neuronal evolutiva ... ----/")
        self.compose_model()
        
        print("\---- Iniciando el entrenamiento del modelo ... ----/")
        self.fit_model(exclude)
        
        while self.performance < 0.25 and contador < 20:
            print("\t/-- Try interaction number {} --/".format(contador))
            
            self.fit_model(exclude)
            contador += self.epochs
        
    def generate_report(self, X_test, y_test):
        
        y_pred = self.neural_model.predict(X_test)
        
        print("Matriz de confusión I:\n")
        
        matrix      = confusion_matrix(y_test, y_pred.argmax(axis=1))
        print(matrix)
        
        
        print("Matriz de confusión II:\n")
       
        matriz      = pd.crosstab(y_test['quality'], y_pred.argmax(axis=1), rownames=['actual'], colnames=['preds'])
        print(matriz)
        
        print("Precision Score:\n")
        
        precision   = precision_score(y_test, y_pred.argmax(axis=1), average=None)
        print(precision)
        
        print("Recall Score:\n")
                
        recall      = recall_score(y_test, y_pred.argmax(axis=1), average=None)
        print(recall)
        
        print("F1 Score:\n")
        
        f1          = f1_score(y_test,y_pred.argmax(axis=1), average=None)
        print(f1)
        
        print("Cohen's kappa Score:\n")

        kappa = cohen_kappa_score(y_test, y_pred.argmax(axis=1))
        print(kappa)
        
        self.calculate_performance(y_test, y_pred, precision_score)
        
    def calculate_performance(self, y_test, y_predict, performance_function):
        performance = eval("performance_function(y_test, y_predict.argmax(axis=1), average=None)")
        accumulate  = 0
        
        for i in range(len(performance)):
            accumulate += performance[i]
            
        self.performance = accumulate/len(performance)
        
        print("performance in this iteration, was set to: {}".format(self.performance))

class Neural_Nature_Manager:
    
    def __init__(self, models_population):
        
        self.population = models_population
    
    def proccess_execution_report():
        pass
    
    def genetic_cross():
        # SELECTION #
        # Cruce
        # Generacion de nuevos individuos
        # Agragacion a la poblacion
        pass
    
    def subjects_selection(self):
        # Calcular Rendimiento de la poblacion
        # Calcular rendimiento de individuo --> Hecho
        selection_probabilities = []
        performance_subjects    = []
        cross_subjects          = []
        
        performance_population  = 0
        cross                   = 999
        
        for model in self.population:
            performance_subjects.append(model.performance)
            performance_population  += model.performance
             
        print(performance_population)
            
        for subject in performance_subjects:
            selection_probabilities.append([subject/performance_population, performance_subjects.index(subject)])

        selection_probabilities.sort()

        for i in range(len(selection_probabilities)):
            if i != 0:
                selection_probabilities[i][0] += selection_probabilities[i-1][0]

        print(selection_probabilities)

        # Generar un aleatorio uniforme de 0 a 1, si es menor o igual a la probabilidad
        # rendimiento_individuo/rendimiento_poblacion, entonces se produce la selección
        for i in range(len(selection_probabilities)):
            rulet = random.uniform(0,1)
            
            for j in range(len(selection_probabilities)):
                
                if selection_probabilities[j-1][0] < rulet and selection_probabilities[j][0] >= rulet and j != 0:
                    cross = selection_probabilities[j][1]
                    
                elif 0 < rulet and selection_probabilities[j][0] > rulet and j == 0:
                    cross = selection_probabilities[j][1]
                    
            print("rulet result: {} | Pairing subject {} with subject {}".format(rulet, i, cross))
            
            while i == cross:
                rulet = random.uniform(0,1)
                print("{} - {} - prob: {}".format(i, cross, rulet))
            
                for j in range(len(selection_probabilities)):
                    
                    if selection_probabilities[j-1][0] < rulet and selection_probabilities[j][0] >= rulet and j != 0:
                        cross = selection_probabilities[j][1]
                        
                    elif 0 < rulet and selection_probabilities[j][0] > rulet and j == 0:
                        cross = selection_probabilities[j][1]
                    
            cross_subjects.append([i, cross])
                    
                    
        print("Subject Pairing List (SPL)\n")
        print(cross_subjects)
                                        
        return cross_subjects
        
# Take a look later

def individual_evaluation(model, data_test, labels_test, number_classes, model_id, route_report, epochs):
	test_loss, test_acc = model.evaluate(data_test, labels_test)
	pre_cls=model.predict_classes(data_test)
	cm1 = confusion_matrix(labels_test, pre_cls)

	true_positive = np.diag(cm1)

	false_positive = []
	for i in range(number_classes):
		false_positive.append(sum(cm1[:,i]) - cm1[i,i])


	false_negative = []
	for i in range(number_classes):
		false_positive.append(sum(cm1[i,:]) - cm1[i,i])


	true_negative = []
	for i in range(number_classes):
		temp = np.delete(cm1, i, 0)
		temp = np.delete(temp, i, 1)  # delete ith column
		true_negative.append(sum(sum(temp)))
	performance_report = 'model_{}, {}, {}'.format(model_id, test_acc, epochs)
	for i in range(number_classes):
		performance_report += ', {}, {}, {}, {}'.format(true_positive[i], true_negative[i], false_positive[i], false_negative[i])

	os.system('echo {} >> {}'.format(performance_report, route_report))
        
        