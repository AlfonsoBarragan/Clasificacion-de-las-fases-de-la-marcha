# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras

import numpy as np
import datetime as dt

def make_population(models_list, file_route, route_report, number_classes, route_data_train, route_data_test):
    # Models format:
    #   [ [ [layers], 
    #       [neurons_per_layer], 
    #       [activation_functions], 
    #       [input_shape, optimizer, loss, metrics], 
    #       [data_train, data_test], 
    #       [fitness],
    #       [time_of_live]
    #   ] ]

    # Input Example:
    #   [ [ ['Dense', 'Dense'], [40, 20], 
    #       ['tf.nn.relu', 'tf.softmax'], 
    #       [20, 'adam', 'sparse_categorical_crossentropy', 'accuracy'], 
    #       ['data_train', 'labels_train', 'data_test', 'labels_test'],
    #       [None]
    #       [0]
    #   ] ]

    file            = open(file_route, 'w')
    counter_model   = 0
    
    file.write("import tensorflow as tf\n"
                +"import numpy as np\n"
                +"import pandas as pd\n\n"
                +"from sklearn import preprocessing\n"
                +"from tensorflow import keras\n\n")

    file.write("# AUXILIAR FUNCTIONS #\n")

    file.write("def make_report_header(number_classes, route_report):\n"
                +"\n\treport = open(route_report, 'w')"
                +"\n\treport.write('model_id, accuracy, epochs_at_moment')"
                +"\n\tfor i in range(number_classes):"
                +"\n\t\treport.write(', TP_class{}, TN_class{}, FP_class{}, FN_class{}')"
                +"\n\treport.close()"
                +"\n\n")

    file.write("def individual_evaluation(model, data_test, labels_test, number_classes, model_id, route_report, epochs):"
                +"\n\ttest_loss, test_acc = model.evaluate(data_test, labels_test)"
                +"\n\tpre_cls=model.predict_classes(data_test)"
                +"\n\tcm1 = confusion_matrix(labels_test, pre_cls)\n"
                +"\n\ttrue_positive = np.diag(cm1)\n" 
                +"\n\tfalse_positive = []"
                +"\n\tfor i in range(number_classes):"
                +"\n\t\tfalse_positive.append(sum(cm1[:,i]) - cm1[i,i])\n"
                +"\n\n\tfalse_negative = []"
                +"\n\tfor i in range(number_classes):"
                +"\n\t\tfalse_positive.append(sum(cm1[i,:]) - cm1[i,i])\n"
                +"\n\n\ttrue_negative = []"
                +"\n\tfor i in range(num_classes):"
                +"\n\t\ttemp = np.delete(cm1, i, 0)"
                +"\n\t\ttemp = np.delete(temp, i, 1)  # delete ith column"
                +"\n\t\ttrue_negative.append(sum(sum(temp)))"
                +"\n\tperformance_report = 'model_{}, {}, {}'.format(model_id, test_acc, epochs)"
                +"\n\tfor i in range(number_classes):"
                +"\n\t\tperformance_report += ', {}, {}, {}, {}'.format(true_positive[i], true_negative[i], false_positive[i], false_negative[i])\n"
                +"\n\tos.system('echo {} >> {}'.format(performance_report, route_report))"
                +"\n\n")
    
    file.write("def randomize_data():\n\tpass\n\n")

    file.write("if __name__ == '__main__':")

    file.write("\t# INICIALITATION #\n\n")

    file.write("\tmake_report_header({}, '{}')\n\n".format(number_classes, route_report))

    file.write("\ttrain = pd.read_csv('data/{}.csv').values".format(route_data_train)
                +"\n\ttest  = pd.read_csv('data/{}.csv').values".format(route_data_test))

    file.write("\t# Reshape and normalize training data\n"
                +"\n\ttrainX = train[:, 1:].reshape(train.shape[0],1,28, 28).astype( 'float32' )"
                +"\n\tX_train = trainX / 255.0"
                +"\n\ty_train = train[:,0]"
                +"\n\t# Reshape and normalize test data"
                +"\n\ttestX = test[:,1:].reshape(test.shape[0],1, 28, 28).astype( 'float32' )"
                +"\n\tX_test = testX / 255.0"
                +"\n\ty_test = test[:,0]\n")

    file.write("\tlb = preprocessing.LabelBinarizer()"
                +"\n\tlb = preprocessing.LabelBinarizer()"
                +"\n\ty_train = lb.fit_transform(y_train)"
                +"\n\ty_test = lb.fit_transform(y_test)")

    file.write("\t# MODEL'S POPULATION #\n")

    for model in models_list:
        file.write("\t### MODEL NUMBER - {} ###\n".format(counter_model))

        file.write("\tmodel_{} = keras.Sequential([\n".format(counter_model))
        
        for i in range(len(model[0])):
            if i == 0:
                if model[1][i] == '':
                    file.write("\t\t\t\tkeras.layers.{0}(input_shape=(1, {1},{1})),\n".format(model[0][i], model[3][0]))
                else:    
                    file.write("\t\t\t\tkeras.layers.{}({}, activation={}, input_shape={}),\n".format(model[0][i], model[1][i], model[2][i], model[3][0]))
            elif i == len(model[0]) - 1:
                file.write("\t\t\t\tkeras.layers.{}({}, activation={})\n".format(model[0][i], model[1][i], model[2][i]))
            else:
                file.write("\t\t\t\tkeras.layers.{}({}, activation={}),\n".format(model[0][i], model[1][i], model[2][i]))
        
        file.write("\t])\n\n")
        
        file.write("\tmodel_{}.compile(optimizer='{}',\n\t\t\t\tloss='{}',\n\t\t\t\tmetrics=['{}'])\n\n".format(counter_model, model[3][1], model[3][2], model[3][3]))

        file.write("\tmodel_{}.fit({}, {}, epochs=20)\n".format(counter_model, model[4][0], model[4][1]))

        file.write("\tepochs_model_{} = 20\n".format(counter_model))

        file.write("\tindividual_evaluation(model_{0}, {1}, {2}, {3}, {4}, '{5}', epochs_model_{0})\n\n".format(counter_model, model[4][2], model[4][3], number_classes, counter_model, route_report))
        
        counter_model += 1

def proccess_execution_report():
    pass

def genetic_cross():
    # SELECTION #
    # Cruce
    # Generacion de nuevos individuos
    # Agragacion a la poblacion

    pass

if __name__ == '__main__':
    a = [
         [['Flatten','Dense','Dense'],['', 80, 40],['','tf.nn.relu', 'tf.nn.softmax'], [28, 'adam', 'sparse_categorical_crossentropy', 'accuracy'], ['X_train', 'y_train', 'X_test', 'y_test']],
         [['Flatten','Dense','Dense'],['', 80, 40],['','tf.nn.relu', 'tf.nn.softmax'], [28, 'adam', 'sparse_categorical_crossentropy', 'accuracy'], ['X_train', 'y_train', 'X_test', 'y_test']]
        ]

    moment = dt.datetime.now()
    make_population(a, './models.py', './report_{}_{}_{}_{}.csv'.format(moment.hour, moment.minute, moment.second, moment.microsecond), 10, 'data_train','data_test')
