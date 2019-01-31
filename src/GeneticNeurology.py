# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras

import numpy as np

def make_population(models_list, file_route):
    # Models format:
    #   [ [ [layers], 
    #       [neurons_per_layer], 
    #       [activation_functions], 
    #       [input_shape, optimizer, loss, metrics], 
    #       [data_train, data_test] 
    #   ] ]

    # Input Example:
    #   [ [ ['Dense', 'Dense'], [40, 20], 
    #       ['tf.nn.relu', 'tf.softmax'], 
    #       [20, 'adam', 'sparse_categorical_crossentropy', 'accuracy'], 
    #       ['data_train', 'labels_train', 'data_test', 'labels_test'] 
    #   ] ]

    file            = open(file_route, 'w')
    counter_model   = 0
    
    file.write("import tensorflow as tf\nimport numpy as np\n\nfrom tensorflow import keras\n\n")

    file.write("# AUXILIAR FUNCTIONS #\n")

    file.write("def individual_evaluation(model, data_test, labels_test):\n\ttest_loss, test_acc = model.evaluate(data_test, labels_test)\n\tprint('Test accuracy:', test_acc)\n\n")
    
    file.write("def randomize_data():\n\tpass")

    file.write("# MODEL'S POPULATION #\n")

    for model in models_list:
        file.write("### MODEL NUMBER - {} ###\n".format(counter_model))

        file.write("model_{} = ([\n".format(counter_model))
        
        for i in range(len(model[0])):
            if i == 0:
                file.write("\t\t\tkeras.layers.{}({}, activation={}, input_shape={}),\n".format(model[0][i], model[1][i], model[2][i], model[3][0]))
            elif i == len(model[0]) - 1:
                file.write("\t\t\tkeras.layers.{}({}, activation={})\n".format(model[0][i], model[1][i], model[2][i]))
            else:
                file.write("\t\t\tkeras.layers.{}({}, activation={}),\n".format(model[0][i], model[1][i], model[2][i]))
        
        file.write("])\n\n")
        
        file.write("model_{}.compile(\n\t\t\t\toptimizer='{}',\n\t\t\t\tloss='{}',\n\t\t\t\tmetrics=['{}'])\n\n".format(counter_model, model[3][1], model[3][2], model[3][3]))

        file.write("model_{}.fit({}, {}, epochs=20)\n".format(counter_model, model[4][0], model[4][1]))

        file.write("individual_evaluation(model_{}, {}, {})\n\n".format(counter_model, model[4][2], model[4][3]))
        
        counter_model += 1   


if __name__ == '__main__':
    a = [[['Dense','Dense'],[80, 40],['tf.nn.relu', 'tf.softmax'], [20, 'adam', 'sparse_categorical_crossentropy', 'accuracy'], ['data_train', 'labels_train', 'data_test', 'labels_test']],
        [['Dense','Dense'],[80, 40],['tf.nn.relu', 'tf.softmax'], [20, 'adam', 'sparse_categorical_crossentropy', 'accuracy'], ['data_train', 'labels_train', 'data_test', 'labels_test']]
        ]
    make_population(a, './models.py')
