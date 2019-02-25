#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 14:09:42 2019

@author: alfonso
"""
from GeneticNeurology import Neural_Network_G, Neural_Nature_Manager
from camera_sincronization import capture_from_webcam

def test_camera():
    capture_from_webcam()
    
def test_neurology():    
    neural_birkin_1 = Neural_Network_G(3, 'Sequential', ['Dense', 'Dense', 'Dense'],
                             [128, 128], 11, 9, 11, ['relu', 'relu', 'softmax'],
                             [0.15, 0.15], 17, 0, 'categorical_crossentropy', 
                             'rmsprop', 100)
    
    
    neural_birkin_2 = Neural_Network_G(3, 'Sequential', ['Dense', 'Dense', 'Dense'],
                             [128, 128], 11, 9, 11, ['relu', 'relu', 'softmax'],
                             [0.15, 0.15], 17, 0, 'categorical_crossentropy', 
                             'rmsprop', 100)
    
    neural_birkin_3 = Neural_Network_G(3, 'Sequential', ['Dense', 'Dense', 'Dense'],
                             [128, 128], 11, 9, 11, ['relu', 'relu', 'softmax'],
                             [0.15, 0.15], 17, 0, 'categorical_crossentropy', 
                             'rmsprop', 100)
    neural_birkin_4 = Neural_Network_G(3, 'Sequential', ['Dense', 'Dense', 'Dense'],
                             [128, 128], 11, 9, 11, ['relu', 'relu', 'softmax'],
                             [0.15, 0.15], 17, 0, 'categorical_crossentropy', 
                             'rmsprop', 100)
    neural_birkin_5 = Neural_Network_G(3, 'Sequential', ['Dense', 'Dense', 'Dense'],
                             [128, 128], 11, 9, 11, ['relu', 'relu', 'softmax'],
                             [0.15, 0.15], 17, 0, 'categorical_crossentropy', 
                             'rmsprop', 100)
    neural_birkin_6 = Neural_Network_G(3, 'Sequential', ['Dense', 'Dense', 'Dense'],
                             [128, 128], 11, 9, 11, ['relu', 'relu', 'softmax'],
                             [0.15, 0.15], 17, 0, 'categorical_crossentropy', 
                             'rmsprop', 100)
    neural_birkin_7 = Neural_Network_G(3, 'Sequential', ['Dense', 'Dense', 'Dense'],
                             [128, 128], 11, 9, 11, ['relu', 'relu', 'softmax'],
                             [0.15, 0.15], 17, 0, 'categorical_crossentropy', 
                             'rmsprop', 100)
    neural_birkin_8 = Neural_Network_G(3, 'Sequential', ['Dense', 'Dense', 'Dense'],
                             [128, 128], 11, 9, 11, ['relu', 'relu', 'softmax'],
                             [0.15, 0.15], 17, 0, 'categorical_crossentropy', 
                             'rmsprop', 100)
    neural_birkin_9 = Neural_Network_G(3, 'Sequential', ['Dense', 'Dense', 'Dense'],
                             [128, 128], 11, 9, 11, ['relu', 'relu', 'softmax'],
                             [0.15, 0.15], 17, 0, 'categorical_crossentropy', 
                             'rmsprop', 100)
    neural_birkin_10 = Neural_Network_G(3, 'Sequential', ['Dense', 'Dense', 'Dense'],
                             [128, 128], 11, 9, 11, ['relu', 'relu', 'softmax'],
                             [0.15, 0.15], 17, 0, 'categorical_crossentropy', 
                             'rmsprop', 100)
    neural_birkin_11 = Neural_Network_G(3, 'Sequential', ['Dense', 'Dense', 'Dense'],
                             [128, 128], 11, 9, 11, ['relu', 'relu', 'softmax'],
                             [0.15, 0.15], 17, 0, 'categorical_crossentropy', 
                             'rmsprop', 100)
    neural_birkin_12 = Neural_Network_G(3, 'Sequential', ['Dense', 'Dense', 'Dense'],
                             [128, 128], 11, 9, 11, ['relu', 'relu', 'softmax'],
                             [0.15, 0.15], 17, 0, 'categorical_crossentropy', 
                             'rmsprop', 100)
    neural_birkin_13 = Neural_Network_G(3, 'Sequential', ['Dense', 'Dense', 'Dense'],
                             [128, 128], 11, 9, 11, ['relu', 'relu', 'softmax'],
                             [0.15, 0.15], 17, 0, 'categorical_crossentropy', 
                             'rmsprop', 100)

    
    neural_birkin_1.init_and_test_class(['quality'])
    neural_birkin_2.init_and_test_class(['quality'])
    neural_birkin_3.init_and_test_class(['quality'])
    neural_birkin_4.init_and_test_class(['quality'])
    neural_birkin_5.init_and_test_class(['quality'])
    neural_birkin_6.init_and_test_class(['quality'])
    neural_birkin_7.init_and_test_class(['quality'])
    neural_birkin_8.init_and_test_class(['quality'])
    neural_birkin_9.init_and_test_class(['quality'])
    neural_birkin_10.init_and_test_class(['quality'])
    neural_birkin_11.init_and_test_class(['quality'])
    neural_birkin_12.init_and_test_class(['quality'])
    neural_birkin_13.init_and_test_class(['quality'])
    

    umbrella_NEST = Neural_Nature_Manager([neural_birkin_1, neural_birkin_2, 
                                           neural_birkin_3, neural_birkin_4, 
                                           neural_birkin_5, neural_birkin_6,
                                           neural_birkin_7, neural_birkin_8,
                                           neural_birkin_9, neural_birkin_10,
                                           neural_birkin_11, neural_birkin_12,
                                           neural_birkin_13])
    
    umbrella_NEST.subjects_selection()


if __name__ == '__main__':
    test_camera()
