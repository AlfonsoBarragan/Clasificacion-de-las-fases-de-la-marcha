#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 14:09:42 2019

@author: alfonso
"""
from GeneticNeurology import Neural_Network_G, Neural_Nature_Manager

if __name__ == '__main__':
    neural_birkin_1 = Neural_Network_G(7, 'Sequential', ['Dense', 'Dense', 'Dense', 'Dense', 'Dense', 'Dense', 'Dense'],
                             [128, 120, 100, 50, 25, 10], 11, 9, 11, ['relu', 'relu','relu','relu', 'softmax', 'softmax', 'softmax', 'softmax'],
                             [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.20], 17, 0, 'categorical_crossentropy', 
                             'rmsprop', 100)
    
    
    neural_birkin_2 = Neural_Network_G(3, 'Sequential', ['Dense', 'Dense', 'Dense'],
                             [128, 128], 11, 9, 11, ['relu', 'relu', 'softmax'],
                             [0.15, 0.15], 17, 0, 'categorical_crossentropy', 
                             'rmsprop', 100)
    
    
    neural_birkin_3 = Neural_Network_G(7, 'Sequential', ['Dense', 'Dense', 'Dense', 'Dense', 'Dense', 'Dense', 'Dense'],
                             [128, 128, 128, 128, 64, 32], 11, 9, 11, ['relu', 'relu','relu','relu', 'softmax', 'softmax', 'softmax', 'softmax'],
                             [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.20], 17, 0, 'categorical_crossentropy', 
                             'rmsprop', 100)
    
    neural_birkin_1.init_and_test_class(['quality'])

    neural_birkin_2.init_and_test_class(['quality'])

    neural_birkin_3.init_and_test_class(['quality'])

    umbrella_NEST = Neural_Nature_Manager([neural_birkin_1, neural_birkin_2, 
                                           neural_birkin_3])
