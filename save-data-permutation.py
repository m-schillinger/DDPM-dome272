#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 18:31:24 2023

@author: mschillinger
"""

# Step 1
import pickle
import numpy as np
 
dataset_size = 10000
data_permutation = np.random.permutation(np.arange(0, dataset_size,1))
# Step 2
with open('data_permutation', 'wb') as data_permutation_file:
    pickle.dump(data_permutation, data_permutation_file)

# Step 2
with open('data_permutation', 'rb') as data_permutation_file: 
    data_permutation = pickle.load(data_permutation_file)
print(data_permutation)
