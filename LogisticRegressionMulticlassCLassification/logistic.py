# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 01:13:39 2023

@author: Sahil Khane
"""

import numpy as mp
import pickle

# checking whether model is working here or not 

# loading the model

data = [[5.1, 3.5, 1.4, 0.2]]
# Load the model using pickle


# Load the machine learning model
with open('C:\\Users\\welcome\\Desktop\\Machine Learning\\LogisticRegressionMulticlassCLassification\\logistic_regression_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

    
prediction = loaded_model.predict(data)

print(prediction )    
    

    
    

