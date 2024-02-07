# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""
import numpy as np
import pickle  
import sklearn

print('hello')

# lets load the model here

# As we have performed standardization on data so along with loading the model we also have to 
# load the standard scaler parameter 
 
# Load SVM model

with open('C:/Users/welcome/Desktop/Machine Learning/SVM - Diabetes Prediction System/diabetesModel.pkl', 'rb') as model_file:
    loaded_svm_model = pickle.load(model_file)

# Load StandardScaler

with open('C:\\Users\\welcome\\Desktop\\Machine Learning\\SVM - Diabetes Prediction System\\standard_scaler.pkl', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)
    
# checking whether loaded model is working here or not
    
data = (5,166,72,19,175,25.8,0.587,51)

input_array_data = np.asarray(data)

# reshaping the data
reshapeInputData = input_array_data.reshape(1,-1)

# (1, -1) specify that you want to reshape it into a 2-dimensional array with one row and as many columns as needed to maintain the total number of elements.


scaleDta =loaded_scaler.transform(reshapeInputData) # using loaded model 

output = loaded_svm_model.predict(scaleDta) # using loaded model 

if output==0:
    print('Person is not having diabetes')
else:
    print('person is having diabetes ')
    
 