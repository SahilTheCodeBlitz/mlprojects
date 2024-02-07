# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 14:50:47 2023

@author: sahil khane

virtual environment name = multiclassclassification'

"""
import numpy as np
import pandas as pd
import pickle 
import streamlit as st


def output(data):
    
    # loading the model 
    
    with open('C:\\Users\\welcome\\Desktop\\Machine Learning\\LogisticRegressionMulticlassCLassification\\logistic_regression_model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)

        
    prediction = loaded_model.predict(data)
    
    if prediction == 0:
        outputs ='Setosa' 
    elif prediction == 1:
        outputs = 'Versicolor'
      
    else:
        outputs = 'Virginica'
    
    return outputs

def main():
    # setting the title of the web application
    
    st.title('Flower Classifier')
        
    
    # taking input
    
    seplength = st.text_input('Enter the sepal length = ')
    sepwidth = st.text_input('Enter the sepal width = ')
    petlength = st.text_input('Enter the petal length = ')
    petwidth = st.text_input('Enter the petal width = ')
        

    
    res = ''
    
    if st.button('Submit'):
        
        ar = np.array([seplength, sepwidth, petlength, petwidth], dtype=float)  # Convert input to float
        ar = ar.reshape(1, -1)  # Reshape to a 2D array
        res=output(ar)
    
    st.success(res)         
         
         
         
# it can only be runned from comand prompt         
if __name__ =='__main__':
    main()