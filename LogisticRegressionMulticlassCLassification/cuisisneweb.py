# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 01:48:54 2023

@author: Sahil Khane

"""

import pandas as pd
import numpy as np
import pickle
import streamlit as st

def ouptput(data):
    
    # loading the model 
    
    with open('C:\\Users\\welcome\\Desktop\\Machine Learning\\LogisticRegressionMulticlassCLassification\\logistic_regression_model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)

        
    prediction = loaded_model.predict(data)
    
    return prediction

def main():
    # setting the title of the web application
    
    st.title('Cuisine Detection')
        
    
    # taking input
    
    
    cusine = st.text_input('Enter the cusine data = ')
        

    
    res = ''
    
    if st.button('Submit'):
        
        ar = np.array([cusine])
        ar = ar.reshape(1, -1)  # Reshape to a 2D array
        res=ouptput(ar)
    
    st.success(res)         
         
         
         
# it can only be runned from comand prompt         
if __name__ =='__main__':
    main()


    
    

