# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 18:22:14 2023

@author: Sahil Khane


"""
# importing some packages

import numpy as np
import pickle  

import streamlit as st


# Loading the model

with open('C:/Users/welcome/Desktop/Machine Learning/SVM - Diabetes Prediction System/diabetesModel.pkl', 'rb') as model_file:
    loaded_svm_model = pickle.load(model_file)

# Load StandardScaler

with open('C:\\Users\\welcome\\Desktop\\Machine Learning\\SVM - Diabetes Prediction System\\standard_scaler.pkl', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

def  prediction(input_data):
    
    input_array_data = np.asarray(input_data)

    # reshaping the data
    reshapeInputData = input_array_data.reshape(1,-1)

    # (1, -1) specify that you want to reshape it into a 2-dimensional array with one row and as many columns as needed to maintain the total number of elements.


    scaleDta =loaded_scaler.transform(reshapeInputData) # using loaded model 

    output = loaded_svm_model.predict(scaleDta) # using loaded model 

     
    print(output)
    
    if output==0:
        
        return 'Person is not having diabetes'
    else:
        return 'person is having diabetes '
        
        
def main():
    
    # setting the title of the web application
    
    st.title('Diabetes Predictor')
    
    # taking input
    
    pregnancy = st.text_input('Number of pregnancies ')
    glucose = st.text_input('Glucose Level ')
    bp = st.text_input('Blood Pressure Value = ')
    skinthickness = st.text_input ('Skin Thickness ')
    insulin = st.text_input('Enter the insulin level = ')
    bmi = st.text_input('Enter body mass index value ')
    dpf = st.text_input('Diabetes pedigree function = ')
    age = st.text_input('Enter your age = ')
    
    # doing prediction
    
    res=''
    
    if st.button('Submit'):
         res=prediction([pregnancy,glucose,bp,skinthickness,insulin
                        ,bmi,dpf,age])
    
    st.success(res)         
         
         
         
# it can only be runned from comand prompt         
if __name__ =='__main__':
    main()
         
    
    
    
    
        
     
