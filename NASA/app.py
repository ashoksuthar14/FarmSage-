import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load your trained model
model_path = r'C:\Users\ashok\OneDrive\Desktop\NASA\RandomForest (1).pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Title of your web app
st.title('Crop Recommendation System')
# User inputs
N = st.number_input('Nitrogen', min_value=0, value=0)
P = st.number_input('Phosphorus', min_value=0, value=0)
K = st.number_input('Potassium', min_value=0, value=0)
temperature = st.number_input('Temperature', value=25.0)
humidity = st.number_input('Humidity', value=50.0)
ph = st.number_input('pH Level', value=7.0)
rainfall = st.number_input('Rainfall', value=100.0)

# Button to make prediction
if st.button('Predict Crop'):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)
    st.write(f'The recommended crop is **{prediction[0]}**.')
