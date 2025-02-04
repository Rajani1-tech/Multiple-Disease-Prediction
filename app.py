import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Health Assistant", layout="wide", page_icon="🧑‍⚕️")

# Getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# Load the trained model
heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person'],
                           default_index=0)

# Page Title
st.title('Heart Disease Prediction using ML')

# Columns for Layout
col1, col2, col3 = st.columns(3)

# Input Fields with Validation
with col1:
    age = st.number_input('Age', min_value=1, max_value=120, step=1)
    if age is None:
        st.error("Age is required.")

with col2:
    sex = st.selectbox('Sex', ['Male', 'Female'])
    sex = 1 if sex == 'Male' else 0

with col3:
    cp = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
    cp = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'].index(cp)

with col1:
    trestbps = st.number_input('Resting Blood Pressure', min_value=80, max_value=200, step=1)

with col2:
    chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=100, max_value=600, step=1)

with col3:
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'])
    fbs = 1 if fbs == 'Yes' else 0

with col1:
    restecg = st.selectbox('Resting ECG Results', ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'])
    restecg = ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'].index(restecg)

with col2:
    thalach = st.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=220, step=1)

with col3:
    exang = st.selectbox('Exercise Induced Angina', ['No', 'Yes'])
    exang = 1 if exang == 'Yes' else 0

with col1:
    oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=6.2, step=0.1)

with col2:
    slope = st.selectbox('Slope of Peak Exercise ST Segment', ['Upsloping', 'Flat', 'Downsloping'])
    slope = ['Upsloping', 'Flat', 'Downsloping'].index(slope)

with col3:
    ca = st.selectbox('Major Vessels Colored by Fluoroscopy', ['0', '1', '2', '3', '4'])
    ca = int(ca)

with col1:
    thal = st.selectbox('Thalassemia Type', ['Normal', 'Fixed Defect', 'Reversible Defect'])
    thal = ['Normal', 'Fixed Defect', 'Reversible Defect'].index(thal)

# Prediction Button
heart_diagnosis = ''
if st.button('Heart Disease Test Result'):
    user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    
    # Check if all values are provided
    if None in user_input:
        st.error("All input values must be provided to make a prediction.")
    else:
        # Perform prediction
        heart_prediction = heart_disease_model.predict([user_input])
        heart_diagnosis = 'The person has heart disease' if heart_prediction[0] == 1 else 'The person does not have heart disease'
        st.success(heart_diagnosis)
