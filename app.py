import os
import streamlit as st
from app_breast_cancer import app_breast_cancer

# Set page configuration
st.set_page_config(page_title="Multiple Disease Prediction", layout="wide", page_icon="🧑‍⚕️")

# Sidebar for navigation
with st.sidebar:
    selected = st.radio(
        'Multiple Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Breast Cancer Prediction'],
        index=2 , # Default to Breast Cancer Prediction
        key="breast_cancer_radio"
        )
    
if selected == 'Breast Cancer Prediction':
    app_breast_cancer()
