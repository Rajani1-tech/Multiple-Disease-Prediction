import streamlit as st
from streamlit_option_menu import option_menu
from app_diabetes import app_diabetes
from app_heart import app_heartdisease, model
from app_breast_cancer import app_breast_cancer

st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")


with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',

                           ['Diabetes Prediction', 'Heart Disease Prediction', 'Breast Cancer Prediction'
                            ],
                           menu_icon='hospital-fill',
                            icons=['activity', 'heart', 'person'],
                           default_index=1) 


if selected == 'Diabetes Prediction':
    app_diabetes()    
if selected == 'Heart Disease Prediction':
    app_heartdisease(model)
elif selected =='Breast Cancer Prediction':
    app_breast_cancer()

