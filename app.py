
import streamlit as st
from streamlit_option_menu import option_menu
from app_heart import app_heartdisease, model

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


if selected == 'Heart Disease Prediction':
    app_heartdisease(model)