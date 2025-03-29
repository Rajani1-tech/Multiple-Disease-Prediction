import streamlit as st
import sqlite3
import pickle
import json
from user import check_recent_predictions

# Function to save predictions in the database
def save_user_prediction(email, disease, input_data, result):
    conn = sqlite3.connect('new_user.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO user_predictions (email, disease, input_parameters, prediction_result) VALUES (?, ?, ?, ?)", 
                   (email, disease, json.dumps(input_data), result))
    conn.commit()
    conn.close()

# Validation function
def validation(user_input):
    return all([
        0 <= user_input[0] <= 20,
        0 <= user_input[1] <= 200,
        0 <= user_input[2] <= 140,
        0 <= user_input[3] <= 100,
        0 <= user_input[4] <= 800,
        0.00 <= user_input[5] <= 70.00,
        0.000 <= user_input[6] <= 3.000,
        0 <= user_input[7] <= 100
    ])

# Diabetes Prediction App
def app_diabetes():
    st.title('Diabetes Prediction using ML')

    if 'input_valid' not in st.session_state:
        st.session_state.input_valid = True

    email = st.session_state.get('user_email', 'Guest')  # Get logged-in user's email

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input('Number of Pregnancies', value=2)
    with col2:
        Glucose = st.number_input('Glucose Level', value=100)
    with col3:
        BloodPressure = st.number_input('Blood Pressure', value=70)
    with col1:
        SkinThickness = st.number_input('Skin Thickness', value=35)
    with col2:
        Insulin = st.number_input('Insulin Level', value=100)
    with col3:
        BMI = st.number_input('BMI', value=32.00)
    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', value=0.25, step=0.001, format="%.3f")
    with col2:
        Age = st.number_input('Age', value=25)

    # Load trained diabetes model
    diabetes_model = pickle.load(open('saved_models/diabetes_model_decision_tree.sav', 'rb'))

    user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                  BMI, DiabetesPedigreeFunction, Age]

    # Validate input
    st.session_state.input_valid = validation(user_input)
    if not st.session_state.input_valid:
        st.error('Please ensure all values are within their valid ranges before proceeding')

    if st.button('Diabetes Test Result', disabled=not st.session_state.input_valid):
        user_input = [float(x) for x in user_input]
        diab_prediction = diabetes_model.predict([user_input])

        diab_diagnosis = (
            'The person **is diabetic** using Decision Tree Model.'
            if diab_prediction[0] == 1
            else 'The person **is not diabetic** using Decision Tree Model.'
        )
        st.success(diab_diagnosis)
          # Save prediction result in the database
        save_user_prediction(email, "Diabetes", user_input, int(diab_prediction[0]))
         # Check if user has predicted Diabetes 3 times within the last 30 days
    if check_recent_predictions(email, 'Diabetes'):
        st.warning("⚠️ **Alert:** You have predicted **Diabetes** 3 or more times in the last 30 days with a positive result. Please consult a doctor.")


