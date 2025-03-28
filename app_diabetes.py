import streamlit as st
import sqlite3
import pickle
import json

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
    if not (0 <= user_input[0] <= 20): return False
    if not (0 <= user_input[1] <= 200): return False
    if not (0 <= user_input[2] <= 140): return False
    if not (0 <= user_input[3] <= 100): return False
    if not (0 <= user_input[4] <= 800): return False
    if not (0.00 <= user_input[5] <= 70.00): return False
    if not (0.000 <= user_input[6] <= 3.000): return False
    if not (0 <= user_input[7] <= 100): return False
    return True

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

    diab_diagnosis = ""

    # Load trained diabetes model
    diabetes_model = pickle.load(open('saved_models/diabetes_model_decision_tree.sav', 'rb'))

    user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                  BMI, DiabetesPedigreeFunction, Age]

    # Validate input
    st.session_state.input_valid = validation(user_input)
    if not st.session_state.input_valid:
        st.error('Please ensure all values are within their valid ranges before proceeding')

    show_performance = False

    if st.button('Diabetes Test Result', disabled=not st.session_state.input_valid):
        user_input = [float(x) for x in user_input]
        diab_prediction = diabetes_model.predict([user_input])

        diab_diagnosis = (
            'The person **is diabetic** using Decision Tree Model.'
            if diab_prediction[0] == 1
            else 'The person **is not diabetic** using Decision Tree Model.'
        )
        st.success(diab_diagnosis)
        show_performance = True

        # Save prediction result in the database
        save_user_prediction(email, "Diabetes", user_input, diab_diagnosis)

    if show_performance:
        st.subheader("Model Performance on Test Data")
        col1, col2 = st.columns([7, 3.27])
        with col1:
            st.image('metrics_plot.png')
        with col2:
            st.image('confusion_matrix_custom.png')
