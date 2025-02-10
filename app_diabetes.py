import streamlit as st
import pickle

def validation(user_input ):
    if user_input[0] < 0 or user_input[0] > 20:
        return False
    if user_input[1] < 0 or user_input[1] > 200:
        return False
    if user_input[2] < 0 or user_input[2] > 140:
        return False
    if user_input[3] < 0 or user_input[3] > 100:
        return False
    if user_input[4] < 0 or user_input[4] > 800:
        return False
    if user_input[5] < 0.00 or user_input[5] > 70.00:
        return False
    if user_input[6] < 0.000 or user_input[6] > 3.000:
        return False
    if user_input[7] < 0 or user_input[7] > 100:
        return False
    return True

def app_diabetes():
    st.title('Diabetes Prediction using ML')
    if not 'input_valid' in st.session_state:
        st.session_state.input_valid = True

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input('Number of Pregnancies', value=2)
   
    with col2:
        Glucose = st.number_input('Glucose Level ', value = 100 )

    with col3:
        BloodPressure = st.number_input('Blood Pressure value', value= 70)

    with col1:
        SkinThickness = st.number_input('Skin Thickness value', value = 35)

    with col2:
        Insulin = st.number_input('Insulin Level', value= 100)

    with col3:
        BMI = st.number_input('BMI value', value = 32.00)

    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value', value=0.25, step=0.001, format="%.3f")

    with col2:
        Age = st.number_input('Age of the Person', value= 25)

    diab_diagnosis = ''

    test_accuracy = 0.753246
    diabetes_model = pickle.load(open('saved_models/diabetes_model_decision_tree.sav', 'rb'))
   
    
    user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]
    
    st.session_state.input_valid = validation(user_input)

    if not st.session_state.input_valid:
        st.error('Please ensure all values are within their valid ranges before proceeding')

    show_performance = False
    
    if st.button('Diabetes Test Result', disabled=not st.session_state.input_valid):
        
       
        user_input = [float(x) for x in user_input]

        diab_prediction = diabetes_model.predict([user_input])

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic using Decision Tree Model.'
        else:
            diab_diagnosis = 'The person is not diabetic using Decision Tree Model.' 
        
        show_performance = True

    st.success(diab_diagnosis)

    if show_performance:

        st.subheader("Model Performance on Test Data")
        col1, col2 = st.columns([7,3.27])
        with col1:
            st.image('metrics_plot.png')

        with col2:
            st.image('confusion_matrix_custom.png')

