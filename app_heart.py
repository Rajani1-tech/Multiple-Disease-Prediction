
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from utils.data_preprocessing import DataPreprocessor
from models.logistic_regression import LogisticRegression
from utils.model_evaluation import ModelEvaluator



preprocessor = DataPreprocessor("dataset/heart.csv")
preprocessor.load_data()
preprocessor.normalize_data()
X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.split_data()
preprocessor.save_data(X_train, y_train, X_val, y_val, X_test, y_test)

model = LogisticRegression(learning_rate=0.01, epochs =1000)
model.fit(X_train, y_train)
model.save_model()


def app_heartdisease(model):
    st.title('Heart Disease Prediction using ML')
    
    # Input Fields
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input('Age', min_value=1, max_value=120, step=1)
    with col2:
        sex = 1 if st.selectbox('Sex', ['Male', 'Female']) == 'Male' else 0
    with col3:
        cp_options = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic']
        cp = cp_options.index(st.selectbox('Chest Pain Type', cp_options))
    
    with col1:
        trestbps = st.number_input('Resting Blood Pressure', min_value=80, max_value=200, step=1)
    with col2:
        chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=100, max_value=600, step=1)
    with col3:
        fbs = 1 if st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes']) == 'Yes' else 0
    
    with col1:
        restecg_options = ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy']
        restecg = restecg_options.index(st.selectbox('Resting ECG Results', restecg_options))
    
    with col2:
        thalach = st.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=220, step=1)
    with col3:
        exang = 1 if st.selectbox('Exercise Induced Angina', ['No', 'Yes']) == 'Yes' else 0
    
    with col1:
        oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=6.2, step=0.1)
    with col2:
        slope_options = ['Upsloping', 'Flat', 'Downsloping']
        slope = slope_options.index(st.selectbox('Slope of Peak Exercise ST Segment', slope_options))
    
    with col3:
        ca = int(st.selectbox('Major Vessels Colored by Fluoroscopy', ['0', '1', '2', '3', '4']))
    
    with col1:
        thal_options = ['Normal', 'Fixed Defect', 'Reversible Defect']
        thal = thal_options.index(st.selectbox('Thalassemia Type', thal_options))
    
    heart_diagnosis = ''
    show_performance = False
    
    if st.button('Heart Disease Test Result'):
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        heart_prediction = model.predict([user_input])
        heart_diagnosis = 'The person has heart disease' if heart_prediction[0] == 1 else 'The person does not have heart disease'
        st.success(heart_diagnosis)
        show_performance = True
    if show_performance:

        st.subheader("Model Performance on Test Data")
        col1, col2 = st.columns([7,3.27])
        with col1:
            st.image('heart_metrics.png')

        with col2:
            st.image('heart_confusion.png')
  
    
    # if show_performance:
    #     st.subheader("📊 Model Performance on Test Data")

    #     evaluator = ModelEvaluator()
    #     evaluator.load_model()
    #     evaluator.load_test_data()
    #     accuracy, y_pred, conf_matrix, report_df = evaluator.evaluate()

    # # Display accuracy
    #     st.write(f"**✅ Model Accuracy:** {accuracy:.2f}%")

    # # Display classification report
    #     st.write("🔹 **Classification Report:**")
    #     st.dataframe(report_df)

    # # Display confusion matrix
    #     st.write("🔹 **Confusion Matrix:**")
    #     fig, ax = plt.subplots(figsize=(5, 3))
    #     sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
    #             xticklabels=['No Disease', 'Disease'], 
    #             yticklabels=['No Disease', 'Disease'])
    #     plt.xlabel("Predicted")
    #     plt.ylabel("Actual")
    #     st.pyplot(fig)
        