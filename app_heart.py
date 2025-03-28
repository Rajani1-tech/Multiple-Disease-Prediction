import streamlit as st
import numpy as np
import pandas as pd
import sqlite3
import json
import matplotlib.pyplot as plt
import seaborn as sns

from utils.data_preprocessing import DataPreprocessor
from models.logistic_regression import LogisticRegression
from utils.model_evaluation import ModelEvaluator

# Load and preprocess data
preprocessor = DataPreprocessor("dataset/heart.csv")
preprocessor.load_data()
preprocessor.normalize_data()
X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.split_data()
preprocessor.save_data(X_train, y_train, X_val, y_val, X_test, y_test)

# Train the model
model = LogisticRegression(learning_rate=0.01, epochs=1000)
model.fit(X_train, y_train)
model.save_model()

def save_user_prediction(email, disease, input_data, result):
    conn = sqlite3.connect('new_user.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO user_predictions (email, disease, input_parameters, prediction_result) VALUES (?, ?, ?, ?)", 
                   (email, disease, json.dumps(input_data), result))
    conn.commit()
    conn.close()

def app_heartdisease(model):
    st.title('Heart Disease Prediction using ML')
    
    email = st.session_state.get('user_email', 'Guest')  # Get logged-in user's email
    
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
        
        # Save data to database
        save_user_prediction(email, "Heart Disease", user_input, heart_diagnosis)
    
    if show_performance:
        st.subheader("Model Performance on Test Data")
        col1, col2 = st.columns([7, 3.27])
        with col1:
            st.image('heart_disease_metrics_vertical.png')
        with col2:
            st.image('heart_disease_confusion_matrix.png')


def show_heart_model_test_result():
    """Display Heart Disease Model Test Results"""
    
    # Explanation about test data
    st.subheader("📊 Why is Test Data Important?")
    st.write(
        "Machine Learning models are trained on historical data, but we need to ensure that they "
        "generalize well to **new, unseen data**. That's why we split the dataset into **training, "
        "validation, and test sets.**"
    )
    
    # Display test data percentage
    test_ratio = 1 - (0.7 + 0.15)  # Assuming train_ratio=0.7, val_ratio=0.15
    st.info(f"🩺 **Test Data Percentage:** {test_ratio * 100:.2f}% of total data.")
    
    st.title("Heart Disease Model Test Results")
    
    # Display model evaluation images and descriptions in separate columns
    col1, col3 = st.columns([7, 3.27])

    with col1:
        # Model performance image
        st.subheader("Model Performance on Test Data")
        st.image('heart_disease_metrics_vertical.png', caption="Heart Disease Model Performance")
        st.write(
            "This figure shows the performance metrics of the Heart Disease model, including accuracy, precision, "
            "recall, and F1-score. These metrics help us assess how well the model predicts the presence or absence "
            "of heart disease."
        )
        
    with col3:
        # Confusion matrix image
        st.subheader("Confusion Matrix")
        st.image('heart_disease_confusion_matrix.png', caption="Confusion Matrix")
        st.write(
            "A confusion matrix helps us evaluate the performance of the classification model. "
            "It shows the counts of true positives (TP), false positives (FP), true negatives (TN), and false negatives (FN)."
        )
        st.write(
    "📌 **Key Terms in Confusion Matrix:**\n"
    "- **True Positives (TP):** Correctly predicted heart disease cases.\n"
    "- **False Positives (FP):** Incorrectly predicted heart disease cases (patients without heart disease predicted to have it).\n"
    "- **True Negatives (TN):** Correctly predicted non-heart disease cases.\n"
    "- **False Negatives (FN):** Incorrectly predicted non-heart disease cases (patients with heart disease predicted not to have it)."
)




    


def show_eda_for_heart_disease():
    """Displays EDA (Exploratory Data Analysis) Results for Heart Disease"""

    st.title("Exploratory Data Analysis (EDA) for Heart Disease")

    # Display Short Description
    st.subheader("📊 What is Exploratory Data Analysis (EDA)?")
    st.write(
        "EDA is an approach to analyzing datasets to summarize their main characteristics, "
        "often with visual methods. It helps us understand the structure of data, uncover "
        "patterns, detect anomalies, and test assumptions before proceeding with modeling."
    )

    st.write(
        "In this section, we present several visualizations that provide insights into the "
        "distribution of data for various heart disease parameters. These plots can help us "
        "understand the relationships between different features and heart disease diagnosis."
    )

    # Display Heart Disease Pie Chart (for diagnosis distribution)
    st.subheader("🫀 Distribution of Heart Disease Diagnoses (Pie Chart)")
    st.image('/home/predator/Desktop/college_project/Multiple-Disease-Prediction/EDA /Heart_Disease_Pie.png', caption="Distribution of Heart Disease Diagnosis (1: Disease Present, 0: Disease Absent)")
    st.write(
        "This pie chart illustrates the proportion of individuals diagnosed with heart disease "
        "versus those without it. A higher percentage of one category over the other may indicate "
        "an imbalance in the dataset that we need to account for when training the model."
    )

    # Display Categorical Data Distribution
    st.subheader("📊 Distribution of Categorical Data")
    st.image('/home/predator/Desktop/college_project/Multiple-Disease-Prediction/EDA /Categorical_data.png', caption="Distribution of Categorical Features (e.g., Chest Pain Type, Fasting Blood Sugar)")
    st.write(
        "This bar plot shows the distribution of categorical features, such as chest pain type and "
        "fasting blood sugar. These features are important for diagnosing heart disease and understanding "
        "how different categories (e.g., chest pain types) are distributed among the individuals."
    )

    # Display Numerical Data Distribution
    st.subheader("📈 Distribution of Numerical Data")
    st.image('/home/predator/Desktop/college_project/Multiple-Disease-Prediction/EDA /Numerical_data.png', caption="Distribution of Numerical Features (e.g., Age, Cholesterol, Blood Pressure)")
    st.write(
        "This plot shows the distribution of numerical features like age, cholesterol levels, and blood pressure. "
        "Understanding the distribution of these features helps in identifying patterns and outliers, which can influence the model's performance."
    )

    # Display Correlation Heatmap
    st.subheader("🔑 Correlation Heatmap of Features")
    st.image('/home/predator/Desktop/college_project/Multiple-Disease-Prediction/EDA /Correlation_Heatmap.png', caption="Correlation Heatmap (Shows how features are related to each other)")
    st.write(
        "The correlation heatmap shows the strength of relationships between different features. "
        "It helps us identify which features are strongly correlated with heart disease diagnosis and with each other. "
        "For example, a high correlation between cholesterol levels and heart disease may suggest that this feature is an important predictor."
    )


   
