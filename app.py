import os
import pickle
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Health Assistant", layout="wide", page_icon="🧑‍⚕️")

# Getting the working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Load dataset (Ensure column names match training data)
dataset_path = "/home/predator/Desktop/college_project/Multiple-Disease-Prediction/dataset/heart.csv"
df = pd.read_csv(dataset_path)

# Convert column names to lowercase to prevent feature mismatch errors
df.columns = df.columns.str.lower()

# Splitting dataset into features (X) and target (y)
X = df.drop(columns=['target'])  # Features
y = df['target']  # Target

# Split into train (70%), validation (15%), test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Load the trained model
model_path = f'{working_dir}/saved_models/heart_disease_model.sav'
heart_disease_model = pickle.load(open(model_path, 'rb'))

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person'],
                           default_index=1)  # Set default to Heart Disease

# Page Title
st.title('Heart Disease Prediction using ML')

# Columns for Layout
col1, col2, col3 = st.columns(3)

# Input Fields
with col1:
    age = st.number_input('Age', min_value=1, max_value=120, step=1)

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
    
    # Ensure all inputs are provided
    if None in user_input:
        st.error("All input values must be provided to make a prediction.")
    else:
        # Perform prediction
        heart_prediction = heart_disease_model.predict([user_input])
        heart_diagnosis = 'The person has heart disease' if heart_prediction[0] == 1 else 'The person does not have heart disease'
        st.success(heart_diagnosis)

# ----- Model Performance Visualization -----
st.subheader("📊 Model Performance on Test Data")

# Predictions on test data
y_pred = heart_disease_model.predict(X_test)

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
st.write(f"**✅ Model Accuracy:** {accuracy:.2f}")

# Classification Report
st.write("🔹 **Classification Report:**")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
st.write("🔹 **Confusion Matrix:**")
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot Confusion Matrix
fig, ax = plt.subplots(figsize=(5, 3))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

# Feature Importance (if using tree-based models)
if hasattr(heart_disease_model, "feature_importances_"):
    st.write("🔹 **Feature Importance:**")
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': heart_disease_model.feature_importances_})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x="Importance", y="Feature", data=feature_importance, palette="viridis")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    st.pyplot(fig)
