import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from models.svm import SVM

# Set page configuration
st.set_page_config(page_title="Health Assistant", layout="wide", page_icon="🧑‍⚕️")

# Load dataset from sklearn
@st.cache_data
def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['diagnosis'] = data.target  # Add target column (0 = Benign, 1 = Malignant)
    return df, data

# Preprocess data
def preprocess_data(df):
    X = df.drop(columns=["diagnosis"]).values
    y = df["diagnosis"].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42), scaler

# Load and preprocess the data
df, data_info = load_data()
(X_train, X_test, y_train, y_test), scaler = preprocess_data(df)

# Train the SVM model
svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
svm.fit(X_train, y_train)

# Sidebar for navigation
with st.sidebar:
    selected = st.radio(
        'Multiple Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Breast Cancer Prediction'],
        index=2  # Default to Breast Cancer Prediction
    )

# Breast Cancer Prediction Page
if selected == 'Breast Cancer Prediction':
    st.title('Breast Cancer Prediction using ML')

    # Collect user inputs
    st.subheader("🔹 Input Patient Details")
    columns = data_info.feature_names
    user_input = {}
    cols = st.columns(3)

    for i, col_name in enumerate(columns):
        with cols[i % 3]:
            user_input[col_name] = st.number_input(
                label=col_name.replace("_", " ").capitalize(),
                value=float(df[col_name].mean()),
                step=0.01
            )

    # Prediction Button
    if st.button('Breast Cancer Test Result'):
        user_input_array = np.array([list(user_input.values())]).reshape(1, -1)
        user_input_scaled = scaler.transform(user_input_array)
        bc_prediction = svm.predict(user_input_scaled)
        diagnosis = 'Malignant (Cancerous)' if bc_prediction[0] == 1 else 'Benign (Non-Cancerous)'
        st.success(f'Prediction: {diagnosis}')

    # Model Performance Section
    st.subheader(" Model Performance on Test Data")
    y_pred = svm.predict(X_test)

    # Accuracy Score
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"**Model Accuracy:** {accuracy:.2f}")

    # Classification Report
    st.write(" **Classification Report:**")
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df = report_df[['precision', 'recall', 'f1-score', 'support']].round(2)
    st.dataframe(report_df)

    # Confusion Matrix
    st.write(" **Confusion Matrix:**")
    conf_matrix = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)
