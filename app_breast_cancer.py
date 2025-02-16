import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from models.svm import SVM
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def app_breast_cancer():

    # Load dataset from sklearn
    @st.cache_data
    def load_data():
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['diagnosis'] = data.target  # Add target column (0 = Benign, 1 = Malignant)
        return df, data

    # Function to get the top features based on PCA
    def get_top_pca_features(df, n_components=10):
        X = df.drop(columns=["diagnosis"])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply PCA with all components
        pca = PCA()
        pca.fit(X_scaled)

        # Get absolute loadings for features
        loadings = abs(pca.components_)
        loadings_df = pd.DataFrame(loadings, columns=df.columns[:-1])

        # Transpose for easier access
        loadings_df = loadings_df.T
        loadings_df.columns = [f'PC{i+1}' for i in range(loadings_df.shape[1])]

        # Extract top 10 unique features
        top_10_features = []
        for i in range(n_components):
            component_loadings = loadings_df.sort_values(by=f'PC{i+1}', ascending=False)
            top_10_features.extend(component_loadings.index[:10].tolist())

        # Remove duplicates while preserving order
        unique_top_10_features = []
        for feature in top_10_features:
            if feature not in unique_top_10_features:
                unique_top_10_features.append(feature)

        return unique_top_10_features[:10], scaler, pca

    # Load and preprocess the data
    df, data_info = load_data()
    top_features, scaler, pca = get_top_pca_features(df)

    # Split data for training and testing
    def preprocess_data(df,selected_features, n_components=10):
        X = df[selected_features]
        y = df["diagnosis"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        return train_test_split(X_pca, y, test_size=0.2, random_state=42), scaler, pca

    (X_train, X_test, y_train, y_test), scaler, pca = preprocess_data(df,selected_features=top_features, n_components=10)

    # Train the SVM model
    svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=2000)
    svm.fit(X_train, y_train)

    st.title('Breast Cancer Prediction using ML')

    # Collect user inputs
    st.subheader("🔹 Input Patient Details")
    user_input = {}
    cols = st.columns(3)

    for i, col_name in enumerate(top_features):
        with cols[i % 3]:
            user_input[col_name] = st.number_input(
                label=col_name.replace("_", " ").capitalize(),
                # value=None,
                value=float(df[col_name].mean()),
                step=0.0001,
                format="%.6f"
            )

    # Prediction Button
    if st.button('Breast Cancer Test Result'):
        if any(value == None for value in user_input.values()):
            st.warning("Please fill in all input fields.")
        else:
            user_input_array = np.array([list(user_input.values())]).reshape(1, -1)
            user_input_scaled = scaler.transform(user_input_array)
            
            bc_prediction = svm.predict(user_input_scaled)

            diagnosis = 'Malignant (Cancerous)' if bc_prediction[0] == 1 else 'Benign (Non-Cancerous)'
            st.success(f'Prediction: {diagnosis}')

            # Model Performance Section
            st.subheader(" Model Performance on Test Data")
            y_pred = svm.predict(X_test)
            y_pred = np.where(y_pred == 1, 1, 0)

            # Accuracy Score
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"**Model Accuracy:** {accuracy:.2f}")

            # Classification Report
            st.write(" **Classification Report:**")
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            report_df = report_df[['precision', 'recall', 'f1-score', 'support']].round(2)
            st.dataframe(report_df)

            # Correlation Matrix Section
            st.subheader("Correlation Matrix")
            corr_matrix = df[top_features].corr()

            # Plot the heatmap of the correlation matrix
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
            plt.title("Correlation Matrix")
            st.pyplot(fig)

            # Confusion Matrix
            st.write(" **Confusion Matrix:**")
            conf_matrix = confusion_matrix(y_test, y_pred)
        
            # Check for unique labels in y_test
            unique_labels = np.unique(y_test)
        
            # Plot the heatmap with dynamic tick labels
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=unique_labels, yticklabels=unique_labels)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            st.pyplot(fig)

# Allow running the app standalone
if __name__ == "__main__":
    app_breast_cancer()