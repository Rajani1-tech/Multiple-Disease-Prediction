import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
class ModelEvaluator:
    def __init__(self, model_folder="saved_models", data_folder="processed_data"):
        self.model_folder = model_folder
        self.data_folder = data_folder
        self.weights = None
        self.bias = 0.0
        self.X_test = None
        self.y_test = None

    def load_model(self):
        self.weights = np.load(f"{self.model_folder}/weights.npy")
        self.bias = np.load(f"{self.model_folder}/bias.npy")
        print(f"Loaded bias: {self.bias}") 

    def load_test_data(self):
        self.X_test = np.load(f"{self.data_folder}/X_test.npy")
        self.y_test = np.load(f"{self.data_folder}/y_test.npy")

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = 1 / (1 + np.exp(-linear_model))
        return (y_pred >= 0.5).astype(int)

    def evaluate(self):
        y_pred = self.predict(self.X_test)
        accuracy = np.mean(y_pred == self.y_test) * 100

        # Compute confusion matrix values
        tp = np.sum((y_pred == 1) & (self.y_test == 1))  
        tn = np.sum((y_pred == 0) & (self.y_test == 0))  
        fp = np.sum((y_pred == 1) & (self.y_test == 0))  
        fn = np.sum((y_pred == 0) & (self.y_test == 1))  

        conf_matrix = np.array([[tn, fp], [fn, tp]])

        # Compute classification metrics
        precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

        precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0
        recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_0 = 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) > 0 else 0

        report_df = pd.DataFrame({
            "precision": [precision_1, precision_0],
            "recall": [recall_1, recall_0],
            "f1-score": [f1_1, f1_0],
            "support": [tp + fn, tn + fp]
        }, index=["Disease (1)", "No Disease (0)"]).round(2)

        return accuracy, y_pred, conf_matrix, report_df
