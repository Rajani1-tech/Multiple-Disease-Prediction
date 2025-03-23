import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib

matplotlib.use('Agg')  # Use a non-GUI backend

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

        tp = np.sum((y_pred == 1) & (self.y_test == 1))  
        tn = np.sum((y_pred == 0) & (self.y_test == 0))  
        fp = np.sum((y_pred == 1) & (self.y_test == 0))  
        fn = np.sum((y_pred == 0) & (self.y_test == 1))  

        conf_matrix = np.array([[tn, fp], [fn, tp]])

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

    def plot_confusion_matrix_heart(self, conf_matrix):
        plt.figure(figsize=(6, 6))
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', cbar=False, square=True,
                    xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])

        plt.title('Heart Disease Confusion Matrix', pad=20, size=26, weight='bold')
        plt.xlabel('Predicted', fontsize=20)
        plt.ylabel('Actual', fontsize=20)

        plt.style.use('dark_background')
        save_path = 'heart_disease_confusion_matrix.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path

    def plot_metrics_heart(self, report_df, accuracy):
        metrics = ['Precision', 'Recall', 'F1-Score']
        positive_metrics = report_df.loc['Disease (1)', ['precision', 'recall', 'f1-score']].values
        negative_metrics = report_df.loc['No Disease (0)', ['precision', 'recall', 'f1-score']].values

        plt.figure(figsize=(12, 8))
        y_pos = np.arange(len(metrics))
        bar_width = 0.3

        plt.barh(y_pos - bar_width/2, positive_metrics, height=bar_width, label='Disease')
        plt.barh(y_pos + bar_width/2, negative_metrics, height=bar_width, label='No Disease')

        plt.barh(len(metrics), accuracy / 100, height=bar_width, label='Overall Accuracy', color='gold')
        metrics.append('Accuracy')

        for i, value in enumerate(positive_metrics):
            plt.text(value + 0.01, i - bar_width/2, f'{value * 100:.1f}%', va='center', fontsize=12, color='white')

        for i, value in enumerate(negative_metrics):
            plt.text(value + 0.01, i + bar_width/2, f'{value * 100:.1f}%', va='center', fontsize=12, color='white')

        plt.text((accuracy/100) + 0.01, len(metrics)- 1, f'{accuracy:.1f}%', va='center', fontsize=12,color='white', fontweight='bold' )
        
        plt.yticks(np.arange(len(metrics)), metrics, fontsize=14)
        plt.xlabel('Value', fontsize=14)
        plt.title('Model Performance Metrics of heart disease', fontsize=18)
        plt.legend(loc='lower right')
        plt.grid(True, axis='x', linestyle='--', alpha=0.7)
        #plt.style.use('')

        save_path = 'heart_disease_metrics.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
