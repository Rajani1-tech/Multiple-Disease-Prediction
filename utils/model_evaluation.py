import numpy as np

class ModelEvaluator:
    def __init__(self, model_folder="saved_models", data_folder="processed_data"):
        self.model_folder = model_folder
        self.data_folder = data_folder
        self.weights = None
        self.bias = None
        self.X_test = None
        self.y_test = None

    def load_model(self):
        self.weights = np.load(f"{self.model_folder}/weights.npy")
        self.bias = np.load(f"{self.model_folder}/bias.npy")

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
        print(f"✅ Test Accuracy: {accuracy:.2f}%")
