import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Convert labels from {0, 1} to {-1, 1} for hinge loss
        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    # No misclassification
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # Misclassification
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)

# Load and preprocess the data
data = load_breast_cancer()
X, y = data.data, data.target
X = StandardScaler().fit_transform(X)  # Normalize features

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM model
svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
svm.fit(X_train, y_train)

# Evaluate the model
train_predictions = svm.predict(X_train)
train_accuracy = np.mean(train_predictions == np.where(y_train <= 0, -1, 1))  # Convert y_train for consistency

test_predictions = svm.predict(X_test)
test_accuracy = np.mean(test_predictions == np.where(y_test <= 0, -1, 1))  # Convert y_test for consistency
