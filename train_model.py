import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from models.decision_tree import DecisionTree

import pickle

def train_diabetic_model():
    diabetes_dataset = pd.read_csv('C:/DiseasePrediction/Multiple-Disease-Prediction/dataset/diabetes.csv') 
    X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
    Y = diabetes_dataset['Outcome']
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.15, stratify=Y, random_state=2)
    
    filename = 'saved_models/diabetes_model_logistic_regression.sav'
    # classifier = LogisticRegression()
    classifier = DecisionTree(max_depth=3)

    classifier.fit(X_train.to_numpy(), Y_train.to_numpy())
    X_train_prediction = classifier.predict(X_train.to_numpy())
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train.to_numpy())
    print(f'Accuracy score of the training data from Decision Tree is: ', training_data_accuracy)
    X_test_prediction = classifier.predict(X_test.to_numpy())
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test.to_numpy())
    print(f'Accuracy score of the test data from Decision Tree is: ', test_data_accuracy)
    pickle.dump(classifier, open(filename, 'wb'))

if __name__ == '__main__':
    train_diabetic_model()
    print('Training done')

