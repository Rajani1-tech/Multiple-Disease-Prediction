import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

import pickle

# Importing the Dependencies

# Data Collection and Analysis
# PIMA Diabetes Dataset
# loading the diabetes dataset to a pandas DataFrame


def train_diabetic_model(which_classifier):
    diabetes_dataset = pd.read_csv('C:/DiseasePrediction/Multiple-Disease-Prediction/dataset/diabetes.csv') 
    X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
    Y = diabetes_dataset['Outcome']
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)
    if which_classifier == 'SVM':
        filename = 'saved_models/diabetes_model_svm.sav'
        classifier = svm.SVC(kernel='linear')
    elif which_classifier == 'DecisionTree':
        filename = 'saved_models/diabetes_model_decision_tree.sav'
        classifier = DecisionTreeClassifier()
    elif which_classifier == 'LogisticRegression':
        filename = 'saved_models/diabetes_model_logistic_regression.sav'
        classifier = LogisticRegression()
    else:
        print('Invalid classifier')
        return None
    
    classifier.fit(X_train, Y_train)
    X_train_prediction = classifier.predict(X_train)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
    print(f'Accuracy score of the training data in {which_classifier} : ', training_data_accuracy)
    X_test_prediction = classifier.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
    print(f'Accuracy score of the test data in {which_classifier}: ', test_data_accuracy)
    pickle.dump(classifier, open(filename, 'wb'))

if __name__ == '__main__':
    train_diabetic_model('SVM')
    train_diabetic_model('DecisionTree')
    train_diabetic_model('LogisticRegression')
    print('Training done')