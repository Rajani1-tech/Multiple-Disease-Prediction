import numpy as np
import pandas as pd
from models.decision_tree import DecisionTree
from plot_metric import plot_metrics, plot_pretty_confusion_matrix
import pickle

def stratified_train_test_split(X, y, test_size=0.2, random_state=None):
   
    if random_state is not None:
        np.random.seed(random_state)
    
    X = np.array(X)
    y = np.array(y)

    classes = np.unique(y)
    class_indices = [np.where(y == c)[0] for c in classes]
    
    train_indices = []
    test_indices = []
    
    for indices in class_indices:
        n_samples = len(indices)
        n_test = int(test_size * n_samples)
        
        shuffled_indices = indices[np.random.permutation(n_samples)]
        
        test_indices.extend(shuffled_indices[:n_test])
        train_indices.extend(shuffled_indices[n_test:])
    
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

def calculate_accuracy(y_true, y_pred):
    
    if len(y_true) != len(y_pred):
        raise ValueError("Arrays must have the same length")
    correct_predictions = np.sum(y_true == y_pred)

    accuracy = correct_predictions / len(y_true)
    
    return accuracy



def calculate_metrics(y_true, y_pred):
    
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
   
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    support = len(y_true)
    
    return {
        'TP': int(TP),
        'TN': int(TN),
        'FP': int(FP),
        'FN': int(FN),
        'Accuracy': float(accuracy),
        'Precision': float(precision),
        'Recall': float(recall),
        'F1_Score': float(f1_score),
        'Support': int(support)
    }
def calculate_metrics_total(y_true, y_pred):
   
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    

    total = len(y_true)
    
    accuracy = (TP + TN) / total
    
    pos_precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    pos_recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    pos_f1 = 2 * (pos_precision * pos_recall) / (pos_precision + pos_recall) if (pos_precision + pos_recall) > 0 else 0
    pos_support = np.sum(y_true == 1)
    
    neg_precision = TN / (TN + FN) if (TN + FN) > 0 else 0
    neg_recall = TN / (TN + FP) if (TN + FP) > 0 else 0
    neg_f1 = 2 * (neg_precision * neg_recall) / (neg_precision + neg_recall) if (neg_precision + neg_recall) > 0 else 0
    neg_support = np.sum(y_true == 0)
    
    return {
            'TP': int(TP),
            'TN': int(TN),
            'FP': int(FP),
            'FN': int(FN), 

            'accuracy': float(accuracy),  

            'pos_precision': float(pos_precision),
            'pos_recall': float(pos_recall),
            'pos_f1_score': float(pos_f1),
            'pos_support': int(pos_support),
        
            'neg_precision': float(neg_precision),
            'neg_recall': float(neg_recall),
            'neg_f1_score': float(neg_f1),
            'neg_support': int(neg_support)
    }

def train_diabetic_model():
    diabetes_dataset = pd.read_csv('dataset/diabetes.csv')
    
    X = diabetes_dataset.drop(columns='Outcome').to_numpy()
    Y = diabetes_dataset['Outcome'].to_numpy()

    X_train, X_test, Y_train, Y_test = stratified_train_test_split(
        X, Y, test_size=0.2, random_state=2
    )
    
    filename = 'saved_models/diabetes_model_decision_tree.sav'
    classifier = DecisionTree(max_depth=3)
  
    classifier.fit(X_train, Y_train)
  
    X_train_prediction = classifier.predict(X_train)
    X_test_prediction = classifier.predict(X_test)
    
    train_metrics = calculate_metrics_total(Y_train, X_train_prediction)
    test_metrics = calculate_metrics_total(Y_test, X_test_prediction)
 
    pickle.dump(classifier, open(filename, 'wb'))
  
    metrics_df = pd.DataFrame({
        'Metric': list(train_metrics.keys()),
        'Training': list(train_metrics.values()),
        'Test': list(test_metrics.values())
    })
    
    metrics_df.to_csv('./results/diabetic_model_metrics.csv', index=False)
    plot_metrics(metrics_file='./results/diabetic_model_metrics.csv')
    plot_pretty_confusion_matrix(metrics_file='./results/diabetic_model_metrics.csv')
    return train_metrics, test_metrics


if __name__ == '__main__':
    train_diabetic_model()
    print('Training done')

