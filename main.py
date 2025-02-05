from utils.data_preprocessing import DataPreprocessor
from models.logistic_regression import LogisticRegression
from utils.model_evaluation import ModelEvaluator

def main():
    print("🚀 Starting Heart Disease Prediction...")

    # Step 1: Preprocess Data
    print("\n🔄 Preprocessing Data...")
    preprocessor = DataPreprocessor("dataset/heart.csv")
    preprocessor.load_data()
    preprocessor.normalize_data()
    X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.split_data()
    preprocessor.save_data(X_train, y_train, X_val, y_val, X_test, y_test)

    # Step 2: Train Logistic Regression Model
    print("\n🧠 Training Model...")
    model = LogisticRegression(learning_rate=0.01, epochs=1000)
    model.fit(X_train, y_train)
    model.save_model()

    # Step 3: Evaluate Model
    print("\n📊 Evaluating Model...")
    evaluator = ModelEvaluator()
    evaluator.load_model()
    evaluator.load_test_data()
    evaluator.evaluate()

if __name__ == "__main__":
    main()
