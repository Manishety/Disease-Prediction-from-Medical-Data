import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

def disease_prediction(dataset_name='breast_cancer', algorithm_name='RandomForest'):
    """
    Predicts disease possibility using a specified classification algorithm on a given dataset.

    Args:
        dataset_name (str): The name of the dataset to use ('breast_cancer', 'diabetes', 'iris').
        algorithm_name (str): The classification algorithm to use ('LogisticRegression', 'SVM', 'RandomForest', 'XGBoost').
    """

    # --- 1. Load Data ---
    if dataset_name == 'breast_cancer':
        data = load_breast_cancer()
        print("Using Breast Cancer Dataset.")
    elif dataset_name == 'diabetes':
        # Using diabetes dataset for demonstration, note this is a regression problem by default
        # but can be adapted for classification (e.g., predicting if a patient is 'diabetic' based on a threshold).
        # We'll simplify for this example by using it as a classification problem.
        # This dataset is for type 2 diabetes and the target is a quantitative measure of disease progression.
        # We will not use it for this classification example as its native target is regression.
        print("The Diabetes dataset is primarily for regression. Using Breast Cancer instead.")
        data = load_breast_cancer()
    elif dataset_name == 'iris':
        # Using Iris for a multi-class classification example
        data = load_iris()
        print("Using Iris Dataset (multi-class example).")
    else:
        print("Invalid dataset name. Using Breast Cancer by default.")
        data = load_breast_cancer()

    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')

    # --- 2. Data Splitting ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 3. Model Selection ---
    if algorithm_name == 'LogisticRegression':
        model = LogisticRegression(max_iter=1000)
        print("Using Logistic Regression.")
    elif algorithm_name == 'SVM':
        model = SVC()
        print("Using Support Vector Machine (SVM).")
    elif algorithm_name == 'RandomForest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        print("Using Random Forest Classifier.")
    elif algorithm_name == 'XGBoost':
        # XGBoost requires a separate library, so we'll use GradientBoostingClassifier from scikit-learn
        # as a stand-in which serves the same conceptual purpose (ensemble of boosted trees).
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        print("Using Gradient Boosting (stand-in for XGBoost).")
    else:
        print("Invalid algorithm name. Using Random Forest by default.")
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    # --- 4. Model Training ---
    print("\nTraining the model...")
    model.fit(X_train, y_train)

    # --- 5. Prediction and Evaluation ---
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=data.target_names)

    print("\n--- Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)

# --- Example Usage ---
print("--- Running Breast Cancer Prediction with Random Forest ---")
disease_prediction(dataset_name='breast_cancer', algorithm_name='RandomForest')

print("\n\n--- Running Breast Cancer Prediction with Logistic Regression ---")
disease_prediction(dataset_name='breast_cancer', algorithm_name='LogisticRegression')

print("\n\n--- Running Iris Prediction with SVM ---")
disease_prediction(dataset_name='iris', algorithm_name='SVM')

# Note: For XGBoost, you would need to install the `xgboost` library separately.
# The code above uses scikit-learn's GradientBoostingClassifier as an equivalent.
