import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


TRAIN_DATA_FILE = "/home/zouhour/ml_project/churn-bigml-80.csv"
TEST_DATA_FILE = "/home/zouhour/ml_project/churn-bigml-20.csv"

def prepare_data():
    """Prepare and preprocess the dataset."""
    print("Loading training and test data...")
    train_data = pd.read_csv(TRAIN_DATA_FILE)
    test_data = pd.read_csv(TEST_DATA_FILE)
    print("Data loaded successfully.")
    
    print("Dropping missing values...")
    train_data = train_data.dropna()
    test_data = test_data.dropna()
    print("Missing values dropped.")

    print("Encoding categorical columns...")
    train_data = pd.get_dummies(train_data)
    test_data = pd.get_dummies(test_data)
    print("Categorical columns encoded.")

    # Ensure both datasets have the same columns after encoding
    train_data, test_data = train_data.align(test_data, join='inner', axis=1)

    print("Splitting features and target...")
    X_train = train_data.drop('Churn', axis=1)
    y_train = train_data['Churn']
    X_test = test_data.drop('Churn', axis=1)
    y_test = test_data['Churn']
    print("Data preparation complete.")
    return X_train, X_test, y_train, y_test

def save_prepared_data(X_train, X_test, y_train, y_test, file_path):
    """Save the prepared data to a file."""
    with open(file_path, 'wb') as file:
        pickle.dump({'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}, file)
    print("Prepared data saved to", file_path)

def load_prepared_data(file_path):
    """Load the prepared data from a file."""
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data['X_train'], data['X_test'], data['y_train'], data['y_test']

def train_model(X_train, y_train):
    """Train a RandomForestClassifier model."""
    print("Training model...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    print("Model training complete.")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model's performance and display metrics.
    """
    print("Evaluating model...")

    # Generate predictions
    predictions = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted")  # Weighted for imbalanced datasets
    recall = recall_score(y_test, predictions, average="weighted")
    f1 = f1_score(y_test, predictions, average="weighted")

    # Print metrics as a table
    print("\nModel Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # Check if model.classes_ is of type numpy.bool_
    if isinstance(model.classes_, np.ndarray) and model.classes_.dtype == np.bool_:
        # Handle boolean class labels by converting them to more descriptive labels
        model.classes_ = ['Negative', 'Positive'] 

    # Ensure target_names is a list or array of class labels
    target_names = model.classes_ if isinstance(model.classes_, (list, np.ndarray)) else []

    # Generate classification report
    report = classification_report(y_test, predictions, target_names=target_names)
    print("\nClassification Report:\n")
    print(report)

    # Return the metrics for further use
    return accuracy, precision, recall, f1

def save_model(model, file_path="model.joblib"):
    """Save the model to a file (default: model.joblib)."""
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {file_path}")

def load_model(file_path):
    """Load the model from a file."""
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    print("Model loaded successfully.")
    return model
