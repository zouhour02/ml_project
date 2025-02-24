import argparse
import mlflow
from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model

def main():
  ##new modi
     # Définir le nom de l'expérience MLflow
    mlflow.set_tracking_uri("file:///home/zouhour/mlruns")
    mlflow.set_experiment("New_Churn_Prediction_Experiment")
    
    parser = argparse.ArgumentParser(description="Run ML tasks")
    parser.add_argument("--prepare", action="store_true", help="Prepare the data")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--save", action="store_true", help="Save the model")
    parser.add_argument("--load", action="store_true", help="Load the model")
    
    args = parser.parse_args()

    if args.prepare:
        prepare_data()
        print("Data prepared successfully.")
    
    elif args.train:
        X_train, X_test, y_train, y_test = prepare_data()

 # Ensure data is available
        model = train_model(X_train, y_train)  # Train the model
        save_model(model)  # Saves as "model.joblib"
        print("Model trained and saved as model.joblib.")

    elif args.evaluate:
        model = load_model("model.joblib")  # Load the trained model
        X_train, X_test, y_train, y_test = prepare_data()  # Ensure data is available
        evaluate_model(model, X_test, y_test)  # Evaluate the model

    elif args.save:
        model = train_model(*prepare_data()[:2])  # Train the model before saving
        save_model(model,"model.joblib")  # Saves as "model.joblib"
        print("Model saved as model.joblib.")

    elif args.load:
        model = load_model("model.joblib")  # Load the model
        print("Model loaded successfully.")

if __name__ == "__main__":
    main()

