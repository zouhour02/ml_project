import argparse
from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model

def main():
    parser = argparse.ArgumentParser(description='Run ML tasks')
    parser.add_argument('--prepare', action='store_true', help='Prepare the data')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--save', action='store_true', help='Save the model')
    parser.add_argument('--load', action='store_true', help='Load the model')

    args = parser.parse_args()

    if args.prepare:
        prepare_data()
    elif args.train:
        X_train, X_test, y_train, y_test = prepare_data()  # Prepare data first
        model = train_model(X_train, y_train)  # Train the model
        save_model(model, 'model.pkl')  # Save the trained model
    elif args.evaluate:
        model = load_model('model.pkl')  # Load the trained model
        X_train, X_test, y_train, y_test = prepare_data()  # Prepare data (or load from a saved file)
        evaluate_model(model, X_test, y_test)  # Evaluate the model
    elif args.save:
        # Optionally implement this functionality
        pass
    elif args.load:
        model = load_model('model.pkl')  # Load the model

if __name__ == '__main__':
    main()

