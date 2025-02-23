from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open('model.pkl', 'rb'))

# Get the model's expected features and their importance
expected_features = model.feature_names_in_

# Assuming the model supports feature importances (like RandomForest)
if hasattr(model, 'feature_importances_'):
    feature_importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': expected_features,
        'Importance': feature_importances
    })
    top_features = feature_importance_df.sort_values(by='Importance', ascending=False).head(4).to_dict(orient='records')
else:
    top_features = None  # Handle cases where the model doesn't provide feature importances

@app.route('/')
def home():
    return render_template('index.html', top_features=top_features)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    input_data = request.form.to_dict()

    # Convert the input data into a DataFrame (ensure numeric values for prediction)
    input_df = pd.DataFrame([input_data])

    # Ensure that the columns match the model's expected features
    for feature in expected_features:
        if feature not in input_df.columns:
            input_df[feature] = 0  # Add missing feature with a default value of 0

    # Reorder columns to match the model's expected order
    input_df = input_df[expected_features]

    # Convert input data to appropriate numeric format
    input_df = input_df.apply(pd.to_numeric, errors='coerce')

    # Make prediction
    prediction = model.predict(input_df)

    # Render result page with prediction and top features
    return render_template('result.html', prediction=prediction[0], top_features=top_features)

if __name__ == '__main__':
    app.run(debug=True)

