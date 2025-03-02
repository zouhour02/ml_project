import pickle
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, render_template,redirect, url_for, flash
from twilio.rest import Client


# Load environment variables
load_dotenv()

# Twilio credentials
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
from_phone = os.getenv("TWILIO_PHONE_NUMBER")

# Debugging - Check if credentials are loaded
if not account_sid or not auth_token or not from_phone:
    print("⚠️ Twilio credentials are missing! Check your .env file.")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "d9f1e8b3c4a65e1b0f3c7e1a2d9f6b8a9c4d7e3a5b6c7d8e9f0a1b2c3d4e5f6a")


# Load the pre-trained model
model = pickle.load(open('model.pkl', 'rb'))

# Get the model's expected features
expected_features = model.feature_names_in_

# Check if the model supports feature importance (e.g., RandomForest)
if hasattr(model, 'feature_importances_'):
    feature_importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': expected_features,
        'Importance': feature_importances
    })
    top_features = feature_importance_df.sort_values(by='Importance', ascending=False).head(4).to_dict(orient='records')
else:
    top_features = None

# ✅ Function to send SMS via Twilio
def send_sms(message):
    """
    Sends an SMS message to a predefined phone number using Twilio.
    """
    try:
        client = Client(account_sid, auth_token)
        to_phone = os.getenv('MY_PHONE_NUMBER')  # Store your phone number in .env

        if not to_phone:
            return "⚠️ Phone number is missing in .env"

        message = client.messages.create(
            body=message,
            from_=from_phone,
            to=to_phone
        )
        return message.sid
    except Exception as e:
        return f"Error sending SMS: {str(e)}"

@app.route('/')
def home():
    return render_template('index.html', top_features=top_features)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles user input, makes a prediction, and displays results.
    """
    input_data = request.form.to_dict()

    # Convert to DataFrame and ensure correct format
    input_df = pd.DataFrame([input_data])
    for feature in expected_features:
        if feature not in input_df.columns:
            input_df[feature] = 0  # Fill missing values with 0

    # Reorder and convert data types
    input_df = input_df[expected_features].apply(pd.to_numeric, errors='coerce')

    # Make prediction
    prediction = model.predict(input_df)[0]

    return render_template('result.html', prediction=prediction, top_features=top_features)
    
# Retrain route
@app.route('/retrain', methods=['GET', 'POST'])
def retrain():
    global model
    
    if request.method == 'POST':
        n_estimators = int(request.form.get('n_estimators', 100))
        max_depth = request.form.get('max_depth')
        max_depth = int(max_depth) if max_depth else None
        
        try:
            # Load new training data
            data_df = pd.read_csv('churn-bigml-80.csv')

            if "Churn" not in data_df.columns:
                flash("Training data must contain a 'Churn' column.", "danger")
                return redirect(url_for('retrain'))

            # Split into features and target
            X = data_df.drop("Churn", axis=1)
            y = data_df["Churn"]
            X = pd.get_dummies(X, drop_first=True)

            # Retrain model
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X, y)

            # Save the retrained model
            pickle.dump(model, open('model.pkl', 'wb'))
            flash("Model retrained successfully!", "success")
        
        except Exception as e:
            flash(f"Error during retraining: {e}", "danger")
        
        return redirect(url_for('home'))
    
    return render_template('retrain.html')

@app.route('/send_sms', methods=['POST'])
def send_sms_route():
    """
    Handles SMS sending after prediction.
    """
    prediction = request.form.get('prediction')

    if not prediction:
        return " Missing prediction data!", 400

    message = f"The predicted value is: {prediction}"

    # Send SMS to your phone
    sms_status = send_sms(message)

    return render_template('sms_sent.html', message_sid=sms_status)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)

