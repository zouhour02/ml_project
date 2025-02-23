from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
from twilio.rest import Client
import os
from dotenv import load_dotenv

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

@app.route('/send_sms', methods=['POST'])
def send_sms_route():
    """
    Handles SMS sending after prediction.
    """
    prediction = request.form.get('prediction')

    if not prediction:
        return "⚠️ Missing prediction data!", 400

    message = f"The predicted value is: {prediction}"

    # Send SMS to your phone
    sms_status = send_sms(message)

    return render_template('sms_sent.html', message_sid=sms_status)

if __name__ == '__main__':
    app.run(debug=True)

