from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from fastapi.responses import JSONResponse

app = FastAPI()

# Try loading the current model. If not available, set to None.
try:
    model = joblib.load("model.joblib")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# ---------------------------
# Prediction Endpoint (Option A)
# ---------------------------
class FeaturesInput(BaseModel):
    features: List[float]  # One-dimensional list of floats
    # Note: For a model retrained with one-hot encoding, the input must match
    # the dummy-encoded feature order and count.

@app.post("/predict")
def predict(input_data: FeaturesInput):
    if model is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded."})
    
    # Convert the features list to a 2D numpy array (1 row, n features)
    data = np.array(input_data.features).reshape(1, -1)
    
    # Print the expected number of features
    print(f"Expected number of features: {model.n_features_in_}")
    print(f"Received features: {data.shape[1]}")
    
    if data.shape[1] != model.n_features_in_:
        return JSONResponse(status_code=400, content={"error": "Feature mismatch"})
    
    prediction = model.predict(data)
    return {"prediction": prediction.tolist()}

# ---------------------------
# Retrain Endpoint
# ---------------------------
class RetrainParams(BaseModel):
    n_estimators: Optional[int] = Field(100, description="Number of trees in the forest")
    max_depth: Optional[int] = Field(None, description="Maximum depth of the tree")
    random_state: Optional[int] = Field(42, description="Random seed for reproducibility")

@app.post("/retrain")
def retrain(params: RetrainParams):
    # Load training data from churn-bigml-80.csv
    try:
        data_df = pd.read_csv("churn-bigml-80.csv")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error reading training data: " + str(e))
    
    # Ensure the target column 'Churn' exists
    if "Churn" not in data_df.columns:
        raise HTTPException(status_code=400, detail="Training data must contain a 'Churn' column.")
    
    # Separate features and target
    X = data_df.drop("Churn", axis=1)
    y = data_df["Churn"]
    
    # Preprocess features:
    # Convert categorical features to dummy/indicator variables
    X = pd.get_dummies(X, drop_first=True)
    
    # Create a new RandomForestClassifier with the provided hyperparameters
    clf = RandomForestClassifier(
        n_estimators=params.n_estimators,
        max_depth=params.max_depth,
        random_state=params.random_state
    )
    
    # Fit the classifier on the preprocessed data
    try:
        clf.fit(X, y)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error during model fitting: " + str(e))
    
    # Save the new model to disk
    joblib.dump(clf, "model.joblib")
    
    # Update the global model so future predictions use the retrained model
    global model
    model = clf
    
    return {"detail": "Model retrained successfully", "new_params": params.dict()}

