#model_api 

import joblib
import numpy as np
import pandas as pd # <-- NEW: Import Pandas to handle structured features
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Add the current directory to Python path to import energy_theft
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- Configuration ---
MODEL_PATH = 'energy_theft_model.joblib'

# The 6 features the model expects, in the EXACT order they were trained:
FEATURE_NAMES = [
    'cons_mean',
    'cons_total',
    'diff_std',
    'lag1_corr',
    'month_std',
    'cons_total_zscore'
]

# 1. Load the model once when the server starts
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    # Note: In a production environment, you would log the error and might not exit immediately.
    exit()


app = Flask(__name__)
# Allow your frontend origin.
CORS(app, origins=["http://localhost:5173"])

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data sent from the client
    data = request.get_json(force=True)
    
    # 2. Extract and structure features using Pandas DataFrame
    try:
        # Create a dictionary to hold the feature values for one customer
        feature_dict = {}
        
        # Pull the values from the input data (assuming input keys match FEATURE_NAMES)
        for name in FEATURE_NAMES:
            if name not in data:
                # If a required feature is missing, raise a clear error
                return jsonify({
                    "error": f"Missing required feature: '{name}'"
                }), 400
            feature_dict[name] = data[name]  # Direct assignment for the prediction function

        # Use the prediction function from energy_theft.py
        try:
            # Import the prediction function
            from energy_theft import predict_energy_theft
            result = predict_energy_theft(feature_dict)
            
            if 'error' in result:
                return jsonify(result), 400
                
            return jsonify(result)
            
        except ImportError:
            # Fallback to direct model prediction if import fails
            print("Warning: Could not import predict_energy_theft function, using direct model prediction")
            
            # Convert to DataFrame for direct model prediction
            features_df = pd.DataFrame([feature_dict], columns=FEATURE_NAMES)
            
            # Make the prediction
            prediction = model.predict(features_df)[0]
            probability = model.predict_proba(features_df)[0][1]
            
            response = {
                'prediction': int(prediction),
                'probability': float(probability),
                'is_theft': bool(prediction)
            }
            
            return jsonify(response)
        
    except Exception as e:
        # Catch JSON parsing errors or other data-related issues
        return jsonify({
            "error": "Invalid input format or data type.",
            "details": str(e)
        }), 400

if __name__ == '__main__':
    # Run the Flask app on a specific port (e.g., 5000)
    app.run(host='127.0.0.1', port=5000)

