#model_api 

import joblib
import numpy as np
import pandas as pd # <-- NEW: Import Pandas to handle structured features
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import shap

# Add the current directory to Python path to import energy_theft
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- Configuration ---
# Updated to IsolationForest v2 artifacts produced by energy_theft.py
MODEL_PATH = r'D:\energy_theft\outputs\energy_theft_isoforest_v2.pkl'
SCALER_PATH = r'D:\energy_theft\outputs\energy_theft_scaler_v2.pkl'

# The IsolationForest expects the following engineered features in EXACT order
# In model_api.py
FEATURE_NAMES = [
    't_kWh', 'z_Avg Voltage (Volt)', 'z_Avg Current (Amp)', 'y_Freq (Hz)',
    'hour', 'day_of_week', 'is_weekend', 'is_peak_hour',
    'kwh_change', 'kwh_vs_yesterday', 'power_ratio',
    'apparent_power', 'load_factor', 'volt_dev', 'freq_dev',
    'group_kwh_zscore', 'rolling_mean_kwh', 'rolling_std_kwh',
    'voltage_drop', 'load_spike', 'supply_instability' 
]

# 1. Load the model and scaler once when the server starts
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    X_BACKGROUND_CENTERED = np.zeros((1, len(FEATURE_NAMES)))
    
    # Initialize KernelExplainer with the model's prediction function
    explainer = shap.KernelExplainer(model.predict, X_BACKGROUND_CENTERED)
    print("✅ SHAP KernelExplainer initialized.")
    print(f"✅ Model loaded from {MODEL_PATH}")
    print(f"✅ Scaler loaded from {SCALER_PATH}")
except Exception as e:
    print(f"❌ Error loading model/scaler: {e}")
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
            feature_dict[name] = data[name]

        # Convert to DataFrame and scale
        features_df = pd.DataFrame([feature_dict], columns=FEATURE_NAMES)
        X_scaled = scaler.transform(features_df)

        # IsolationForest outputs: decision_function (higher is normal), predict (1 normal, -1 anomaly)
        anomaly_score = float(model.decision_function(X_scaled)[0])
        raw_label = int(model.predict(X_scaled)[0])  # 1 normal, -1 anomaly
        # Map to theft label: 1 = theft, 0 = normal (align to prior API contract is_theft)
        theft_label = 1 if raw_label == -1 else 0

        # --- Robust SHAP computation with fallback ---
        try:
            # shap_values can be list or array depending on model/explainer version
            sv = explainer.shap_values(X_scaled, nsamples=100)
            if isinstance(sv, list):
                # choose first output; shape often (n_samples, n_features)
                sv_arr = np.array(sv[0])
            else:
                sv_arr = np.array(sv)
            # ensure we take first sample
            shap_vals_sample = sv_arr[0]
            shap_dict = dict(zip(FEATURE_NAMES, shap_vals_sample))
        except Exception as e_shap:
            # fallback: return empty/shap zeros and log the error
            print(f"⚠️ SHAP computation failed: {e_shap}")
            shap_dict = {fn: 0.0 for fn in FEATURE_NAMES}
        
        # Get top 5 features contributing to the prediction (by absolute SHAP value)
        top_features = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        
        # Build the final contribution list
        contributions = []
        for f, val in top_features:
            contributions.append({
                'feature': f,
                'value': float(features_df.iloc[0][f]), 
                'shap_value': float(val) 
            })
            
        # Generate simple explanation text (clearer for frontend)
        top_feature_names = [c['feature'] for c in contributions]
        explanation_text = (
             ("⚠️ Theft suspected. " if theft_label == 1 else "✅ Normal usage detected. ")
             + "Top factors: " + ", ".join(top_feature_names) + "."
        )

        response = {
            'prediction': theft_label,
            'probability': None,  # Not applicable for IsolationForest; left as None
            'is_theft': bool(theft_label),
            'anomaly_score': anomaly_score,
            'explanation': explanation_text,
            'contributions': contributions
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

