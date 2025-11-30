// backend/server.js

const express = require('express');
const cors = require('cors');
const axios = require('axios'); 

const app = express();
const PORT = 3001; // Node.js Server Port

// Configuration for Python API running on port 5000
const PYTHON_API_URL = 'http://127.0.0.1:5000/predict'; 

const corsOptions = {
  origin: 'http://localhost:5173', // <--- SPECIFY the exact address of your frontend
  methods: 'GET,HEAD,PUT,PATCH,POST,DELETE',
  credentials: true, // Allow cookies to be sent (good practice)
};

// --- Middleware setup ---
app.use(cors()); 
app.use(express.json()); 
app.use(cors(corsOptions));
app.use(express.json());
// --- API Endpoint: Prediction Gateway ---
app.post('/api/predict_theft', async (req, res) => {
    try {
        const inputData = req.body; 

        // Accept either { features: [...] } OR { 't_kWh': ..., 'z_Avg Voltage (Volt)': ..., ... }
        const featureNames = [
          't_kWh', 'z_Avg Voltage (Volt)', 'z_Avg Current (Amp)', 'y_Freq (Hz)',
          'hour', 'day_of_week', 'is_weekend', 'is_peak_hour',
          'kwh_change', 'kwh_vs_yesterday', 'power_ratio',
          'apparent_power', 'load_factor', 'volt_dev', 'freq_dev',
          'group_kwh_zscore', 'rolling_mean_kwh', 'rolling_std_kwh',
          'voltage_drop', 'load_spike', 'supply_instability'
        ];

        let pythonPayload = {};

        if (inputData && Array.isArray(inputData.features)) {
            // Array -> map by position
            for (let i = 0; i < featureNames.length; i++) {
                pythonPayload[featureNames[i]] = inputData.features[i] ?? null;
            }
        } else if (inputData && typeof inputData === 'object') {
            // Object -> pick values by name (preferred)
            for (const k of featureNames) {
                // if missing, set null - Python API will validate and return an informative error
                pythonPayload[k] = Object.prototype.hasOwnProperty.call(inputData, k) ? inputData[k] : null;
            }
        } else {
            return res.status(400).json({ error: "Invalid input format. Send { features: [...] } or an object with feature keys." });
        }

        console.log('Transformed payload for Python API:', pythonPayload);

        // Call the running Python Flask API
        const pythonResponse = await axios.post(PYTHON_API_URL, pythonPayload);

        // Forward the prediction result back to the React frontend
        res.status(pythonResponse.status).json(pythonResponse.data);

    } catch (error) {
                console.error('Error calling Python API or processing request:', error?.message);
                if (error?.response) {
                  const status = error.response.status || 500;
                  const payload = typeof error.response.data === 'object' ? error.response.data : { error: String(error.response.data) };
                  return res.status(status).json(payload);
                }
                res.status(500).json({ error: 'Failed to get prediction.', detail: error?.message });
    }
});

// Start the server
app.listen(PORT, () => {
    console.log(`✅ Node.js server running on http://localhost:${PORT}`);
    console.log(`Communicating with Python API at ${PYTHON_API_URL}`);
});
