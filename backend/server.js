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

        if (!inputData.features || !Array.isArray(inputData.features)) {
            return res.status(400).json({ error: "Invalid data format. Expected an array of features." });
        }

        console.log('Forwarding data to Python API...');

        // Transform the features array into the format expected by Python API (IsolationForest v2)
        const featureNames = [
          't_kWh', 'z_Avg Voltage (Volt)', 'z_Avg Current (Amp)', 'y_Freq (Hz)',
          'hour', 'day_of_week', 'is_weekend', 'is_peak_hour',
          'kwh_change', 'kwh_vs_yesterday', 'power_ratio',
          'apparent_power', 'load_factor', 'volt_dev', 'freq_dev',
          'group_kwh_zscore', 'rolling_mean_kwh', 'rolling_std_kwh',
          'voltage_drop', 'load_spike', 'supply_instability'
        ];
        const pythonPayload = {};
        
        for (let i = 0; i < featureNames.length && i < inputData.features.length; i++) {
            pythonPayload[featureNames[i]] = inputData.features[i];
        }

        console.log('Transformed payload for Python API:', pythonPayload);

        // Call the running Python Flask API
        const pythonResponse = await axios.post(PYTHON_API_URL, pythonPayload);

        // Forward the prediction result back to the React frontend
        res.json(pythonResponse.data);

    } catch (error) {
        console.error('Error calling Python API or processing request:', error.message);
        res.status(500).json({ error: 'Failed to get prediction.', detail: error.message });
    }
});

// Start the server
app.listen(PORT, () => {
    console.log(`✅ Node.js server running on http://localhost:${PORT}`);
    console.log(`Communicating with Python API at ${PYTHON_API_URL}`);
});
