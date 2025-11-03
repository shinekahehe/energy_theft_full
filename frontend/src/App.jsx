import React, { useState } from 'react';

// Configuration
const API_URL = 'http://localhost:3001/api/predict_theft'; // Local Node.js backend

// IsolationForest feature schema (exact order must match backend/API)
const featureOrder = [
  't_kWh', 'z_Avg Voltage (Volt)', 'z_Avg Current (Amp)', 'y_Freq (Hz)',
  'hour', 'day_of_week', 'is_weekend', 'is_peak_hour',
  'kwh_change', 'kwh_vs_yesterday', 'power_ratio',
  'apparent_power', 'load_factor', 'volt_dev', 'freq_dev',
  'group_kwh_zscore', 'rolling_mean_kwh', 'rolling_std_kwh',
  'voltage_drop', 'load_spike', 'supply_instability'
];

// Initial values
const initialFeatures = {
  t_kWh: 0,
  'z_Avg Voltage (Volt)': 230,
  'z_Avg Current (Amp)': 5,
  'y_Freq (Hz)': 50,
  hour: 12,
  day_of_week: 2,
  is_weekend: 0,
  is_peak_hour: 1,
  kwh_change: 0,
  kwh_vs_yesterday: 0,
  power_ratio: 0,
  apparent_power: 1.15,
  load_factor: 1,
  volt_dev: 0,
  freq_dev: 0,
  group_kwh_zscore: 0,
  rolling_mean_kwh: 0,
  rolling_std_kwh: 0,
  voltage_drop: 0,
  load_spike: 0,
  supply_instability: 0,
};

const LABELS = {
  t_kWh: 'Energy (t_kWh)',
  'z_Avg Voltage (Volt)': 'Avg Voltage (V)',
  'z_Avg Current (Amp)': 'Avg Current (A)',
  'y_Freq (Hz)': 'Frequency (Hz)',
  hour: 'Hour of Day (0-23)',
  day_of_week: 'Day of Week (0=Mon)',
  is_weekend: 'Is Weekend (0/1)',
  is_peak_hour: 'Is Peak Hour (0/1)',
  kwh_change: 'Δ kWh (prev)',
  kwh_vs_yesterday: 'Δ kWh (t-96)',
  power_ratio: 'Power Ratio',
  apparent_power: 'Apparent Power (kVA)',
  load_factor: 'Load Factor',
  volt_dev: 'Voltage Deviation',
  freq_dev: 'Frequency Deviation',
  group_kwh_zscore: 'Peer kWh Z-Score',
  rolling_mean_kwh: 'Rolling Mean kWh',
  rolling_std_kwh: 'Rolling Std kWh',
  voltage_drop: 'Voltage Drop (0/1)',
  load_spike: 'Load Spike (0/1)',
  supply_instability: 'Supply Instability (0/1)',
};

const App = () => {
  const [features, setFeatures] = useState(initialFeatures);
  const [predictionResult, setPredictionResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFeatureChange = (e) => {
    const { id, value } = e.target;
    setFeatures((prev) => ({ ...prev, [id]: parseFloat(value) || 0 }));
  };

  const handlePrediction = async () => {
    setIsLoading(true);
    setError(null);
    setPredictionResult(null);

    // Build payload array in correct order
    const payload = { features: [] };
    for (const key of featureOrder) {
      const val = features[key];
      if (isNaN(val) || val === null) {
        setError('Please ensure all input fields contain valid numbers.');
        setIsLoading(false);
        return;
      }
      payload.features.push(val);
    }

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await response.json();

      if (!response.ok) {
        const msg = data.error || data.detail || `Server returned status ${response.status}`;
        setError(`Prediction Failed: ${msg}`);
        return;
      }

      setPredictionResult(data);
    } catch (err) {
      setError(`Network Error: Could not reach the API at ${API_URL}.`);
    } finally {
      setIsLoading(false);
    }
  };

  const isTheft = predictionResult?.is_theft;
  const resultColor = isTheft ? 'text-red-600 border-red-400' : 'text-green-600 border-green-400';
  const resultText = isTheft ? '🚨 ANOMALY DETECTED' : '✅ NORMAL';
  const anomalyScore = predictionResult?.anomaly_score ?? 'N/A';
  const explanation = predictionResult?.explanation;
  const contributions = predictionResult?.contributions || [];

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <div className="w-full max-w-5xl bg-white p-8 rounded-xl shadow-2xl space-y-8">
        <h1 className="text-3xl font-bold text-center text-gray-800">Energy Theft Anomaly Analysis</h1>
        <p className="text-gray-600 text-center">Enter engineered features for IsolationForest analysis.</p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {featureOrder.map((key, idx) => (
            <div key={key} className="flex items-center space-x-3">
              <label htmlFor={key} className="text-gray-700 font-medium w-[240px]">
                {idx + 1}. {LABELS[key]}
              </label>
              <input
                type="number"
                id={key}
                value={features[key]}
                onChange={handleFeatureChange}
                className="flex-grow p-2 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500"
                required
              />
            </div>
          ))}
        </div>

        <div>
          <button
            onClick={handlePrediction}
            disabled={isLoading}
            className="w-full py-3 mt-2 bg-blue-600 text-white font-bold rounded-lg hover:bg-blue-700 transition duration-150 shadow-md disabled:bg-blue-400"
          >
            {isLoading ? 'Predicting...' : 'Analyze Anomaly'}
          </button>
        </div>

        {(predictionResult || error) && (
          <div
            className={`mt-6 p-4 border rounded-lg ${error ? 'border-red-400' : ''}`}
            style={!error && predictionResult ? { borderColor: isTheft ? '#f87171' : '#4ade80' } : {}}
          >
            {error && <p className="text-sm text-red-500 font-medium">{error}</p>}

            {predictionResult && !error && (
              <>
                <p className={`font-bold text-lg ${resultColor}`}>{resultText}</p>
                <p className="text-sm text-gray-600 mt-1">Anomaly Score: {anomalyScore}</p>
                {explanation && (
                  <pre className="text-sm text-gray-800 mt-3 whitespace-pre-wrap leading-6 bg-gray-50 p-4 rounded-lg border border-gray-200">{explanation}</pre>
                )}
                {contributions.length > 0 && (
                  <div className="mt-3">
                    <p className="text-sm font-semibold text-gray-800 mb-1">Top contributing features:</p>
                    <ul className="text-sm text-gray-700 list-disc pl-5 space-y-1">
                      {contributions.map((c) => (
                        <li key={c.feature}>
                          <span className="font-medium">{c.feature}</span>: value={c.value}, shap={
                            typeof c.shap_value === 'number' ? c.shap_value.toFixed(3) : c.shap_value
                          }
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default App;