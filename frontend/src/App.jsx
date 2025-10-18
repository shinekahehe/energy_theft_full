import React, { useState } from 'react';

// Configuration
const API_URL = 'http://localhost:3001/api/predict_theft'; // Local Node.js backend

// Define the structure and initial state for the 6 features
const initialFeatures = {
    cons_mean: 250.5,
    cons_total: 75150.0,
    diff_std: 5.12,
    lag1_corr: 0.95,
    month_std: 1.5,
    cons_total_zscore: 0.55,
};

// Define the display labels and descriptions for the features
const FEATURE_INFO = {
    cons_mean: {
        label: 'Average Consumption',
        description: 'Mean energy consumption over time',
        unit: 'kWh',
        icon: '📊'
    },
    cons_total: {
        label: 'Total Consumption',
        description: 'Cumulative energy consumption',
        unit: 'kWh',
        icon: '🔋'
    },
    diff_std: {
        label: 'Daily Change StDev',
        description: 'Variability in daily consumption changes',
        unit: 'kWh',
        icon: '📈'
    },
    lag1_corr: {
        label: 'Lag-1 Correlation',
        description: 'Correlation between consecutive days',
        unit: 'ratio',
        icon: '🔗'
    },
    month_std: {
        label: 'Monthly StDev',
        description: 'Seasonal variability in consumption',
        unit: 'kWh',
        icon: '📅'
    },
    cons_total_zscore: {
        label: 'Total Consumption Z-Score',
        description: 'Standardized total consumption score',
        unit: 'z-score',
        icon: '⚡'
    },
};

// Main App Component
const App = () => {
    const [features, setFeatures] = useState(initialFeatures);
    const [predictionResult, setPredictionResult] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [showResult, setShowResult] = useState(false);

    // Handler for updating any feature input field
    const handleFeatureChange = (e) => {
        const { id, value } = e.target;
        setFeatures(prev => ({
            ...prev,
            [id]: parseFloat(value) || 0,
        }));
    };

    const handlePrediction = async () => {
        setIsLoading(true);
        setError(null);
        setPredictionResult(null);
        setShowResult(false);

        // Validate input and prepare payload
        const payload = { features: [] };
        let isValid = true;
        
        const featureOrder = ['cons_mean', 'cons_total', 'diff_std', 'lag1_corr', 'month_std', 'cons_total_zscore'];
        
        for (const key of featureOrder) {
            const value = features[key];
            if (isNaN(value) || value === null) {
                isValid = false;
                break;
            }
            payload.features.push(value);
        }
        
        if (!isValid) {
            setError("Please ensure all 6 input fields contain valid numbers.");
            setIsLoading(false);
            return;
        }

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (!response.ok) {
                const errorMessage = data.error || `Server returned status ${response.status}`;
                setError(`Prediction Failed: ${errorMessage}`);
                return;
            }

            setPredictionResult(data);
            setShowResult(true);

        } catch (err) {
            setError(`Network Error: Could not reach the API at ${API_URL}. Is your Flask app running?`);
        } finally {
            setIsLoading(false);
        }
    };

    // Determine result styling
    const isTheft = predictionResult?.is_theft;
    const probabilityScore = predictionResult ? (predictionResult.probability * 100).toFixed(2) : 'N/A';
    
    return (
        <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
            {/* Background Pattern */}
            <div className="absolute inset-0 bg-[url('data:image/svg+xml,%3Csvg width="60" height="60" viewBox="0 0 60 60" xmlns="http://www.w3.org/2000/svg"%3E%3Cg fill="none" fill-rule="evenodd"%3E%3Cg fill="%239C92AC" fill-opacity="0.05"%3E%3Ccircle cx="30" cy="30" r="2"/%3E%3C/g%3E%3C/g%3E%3C/svg%3E')] opacity-40"></div>
            
            <div className="relative flex items-center justify-center min-h-screen p-4">
                <div className="w-full max-w-4xl">
                    {/* Header Section */}
                    <div className="text-center mb-12">
                        <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full mb-6 shadow-lg">
                            <svg className="w-10 h-10 text-white" fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M11.3 1.046A1 1 0 0112 2v2.102a.853.853 0 001.386.581L16 6.882V18a1 1 0 01-2 0V7.22l-2.614-1.2A.853.853 0 0010 6v3.118a1 1 0 01-2 0V2a1 1 0 011.3-.954z" clipRule="evenodd"></path>
                            </svg>
                        </div>
                        <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-4">
                            Energy Theft Detection
                        </h1>
                        <p className="text-xl text-gray-600 max-w-2xl mx-auto leading-relaxed">
                            Advanced AI-powered system to detect and predict energy theft patterns using machine learning algorithms
                        </p>
                    </div>

                    {/* Main Card */}
                    <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-2xl border border-white/20 overflow-hidden">
                        <div className="p-8">
                            {/* Input Section */}
                            <div className="mb-8">
                                <h2 className="text-2xl font-semibold text-gray-800 mb-6 flex items-center">
                                    <span className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center mr-3">
                                        <span className="text-blue-600 font-bold">1</span>
                                    </span>
                                    Energy Consumption Parameters
                                </h2>
                                
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    {Object.keys(initialFeatures).map((key, index) => (
                                        <div key={key} className="group">
                                            <div className="bg-gradient-to-r from-gray-50 to-gray-100 rounded-2xl p-6 border border-gray-200 hover:border-blue-300 transition-all duration-300 hover:shadow-lg">
                                                <div className="flex items-center mb-3">
                                                    <span className="text-2xl mr-3">{FEATURE_INFO[key].icon}</span>
                                                    <div>
                                                        <label htmlFor={key} className="block text-sm font-semibold text-gray-700">
                                                            {FEATURE_INFO[key].label}
                                                        </label>
                                                        <span className="text-xs text-gray-500">{FEATURE_INFO[key].unit}</span>
                                                    </div>
                                                </div>
                                                <input
                                                    type="number"
                                                    id={key}
                                                    value={features[key]}
                                                    onChange={handleFeatureChange}
                                                    className="w-full px-4 py-3 bg-white border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 text-lg font-medium"
                                                    placeholder={`Enter ${FEATURE_INFO[key].label.toLowerCase()}`}
                                                    step="0.01"
                                                />
                                                <p className="text-xs text-gray-500 mt-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                                                    {FEATURE_INFO[key].description}
                                                </p>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>

                            {/* Action Button */}
                            <div className="text-center mb-8">
                                <button 
                                    onClick={handlePrediction} 
                                    disabled={isLoading}
                                    className="group relative inline-flex items-center justify-center px-12 py-4 text-lg font-semibold text-white bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl shadow-lg hover:shadow-xl transform hover:-translate-y-1 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
                                >
                                    {isLoading ? (
                                        <>
                                            <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                            </svg>
                                            Analyzing Data...
                                        </>
                                    ) : (
                                        <>
                                            <svg className="w-5 h-5 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                                            </svg>
                                            Predict Theft Likelihood
                                        </>
                                    )}
                                </button>
                            </div>

                            {/* Result Section */}
                            {(predictionResult || error) && (
                                <div className={`transition-all duration-500 ${showResult ? 'opacity-100 transform translate-y-0' : 'opacity-0 transform translate-y-4'}`}>
                                    <h2 className="text-2xl font-semibold text-gray-800 mb-6 flex items-center">
                                        <span className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center mr-3">
                                            <span className="text-green-600 font-bold">2</span>
                                        </span>
                                        Analysis Results
                                    </h2>
                                    
                                    {error ? (
                                        <div className="bg-red-50 border border-red-200 rounded-2xl p-6">
                                            <div className="flex items-center">
                                                <svg className="w-6 h-6 text-red-500 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                                </svg>
                                                <p className="text-red-700 font-medium">{error}</p>
                                            </div>
                                        </div>
                                    ) : predictionResult && (
                                        <div className={`rounded-2xl p-8 border-2 ${
                                            isTheft 
                                                ? 'bg-gradient-to-r from-red-50 to-orange-50 border-red-200' 
                                                : 'bg-gradient-to-r from-green-50 to-emerald-50 border-green-200'
                                        }`}>
                                            <div className="text-center">
                                                <div className={`inline-flex items-center justify-center w-20 h-20 rounded-full mb-6 ${
                                                    isTheft 
                                                        ? 'bg-gradient-to-r from-red-500 to-orange-500' 
                                                        : 'bg-gradient-to-r from-green-500 to-emerald-500'
                                                }`}>
                                                    {isTheft ? (
                                                        <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
                                                        </svg>
                                                    ) : (
                                                        <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                                                        </svg>
                                                    )}
                                                </div>
                                                
                                                <h3 className={`text-3xl font-bold mb-4 ${
                                                    isTheft ? 'text-red-700' : 'text-green-700'
                                                }`}>
                                                    {isTheft ? '🚨 HIGH RISK DETECTED' : '✅ LOW RISK - NORMAL'}
                                                </h3>
                                                
                                                <div className="bg-white/60 backdrop-blur-sm rounded-xl p-6 mb-6">
                                                    <div className="text-4xl font-bold text-gray-800 mb-2">
                                                        {probabilityScore}%
                                                    </div>
                                                    <div className="text-gray-600 font-medium">
                                                        Theft Probability
                                                    </div>
                                                </div>
                                                
                                                <div className={`inline-flex items-center px-6 py-3 rounded-full text-sm font-semibold ${
                                                    isTheft 
                                                        ? 'bg-red-100 text-red-800' 
                                                        : 'bg-green-100 text-green-800'
                                                }`}>
                                                    {isTheft 
                                                        ? '⚠️ Immediate investigation recommended' 
                                                        : '✅ Customer behavior appears normal'
                                                    }
                                                </div>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Footer */}
                    <div className="text-center mt-12">
                        <p className="text-gray-500 text-sm">
                            Powered by Advanced Machine Learning • XGBoost Algorithm • Real-time Analysis
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default App;