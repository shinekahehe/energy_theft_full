# ================================================================
# 🌟 AI-Based Reliable and Explainable Energy Theft Detection
# Phase 1: Unsupervised Detection (Isolation Forest)
# ================================================================

# --- Imports ---
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib

# ================================================================
# STEP 1: Load and Prepare Dataset
# ================================================================
file_path = r"C:\Users\srish\OneDrive\Documents\energy_theft_data\CEEW - Smart meter data Mathura 2019.csv"

df = pd.read_csv(
    file_path,
    parse_dates=['x_Timestamp'],
    dtype={
        't_kWh': np.float32,
        'z_Avg Voltage (Volt)': np.float32,
        'z_Avg Current (Amp)': np.float32,
        'y_Freq (Hz)': np.float16,
        'meter': 'category'
    }
)

df = df.set_index('x_Timestamp').dropna()
print("✅ SUCCESS! Data loaded.")
print(df.head())
print(df.info())

# ================================================================
# STEP 2: Downsample to 15-Minute Intervals
# ================================================================
df = df.reset_index()

aggregation_rules = {
    't_kWh': 'sum',
    'z_Avg Voltage (Volt)': 'mean',
    'z_Avg Current (Amp)': 'mean',
    'y_Freq (Hz)': 'mean',
}

df_15min = (
    df.groupby('meter')
    .resample('15min', on='x_Timestamp')
    .agg(aggregation_rules)
    .reset_index()
    .dropna()
    .set_index('x_Timestamp')
)

print("✅ Step 2: Downsampling Complete.")
print(f"Readings reduced from {len(df):,} → {len(df_15min):,}")

# ================================================================
# STEP 3: Feature Engineering
# ================================================================

# --- A. Time-Based Context Features ---
PEAK_START, PEAK_END = 7, 22

df_15min['hour'] = df_15min.index.hour
df_15min['day_of_week'] = df_15min.index.dayofweek
df_15min['is_weekend'] = (df_15min['day_of_week'] >= 5).astype(int)
df_15min['is_peak_hour'] = ((df_15min['hour'] >= PEAK_START) &
                            (df_15min['hour'] < PEAK_END)).astype(int)

# --- B. Lag Features (Per Meter) ---
df_15min['kwh_change'] = (
    df_15min.groupby('meter')['t_kWh'].diff(periods=1).fillna(0)
)
df_15min['kwh_vs_yesterday'] = (
    df_15min.groupby('meter')['t_kWh'].diff(periods=96).fillna(0)
)

# --- C. Apparent Power, Load Factor & Power Ratio ---
df_15min['apparent_power'] = (
    df_15min['z_Avg Voltage (Volt)'] * df_15min['z_Avg Current (Amp)'] / 1000
)

df_15min['load_factor'] = (
    df_15min['t_kWh'] / (df_15min['apparent_power'].replace(0, 1e-6))
).clip(0, 2)

df_15min['power_ratio'] = (
    (df_15min['z_Avg Voltage (Volt)'] * df_15min['z_Avg Current (Amp)'])
    / (df_15min['t_kWh'].replace(0, 1e-6))
).replace([np.inf, -np.inf], 0)

# --- D. Voltage & Frequency Deviations ---
df_15min['volt_dev'] = np.abs(df_15min['z_Avg Voltage (Volt)'] - 230) / 230
df_15min['freq_dev'] = np.abs(df_15min['y_Freq (Hz)'] - 50) / 50

# --- E. Peer Comparison (Z-score within time window) ---
group_stats = df_15min.groupby(df_15min.index)['t_kWh'].agg(['mean', 'std'])
group_stats.rename(columns={'mean': 'group_mean_kwh', 'std': 'group_std_kwh'}, inplace=True)
df_15min = df_15min.merge(group_stats, left_index=True, right_index=True, how='left')

df_15min['group_kwh_zscore'] = (
    df_15min['t_kWh'] - df_15min['group_mean_kwh']
) / df_15min['group_std_kwh'].replace(0, 1e-6)

df_15min['group_kwh_zscore'] = (
    df_15min['group_kwh_zscore'].replace([np.inf, -np.inf], 0).fillna(0)
)
df_15min.drop(columns=['group_mean_kwh', 'group_std_kwh'], inplace=True)

# --- F. Rolling Statistics (24-hour window = 96 samples) ---
df_15min['rolling_mean_kwh'] = (
    df_15min.groupby('meter')['t_kWh']
    .transform(lambda x: x.rolling(window=96, min_periods=1).mean())
)
df_15min['rolling_std_kwh'] = (
    df_15min.groupby('meter')['t_kWh']
    .transform(lambda x: x.rolling(window=96, min_periods=1).std())
)

# Fill early NaNs
df_15min[['rolling_mean_kwh', 'rolling_std_kwh']] = df_15min[['rolling_mean_kwh', 'rolling_std_kwh']].fillna(0)

# --- G. Future Additions (Voltage Drop & Load Spike Placeholders) ---
# TODO: Add voltage_drop and load_spike detection next
# df_15min['voltage_drop'] = ...
# df_15min['load_spike'] = ...

print("✅ Step 3: Feature Engineering Complete.")
print("\nFeature summary:\n", df_15min.describe().T[['mean', 'std', 'min', 'max']].head(10))

# ================================================================
# STEP 4: Unsupervised Anomaly Detection (Isolation Forest)
# ================================================================

# --- Select Features for Model ---
features = [
    't_kWh', 'z_Avg Voltage (Volt)', 'z_Avg Current (Amp)', 'y_Freq (Hz)',
    'hour', 'day_of_week', 'is_weekend', 'is_peak_hour',
    'kwh_change', 'kwh_vs_yesterday', 'power_ratio',
    'apparent_power', 'load_factor', 'volt_dev', 'freq_dev',
    'group_kwh_zscore', 'rolling_mean_kwh', 'rolling_std_kwh'
]

X = df_15min[features].copy()

# --- Scale Features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Isolation Forest Model ---
model = IsolationForest(
    contamination=0.00923,  # ~0.9% anomalies
    random_state=42,
    n_jobs=-1,
    max_samples=0.8  # adds robustness
)
model.fit(X_scaled)

# --- Predictions ---
df_15min['anomaly_score'] = model.decision_function(X_scaled)
df_15min['anomaly_label'] = model.predict(X_scaled)  # -1 = Anomaly

theft_candidates = df_15min[df_15min['anomaly_label'] == -1]
theft_candidates = theft_candidates.sort_values(by='anomaly_score')

# --- Save Model, Scaler, and Results ---
# import os

# output_dir = r"D:\energy_theft\outputs"
# os.makedirs(output_dir, exist_ok=True)

# joblib.dump(model, os.path.join(output_dir, "energy_theft_isoforest.pkl"))
# joblib.dump(scaler, os.path.join(output_dir, "energy_theft_scaler.pkl"))
# df_15min.reset_index().to_csv(os.path.join(output_dir, "energy_theft_results.csv"), index=False)

# print(f"\n📁 Results saved in: {output_dir}")


# --- Results Summary ---
print("\n✅ Step 4: Isolation Forest Model Trained.")
print(f"Total potential theft events flagged: {len(theft_candidates):,}")
print("\n--- TOP 10 Potential Theft Candidates ---")
print(
    theft_candidates.reset_index().head(10)[
        ['meter', 'x_Timestamp', 't_kWh', 'group_kwh_zscore', 'anomaly_score']
    ]
)
print(theft_candidates.head(10).to_string(index=False))


# ================================================================
# STEP 3.1: Voltage Drop & Load Spike Detection
# ================================================================

# --- Nominal voltage (can be 230V phase or 415V line) ---
NOMINAL_VOLTAGE = 230.0

# --- Voltage Drop: Detect sudden fall below nominal levels ---
df_15min['voltage_drop'] = (
    (df_15min['z_Avg Voltage (Volt)'] < 0.9 * NOMINAL_VOLTAGE).astype(int)
)

# --- Sudden Voltage Change Rate ---
df_15min['volt_change_rate'] = (
    df_15min.groupby('meter')['z_Avg Voltage (Volt)'].diff().abs()
)

# --- Load Spike: Detect sharp kWh or current jumps ---
df_15min['load_spike'] = (
    (
        (df_15min['kwh_change'] > (df_15min['rolling_std_kwh'] * 3))
        | (df_15min.groupby('meter')['z_Avg Current (Amp)'].diff().abs() > 15)
    ).astype(int)
)

# --- Combined Power Quality Indicator ---
df_15min['supply_instability'] = (
    df_15min['voltage_drop'] + df_15min['load_spike']
)

print("✅ Step 3.1: Voltage Drop and Load Spike Detection Added.")
print(df_15min[['voltage_drop', 'load_spike', 'supply_instability']].sum())

# ================================================================
# STEP 3.2: Update Feature List & Proceed to Anomaly Detection
# ================================================================

# --- Add new features to model input ---
features = [
    't_kWh', 'z_Avg Voltage (Volt)', 'z_Avg Current (Amp)', 'y_Freq (Hz)',
    'hour', 'day_of_week', 'is_weekend', 'is_peak_hour',
    'kwh_change', 'kwh_vs_yesterday',
    'power_ratio', 'apparent_power', 'load_factor',
    'volt_dev', 'freq_dev', 'group_kwh_zscore',
    'rolling_mean_kwh', 'rolling_std_kwh',
    'voltage_drop', 'load_spike', 'supply_instability'   # 👈 newly added
]

# --- Scale and train Isolation Forest as before ---
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

X = df_15min[features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = IsolationForest(
    contamination=0.00923,  # ~0.9% anomalies
    random_state=42,
    n_jobs=-1,
    max_samples=0.8
)
model.fit(X_scaled)

# --- Predictions ---
df_15min['anomaly_score'] = model.decision_function(X_scaled)
df_15min['anomaly_label'] = model.predict(X_scaled)  # -1 = anomaly
df_15min['anomaly_label'] = df_15min['anomaly_label'].replace({1: 0, -1: 1})

print("✅ Step 3.2: Model retrained with new voltage/load features.")
print("Potential theft events flagged:", df_15min['anomaly_label'].sum())

# --- Save updated artifacts ---
import os

output_dir = r"D:\energy_theft\outputs"
os.makedirs(output_dir, exist_ok=True)

joblib.dump(model, os.path.join(output_dir, "energy_theft_isoforest_v2.pkl"))
joblib.dump(scaler, os.path.join(output_dir, "energy_theft_scaler_v2.pkl"))
df_15min.reset_index().to_csv(os.path.join(output_dir, "energy_theft_results_v2.csv"), index=False)

print(f"""
✅ Step 3.2: Model retrained with new voltage/load features.
📁 Saved files:
   • Model:   {os.path.join(output_dir, 'energy_theft_isoforest_v2.pkl')}
   • Scaler:  {os.path.join(output_dir, 'energy_theft_scaler_v2.pkl')}
   • Results: {os.path.join(output_dir, 'energy_theft_results_v2.csv')}
""")


# --- Quick peek at top anomalies ---
theft_candidates = (
    df_15min[df_15min['anomaly_label'] == 1]
    .sort_values(by='anomaly_score')
    .head(10)
)[['meter', 't_kWh', 'voltage_drop', 'load_spike', 'supply_instability', 'anomaly_score']]

print("\n--- TOP 10 Potential Theft Candidates ---")
print(theft_candidates.to_string(index=False))
