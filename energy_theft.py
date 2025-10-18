# Data handling
import pandas as pd
import numpy as np
import os
import datetime
import warnings

# Visualization and Modeling
import matplotlib
# CRITICAL FIX for headless environment: Use the 'Agg' backend to save plots to files
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from imblearn.over_sampling import SMOTE 
import optuna
import joblib
import sklearn.utils

warnings.filterwarnings("ignore")

# --- CONFIGURATION & FILE PATHS ---
PROJECT_ROOT = r"C:\Users\srish\OneDrive\Documents\data set.csv\data set.csv"
RAW_INPUT_FILE = PROJECT_ROOT
CLEANED_OUTPUT_FILE = r"C:\Users\srish\OneDrive\Documents\cleaned_features.csv"
PLOTS_DIR = r"C:\Users\srish\OneDrive\Documents\energy_theft_data\plots"

CUSTOMER_ID_COL_NAME = 'CONS_NO'
FLAG_COLUMN = 'FLAG'
VALUE_COL_NAME = 'Consumption' 

# Ensure plots directory exists
os.makedirs(PLOTS_DIR, exist_ok=True)

print(f"--- STARTING DATA PIPELINE ---")

# --- 1. DATA LOADING AND TRANSFORMATION (FIXED FOR REAL FLAG) ---

# 1. Load the CSV, assuming no proper header
df = pd.read_csv(RAW_INPUT_FILE, header=None, low_memory=False)

# 2. Promote first row to headers and clean
# NOTE: We assume the FLAG column is now one of the columns in the first row.
new_header = df.iloc[0].tolist()
new_header[0] = CUSTOMER_ID_COL_NAME 
df.columns = new_header
df = df[1:].reset_index(drop=True)
df.columns = df.columns.astype(str).str.strip()
df = df.loc[:, ~df.columns.duplicated()]

# 3. Reshape dataset (Melt)
# CRITICAL CHANGE: Keep both CONS_NO and FLAG as ID variables
df_long = df.melt(
    id_vars=[CUSTOMER_ID_COL_NAME, FLAG_COLUMN], 
    var_name="Date",
    value_name=VALUE_COL_NAME
)

# 4. Convert and clean consumption values
df_long[VALUE_COL_NAME] = pd.to_numeric(df_long[VALUE_COL_NAME], errors='coerce')
df_long.dropna(subset=[VALUE_COL_NAME], inplace=True)

# 5. Convert date strings to datetime
df_long["Date"] = pd.to_datetime(df_long["Date"], errors="coerce", dayfirst=True)
df_long.dropna(subset=["Date"], inplace=True)

# 6. Convert FLAG to numeric (if it wasn't already)
df_long[FLAG_COLUMN] = pd.to_numeric(df_long[FLAG_COLUMN], errors='coerce').astype('Int64')
df_long.dropna(subset=[FLAG_COLUMN], inplace=True)

# --- 2. DATA CLEANING AND LABELING (REMOVED RANDOM ASSIGNMENT) ---

# 7. Imputation and Outlier Treatment
df_long[VALUE_COL_NAME] = df_long.groupby(CUSTOMER_ID_COL_NAME)[VALUE_COL_NAME].transform(
    lambda x: x.interpolate(method='linear', limit_direction="both")
)
valid_customers = df_long.groupby(CUSTOMER_ID_COL_NAME)[VALUE_COL_NAME].sum()
valid_customers = valid_customers[valid_customers > 0].index
df_long = df_long[df_long[CUSTOMER_ID_COL_NAME].isin(valid_customers)]

q_low, q_high = df_long[VALUE_COL_NAME].quantile([0.01, 0.99])
df_long[VALUE_COL_NAME] = df_long[VALUE_COL_NAME].clip(lower=q_low, upper=q_high)

# --- 3. FEATURE ENGINEERING (Z-SCORE & Simplified Time-Series) ---

# 1. Feature Creation on df_long
df_long['Consumption_Diff'] = df_long.groupby(CUSTOMER_ID_COL_NAME)[VALUE_COL_NAME].diff()
df_long['Consumption_Lag1'] = df_long.groupby(CUSTOMER_ID_COL_NAME)[VALUE_COL_NAME].shift(1)
df_long['Month'] = df_long['Date'].dt.month

# 2. Aggregation (Creating the Features DataFrame)
features = df_long.groupby(CUSTOMER_ID_COL_NAME).agg(
    # Basic Aggregates
    cons_mean=(VALUE_COL_NAME, 'mean'),
    cons_total=(VALUE_COL_NAME, 'sum'),
    
    # Advanced: Volatility of Daily Changes
    diff_std=('Consumption_Diff', 'std'),
    
    # Advanced: Stability (Correlation between today and yesterday)
    lag1_corr=('Consumption_Lag1', lambda x: x.corr(df_long.loc[x.index, VALUE_COL_NAME])),
    
    # Advanced: Seasonal Variability 
    month_std=('Consumption', lambda x: x.groupby(df_long.loc[x.index, 'Month']).std().mean()),
    
    # Target Flag (Max works because 0/1 are constant per customer)
    FLAG=(FLAG_COLUMN, 'max')
).reset_index()

# --- NEW: Z-SCORE OUTLIER FEATURE ---
features['cons_total_zscore'] = (
    features['cons_total'] - features['cons_total'].mean()
) / features['cons_total'].std()
# --- END NEW Z-SCORE FEATURE ---

# Final cleaning and saving
features = features.fillna(0)

features.columns = ['CONS_NO', 'cons_mean', 'cons_total', 
                    'diff_std', 'lag1_corr', 'month_std', 'FLAG', 'cons_total_zscore']

print("Successfully created advanced Z-Score and time-series features.")
features.to_csv(CLEANED_OUTPUT_FILE, index=False)

# --- 4. DATA AUGMENTATION FOR BALANCED TRAINING ---

print("\n--- DATA AUGMENTATION ---")
print("FLAG distribution in original data:")
print(df["FLAG"].value_counts())

# Separate theft rows
theft_data = df[df["FLAG"] == 1]

# Set ratio (1 for equal, 2 for double, etc.)
ratio = 1
n_synthetic = int(len(theft_data) * ratio)

# Sample with replacement
theft_sampled = theft_data.sample(n=n_synthetic, replace=True, random_state=42)

# Add small noise to numeric features
for col in theft_sampled.select_dtypes(include=['float64', 'int64']).columns:
    if col != 'FLAG':  # Don't add noise to the target variable
        theft_sampled[col] += np.random.normal(0, 0.05 * df[col].std(), size=n_synthetic)

# Combine original + synthetic
df_augmented = pd.concat([df, theft_sampled], ignore_index=True)
print("Original theft:", len(theft_data))
print("Synthetic theft:", n_synthetic)
print("Final dataset shape:", df_augmented.shape)

# --- 5. PREPARE DATA FOR MODELING ---

# Separate features and target
X = df_augmented.drop("FLAG", axis=1)
y = df_augmented["FLAG"]

# Shuffle entire dataset first
X, y = sklearn.utils.shuffle(X, y, random_state=42)

# Ensure target is integer
y = y.astype(int)

# Drop ID column
X = X.drop(columns=['CONS_NO'])

# Shuffle entire dataset
X, y = sklearn.utils.shuffle(X, y, random_state=42)

# Train/test split (stratify by FLAG)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Train class counts:\n", y_train.value_counts())
print("Test class counts:\n", y_test.value_counts())

# Ensure target is integer 0/1
y_train = y_train.astype(int)
y_test = y_test.astype(int)

# Automatically convert object columns to category
for col in X_train.select_dtypes(include='object').columns:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# --- 6. MODEL TRAINING ---

print("\n--- TRAINING XGBOOST MODEL ---")

# Calculate scale_pos_weight for imbalanced data
scale_pos_weight = (y_train==0).sum() / (y_train==1).sum()
print(f"Scale pos weight: {scale_pos_weight}")

xgb_clf = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    enable_categorical=True  # important
)

xgb_clf.fit(X_train, y_train)

# Predict classes
y_pred = xgb_clf.predict(X_test)

# Predict probabilities for ROC-AUC
y_proba = xgb_clf.predict_proba(X_test)[:, 1]  # probability of FLAG=1

# --- 7. MODEL EVALUATION ---

print("\n--- MODEL EVALUATION ---")

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC Score: {roc_auc}")

# --- 8. SAVE THE MODEL ---

model_filename = 'energy_theft_model.joblib'
joblib.dump(xgb_clf, model_filename)
print(f"\n✅ Trained model successfully saved to {model_filename}")

# --- 9. VISUALIZATION ---

print(f"\n--- VISUALIZATION (Saving Plots to files in the '{PLOTS_DIR}' directory) ---")

# Plot 1: Class Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x="FLAG", data=features)
plt.title("Class Distribution (Normal vs Theft)")
plt.savefig(os.path.join(PLOTS_DIR, '01_class_distribution_final.png'))
plt.close()

# Plot 2: Distribution of Daily Consumption Variability (diff_std)
plt.figure(figsize=(8, 5))
sns.kdeplot(data=features, x="diff_std", hue="FLAG", fill=True)
plt.title("Distribution of Daily Consumption Variability (diff_std)")
plt.savefig(os.path.join(PLOTS_DIR, '02_diff_std_distribution.png'))
plt.close()

# Plot 3: Distribution of Total Consumption Z-Score (cons_total_zscore)
plt.figure(figsize=(8, 5))
sns.kdeplot(data=features, x="cons_total_zscore", hue="FLAG", fill=True)
plt.title("Distribution of Total Consumption Z-Score")
plt.savefig(os.path.join(PLOTS_DIR, '03_zscore_distribution.png'))
plt.close()

# Plot 4: Feature Correlation Heatmap
corr = features.drop([CUSTOMER_ID_COL_NAME, FLAG_COLUMN], axis=1).corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap (Advanced Features)")
plt.savefig(os.path.join(PLOTS_DIR, '04_feature_correlation_heatmap.png'))
plt.close()

# Plot 5: Consumption Trends for Sample Customers
sample_customers = df_long[CUSTOMER_ID_COL_NAME].unique()[:3]
for i, cust in enumerate(sample_customers):
    cust_data = df_long[df_long[CUSTOMER_ID_COL_NAME] == cust]

    if not cust_data.empty:
        plt.figure(figsize=(12,4))
        plt.plot(cust_data["Date"], cust_data[VALUE_COL_NAME])
        plt.title(f"Consumption Trend - Customer {cust}")
        plt.xlabel("Date")
        plt.ylabel("Consumption (kWh)")
        plt.savefig(os.path.join(PLOTS_DIR, f'05_customer_{i+1}_trend.png'))
        plt.close()

# Plot 6: Total Daily Energy Consumption
daily_total = df_long.groupby("Date")[VALUE_COL_NAME].sum()
plt.figure(figsize=(12,5))
plt.plot(daily_total.index, daily_total.values)
plt.title("Total Daily Energy Consumption (Aggregate)")
plt.xlabel("Date")
plt.ylabel("Total Consumption (kWh)")
plt.savefig(os.path.join(PLOTS_DIR, '06_total_daily_consumption.png'))
plt.close()

print(f"✅ All 6 plots saved to the '{PLOTS_DIR}' directory.")
print("--- PIPELINE COMPLETE ---")

# --- 10. PREDICTION FUNCTION FOR API ---

def predict_energy_theft(features_dict):
    """
    Function to predict energy theft for new data
    Args:
        features_dict: Dictionary with keys ['cons_mean', 'cons_total', 'diff_std', 'lag1_corr', 'month_std', 'cons_total_zscore']
    Returns:
        dict: Prediction result with 'prediction', 'probability', 'is_theft'
    """
    try:
        # Convert to DataFrame
        feature_df = pd.DataFrame([features_dict])
        
        # Make prediction
        prediction = xgb_clf.predict(feature_df)[0]
        probability = xgb_clf.predict_proba(feature_df)[0][1]
        
        return {
            'prediction': int(prediction),
            'probability': float(probability),
            'is_theft': bool(prediction)
        }
    except Exception as e:
        return {
            'error': f'Prediction failed: {str(e)}',
            'prediction': 0,
            'probability': 0.0,
            'is_theft': False
        }

print("\n✅ Prediction function ready for API integration")