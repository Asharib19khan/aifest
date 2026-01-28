"""
Fraud Detection - Exploratory Data Analysis (EDA)
This script performs comprehensive EDA on train.csv and suggests ML models
for predicting the 'is_fraud' column
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("="*80)
print("FRAUD DETECTION - EXPLORATORY DATA ANALYSIS")
print("="*80)

# Load data
print("\n[1] Loading data...")
train_df = pd.read_csv('train.csv')
print(f"[OK] Data loaded successfully: {train_df.shape[0]:,} rows, {train_df.shape[1]} columns")

# ============================================================================
# BASIC DATA OVERVIEW
# ============================================================================
print("\n" + "="*80)
print("[2] BASIC DATA OVERVIEW")
print("="*80)

print("\n[DATA] Dataset Shape:", train_df.shape)
print("\n[INFO] Column Names and Types:")
print(train_df.dtypes)

print("\n[STATS] Basic Statistics:")
print(train_df.describe())

print("\n[CHECK] First 5 rows:")
print(train_df.head())

# ============================================================================
# MISSING VALUES ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("[3] MISSING VALUES ANALYSIS")
print("="*80)

missing_data = pd.DataFrame({
    'Column': train_df.columns,
    'Missing_Count': train_df.isnull().sum(),
    'Missing_Percentage': (train_df.isnull().sum() / len(train_df) * 100).round(2)
})
missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

if len(missing_data) > 0:
    print("\n[WARNING]  Columns with missing values:")
    print(missing_data.to_string(index=False))
else:
    print("\n[OK] No missing values found!")

# ============================================================================
# TARGET VARIABLE ANALYSIS (is_fraud)
# ============================================================================
print("\n" + "="*80)
print("[4] TARGET VARIABLE ANALYSIS - 'is_fraud'")
print("="*80)

fraud_counts = train_df['is_fraud'].value_counts()
fraud_pct = train_df['is_fraud'].value_counts(normalize=True) * 100

print("\n[TARGET] Target Distribution:")
print(f"   Non-Fraud (0): {fraud_counts[0]:,} ({fraud_pct[0]:.2f}%)")
print(f"   Fraud (1):     {fraud_counts[1]:,} ({fraud_pct[1]:.2f}%)")
print(f"\n[BALANCE]  Imbalance Ratio: 1:{fraud_counts[0]/fraud_counts[1]:.1f}")
print("   [WARNING]  This is a highly imbalanced dataset!")

# ============================================================================
# FEATURE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("[5] FEATURE ANALYSIS")
print("="*80)

# Categorical features
categorical_features = ['source', 'browser', 'sex', 'device_id']
print("\n[DATA] CATEGORICAL FEATURES:")
print("-" * 80)

for col in categorical_features:
    if col in train_df.columns:
        unique_count = train_df[col].nunique()
        print(f"\n{col.upper()}:")
        print(f"   Unique values: {unique_count}")
        if unique_count <= 10:
            print(f"   Value distribution:")
            print(train_df[col].value_counts().head(10).to_string())
            
            # Fraud rate by category
            print(f"\n   Fraud rate by {col}:")
            fraud_rate = train_df.groupby(col)['is_fraud'].agg(['sum', 'count', 'mean'])
            fraud_rate.columns = ['Fraud_Count', 'Total', 'Fraud_Rate']
            fraud_rate['Fraud_Rate'] = (fraud_rate['Fraud_Rate'] * 100).round(2)
            print(fraud_rate.sort_values('Fraud_Rate', ascending=False).to_string())

# Numerical features
numerical_features = ['purchase_value', 'age']
print("\n\n[STATS] NUMERICAL FEATURES:")
print("-" * 80)

for col in numerical_features:
    if col in train_df.columns:
        print(f"\n{col.upper()}:")
        print(f"   Min: {train_df[col].min()}")
        print(f"   Max: {train_df[col].max()}")
        print(f"   Mean: {train_df[col].mean():.2f}")
        print(f"   Median: {train_df[col].median():.2f}")
        print(f"   Std: {train_df[col].std():.2f}")
        
        # Compare fraud vs non-fraud
        print(f"\n   Comparison (Fraud vs Non-Fraud):")
        print(f"   Non-Fraud Mean: {train_df[train_df['is_fraud']==0][col].mean():.2f}")
        print(f"   Fraud Mean:     {train_df[train_df['is_fraud']==1][col].mean():.2f}")

# ============================================================================
# DATETIME FEATURES ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("[6] DATETIME FEATURES ANALYSIS")
print("="*80)

# Convert datetime columns
datetime_cols = ['signup_time', 'purchase_time']
for col in datetime_cols:
    if col in train_df.columns:
        train_df[col] = pd.to_datetime(train_df[col])

# Time difference between signup and purchase
train_df['time_to_purchase'] = (train_df['purchase_time'] - train_df['signup_time']).dt.total_seconds() / 3600  # in hours

print("\n[TIME] Time to Purchase (hours):")
print(f"   Mean: {train_df['time_to_purchase'].mean():.2f} hours")
print(f"   Median: {train_df['time_to_purchase'].median():.2f} hours")
print(f"\n   Non-Fraud Mean: {train_df[train_df['is_fraud']==0]['time_to_purchase'].mean():.2f} hours")
print(f"   Fraud Mean:     {train_df[train_df['is_fraud']==1]['time_to_purchase'].mean():.2f} hours")

# Extract temporal features
train_df['signup_hour'] = train_df['signup_time'].dt.hour
train_df['signup_day'] = train_df['signup_time'].dt.dayofweek
train_df['purchase_hour'] = train_df['purchase_time'].dt.hour

print("\n[DATE] Fraud rate by hour of day:")
fraud_by_hour = train_df.groupby('purchase_hour')['is_fraud'].mean() * 100
print(fraud_by_hour.sort_values(ascending=False).head(10).to_string())

# ============================================================================
# CORRELATIONS
# ============================================================================
print("\n" + "="*80)
print("[7] CORRELATION ANALYSIS")
print("="*80)

# Select numerical columns for correlation
numerical_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
if 'ID' in numerical_cols:
    numerical_cols.remove('ID')

correlation_with_target = train_df[numerical_cols].corr()['is_fraud'].sort_values(ascending=False)
print("\n[LINK] Correlation with 'is_fraud':")
print(correlation_with_target.to_string())

# ============================================================================
# KEY INSIGHTS
# ============================================================================
print("\n" + "="*80)
print("[8] KEY INSIGHTS & OBSERVATIONS")
print("="*80)

print("\n[CHECK] Key Findings:")
print("   1. Dataset is highly imbalanced (~7.8% fraud rate)")
print("   2. Time-to-purchase might be a strong indicator")
print("   3. Certain sources, browsers, and devices may have higher fraud rates")
print("   4. Age and purchase value distributions differ between fraud/non-fraud")
print("   5. Temporal patterns (hour of day) show varying fraud rates")

# ============================================================================
# ML MODEL RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("[9] MACHINE LEARNING MODEL RECOMMENDATIONS")
print("="*80)

print("\n[ML] RECOMMENDED MODELS FOR FRAUD DETECTION:")
print("\n" + "="*80)

recommendations = [
    {
        "rank": 1,
        "model": "LightGBM (Light Gradient Boosting Machine)",
        "reasons": [
            "[OK] Handles imbalanced data well with scale_pos_weight parameter",
            "[OK] Fast training and prediction",
            "[OK] Automatically handles categorical features",
            "[OK] Robust to outliers",
            "[OK] Can capture complex non-linear relationships"
        ],
        "params": "scale_pos_weight, max_depth, learning_rate, num_leaves"
    },
    {
        "rank": 2,
        "model": "XGBoost",
        "reasons": [
            "[OK] Excellent performance on tabular data",
            "[OK] Built-in handling for imbalanced data",
            "[OK] Feature importance analysis",
            "[OK] Regularization to prevent overfitting"
        ],
        "params": "scale_pos_weight, max_depth, learning_rate, subsample"
    },
    {
        "rank": 3,
        "model": "Random Forest",
        "reasons": [
            "[OK] Robust and interpretable",
            "[OK] Handles mixed data types well",
            "[OK] Can use class_weight='balanced' for imbalanced data",
            "[OK] Less prone to overfitting with enough trees"
        ],
        "params": "n_estimators, max_depth, class_weight='balanced'"
    },
    {
        "rank": 4,
        "model": "CatBoost",
        "reasons": [
            "[OK] Excellent with categorical features (native support)",
            "[OK] Auto handling of imbalanced data",
            "[OK] Minimal hyperparameter tuning needed",
            "[OK] Robust to overfitting"
        ],
        "params": "auto_class_weights, depth, learning_rate"
    },
    {
        "rank": 5,
        "model": "Logistic Regression with SMOTE",
        "reasons": [
            "[OK] Good baseline model",
            "[OK] SMOTE helps with imbalance",
            "[OK] Fast and interpretable",
            "[OK] Provides probability scores"
        ],
        "params": "C, class_weight='balanced', penalty='l2'"
    }
]

for rec in recommendations:
    print(f"\n{rec['rank']}. {rec['model']}")
    print(f"   {'-' * 70}")
    print("   Why use this model:")
    for reason in rec['reasons']:
        print(f"   {reason}")
    print(f"\n   Key parameters to tune: {rec['params']}")

print("\n" + "="*80)
print("[WARNING]  IMPORTANT CONSIDERATIONS:")
print("="*80)
print("""
1. IMBALANCE HANDLING:
   - Use class_weight or scale_pos_weight parameters
   - Consider SMOTE/ADASYN for oversampling
   - Use stratified K-fold cross-validation
   
2. EVALUATION METRICS (Don't rely on accuracy!):
   - Precision, Recall, F1-Score
   - ROC-AUC and PR-AUC (Precision-Recall AUC)
   - Confusion Matrix
   - Focus on reducing False Negatives (missing fraud)
   
3. FEATURE ENGINEERING:
   - Time-to-purchase (already calculated)
   - Frequency of user_id, device_id, ip_address
   - One-hot encoding for categorical features
   - Velocity features (purchases per time period)
   
4. CROSS-VALIDATION:
   - Use StratifiedKFold (5-10 folds)
   - Ensure both classes represented in each fold
   
5. ENSEMBLE METHODS:
   - Combine multiple models for better results
   - Stacking: LightGBM + XGBoost + Random Forest
""")

print("\n" + "="*80)
print("[DONE] EDA COMPLETE!")
print("="*80)
print("\nNext steps:")
print("1. Implement feature engineering")
print("2. Try LightGBM as the primary model")
print("3. Use proper evaluation metrics (ROC-AUC, PR-AUC)")
print("4. Apply cross-validation with stratification")
print("5. Handle class imbalance appropriately")
print("\n" + "="*80)
