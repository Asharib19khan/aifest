"""
Fraud Detection with LightGBM
Complete pipeline including data cleaning, feature engineering, and model training
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, 
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, average_precision_score
)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("FRAUD DETECTION - LIGHTGBM MODEL")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1] Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# ============================================================================
# 2. DATA CLEANING & PREPROCESSING
# ============================================================================
print("\n[2] Data cleaning and preprocessing...")

def preprocess_data(df, is_train=True):
    """Clean and preprocess the data"""
    df = df.copy()
    
    # Convert datetime columns
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    
    # Handle missing values (if any)
    if df.isnull().sum().sum() > 0:
        print(f"   Handling {df.isnull().sum().sum()} missing values...")
        # Fill numerical columns with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = [c for c in categorical_cols if c not in ['signup_time', 'purchase_time']]
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df

train_df = preprocess_data(train_df, is_train=True)
test_df = preprocess_data(test_df, is_train=False)
print("   [OK] Data cleaned successfully")

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================
print("\n[3] Feature engineering...")

def create_features(df, is_train=True):
    """Create new features from existing data"""
    df = df.copy()
    
    # Time-based features
    df['time_to_purchase'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600  # hours
    df['signup_hour'] = df['signup_time'].dt.hour
    df['signup_day'] = df['signup_time'].dt.dayofweek
    df['signup_month'] = df['signup_time'].dt.month
    df['purchase_hour'] = df['purchase_time'].dt.hour
    df['purchase_day'] = df['purchase_time'].dt.dayofweek
    
    # Fast purchase flag (fraud tends to purchase quickly)
    df['is_fast_purchase'] = (df['time_to_purchase'] < 24).astype(int)
    
    # Hour features
    df['is_night_signup'] = ((df['signup_hour'] >= 22) | (df['signup_hour'] <= 6)).astype(int)
    df['is_night_purchase'] = ((df['purchase_hour'] >= 22) | (df['purchase_hour'] <= 6)).astype(int)
    
    # Weekend features
    df['is_weekend_signup'] = (df['signup_day'] >= 5).astype(int)
    df['is_weekend_purchase'] = (df['purchase_day'] >= 5).astype(int)
    
    # Age bins
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 100], labels=['young', 'mid', 'mature', 'senior'])
    
    # Purchase value bins
    df['purchase_category'] = pd.cut(df['purchase_value'], bins=[0, 20, 40, 60, 200], labels=['low', 'medium', 'high', 'very_high'])
    
    return df

train_df = create_features(train_df, is_train=True)
test_df = create_features(test_df, is_train=False)
print("   [OK] Features created successfully")
print(f"   New feature count: {train_df.shape[1] - 12} additional features")

# ============================================================================
# 4. ENCODE CATEGORICAL VARIABLES
# ============================================================================
print("\n[4] Encoding categorical variables...")

# Define categorical columns to encode
cat_cols = ['source', 'browser', 'sex', 'age_group', 'purchase_category']

# Label encoding
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    # Fit on combined train+test to handle all categories
    combined_values = pd.concat([train_df[col], test_df[col]]).astype(str)
    le.fit(combined_values)
    train_df[col] = le.transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))
    label_encoders[col] = le

print(f"   [OK] Encoded {len(cat_cols)} categorical columns")

# ============================================================================
# 5. PREPARE FEATURES AND TARGET
# ============================================================================
print("\n[5] Preparing features and target...")

# Drop unnecessary columns
drop_cols = ['signup_time', 'purchase_time', 'user_id', 'device_id', 'ip_address', 'ID']
if 'is_fraud' in drop_cols:
    drop_cols.remove('is_fraud')

# Feature columns
feature_cols = [col for col in train_df.columns if col not in drop_cols and col != 'is_fraud']

X = train_df[feature_cols]
y = train_df['is_fraud']
X_test = test_df[feature_cols]

print(f"   Features shape: {X.shape}")
print(f"   Target distribution:")
print(f"      Non-Fraud: {(y==0).sum():,} ({(y==0).mean()*100:.2f}%)")
print(f"      Fraud:     {(y==1).sum():,} ({(y==1).mean()*100:.2f}%)")
print(f"\n   Feature columns ({len(feature_cols)}):")
for i, col in enumerate(feature_cols, 1):
    print(f"      {i}. {col}")

# ============================================================================
# 6. LIGHTGBM MODEL WITH CROSS-VALIDATION
# ============================================================================
print("\n[6] Training LightGBM model with cross-validation...")

# Calculate scale_pos_weight for imbalanced data
scale_pos_weight = (y == 0).sum() / (y == 1).sum()
print(f"   Scale pos weight: {scale_pos_weight:.2f}")

# LightGBM parameters optimized for fraud detection
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'max_depth': 6,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'scale_pos_weight': scale_pos_weight,
    'verbose': -1,
    'random_state': 42
}

# Cross-validation setup
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Store metrics
cv_scores = []
cv_pr_auc_scores = []
cv_f1_scores = []
oof_predictions = np.zeros(len(X))
test_predictions = np.zeros(len(X_test))

print(f"\n   Performing {n_folds}-fold cross-validation...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n   Fold {fold}/{n_folds}:")
    
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)
    
    # Train model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )
    
    # Predictions
    val_pred = model.predict(X_val_fold, num_iteration=model.best_iteration)
    test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    # Store out-of-fold predictions
    oof_predictions[val_idx] = val_pred
    test_predictions += test_pred / n_folds
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_val_fold, val_pred)
    pr_auc = average_precision_score(y_val_fold, val_pred)
    
    # Convert to binary predictions for F1
    val_pred_binary = (val_pred > 0.5).astype(int)
    f1 = f1_score(y_val_fold, val_pred_binary)
    
    cv_scores.append(roc_auc)
    cv_pr_auc_scores.append(pr_auc)
    cv_f1_scores.append(f1)
    
    print(f"      ROC-AUC: {roc_auc:.4f}")
    print(f"      PR-AUC:  {pr_auc:.4f}")
    print(f"      F1-Score: {f1:.4f}")

# ============================================================================
# 7. FINAL MODEL EVALUATION
# ============================================================================
print("\n" + "="*80)
print("[7] FINAL MODEL EVALUATION")
print("="*80)

# Overall ROC-AUC
overall_roc_auc = roc_auc_score(y, oof_predictions)
overall_pr_auc = average_precision_score(y, oof_predictions)

print(f"\n   Cross-Validation Results:")
print(f"   ROC-AUC:  {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
print(f"   PR-AUC:   {np.mean(cv_pr_auc_scores):.4f} (+/- {np.std(cv_pr_auc_scores):.4f})")
print(f"   F1-Score: {np.mean(cv_f1_scores):.4f} (+/- {np.std(cv_f1_scores):.4f})")

print(f"\n   Overall Out-of-Fold Performance:")
print(f"   ROC-AUC: {overall_roc_auc:.4f}")
print(f"   PR-AUC:  {overall_pr_auc:.4f}")

# Find optimal threshold
precisions, recalls, thresholds = precision_recall_curve(y, oof_predictions)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"\n   Optimal Threshold: {optimal_threshold:.4f}")

# Predictions with optimal threshold
oof_predictions_binary = (oof_predictions > optimal_threshold).astype(int)

# Classification report
print("\n   Classification Report (with optimal threshold):")
print(classification_report(y, oof_predictions_binary, target_names=['Non-Fraud', 'Fraud']))

# Confusion Matrix
print("\n   Confusion Matrix:")
cm = confusion_matrix(y, oof_predictions_binary)
print(f"                 Predicted")
print(f"                 Non-Fraud  Fraud")
print(f"   Actual Non-Fraud  {cm[0][0]:6d}    {cm[0][1]:6d}")
print(f"   Actual Fraud      {cm[1][0]:6d}    {cm[1][1]:6d}")

# Precision, Recall, F1
precision = precision_score(y, oof_predictions_binary)
recall = recall_score(y, oof_predictions_binary)
f1 = f1_score(y, oof_predictions_binary)

print(f"\n   Detailed Metrics:")
print(f"   Precision: {precision:.4f} (of predicted frauds, how many are actual frauds)")
print(f"   Recall:    {recall:.4f} (of actual frauds, how many we detected)")
print(f"   F1-Score:  {f1:.4f} (harmonic mean of precision and recall)")

# ============================================================================
# 8. FEATURE IMPORTANCE
# ============================================================================
print("\n" + "="*80)
print("[8] FEATURE IMPORTANCE")
print("="*80)

# Train final model on all data to get feature importance
final_train_data = lgb.Dataset(X, label=y)
final_model = lgb.train(
    params,
    final_train_data,
    num_boost_round=model.best_iteration
)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': final_model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

print("\n   Top 15 Most Important Features:")
print(feature_importance.head(15).to_string(index=False))

# ============================================================================
# 9. GENERATE PREDICTIONS FOR TEST DATA
# ============================================================================
print("\n" + "="*80)
print("[9] GENERATING TEST PREDICTIONS")
print("="*80)

# Create submission file
submission = pd.DataFrame({
    'ID': test_df['ID'],
    'is_fraud': test_predictions
})

submission.to_csv('submission_lightgbm.csv', index=False)
print(f"\n   [OK] Predictions saved to 'submission_lightgbm.csv'")
print(f"   Predicted fraud rate: {(test_predictions > 0.5).mean()*100:.2f}%")

# Also save with optimal threshold
submission_optimal = pd.DataFrame({
    'ID': test_df['ID'],
    'is_fraud': (test_predictions > optimal_threshold).astype(int)
})
submission_optimal.to_csv('submission_lightgbm_optimal.csv', index=False)
print(f"   [OK] Predictions with optimal threshold saved to 'submission_lightgbm_optimal.csv'")
print(f"   Predicted fraud rate (optimal): {(test_predictions > optimal_threshold).mean()*100:.2f}%")

print("\n" + "="*80)
print("[DONE] MODEL TRAINING COMPLETE!")
print("="*80)
print("\nSummary:")
print(f"   - Model: LightGBM with {n_folds}-fold CV")
print(f"   - ROC-AUC: {overall_roc_auc:.4f}")
print(f"   - PR-AUC: {overall_pr_auc:.4f}")
print(f"   - Features used: {len(feature_cols)}")
print(f"   - Best iteration: {model.best_iteration}")
print(f"   - Submission files created: 2")
print("="*80)
