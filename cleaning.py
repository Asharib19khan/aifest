import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv('data.csv')

def quick_clean(df, target_col):
    print("-- Initial Data Check --")
    print(df.info())
    
    df = df.drop(columns=[col for col in df.columns if 'id' in col.lower() or 'name' in col.lower()], errors='ignore')

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].fillna('Missing')

    le = LabelEncoder()
    for col in cat_cols:
        if col != target_col:
            df[col] = le.fit_transform(df[col].astype(str))

    return df

target = 'TARGET_COLUMN_NAME' 
df_clean = quick_clean(df, target)

X = df_clean.drop(target, axis=1)
y = df_clean[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("/n Data is cleaned, encoded, and scaled. Ready for modeling!")