import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "datasets")

# Create models directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

print("=============================================================")
print("      Starting Machine Learning Model Training Pipeline      ")
print("=============================================================\n")


# ─────────────────────────────────────────────────────────────
# 1. DIABETES MODEL
# ─────────────────────────────────────────────────────────────
def train_diabetes():
    print("--- Training Diabetes Prediction Model ---")
    csv_path = os.path.join(DATASET_DIR, "diabetes.csv")
    if not os.path.exists(csv_path):
        print(f"Error: Dataset {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    # Target and Features
    target_col = "Outcome"
    feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    # Drop rows with missing target
    df = df.dropna(subset=[target_col])
    
    X = df[feature_cols].copy()
    y = df[target_col].astype(int)
    
    # Cast to numeric
    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Impute
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    
    # Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_imp)
    X_test_sc = scaler.transform(X_test_imp)
    
    # Train
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
    model.fit(X_train_sc, y_train)
    
    # Evaluate
    train_acc = accuracy_score(y_train, model.predict(X_train_sc))
    test_acc = accuracy_score(y_test, model.predict(X_test_sc))
    print(f"Diabetes Model Trained. Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    # Save bundle
    joblib.dump(model, os.path.join(MODEL_DIR, "diabetes_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "diabetes_scaler.pkl"))
    joblib.dump(imputer, os.path.join(MODEL_DIR, "diabetes_imputer.pkl"))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "diabetes_features.pkl"))
    print("[SUCCESS] Diabetes bundle saved successfully.\n")


# ─────────────────────────────────────────────────────────────
# 2. HEART DISEASE MODEL
# ─────────────────────────────────────────────────────────────
def train_heart():
    print("--- Training Heart Disease Prediction Model ---")
    csv_path = os.path.join(DATASET_DIR, "heart_disease.csv")
    if not os.path.exists(csv_path):
        print(f"Error: Dataset {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    df = df.rename(columns={'thalch': 'thalach'})
    
    # Map categorical text fields to numeric encodings safely
    gender_map = {"male": 1.0, "female": 0.0, "male ": 1.0, "female ": 0.0}
    cp_map = {
        "typical angina": 0.0, 
        "atypical angina": 1.0, 
        "non-anginal": 2.0, 
        "asymptomatic": 3.0
    }
    restecg_map = {
        "normal": 0.0,
        "st-t abnormality": 1.0,
        "st-t": 1.0,
        "lv hypertrophy": 2.0
    }
    slope_map = {"upsloping": 0.0, "flat": 1.0, "downsloping": 2.0}
    thal_map = {"normal": 0.0, "fixed defect": 1.0, "reversable defect": 2.0}
    
    # Apply clean categorical mappings
    if 'sex' in df.columns:
        df['sex'] = df['sex'].astype(str).str.strip().str.lower().map(gender_map)
    if 'cp' in df.columns:
        df['cp'] = df['cp'].astype(str).str.strip().str.lower().map(cp_map)
    if 'restecg' in df.columns:
        df['restecg'] = df['restecg'].astype(str).str.strip().str.lower().map(restecg_map)
    if 'slope' in df.columns:
        df['slope'] = df['slope'].astype(str).str.strip().str.lower().map(slope_map)
    if 'thal' in df.columns:
        df['thal'] = df['thal'].astype(str).str.strip().str.lower().map(thal_map)
        
    # Map boolean TRUE/FALSE fields
    bool_cols = ['fbs', 'exang']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper().map({"TRUE": 1.0, "FALSE": 0.0})
            
    # Target column is 'num'. num = 0 means normal, > 0 means heart disease
    target_col = "num"
    feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    df = df.dropna(subset=[target_col])
    
    X = df[feature_cols].copy()
    y = (df[target_col] > 0).astype(int)
    
    # Cast all columns to float
    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Impute missing values with median
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    
    # Scale values
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_imp)
    X_test_sc = scaler.transform(X_test_imp)
    
    # Train classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train_sc, y_train)
    
    train_acc = accuracy_score(y_train, model.predict(X_train_sc))
    test_acc = accuracy_score(y_test, model.predict(X_test_sc))
    print(f"Heart Disease Model Trained. Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    # Save bundle
    joblib.dump(model, os.path.join(MODEL_DIR, "heart_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "heart_scaler.pkl"))
    joblib.dump(imputer, os.path.join(MODEL_DIR, "heart_imputer.pkl"))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "heart_features.pkl"))
    print("[SUCCESS] Heart Disease bundle saved successfully.\n")


# ─────────────────────────────────────────────────────────────
# 3. KIDNEY DISEASE MODEL
# ─────────────────────────────────────────────────────────────
def train_kidney():
    print("--- Training Kidney Disease Prediction Model ---")
    csv_path = os.path.join(DATASET_DIR, "kidney_disease.csv")
    if not os.path.exists(csv_path):
        print(f"Error: Dataset {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    # Target preprocessing
    target_col = "classification"
    feature_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
    
    df = df.dropna(subset=[target_col])
    # Clean classification target
    df[target_col] = df[target_col].astype(str).str.strip().str.replace(r'\t', '', regex=True).str.lower()
    df['target'] = df[target_col].map({'ckd': 1, 'notckd': 0})
    
    # Drop rows where target is invalid/nan
    df = df.dropna(subset=['target'])
    
    X = df[feature_cols].copy()
    y = df['target'].astype(int)
    
    # Cast all columns to float safely
    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Impute missing values with mean
    imputer = SimpleImputer(strategy='mean')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    
    # Scale values
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_imp)
    X_test_sc = scaler.transform(X_test_imp)
    
    # Train
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train_sc, y_train)
    
    train_acc = accuracy_score(y_train, model.predict(X_train_sc))
    test_acc = accuracy_score(y_test, model.predict(X_test_sc))
    print(f"Kidney Disease Model Trained. Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    # Save bundle
    joblib.dump(model, os.path.join(MODEL_DIR, "kidney_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "kidney_scaler.pkl"))
    joblib.dump(imputer, os.path.join(MODEL_DIR, "kidney_imputer.pkl"))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "kidney_features.pkl"))
    print("[SUCCESS] Kidney Disease bundle saved successfully.\n")


# ─────────────────────────────────────────────────────────────
# 4. LIVER DISEASE MODEL
# ─────────────────────────────────────────────────────────────
def train_liver():
    print("--- Training Liver Disease Prediction Model ---")
    csv_path = os.path.join(DATASET_DIR, "liver_diseases.csv")
    if not os.path.exists(csv_path):
        print(f"Error: Dataset {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    # Map Gender column
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].astype(str).str.strip().str.lower().map({"male": 1.0, "female": 0.0})
        
    # Target column is Dataset (1 = liver disease, 2 = healthy)
    target_col = "Dataset"
    feature_cols = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio']
    
    df = df.dropna(subset=[target_col])
    # Map target: 1 -> 1 (Positive), 2 -> 0 (Negative)
    y = df[target_col].map({1: 1, 2: 0}).astype(int)
    
    X = df[feature_cols].copy()
    
    # Cast all float features to numeric
    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    
    # Scale values
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_imp)
    X_test_sc = scaler.transform(X_test_imp)
    
    # Train RandomForest model
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train_sc, y_train)
    
    train_acc = accuracy_score(y_train, model.predict(X_train_sc))
    test_acc = accuracy_score(y_test, model.predict(X_test_sc))
    print(f"Liver Disease Model Trained. Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    # Save bundle
    joblib.dump(model, os.path.join(MODEL_DIR, "liver_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "liver_scaler.pkl"))
    joblib.dump(imputer, os.path.join(MODEL_DIR, "liver_imputer.pkl"))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "liver_features.pkl"))
    print("[SUCCESS] Liver Disease bundle saved successfully.\n")


if __name__ == "__main__":
    train_diabetes()
    train_heart()
    train_kidney()
    train_liver()
    print("=============================================================")
    print("      All Machine Learning Models Trained Successfully!       ")
    print("=============================================================")
