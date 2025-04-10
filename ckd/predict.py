import pandas as pd
import numpy as np
import joblib

# Load the saved model and preprocessing objects
model = joblib.load('ckd_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')
target_encoder = joblib.load('target_encoder.pkl')
feature_names = joblib.load('feature_names.pkl')

# Predict on new data
def predict_new_data(new_data):
    df_new = pd.DataFrame(new_data)
    
    # Validate features
    missing_features = [f for f in feature_names if f not in df_new.columns]
    extra_features = [f for f in df_new.columns if f not in feature_names]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")
    if extra_features:
        raise ValueError(f"Extra features: {extra_features}")
    
    df_new = df_new[feature_names]  # Reorder to match training
    
    numeric_cols = df_new.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df_new.select_dtypes(include=['object']).columns
    df_new[numeric_cols] = df_new[numeric_cols].fillna(df_new[numeric_cols].mean())
    for col in categorical_cols:
        df_new[col] = df_new[col].fillna(df_new[col].mode()[0])
        df_new[col] = label_encoders[col].transform(df_new[col])
    
    X_new_scaled = scaler.transform(df_new)
    predictions = model.predict(X_new_scaled)
    return predictions

# Generator to predict from dataset
def predict_dataset_generator(file_path):
    data = pd.read_csv(file_path)
    data['classification'] = data['classification'].str.strip()
    data = data[data['classification'].isin(['ckd', 'notckd'])]
    
    X = data.drop(['id', 'classification'], axis=1)
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
    for col in categorical_cols:
        X[col] = X[col].fillna(X[col].mode()[0])
        X[col] = label_encoders[col].transform(X[col])
    
    X_scaled = scaler.transform(X)
    for i in range(len(X_scaled)):
        prediction = model.predict([X_scaled[i]])[0]
        actual = target_encoder.transform([data['classification'].iloc[i]])[0]
        yield {
            'row_index': i,
            'prediction': prediction,
            'actual': actual,
            'features': data.iloc[i].drop(['id', 'classification']).to_dict()
        }

if __name__ == "__main__":
    # Example: Predict on new data
    new_data = {
        'age': [48.0], 'bp': [80.0], 'sg': [1.020], 'al': [1.0], 'su': [0.0],
        'rbc': ['normal'], 'pc': ['normal'], 'pcc': ['notpresent'], 'ba': ['notpresent'],
        'bgr': [121.0], 'bu': [36.0], 'sc': [1.2], 'sod': [137.0], 'pot': [4.4],
        'hemo': [15.4], 'pcv': [44.0], 'wc': [7800.0], 'rc': [5.2],
        'htn': ['yes'], 'dm': ['no'], 'cad': ['no'], 'appet': ['good'], 'pe': ['no'], 'ane': ['no']
    }
    predictions = predict_new_data(new_data)
    print("New Data Prediction (0 = CKD, 1 = no CKD):", predictions[0])

    # Example: Generator over dataset
    print("\nGenerating predictions from dataset:")
    file_path = 'kidney_disease (1).csv'  # Update if needed
    prediction_gen = predict_dataset_generator(file_path)
    for i, pred in enumerate(prediction_gen):
        print(f"Row {pred['row_index']}: Predicted={pred['prediction']}, Actual={pred['actual']}")
        if i == 2:  # Show 3 rows
            break