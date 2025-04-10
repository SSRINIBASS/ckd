import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

def train_and_save_model(file_path):
    # Load and clean data
    data = pd.read_csv(file_path)
    data['classification'] = data['classification'].str.strip()
    data = data[data['classification'].isin(['ckd', 'notckd'])]

    # Preprocessing
    def preprocess_data(df):
        X = df.drop(['id', 'classification'], axis=1)
        y = df['classification']
        numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
        
        label_encoders = {}
        for col in categorical_cols:
            label_encoders[col] = LabelEncoder()
            X[col] = X[col].fillna(X[col].mode()[0])
            X[col] = label_encoders[col].fit_transform(X[col])
        
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y, scaler, label_encoders, target_encoder, X.columns

    X_scaled, y, scaler, label_encoders, target_encoder, feature_names = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate accuracy
    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy on Test Set: {accuracy:.2f}")

    # Save model and objects
    joblib.dump(model, 'ckd_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    joblib.dump(target_encoder, 'target_encoder.pkl')
    joblib.dump(feature_names, 'feature_names.pkl')  # Save feature names for prediction
    print("Model and preprocessing objects saved!")

if __name__ == "__main__":
    file_path = 'kidney_disease (1).csv'  # Update if needed
    train_and_save_model(file_path)