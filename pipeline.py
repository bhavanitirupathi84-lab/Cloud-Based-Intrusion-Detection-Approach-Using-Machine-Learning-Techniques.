# pipeline.py
# Complete preprocessing pipeline

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Import our preprocessing functions
from preprocessing import (
    load_data,
    encode_categorical,
    map_attack_types,
    engineer_features
)

def preprocess_pipeline():
    """
    Run complete preprocessing pipeline
    """
    print("\n" + "="*70)
    print(" "*20 + "PREPROCESSING PIPELINE")
    print("="*70)
    
    # STEP 1: Load data
    print("\n[1/6] Loading data...")
    train_df, test_df = load_data()
    
    # STEP 2: Encode categorical
    print("\n[2/6] Encoding categorical features...")
    train_df = encode_categorical(train_df)
    test_df = encode_categorical(test_df)
    
    # STEP 3: Map attack types
    print("\n[3/6] Mapping attack types...")
    train_df = map_attack_types(train_df)
    test_df = map_attack_types(test_df)
    
    # STEP 4: Engineer features
    print("\n[4/6] Engineering features...")
    train_df, features = engineer_features(train_df)
    test_df, _ = engineer_features(test_df)
    
    # STEP 5: Scale features
    print("\n[5/6] Scaling features...")
    print("  (Making all features have similar ranges)")
    
    scaler = StandardScaler()
    
    X_train = train_df[features].values
    y_train = train_df['attack_category'].values
    
    X_test = test_df[features].values
    y_test = test_df['attack_category'].values
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"  ✓ Training shape: {X_train_scaled.shape}")
    print(f"  ✓ Test shape: {X_test_scaled.shape}")
    
    # STEP 6: Save everything
    print("\n[6/6] Saving processed data...")
    
    # Create processed directory if it doesn't exist
    os.makedirs('data/processed', exist_ok=True)
    
    # Save arrays
    np.save('data/processed/X_train.npy', X_train_scaled)
    np.save('data/processed/y_train.npy', y_train)
    np.save('data/processed/X_test.npy', X_test_scaled)
    np.save('data/processed/y_test.npy', y_test)
    
    # Save scaler
    with open('data/processed/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature names
    with open('data/processed/feature_names.txt', 'w') as f:
        f.write('\n'.join(features))
    
    print("\n  Saved files:")
    print("    ✓ X_train.npy")
    print("    ✓ y_train.npy")
    print("    ✓ X_test.npy")
    print("    ✓ y_test.npy")
    print("    ✓ scaler.pkl")
    print("    ✓ feature_names.txt")
    
    # Print summary
    print("\n" + "="*70)
    print("PREPROCESSING SUMMARY")
    print("="*70)
    print(f"\nTraining samples: {len(X_train):,}")
    print(f"Test samples:     {len(X_test):,}")
    print(f"Features:         {len(features)}")
    
    print("\nClass distribution (Training):")
    category_names = [
        'normal', 'brute_force', 'data_exfiltration',
        'geo_anomaly', 'privilege_escalation', 'insider_threat'
    ]
    for i, name in enumerate(category_names):
        count = (y_train == i).sum()
        pct = count / len(y_train) * 100
        print(f"  {i}: {name:20s} - {count:6,d} ({pct:5.2f}%)")
    
    print("\n" + "="*70)
    print("✓ PREPROCESSING PIPELINE COMPLETE!")
    print("="*70)
    print("\nYou can now train your machine learning models!")
    
    return X_train_scaled, y_train, X_test_scaled, y_test, features

if __name__ == "__main__":
    X_train, y_train, X_test, y_test, features = preprocess_pipeline()

