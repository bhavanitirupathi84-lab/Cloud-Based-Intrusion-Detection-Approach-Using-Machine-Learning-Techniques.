
# train_isolation_forest.py
# Train Isolation Forest for anomaly detection

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix
)
import pickle
import os

def train_isolation_forest():
    """
    Train Isolation Forest on NORMAL data only
    """
    print("\n" + "="*70)
    print(" "*15 + "ISOLATION FOREST TRAINING")
    print("="*70)
    
    # STEP 1: Load preprocessed data
    print("\n[1/5] Loading data...")
    X_train = np.load('data/processed/X_train.npy')
    y_train = np.load('data/processed/y_train.npy')
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    print(f"  ✓ Training samples: {len(X_train):,}")
    print(f"  ✓ Test samples: {len(X_test):,}")
    
    # STEP 2: Extract normal samples only
    print("\n[2/5] Extracting normal samples...")
    print("  (Isolation Forest learns ONLY from normal traffic)")
    
    X_normal = X_train[y_train == 0]
    
    print(f"  ✓ Normal samples: {len(X_normal):,}")
    print(f"  ✓ Attack samples: {len(X_train) - len(X_normal):,}")
    
    contamination = 1 - len(X_normal) / len(X_train)
    print(f"  ✓ Contamination rate: {contamination:.3f} ({contamination*100:.1f}%)")
    
    # STEP 3: Train model
    print("\n[3/5] Training Isolation Forest...")
    print("  Model parameters:")
    print("    • n_estimators: 200 (number of trees)")
    print("    • contamination: 0.15 (expected % of anomalies)")
    print("    • max_samples: auto")
    print("    • random_state: 42 (for reproducibility)")
    
    iso_model = IsolationForest(
        n_estimators=200,
        contamination=0.15,
        max_samples='auto',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    print("\n  Training... (this may take 1-2 minutes)")
    iso_model.fit(X_normal)
    print("  ✓ Training complete!")
    
    # STEP 4: Evaluate
    print("\n[4/5] Evaluating on test set...")
    
    # Predict on test set
    y_pred_iso = iso_model.predict(X_test)
    
    # Convert predictions:
    # Isolation Forest: -1 = anomaly, 1 = normal
    # We convert to: 1 = attack, 0 = normal
    y_pred_binary = np.where(y_pred_iso == -1, 1, 0)
    y_test_binary = np.where(y_test == 0, 0, 1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_binary, y_pred_binary)
    precision = precision_score(y_test_binary, y_pred_binary)
    recall = recall_score(y_test_binary, y_pred_binary)
    f1 = f1_score(y_test_binary, y_pred_binary)
    
    print("\n" + "="*70)
    print("ISOLATION FOREST PERFORMANCE")
    print("="*70)
    print(f"\n  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    # Confusion matrix
    cm = confusion_matrix(y_test_binary, y_pred_binary)
    print("\n  Confusion Matrix:")
    print("                   Predicted")
    print("                 Normal  Attack")
    print(f"  Actual Normal  {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"  Actual Attack  {cm[1,0]:6d}  {cm[1,1]:6d}")
    
    # Detection stats
    total_attacks = np.sum(y_test_binary == 1)
    detected_attacks = np.sum((y_pred_binary == 1) & (y_test_binary == 1))
    missed_attacks = total_attacks - detected_attacks
    
    print(f"\n  Total attacks in test set: {total_attacks:,}")
    print(f"  Attacks detected: {detected_attacks:,}")
    print(f"  Attacks missed: {missed_attacks:,}")
    print(f"  Detection rate: {detected_attacks/total_attacks*100:.2f}%")
    
    # STEP 5: Save model
    print("\n[5/5] Saving model...")
    
    os.makedirs('models', exist_ok=True)
    
    with open('models/isolation_forest.pkl', 'wb') as f:
        pickle.dump(iso_model, f)
    
    print("  ✓ Model saved to: models/isolation_forest.pkl")
    
    print("\n" + "="*70)
    print("✓ ISOLATION FOREST TRAINING COMPLETE!")
    print("="*70)
    
    return iso_model, accuracy, precision, recall, f1

if __name__ == "__main__":
    model, acc, prec, rec, f1 = train_isolation_forest()
