```python
# dual_model_pipeline.py
# Use Isolation Forest + Random Forest together

import numpy as np
import pickle
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

def load_models():
    """Load both trained models"""
    print("Loading models...")
    
    with open('models/isolation_forest.pkl', 'rb') as f:
        iso_model = pickle.load(f)
    print("  ✓ Isolation Forest loaded")
    
    with open('models/random_forest.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    print("  ✓ Random Forest loaded")
    
    return iso_model, rf_model

def dual_model_predict(X, iso_model, rf_model):
    """
    Sequential dual-model prediction
    
    STAGE 1: Isolation Forest (is it normal or anomaly?)
    STAGE 2: Random Forest (what type of attack?)
    """
    # STAGE 1: Isolation Forest
    iso_predictions = iso_model.predict(X)
    # -1 = anomaly, 1 = normal
    
    # Initialize all as normal (0)
    final_predictions = np.zeros(len(X), dtype=int)
    
    # STAGE 2: Random Forest (only for anomalies)
    anomaly_indices = np.where(iso_predictions == -1)[0]
    
    if len(anomaly_indices) > 0:
        X_anomalies = X[anomaly_indices]
        rf_predictions = rf_model.predict(X_anomalies)
        final_predictions[anomaly_indices] = rf_predictions
    
    return final_predictions, iso_predictions

def evaluate_dual_model():
    """
    Evaluate the complete dual-model pipeline
    """
    print("\n" + "="*70)
    print(" "*15 + "DUAL-MODEL PIPELINE EVALUATION
print("="*70)
    
    # STEP 1: Load data
    print("\n[1/4] Loading test data...")
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    print(f"  ✓ Test samples: {len(X_test):,}")
    
    # STEP 2: Load models
    print("\n[2/4] Loading models...")
    iso_model, rf_model = load_models()
    
    # STEP 3: Run predictions
    print("\n[3/4] Running dual-model prediction...")
    print("  • Stage 1: Isolation Forest detects anomalies")
    print("  • Stage 2: Random Forest classifies attack types")
    
    y_pred, iso_pred = dual_model_predict(X_test, iso_model, rf_model)
    print("  ✓ Predictions complete!")
    
    # STEP 4: Evaluate
    print("\n[4/4] Calculating metrics...")
    
    accuracy = accuracy_score(y_test, y_pred)
    
    category_names = [
        'normal', 'brute_force', 'data_exfiltration',
        'geo_anomaly', 'privilege_escalation', 'insider_threat'
    ]
    
    print("\n" + "="*70)
    print("DUAL-MODEL PIPELINE PERFORMANCE")
    print("="*70)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\n" + "="*70)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*70)
    print()
    print(classification_report(y_test, y_pred,
                                 target_names=category_names,
                                 digits=4))
    
    print("\n" + "="*70)
    print("CONFUSION MATRIX")
    print("="*70)
    print("\n  Rows = Actual, Columns = Predicted")
    print("\n     " + "    ".join([f"{i}" for i in range(6)]))
    cm = confusion_matrix(y_test, y_pred)
    for i, row in enumerate(cm):
        print(f"  {i}  " + "  ".join([f"{val:5d}" for val in row]))
    
    # Stage analysis
    print("\n" + "="*70)
    print("STAGE ANALYSIS")
    print("="*70)
    
    normal_mask = (y_test == 0)
    attack_mask = (y_test != 0)
    
    total_attacks = np.sum(attack_mask)
    attacks_flagged = np.sum((iso_pred == -1) & attack_mask)
    
    print("\nStage 1 (Isolation Forest):")
    print(f"  Total attacks in test set: {total_attacks:,}")
    print(f"  Attacks flagged as anomalies: {attacks_flagged:,}")
    print(f"  Detection rate: {attacks_flagged/total_attacks*100:.2f}%")
    
    correctly_classified = np.sum((y_pred == y_test) & attack_mask)
    
    print("\nStage 2 (Random Forest):")
    print(f"  Anomalies to classify: {np.sum(iso_pred == -1):,}")
    print(f"  Correctly classified attacks: {correctly_classified:,}")
    print(f"  Classification accuracy: {correctly_classified/total_attacks*100:.2f}%")
    
    # Per-attack-type accuracy
    print("\n" + "="*70)
    print("PER-ATTACK-TYPE DETECTION")
    print("="*70)
    print()
    for i, name in enumerate(category_names):
        if i == 0:  # Skip normal
            continue
        mask = (y_test == i)
        if np.sum(mask) > 0:
            detected = np.sum((y_pred == i) & mask)
            total = np.sum(mask)
            rate = detected / total * 100
            print(f"  {name:25s}: {detected:4d}/{total:4d} ({rate:5.2f}%)")
    
    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    results = {
        'accuracy': accuracy,
        'y_pred': y_pred,
        'y_test': y_test,
        'iso_pred': iso_pred,
        'confusion_matrix': cm
    }
    
    import os
    os.makedirs('data/results', exist_ok=True)
    
    with open('data/results/dual_model_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\n  ✓ Results saved to: data/results/dual_model_results.pkl")
    
    print("\n" + "="*70)
    print("✓ DUAL-MODEL PIPELINE EVALUATION COMPLETE!")
    print("="*70)
    
    return accuracy, y_pred, y_test

if __name__ == "__main__":
    accuracy, y_pred, y_test = evaluate_dual_model()


