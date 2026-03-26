import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, accuracy_score
# from sklearn.ensemble import IsolationForest  # Replaced with Autoencoder
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

def run_pipeline():
    # -------------------------------
    # LOAD DATA
    # -------------------------------
    data = pd.read_csv("data/claims_data.csv")

    # Preprocessing: Handle Categorical 'specialty'
    data_encoded = pd.get_dummies(data, columns=["specialty"])
    X = data_encoded.drop(["fraud_label", "provider_id"], axis=1).astype(float)
    y = data_encoded["fraud_label"]

    # -------------------------------
    # TRAIN-TEST SPLIT
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # -------------------------------
    # SUPERVISED MODEL – XGBOOST (with GridSearchCV)
    # -------------------------------
    # Define parameter grid for tuning to achieve realistic accuracy
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Calculate class weight based on imbalance
    fraud_ratio = (y == 0).sum() / (y == 1).sum()
    
    xgb_base = XGBClassifier(
        scale_pos_weight=fraud_ratio, # Balanced weighting
        eval_metric="logloss"
    )

    print(f"\n--- Tuning XGBoost for F1-SCORE (Balanced Precision/Recall) ---")
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        scoring='f1', # Balance precision and recall
        cv=skf,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    
    xgb_model = grid_search.best_estimator_
    cv_score = grid_search.best_score_
    
    print(f"Best CV F1-Score: {cv_score:.4f}")
    print(f"Best Parameters: {grid_search.best_params_}")

    y_pred = xgb_model.predict(X_test)
    print("\n--- XGBoost Test Set Evaluation ---")
    print(classification_report(y_test, y_pred))

    # -------------------------------
    # UNSUPERVISED – AUTOENCODER
    # -------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train_normal = scaler.transform(X_train[y_train == 0]) # Train AE ONLY on normal data

    # Define Autoencoder Architecture
    input_dim = X_scaled.shape[1]
    input_layer = Input(shape=(input_dim,))
    
    # Encoder: Compress features
    encoder = Dense(16, activation='relu')(input_layer)
    encoder = Dense(8, activation='relu')(encoder)
    
    # Bottleneck (Latent Representation)
    latent = Dense(4, activation='relu')(encoder)
    
    # Decoder: Reconstruct features
    decoder = Dense(8, activation='relu')(latent)
    decoder = Dense(16, activation='relu')(decoder)
    output_layer = Dense(input_dim, activation='linear')(decoder)
    
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')

    # Train Autoencoder (Input = Output)
    print("\n--- Training Autoencoder (on Normal Samples ONLY) ---")
    autoencoder.fit(
        X_train_normal, X_train_normal,
        epochs=100,
        batch_size=64,
        shuffle=True,
        validation_split=0.1,
        verbose=0
    )

    # Compute Reconstruction Error for ALL data
    predictions = autoencoder.predict(X_scaled)
    reconstruction_error = np.mean(np.square(X_scaled - predictions), axis=1)

    # Normalize reconstruction error to 0-1 anomaly score
    # Use a quantile-based normalization to handle outliers better
    error_min = np.percentile(reconstruction_error, 1)
    error_max = np.percentile(reconstruction_error, 99)
    anomaly_norm = np.clip((reconstruction_error - error_min) / (error_max - error_min), 0, 1)

    # -------------------------------
    # HYBRID FRAUD SCORING
    # -------------------------------
    data["supervised_score"] = xgb_model.predict_proba(X)[:, 1]
    data["unsupervised_score"] = anomaly_norm

    data["final_risk_score"] = (
        0.7 * data["supervised_score"] +
        0.3 * data["unsupervised_score"]
    )

    data["fraud_flag"] = data["final_risk_score"] > 0.65

    # -------------------------------
    # PROVIDER-LEVEL AGGREGATION
    # -------------------------------
    provider_risk = data.groupby("provider_id").agg(
        total_claims=("final_risk_score", "count"),
        avg_risk_score=("final_risk_score", "mean"),
        fraud_rate=("fraud_flag", "mean"),
        avg_amount=("claim_amount", "mean"),
        max_frequency=("claim_frequency", "max")
    ).reset_index()

    # -------------------------------
    # REGULATORY REPORT GENERATION
    # -------------------------------
    def generate_evidence(row):
        reasons = []
        if row["avg_risk_score"] > 0.8:
            reasons.append("Critical hybrid risk score exceeding threshold (0.8).")
        if row["fraud_rate"] > 0.5:
            reasons.append(f"High anomaly rate: {row['fraud_rate']:.1%} of claims flagged.")
        if row["avg_amount"] > 8000:
            reasons.append("Extreme claim amounts detected (Potential Upcoding).")
        if row["max_frequency"] > 25:
            reasons.append("Abnormal claim frequency (Potential Phantom Billing).")
        
        if not reasons:
            return "Normal activity patterns observed."
        return " | ".join(reasons)

    provider_risk["evidence_statement"] = provider_risk.apply(generate_evidence, axis=1)
    provider_risk["audit_priority"] = pd.cut(
        provider_risk["avg_risk_score"], 
        bins=[0, 0.4, 0.7, 1.0], 
        labels=["Low", "Medium", "High"]
    )

    provider_risk.to_csv("regulatory_report.csv", index=False)
    provider_risk.to_csv("provider_risk.csv", index=False)

    # -------------------------------
    # SHAP EXPLAINABILITY & META DATA
    # -------------------------------
    # Save top feature and accuracy for dashboard
    import json
    from sklearn.metrics import accuracy_score
    
    importances = xgb_model.feature_importances_
    features = X.columns
    top_feature = features[np.argmax(importances)]
    
    # Calculate Calibration Threshold for ~80% Recall
    probs_test = xgb_model.predict_proba(X_test)[:, 1]
    # We want top X% where Recall is >= 0.8
    # Sorting descending
    sorted_probs = np.sort(probs_test)[::-1]
    idx_80 = int(len(y_test[y_test==1]) * 0.8)
    # Ensure it doesn't go out of bounds
    idx_80 = min(idx_80, len(sorted_probs)-1)
    recommended_threshold = float(sorted_probs[idx_80])
    
    accuracy = accuracy_score(y_test, y_pred)
    
    with open("model_meta.json", "w") as f:
        json.dump({
            "top_feature": top_feature,
            "accuracy": float(accuracy),
            "recommended_threshold": round(recommended_threshold, 2),
            "best_params": grid_search.best_params_,
            "unsupervised_method": "Deep Learning Autoencoder (4-layer Bottleneck)",
            "cv_recall": float(cv_score)
        }, f)

    try:
        explainer = shap.Explainer(xgb_model, X_train)
        shap_values = explainer(X_test)
        plt.figure()
        shap.summary_plot(shap_values, X_test, show=False)
        plt.savefig("shap_summary.png")
        print(f"\nSaved SHAP summary and Regulatory Report. Accuracy: {accuracy:.2%}")
    except Exception as e:
        print(f"\nSHAP error: {e}")

    # SAVE RESULTS
    data.to_csv("fraud_results.csv", index=False)
    print("\nPipeline Complete: fraud_results.csv, provider_risk.csv, regulatory_report.csv, model_meta.json saved.")

if __name__ == "__main__":
    run_pipeline()
