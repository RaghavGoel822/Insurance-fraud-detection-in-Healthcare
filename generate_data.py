import pandas as pd
import numpy as np
import random
import os

def generate_synthetic_data(
    n_samples=10000, 
    base_amount=500, 
    base_freq=15, 
    base_rarity=0.5, 
    base_gap=5,
    fraud_intensity=1.0,
    noise_level=0.04
):
    """
    Generates synthetic healthcare claims data with adjustable parameters.
    """
    providers = [f"P{i:03d}" for i in range(1, 151)]
    specialties = ["General Practice", "Cardiology", "Orthopedics", "Radiology", "Dentistry", "Neurology", "Oncology"]
    
    # Decide on a "Primary Fraud Mode" for this run
    fraud_scenario = random.choice(['amount', 'frequency', 'rarity', 'velocity'])
    
    # Dynamically select high-risk providers
    high_risk_providers = random.sample(providers, k=random.randint(3, 10))
    
    data = []
    
    for i in range(n_samples):
        provider = random.choice(providers)
        specialty = random.choice(specialties)
        
        # Base features (Parameterized)
        claim_amount = np.random.gamma(shape=2, scale=base_amount) 
        claim_frequency = np.random.randint(1, base_freq + 1)
        procedure_rarity = np.random.uniform(0, base_rarity)
        days_between_claims = np.random.randint(base_gap, 60)
        patient_age = np.random.randint(18, 90)
        
        # Initialize fraud label
        fraud_label = 0
        
        # Pattern Injection (Scaled by fraud_intensity)
        if provider in high_risk_providers:
            if fraud_scenario == 'amount':
                claim_amount += np.random.uniform(1000, 3500) * fraud_intensity
                fraud_label = 1
            elif fraud_scenario == 'frequency':
                claim_frequency += int(np.random.randint(6, 12) * fraud_intensity)
                fraud_label = 1
            elif fraud_scenario == 'rarity':
                # Shift rarity towards 1.0
                shift = (1.0 - procedure_rarity) * 0.5 * fraud_intensity
                procedure_rarity = np.clip(procedure_rarity + shift + np.random.uniform(0.1, 0.3), 0, 1)
                fraud_label = 1
            elif fraud_scenario == 'velocity':
                # Reduce days drastically
                reduction = int((days_between_claims - 1) * 0.8 * fraud_intensity)
                days_between_claims = max(1, days_between_claims - reduction)
                claim_frequency += int(2 * fraud_intensity)
                fraud_label = 1
        
        # Organic patterns (also slightly affected by intensity)
        if specialty == "General Practice" and procedure_rarity > 0.8:
            if random.random() > (0.4 - 0.1 * fraud_intensity):
                fraud_label = 1
                
        if days_between_claims < 3 and claim_amount > (base_amount * 5):
             if random.random() > (0.6 - 0.1 * fraud_intensity):
                fraud_label = 1

        # Label Noise (Parameterized)
        if random.random() < noise_level:
            fraud_label = 1 - fraud_label 

        data.append({
            "provider_id": provider,
            "specialty": specialty,
            "claim_amount": claim_amount,
            "claim_frequency": claim_frequency,
            "procedure_rarity": procedure_rarity,
            "days_between_claims": days_between_claims,
            "patient_age": patient_age,
            "fraud_label": fraud_label
        })
        
    df = pd.DataFrame(data)
    
    if not os.path.exists("data"):
        os.makedirs("data")
        
    df.to_csv("data/claims_data.csv", index=False)
    print(f"Generated {n_samples} samples for scenario: {fraud_scenario.upper()}")
    print(f"Noise Level: {noise_level:.1%}, Fraud Intensity: {fraud_intensity:.1f}")
    return fraud_scenario

if __name__ == "__main__":
    generate_synthetic_data()
