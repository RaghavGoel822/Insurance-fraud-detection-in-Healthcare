import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import generate_data
import fraud_model
import json

st.set_page_config(page_title="Insurance Fraud Detection", layout="wide")

# -------------------------------
# SIDEBAR CONTROLS
# -------------------------------
with st.sidebar:
    st.divider()
    st.header("🧬 Simulation Parameters")
    with st.expander("🛠️ Adjust Dataset Realism", expanded=False):
        s_base_amount = st.slider("Base Claim Amount ($)", 100, 1000, 500, step=50)
        s_base_freq = st.slider("Max Base Frequency", 5, 30, 15)
        s_base_rarity = st.slider("Base Rarity Range", 0.1, 0.9, 0.5, step=0.05)
        s_base_gap = st.slider("Min Days Gap", 1, 10, 5)
        s_fraud_intensity = st.slider("Fraud Signal Intensity", 0.5, 2.0, 1.0, step=0.1)
        s_noise_level = st.slider("Label Noise (%)", 0.0, 10.0, 4.0, step=0.5) / 100.0

    if st.button("🚀 Generate New Data & Retrain Model", use_container_width=True):
        with st.status("Initializing data generation..."):
            st.write("Generating 10,000 new samples with custom parameters...")
            generate_data.generate_synthetic_data(
                base_amount=s_base_amount,
                base_freq=s_base_freq,
                base_rarity=s_base_rarity,
                base_gap=s_base_gap,
                fraud_intensity=s_fraud_intensity,
                noise_level=s_noise_level
            )
            st.write("Retraining Hybrid AI Model (XGBoost + Autoencoder)...")
            fraud_model.run_pipeline()
            st.write("Updating reports...")
            st.cache_data.clear() # Clear cache to load new data
        st.toast("System Updated with New Parameters!", icon="✅")
        st.rerun()

    st.divider()
    st.divider()
    st.header("🎯 Detection Sensitivity")
    
    # Load metadata for recommendation
    recommended_t = 0.65
    if os.path.exists("model_meta.json"):
        with open("model_meta.json", "r") as f:
            recommended_t = json.load(f).get("recommended_threshold", 0.65)

    risk_threshold = st.slider(
        "Risk Score Threshold", 
        min_value=0.05, 
        max_value=0.95, 
        value=recommended_t, 
        step=0.05,
        help="Lowering the threshold catches more potential fraud (higher sensitivity). The default is calibrated for ~80% detection."
    )
    st.info(f"Threshold set to **{risk_threshold}**. {f'(AI Recommended: **{recommended_t}**)' if recommended_t != 0.65 else ''}")
    st.markdown(f"**Target:** Catching at least 80% of fraudulent claims.")

    st.divider()
    st.info("""
    **Pro Tip:** Use the **Simulation Parameters** above to make the fraud harder or easier to detect, then adjust the **Sensitivity** slider to optimize your detection rate.
    """)

st.title("🏥 Health Insurance Fraud Detection Dashboard")

# Check if data files exist
if not os.path.exists("fraud_results.csv") or not os.path.exists("regulatory_report.csv"):
    st.warning("⚠️ Result files not found. Please click the sidebar button to initialize.")
    st.stop()

# Load data
@st.cache_data
def load_data():
    claims = pd.read_csv("fraud_results.csv")
    # We will recalculate the report dynamically based on the slider
    return claims

claims = load_data()

# -------------------------------
# DYNAMIC FLAGGING LOGIC
# -------------------------------
# Apply the threshold from the slider
claims["fraud_flag"] = claims["final_risk_score"] > risk_threshold

# Recalculate provider-level metrics based on the current threshold
report = claims.groupby("provider_id").agg(
    total_claims=("final_risk_score", "count"),
    avg_risk_score=("final_risk_score", "mean"),
    fraud_rate=("fraud_flag", "mean"),
    avg_amount=("claim_amount", "mean"),
    max_frequency=("claim_frequency", "max")
).reset_index()

# Re-generate evidence statements and priority
def generate_evidence(row):
    reasons = []
    if row["avg_risk_score"] > risk_threshold:
        reasons.append(f"High risk score ({row['avg_risk_score']:.2f}).")
    if row["fraud_rate"] > 0.15: # If more than 15% of claims are flagged
        reasons.append(f"Suspicious flag density: {row['fraud_rate']:.1%} of claims flagged.")
    if row["avg_amount"] > 6000:
        reasons.append("High claim amounts detected.")
    if row["max_frequency"] > 25:
        reasons.append("Abnormal claim frequency.")
    return " | ".join(reasons) if reasons else "Normal activity patterns observed."

report["evidence_statement"] = report.apply(generate_evidence, axis=1)

# Dynamic Priority Logic
def calculate_priority(row):
    # If average risk is very high OR a significant portion is flagged
    if row["avg_risk_score"] > risk_threshold + 0.05 or row["fraud_rate"] > 0.20:
        return "High"
    elif row["avg_risk_score"] > risk_threshold - 0.15 or row["fraud_rate"] > 0.05:
        return "Medium"
    else:
        return "Low"

report["audit_priority"] = report.apply(calculate_priority, axis=1)

# -------------------------------
# OVERALL METRICS
# -------------------------------
st.subheader("📊 Overall Risk Summary")

# Calculate Ground Truth Metrics (if fraud_label exists)
has_ground_truth = "fraud_label" in claims.columns
if has_ground_truth:
    total_actual_fraud = int(claims["fraud_label"].sum())
    captured_fraud = int(claims[(claims["fraud_flag"]==True) & (claims["fraud_label"]==1)].shape[0])
    detection_rate = captured_fraud / total_actual_fraud if total_actual_fraud > 0 else 1.0
else:
    total_actual_fraud = 0
    detection_rate = 0.0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Claims", f"{len(claims):,}")
col2.metric("Flagged Claims", int(claims["fraud_flag"].sum()))
col3.metric("High Risk Providers", len(report[report["audit_priority"] == "High"]))

if has_ground_truth:
    col4.metric("Actual Fraud (Hidden)", total_actual_fraud)
    col5.metric("Detection Rate", f"{detection_rate:.1%}")
else:
    col4.metric("Avg Risk Score", round(claims["final_risk_score"].mean(), 3))

st.info("💡 **Note:** 10,000 claims are generated across 150 unique providers. Flagging 10 high-risk providers means 6.7% of your provider network is identified as suspicious, which is a significant volume for investigative teams.")

# -------------------------------
# REGULATORY AUDIT REPORT
# -------------------------------
st.divider()
st.subheader("⚖️ Regulatory Audit Report (Evidence Statements)")
st.markdown("""
**Purpose:** This report is designed for submission to regulatory authorities. It provides an automated **Evidence Statement** 
explaining *why* a provider has been flagged, including anomalies in billing frequency and claim amounts.
""")

# Styling the report table
def color_priority(val):
    color = 'red' if val == 'High' else 'orange' if val == 'Medium' else 'green'
    return f'color: {color}; font-weight: bold'

st.dataframe(
    report.style.applymap(color_priority, subset=['audit_priority']),
    use_container_width=True
)

# -------------------------------
# CLAIM-LEVEL VIEW
# -------------------------------
st.divider()
st.subheader("🚩 Flagged Claims (High Risk)")
st.markdown("""
**Explanation:** This table lists individual claims that exceeded the risk threshold. 
- **Supervised Score:** Probability of fraud based on known historical patterns.
- **Unsupervised Score:** Degree of "strangeness" compared to typical billing behavior.
""")
flagged_claims = claims[claims["fraud_flag"] == True].sort_values("final_risk_score", ascending=False)
st.dataframe(flagged_claims, use_container_width=True)

# -------------------------------
# PROVIDER VISUALIZATION
# -------------------------------
st.divider()
st.subheader("👨‍⚕️ Provider Risk Distribution")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("""
    **Bar Chart Explanation:** Each bar represents a unique Provider ID. The height shows their average risk score.
    - **Red Bars:** Providers with scores > 0.7 (Immediate Audit Recommended).
    - **Blue Bars:** Lower risk providers.
    """)
    fig, ax = plt.subplots(figsize=(10, 6))
    priority_colors = {'High': '#ff4b4b', 'Medium': '#ffa500', 'Low': '#1f77b4'}
    colors = [priority_colors.get(p, '#1f77b4') for p in report["audit_priority"]]
    ax.bar(report["provider_id"], report["avg_risk_score"], color=colors)
    ax.set_ylabel("Risk Score")
    ax.set_xlabel("Provider ID")
    plt.xticks(rotation=90)
    st.pyplot(fig)

with col_right:
    st.markdown("""
    **Audit Priority Breakdown:**
    Distribution of providers across risk tiers. This helps in allocating investigative resources efficiently.
    """)
    priority_counts = report["audit_priority"].value_counts()
    fig2, ax2 = plt.subplots()
    ax2.pie(priority_counts, labels=priority_counts.index, autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99'])
    st.pyplot(fig2)

# -------------------------------
# FRADUSTERS SPOTLIGHT (Detailed Case List)
# -------------------------------
st.divider()
st.subheader("🕵️ Fraudster Spotlight: Case-by-Case Justification")
st.markdown("""
**Regulatory Requirement:** Below is the list of providers flagged as high-risk fraudsters. 
The **Justification** column details the specific billing anomalies and statistical outliers that led to this classification.
""")

fraudsters = report[report["audit_priority"] == "High"].sort_values("avg_risk_score", ascending=False)

if not fraudsters.empty:
    for _, row in fraudsters.iterrows():
        with st.expander(f"Case Report: {row['provider_id']} (Risk Score: {row['avg_risk_score']:.2f})"):
            st.write(f"**Primary Evidence:** {row['evidence_statement']}")
            st.write("---")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Avg Claim Amount", f"${row['avg_amount']:,.2f}")
            col_b.metric("Max Frequency", f"{row['max_frequency']} claims/day")
            col_c.metric("Anomaly Rate", f"{row['fraud_rate']:.1%}")
            st.info(f"**Recommendation:** Provider {row['provider_id']} displays behavior consistent with phantom billing and upcoding. Immediate suspension for internal audit is recommended.")
else:
    st.success("No high-risk fraudsters identified in the current batch.")

# -------------------------------
# EXPLAINABILITY (Deeper SHAP Analysis)
# -------------------------------
st.divider()
st.subheader("🔍 Deep Dive: How the AI Thinks (SHAP)")
st.markdown("""
### Understanding the SHAP Summary Plot
SHAP (SHapley Additive exPlanations) values help us break down the "Black Box" of machine learning. 

**How to read this plot:**
1. **Vertical Axis:** Features are ranked by importance. Features at the top had the most impact on whether a claim was flagged as fraud.
2. **Color (Red/Blue):** 
   - **Red** indicates a *high* value for that feature.
   - **Blue** indicates a *low* value.
3. **Horizontal Axis (SHAP Value):**
   - Dots to the **right** of the center line indicate features that *increased* the fraud risk.
   - Dots to the **left** indicate features that *decreased* the risk.

**Clinical Interpretation:**
""")

import json
if os.path.exists("model_meta.json"):
    with open("model_meta.json", "r") as f:
        meta = json.load(f)
        top_feature = meta.get("top_feature", "claim_amount")
        
        explanations = {
            "claim_amount": f"The feature `{top_feature}` is at the top with many **red dots on the right**, which means the model has learned that high-dollar claims are the primary indicator of fraud in this specific dataset (Potential Upcoding/Phantom Billing).",
            "claim_frequency": f"The feature `{top_feature}` is the primary driver. The AI has identified that an unusually high number of claims per provider is the most significant signal for fraudulent activity.",
            "procedure_rarity": f"The model is primarily looking at `{top_feature}`. This suggests that billing for rare or complex procedures that don't match the provider's typical profile is the main red flag here.",
            "days_between_claims": f"The feature `{top_feature}` (Velocity) is dominant. The model has learned that very short intervals between claims for similar services are the strongest predictor of fraud.",
            "patient_age": f"Surprisingly, `{top_feature}` is a key indicator. This might suggest the model found age-related billing anomalies or systematic targeting of specific demographics."
        }
        
        # Fallback for specialty-related dummies
        if "specialty_" in top_feature:
            st.info(f"**Primary Indicator Identified:** The model found that claims from the `{top_feature.replace('specialty_', '')}` specialty are being flagged at a higher rate, possibly due to higher baseline risk in those procedures.")
        else:
            st.info(f"**Primary Indicator Identified:** {explanations.get(top_feature, f'The model has prioritized `{top_feature}` as the most significant factor in identifying fraud.')}")
else:
    st.info("Clinical interpretation will appear here once the model is trained and metadata is available.")

st.markdown("""
""")

if os.path.exists("shap_summary.png"):
    st.image("shap_summary.png", caption="Global Behavioral Patterns identified by the AI")
else:
    st.info("SHAP plot not found. Please run the model to generate the visualization.")

# -------------------------------
# MODEL PERFORMANCE METRICS
# -------------------------------
st.divider()
st.subheader("⚙️ AI System Performance")
if os.path.exists("model_meta.json"):
    with open("model_meta.json", "r") as f:
        meta = json.load(f)
        accuracy = meta.get("accuracy", 0.0)
        st.write(f"### Model Accuracy: `{accuracy:.2%}`")
        st.progress(accuracy)
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Unsupervised Method", meta.get("unsupervised_method", "N/A"))
        with col_m2:
            st.metric("Best CV F1-Score", f"{meta.get('cv_f1_score', 0.0):.4f}")
            

            
        st.caption("This system uses a Hybrid AI approach combining Supervised XGBoost (tuned via CV) and an Unsupervised Deep Learning Autoencoder.")
else:
    st.info("Metrics will appear here after the model has been trained.")
