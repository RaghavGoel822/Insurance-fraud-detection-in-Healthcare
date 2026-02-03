# 🏥 Healthcare Fraud Detection System using Hybrid AI

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![TensorFlow](https://img.shields.io/badge/Model-Autoencoder-orange)

## 📌 Project Overview

This project implements a **Hybrid AI System** designed to detect fraudulent healthcare claims with high accuracy. By combining **Supervised Learning (XGBoost)** for known fraud patterns and **Unsupervised Learning (Deep Autoencoders)** for novel anomalies, the system provides a robust defense against financial loss in the healthcare sector.

The project includes a fully interactive **Dashboard** for analysts, an **Automated Regulatory Reporting** module, and a **Synthetic Data Generator** to simulate realistic fraud scenarios (Phantom Billing, Upcoding, Unbundling).

## 🚀 Key Features

-   **Hybrid AI Architecture**:
    -   **XGBoost**: Detects known fraud patterns (e.g., Upcoding) with high precision.
    -   **Autoencoder**: A Deep Neural Network that flags "unknown unknowns" by measuring reconstruction error.
    -   **Ensemble Scoring**: Combines both models (`0.7 * Supervised + 0.3 * Unsupervised`) for a final Risk Score.
-   **Explainable AI (XAI)**:
    -   Integrates **SHAP (SHapley Additive exPlanations)** to explain *why* a specific provider was flagged.
-   **Interactive Dashboard**:
    -   Built with **Streamlit**.
    -   Visualizes fraud distribution, high-risk providers, and financial impact.
    -   Allows dynamic "What-If" simulation (adjusting fraud intensity and thresholds).
-   **Automated Reporting**:
    -   Generates `regulatory_report.csv` with natural language evidence statements for auditors.
    -   Generates `Healthcare_Fraud_Detection.pptx` (Presentation) and `Project_Report.docx` (Word Report) programmatically.

## 📂 Project Structure

```
├── dashboard.py             # Main Streamlit Dashboard application
├── fraud_model.py           # core AI training pipeline (XGBoost + Autoencoder)
├── generate_data.py         # Synthetic data generator (HIPAA-compliant simulation)
├── create_pptx.py           # Script to generate the PowerPoint presentation
├── create_report.py         # Script to generate the detailed Word report
├── requirements.txt         # Project dependencies
├── data/
│   └── claims_data.csv      # Generated dataset
├── fraud_results.csv        # Model predictions and scoring
├── regulatory_report.csv    # Audit evidence for flagged providers
├── model_meta.json          # Model metadata and accuracy metrics
├── shap_summary.png         # Global feature importance plot
├── Healthcare_Fraud_Detection.pptx # Generated Slide Deck
└── Project_Report.docx      # Generated Technical Report
```

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/healthcare-fraud-detection.git
    cd healthcare-fraud-detection
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## ⚡ Usage

### 1. Generate Data & Train Model
Run the pipeline to generate synthetic data and train the Hybrid AI models:
```bash
python fraud_model.py
```
*This will create `fraud_results.csv`, `model_meta.json`, and `shap_summary.png`.*

### 2. Launch the Dashboard
Start the interactive application:
```bash
streamlit run dashboard.py
```
*Access the dashboard in your browser at `http://localhost:8501`.*

### 3. Generate Documentation (Optional)
To regenerate the Presentation or Technical Report:
```bash
python create_pptx.py   # Generates PPTX
python create_report.py # Generates DOCX
```

## 📊 Methodology Details

### Synthetic Data Generation
We use `generate_data.py` to create realistic claims data using statistical distributions:
-   **Cost**: Gamma Distribution (Right-skewed).
-   **Frequency**: Poisson Distribution.
-   **Fraud Injection**: Targeted injection of patterns like *Phantom Billing* (High Freq, Normal Cost) and *Upcoding* (High Cost, Low Complexity).

### Hybrid Scoring Logic
The final risk score is calculated as:
$$ \text{Risk} = 0.7 \times P(\text{XGB}) + 0.3 \times \text{Norm}(\text{Reconstruction Error}) $$

This ensures that we catch obvious fraud (XGBoost) while having a safety net for new, creative fraud schemes (Autoencoder).

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).
