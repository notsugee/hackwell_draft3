# WellDoc AI Risk Prediction Dashboard

An AI-driven risk prediction engine for chronic care patients that processes 30-180 days of patient data to predict the probability of deterioration within 90 days.

## Overview

This dashboard consists of three main pages:

1. **Data Upload**: Upload patient data in CSV or JSON format
2. **Data Visualization**: Visualize patient data with interactive charts
3. **Risk Prediction**: Predict patient deterioration risk and explore model explanations

## Features

- Upload and preview patient data
- Interactive data visualizations
- AI-powered risk prediction
- Model performance metrics (AUROC, AUPRC, Calibration, Confusion Matrix)
- Model explainability with global and local feature importance
- Clinician-friendly AI-generated summaries

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:
   ```
   streamlit run app.py
   ```

## Usage

1. Navigate to the Data Upload page and upload your patient data in CSV or JSON format
2. Explore the data visualizations on the Data Visualization page
3. Run the risk prediction model on the Risk Prediction page to get patient risk assessments and model explanations

## Project Structure

```
├── app.py                 # Main Streamlit application
├── model_utils.py         # Utility functions for data processing and prediction
├── requirements.txt       # Python dependencies
├── assets/                # Placeholder images and resources
└── README.md              # This file
```

## Model Approach

Our model processes 30-180 days of patient data, including vitals, medication adherence, activity metrics, and lab results to predict the probability of deterioration within 90 days.

## Evaluation Metrics

The model is evaluated using:
- AUROC (Area Under Receiver Operating Characteristic curve)
- AUPRC (Area Under Precision-Recall Curve)
- Calibration metrics
- Confusion matrix
