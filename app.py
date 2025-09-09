import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from PIL import Image
import os

st.set_page_config(
    page_title="WellDoc AI Risk Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_placeholder_image(image_name):
    try:
        image_path = os.path.join("assets", image_name)
        return Image.open(image_path)
    except:
        img = Image.new('RGB', (600, 400), color=(73, 109, 137))
        return img

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Upload", "Data Visualization", "Risk Prediction"])

if "patient_data" not in st.session_state:
    st.session_state.patient_data = None

if "prediction_results" not in st.session_state:
    st.session_state.prediction_results = None

if page == "Data Upload":
    st.title("AI-Driven Risk Prediction Engine for Chronic Care Patients")
    st.header("Data Upload")
    
    st.markdown("""
    Welcome to the WellDoc AI Risk Prediction Dashboard. This tool helps healthcare providers 
    predict the risk of deterioration for chronic care patients within 90 days.
    
    Please upload your patient data in CSV or JSON format.
    """)
    
    upload_option = st.radio("Select file format", ["CSV", "JSON"])
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_file = st.file_uploader(f"Upload patient data ({upload_option})", 
                                        type=['csv'] if upload_option == "CSV" else ['json'])
    
    with col2:
        st.write("")
        st.write("")
        if st.button("📊 Load Sample Data"):
            try:
                if upload_option == "CSV":
                    sample_path = "assets/sample_patient_data.csv"
                    data = pd.read_csv(sample_path)
                else:
                    sample_path = "assets/sample_patient_data.json"
                    with open(sample_path, 'r') as f:
                        data = pd.DataFrame(json.load(f))
                
                st.session_state.patient_data = data
                st.success(f"Sample data loaded successfully!")
                
            except Exception as e:
                st.error(f"Error loading sample data: {e}")
    
    if uploaded_file is not None:
        try:
            if upload_option == "CSV":
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.DataFrame(json.load(uploaded_file))
                
            st.session_state.patient_data = data
            
            st.success("Data uploaded successfully!")
            
            st.subheader("Data Preview")
            st.dataframe(data.head())
            
            st.subheader("Data Statistics")
            st.write(f"Total patients: {data.shape[0]}")
            st.write(f"Features available: {data.shape[1]}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Data Types")
                st.write(data.dtypes)
            
            with col2:
                st.subheader("Missing Values")
                st.write(data.isnull().sum())
                
        except Exception as e:
            st.error(f"Error: {e}")

elif page == "Data Visualization":
    st.title("Data Visualization")
    
    if st.session_state.patient_data is None:
        st.warning("Please upload patient data first on the Data Upload page.")
        if st.button("Go to Data Upload"):
            st.session_state.page = "Data Upload"
    else:
        data = st.session_state.patient_data
        
        st.subheader("Patient Data Overview")
        
        tab1, tab2, tab3 = st.tabs(["Demographics", "Clinical Metrics", "Temporal Patterns"])
        
        with tab1:
            st.write("### Patient Demographics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(load_placeholder_image("age_distribution.png"), caption="Age Distribution")
                
            with col2:
                st.image(load_placeholder_image("gender_distribution.png"), caption="Gender Distribution")
                
            st.image(load_placeholder_image("geographic_distribution.png"), caption="Geographic Distribution")
        
        with tab2:
            st.write("### Clinical Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(load_placeholder_image("clinical_metric1.png"), caption="Blood Glucose Levels")
                
            with col2:
                st.image(load_placeholder_image("clinical_metric2.png"), caption="Blood Pressure Readings")
                
            st.image(load_placeholder_image("clinical_correlation.png"), caption="Clinical Metric Correlations")
        
        with tab3:
            st.write("### Temporal Patterns")
            
            st.image(load_placeholder_image("temporal_pattern.png"), caption="Metric Changes Over Time")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(load_placeholder_image("trend_analysis.png"), caption="Trend Analysis")
                
            with col2:
                st.image(load_placeholder_image("seasonal_patterns.png"), caption="Seasonal Patterns")

elif page == "Risk Prediction":
    st.title("Risk Prediction")
    
    if st.session_state.patient_data is None:
        st.warning("Please upload patient data first on the Data Upload page.")
        if st.button("Go to Data Upload"):
            st.session_state.page = "Data Upload"
    else:        
        if st.button("Run Prediction"):
            with st.spinner("Running AI prediction model..."):
                import time
                time.sleep(2)
                
                st.session_state.prediction_results = True
            
            st.success("Prediction completed!")
        
        if st.session_state.prediction_results:
            st.subheader("Prediction Results")
            
            tab1, tab2, tab3 = st.tabs(["Risk Assessment", "Model Performance", "Explainability"])
            
            with tab1:
                st.write("### Patient Risk Assessment")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.metric(label="Overall Risk Score", value="76%", delta="12%")
                    st.metric(label="Time Horizon", value="90 days")
                    st.metric(label="Confidence Level", value="High")
                
                with col2:
                    st.image(load_placeholder_image("risk_gauge.png"), caption="Risk Assessment Gauge")
                
                st.subheader("Patient-Specific Risk Factors")
                st.image(load_placeholder_image("patient_risk_factors.png"), caption="Top Risk Factors")
            
            with tab2:
                st.write("### Model Performance Metrics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(load_placeholder_image("roc_curve.png"), caption="ROC Curve (AUROC: 0.87)")
                
                with col2:
                    st.image(load_placeholder_image("pr_curve.png"), caption="Precision-Recall Curve (AUPRC: 0.82)")
                
                st.subheader("Confusion Matrix")
                st.image(load_placeholder_image("confusion_matrix.png"), caption="Confusion Matrix")
                
                st.subheader("Calibration Plot")
                st.image(load_placeholder_image("calibration_plot.png"), caption="Model Calibration")
            
            with tab3:
                st.write("### Model Explainability")
                
                st.subheader("Global Feature Importance")
                st.image(load_placeholder_image("global_feature_importance.png"), caption="Top Features Driving Predictions")
                
                st.subheader("Patient-Specific Insights")
                st.image(load_placeholder_image("local_feature_importance.png"), caption="Factors Affecting This Patient")
                
                st.subheader("Clinical Summary")
                st.info("""
                **AI-Generated Clinical Interpretation:**
                
                This patient shows elevated risk (76%) for deterioration within 90 days. 
                The primary contributing factors are:
                
                1. Consistent elevation in blood glucose levels over the past 45 days
                2. Missed medication adherence (17 instances in last 60 days)
                3. Declining physical activity metrics compared to baseline
                
                Recommended interventions include medication regimen review, 
                diabetes management education, and increased frequency of glucose monitoring.
                """)

st.sidebar.markdown("---")
st.sidebar.caption("WellDoc AI Prediction Engine v1.0")
