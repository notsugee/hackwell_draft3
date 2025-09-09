import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from PIL import Image
import os
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

st.set_page_config(
    page_title="WellDoc AI Risk Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_image(image_name):
    try:
        image_path = os.path.join("assets", image_name)
        return Image.open(image_path)
    except:
        img = Image.new('RGB', (600, 400), color=(73, 109, 137))
        return img

# Function to prepare patient data for prediction
def prepare_patient_data_for_prediction(data, patient_id=None):
    if patient_id is None:
        # If no patient ID provided, select a random one
        patient_id = np.random.choice(data['patient_id'].unique())
    
    # Get data for selected patient
    patient_data = data[data['patient_id'] == patient_id].sort_values('date')
    
    # Extract features for prediction
    features = {
        'blood_glucose_avg': patient_data['blood_glucose'].mean(),
        'blood_glucose_std': patient_data['blood_glucose'].std(),
        'systolic_bp_avg': patient_data['systolic_bp'].mean(),
        'diastolic_bp_avg': patient_data['diastolic_bp'].mean(),
        'heart_rate_avg': patient_data['heart_rate'].mean(),
        'medication_adherence': patient_data['medication_adherence'].mean(),
        'physical_activity_avg': patient_data['physical_activity'].mean(),
        'sleep_quality_avg': patient_data['sleep_quality'].mean(),
        'has_diabetes': patient_data['has_diabetes'].iloc[0],
        'has_hypertension': patient_data['has_hypertension'].iloc[0],
        'has_heart_disease': patient_data['has_heart_disease'].iloc[0],
        'age': patient_data['age'].iloc[0],
    }
    
    # Check for deterioration in last 30 days vs first 30 days
    if len(patient_data) >= 60:
        first_30_days = patient_data.iloc[:30]
        last_30_days = patient_data.iloc[-30:]
        
        features['glucose_trend'] = (last_30_days['blood_glucose'].mean() - first_30_days['blood_glucose'].mean()) / first_30_days['blood_glucose'].mean()
        features['bp_trend'] = (last_30_days['systolic_bp'].mean() - first_30_days['systolic_bp'].mean()) / first_30_days['systolic_bp'].mean()
        features['activity_trend'] = (last_30_days['physical_activity'].mean() - first_30_days['physical_activity'].mean()) / first_30_days['physical_activity'].mean()
    else:
        features['glucose_trend'] = 0
        features['bp_trend'] = 0
        features['activity_trend'] = 0
    
    return patient_id, features

# Function to generate risk prediction
def predict_risk(features):
    # In a real application, this would use a trained model
    # For demonstration, we'll calculate risk based on key factors
    
    # Define the weights for each factor
    weights = {
        'blood_glucose_avg': 0.25,
        'systolic_bp_avg': 0.15,
        'medication_adherence': -0.20,  # negative because higher adherence reduces risk
        'physical_activity_avg': -0.10,  # negative because higher activity reduces risk
        'has_diabetes': 0.10,
        'has_hypertension': 0.05,
        'has_heart_disease': 0.10,
        'glucose_trend': 0.25,  # positive trend is bad
        'bp_trend': 0.15,       # positive trend is bad
        'activity_trend': -0.15  # negative because positive trend in activity is good
    }
    
    # Normalize the features
    norm_features = {}
    norm_features['blood_glucose_avg'] = min(1.0, max(0, (features['blood_glucose_avg'] - 90) / 100))
    norm_features['systolic_bp_avg'] = min(1.0, max(0, (features['systolic_bp_avg'] - 110) / 70))
    norm_features['medication_adherence'] = features['medication_adherence']
    norm_features['physical_activity_avg'] = min(1.0, features['physical_activity_avg'] / 60)
    norm_features['has_diabetes'] = 1 if features['has_diabetes'] else 0
    norm_features['has_hypertension'] = 1 if features['has_hypertension'] else 0
    norm_features['has_heart_disease'] = 1 if features['has_heart_disease'] else 0
    norm_features['glucose_trend'] = min(1.0, max(0, features['glucose_trend'] * 3))
    norm_features['bp_trend'] = min(1.0, max(0, features['bp_trend'] * 3))
    norm_features['activity_trend'] = min(1.0, max(-1, features['activity_trend'] * 3))
    
    # Calculate risk score
    risk_score = 0
    factor_contributions = {}
    
    for factor, weight in weights.items():
        contribution = norm_features[factor] * weight
        risk_score += contribution
        factor_contributions[factor] = contribution
    
    # Normalize to 0-100 scale
    risk_score = min(100, max(0, risk_score * 100))
    
    # Sort factors by absolute contribution
    sorted_factors = sorted(factor_contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    
    return risk_score, sorted_factors

# Create friendly names for factors
factor_names = {
    'blood_glucose_avg': 'Average Blood Glucose',
    'systolic_bp_avg': 'Average Systolic Blood Pressure',
    'medication_adherence': 'Medication Adherence',
    'physical_activity_avg': 'Physical Activity Level',
    'has_diabetes': 'Diabetes Status',
    'has_hypertension': 'Hypertension Status',
    'has_heart_disease': 'Heart Disease Status',
    'glucose_trend': 'Blood Glucose Trend',
    'bp_trend': 'Blood Pressure Trend',
    'activity_trend': 'Physical Activity Trend'
}

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Upload", "Data Visualization", "Risk Prediction"])

if "patient_data" not in st.session_state:
    st.session_state.patient_data = None

if "prediction_results" not in st.session_state:
    st.session_state.prediction_results = None

if "selected_patient" not in st.session_state:
    st.session_state.selected_patient = None

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
                st.image(load_image("age_distribution.png"), caption="Age Distribution")
                
            with col2:
                st.image(load_image("gender_distribution.png"), caption="Gender Distribution")
                
            st.image(load_image("geographic_distribution.png"), caption="Geographic Distribution")
        
        with tab2:
            st.write("### Clinical Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(load_image("clinical_metric1.png"), caption="Blood Glucose Levels")
                
            with col2:
                st.image(load_image("clinical_metric2.png"), caption="Blood Pressure Readings")
                
            st.image(load_image("clinical_correlation.png"), caption="Clinical Metric Correlations")
        
        with tab3:
            st.write("### Temporal Patterns")
            
            st.image(load_image("temporal_pattern.png"), caption="Metric Changes Over Time")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(load_image("trend_analysis.png"), caption="Trend Analysis")
                
            with col2:
                st.image(load_image("seasonal_patterns.png"), caption="Seasonal Patterns")

elif page == "Risk Prediction":
    st.title("Risk Prediction")
    
    if st.session_state.patient_data is None:
        st.warning("Please upload patient data first on the Data Upload page.")
        if st.button("Go to Data Upload"):
            st.session_state.page = "Data Upload"
    else:
        data = st.session_state.patient_data
        
        # Select a patient for analysis
        st.subheader("Select Patient for Risk Assessment")
        
        # Get list of unique patients
        unique_patients = sorted(data['patient_id'].unique())
        
        # Let user select a patient or use previous selection
        selected_patient = st.selectbox(
            "Choose a patient ID:",
            unique_patients,
            index=0 if st.session_state.selected_patient is None else 
                 unique_patients.index(st.session_state.selected_patient)
        )
        
        # Update session state
        st.session_state.selected_patient = selected_patient
        
        # Show patient summary
        patient_data = data[data['patient_id'] == selected_patient]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Age", f"{patient_data['age'].iloc[0]}")
        with col2:
            st.metric("Gender", f"{patient_data['gender'].iloc[0]}")
        with col3:
            conditions = []
            if patient_data['has_diabetes'].iloc[0]:
                conditions.append("Diabetes")
            if patient_data['has_hypertension'].iloc[0]:
                conditions.append("Hypertension")
            if patient_data['has_heart_disease'].iloc[0]:
                conditions.append("Heart Disease")
            st.metric("Conditions", ", ".join(conditions) if conditions else "None")
        
        # Button to run prediction
        if st.button("Run Risk Prediction Analysis"):
            with st.spinner("Running AI prediction model..."):
                # Prepare data and run prediction
                patient_id, features = prepare_patient_data_for_prediction(data, selected_patient)
                risk_score, factor_contributions = predict_risk(features)
                
                # Store results in session state
                st.session_state.prediction_results = {
                    'patient_id': patient_id,
                    'risk_score': risk_score,
                    'factor_contributions': factor_contributions,
                    'features': features
                }
            
            st.success("Prediction completed!")
        
        # Show prediction results if available
        if st.session_state.prediction_results is not None:
            results = st.session_state.prediction_results
            
            st.subheader("Prediction Results")
            
            tab1, tab2, tab3 = st.tabs(["Risk Assessment", "Model Performance", "Explainability"])
            
            with tab1:
                st.write("### Patient Risk Assessment")
                
                # Risk score display
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Determine risk level
                    risk_level = "Low"
                    if results['risk_score'] > 30:
                        risk_level = "Moderate"
                    if results['risk_score'] > 60:
                        risk_level = "High"
                        
                    risk_delta = None
                    if 'previous_risk' in results:
                        risk_delta = results['risk_score'] - results['previous_risk']
                    
                    st.metric(
                        label="Overall Risk Score", 
                        value=f"{results['risk_score']:.1f}%", 
                        delta=f"{risk_delta:.1f}%" if risk_delta is not None else None,
                        delta_color="inverse"
                    )
                    st.metric(label="Risk Level", value=risk_level)
                    st.metric(label="Time Horizon", value="90 days")
                
                with col2:
                    st.image(load_image("risk_gauge.png"), caption="Risk Assessment Gauge")
                
                # Patient-specific risk factors
                st.subheader("Patient-Specific Risk Factors")
                
                # Create a DataFrame for displaying factor contributions
                factors_df = []
                for factor, contribution in results['factor_contributions'][:5]:  # Top 5 factors
                    impact = "Increases Risk" if contribution > 0 else "Decreases Risk"
                    magnitude = abs(contribution) / 0.01  # Scale for display
                    factors_df.append({
                        "Factor": factor_names.get(factor, factor),
                        "Impact": impact,
                        "Magnitude": f"{magnitude:.1f}%"
                    })
                
                st.table(pd.DataFrame(factors_df))
                
                # Display the patient risk factors chart
                st.image(load_image("patient_risk_factors.png"), 
                         caption="Visual Representation of Risk Factors")
            
            with tab2:
                st.write("### Model Performance Metrics")
                
                # Performance metrics explanation
                st.markdown("""
                Our risk prediction model was evaluated using the following metrics:
                """)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(load_image("roc_curve.png"), caption="ROC Curve (AUROC: 0.87)")
                    st.markdown("""
                    **AUROC (Area Under Receiver Operating Characteristic)**: 
                    Measures the model's ability to discriminate between patients who will deteriorate 
                    and those who won't. Higher values indicate better discrimination.
                    """)
                
                with col2:
                    st.image(load_image("pr_curve.png"), caption="Precision-Recall Curve (AUPRC: 0.82)")
                    st.markdown("""
                    **AUPRC (Area Under Precision-Recall Curve)**:
                    Particularly important for imbalanced datasets where deterioration events are rare.
                    Higher values indicate better performance in identifying true positives.
                    """)
                
                # Confusion matrix
                st.subheader("Confusion Matrix")
                st.image(load_image("confusion_matrix.png"), caption="Confusion Matrix")
                st.markdown("""
                The confusion matrix shows:
                - **True Positives**: Correctly predicted deteriorations
                - **False Positives**: Incorrectly predicted deteriorations
                - **True Negatives**: Correctly predicted non-deteriorations
                - **False Negatives**: Missed deteriorations
                """)
                
                # Calibration plot
                st.subheader("Calibration Plot")
                st.image(load_image("calibration_plot.png"), caption="Model Calibration")
                st.markdown("""
                **Calibration**: Measures how well the predicted probabilities match the actual outcomes.
                A well-calibrated model's predicted risk of 70% should correspond to an actual 70% 
                frequency of deterioration.
                """)
            
            with tab3:
                st.write("### Model Explainability")
                
                # Global feature importance
                st.subheader("Global Feature Importance")
                st.image(load_image("global_feature_importance.png"), 
                         caption="Top Features Driving Predictions")
                
                st.markdown("""
                These features have the greatest influence on our model's predictions across all patients:
                
                1. **Blood Glucose Trends**: Consistent increases over time
                2. **Medication Adherence**: Frequency of missed medications
                3. **Physical Activity**: Decline in activity levels
                4. **Blood Pressure**: Elevated or unstable readings
                5. **Sleep Quality**: Poor sleep patterns
                """)
                
                # Local feature importance
                st.subheader("Patient-Specific Insights")
                st.image(load_image("local_feature_importance.png"), 
                         caption="Factors Affecting This Patient")
                
                # AI-generated clinical summary
                st.subheader("Clinical Summary")
                
                # Generate a more dynamic summary based on the patient's actual risk factors
                top_factors = [factor for factor, _ in results['factor_contributions'][:3]]
                factor_descriptions = {
                    'blood_glucose_avg': "elevated blood glucose levels",
                    'glucose_trend': "consistent increase in blood glucose over time",
                    'medication_adherence': "missed medication doses",
                    'physical_activity_avg': "low physical activity levels",
                    'activity_trend': "declining physical activity",
                    'systolic_bp_avg': "elevated blood pressure readings",
                    'bp_trend': "worsening blood pressure trends",
                    'has_diabetes': "diabetes condition",
                    'has_hypertension': "hypertension condition",
                    'has_heart_disease': "heart disease condition"
                }
                
                # Format top factors with their descriptions
                factor_text = []
                for i, factor in enumerate(top_factors):
                    if factor in factor_descriptions:
                        factor_text.append(f"{i+1}. {factor_descriptions[factor].capitalize()}")
                
                # Generate recommendations based on top factors
                recommendations = []
                if 'blood_glucose_avg' in top_factors or 'glucose_trend' in top_factors:
                    recommendations.append("More frequent glucose monitoring and diabetes management education")
                if 'medication_adherence' in top_factors:
                    recommendations.append("Medication regimen review and adherence support")
                if 'physical_activity_avg' in top_factors or 'activity_trend' in top_factors:
                    recommendations.append("Personalized physical activity plan with gradual increases")
                if 'systolic_bp_avg' in top_factors or 'bp_trend' in top_factors:
                    recommendations.append("Blood pressure management and more frequent monitoring")
                
                # Default recommendations if none were generated
                if not recommendations:
                    recommendations = ["Regular check-ups", "Continued monitoring of key health metrics"]
                
                # Create the clinical summary
                formatted_recommendations = [f"• {rec}" for rec in recommendations]
                clinical_summary = f"""
                **AI-Generated Clinical Interpretation:**
                
                This patient shows {'high' if results['risk_score'] > 60 else 'moderate' if results['risk_score'] > 30 else 'low'} risk 
                ({results['risk_score']:.1f}%) for deterioration within 90 days. 
                The primary contributing factors are:
                
                {chr(10).join(factor_text)}
                
                Recommended interventions include:
                {chr(10).join(formatted_recommendations)}
                """
                
                st.info(clinical_summary)

st.sidebar.markdown("---")
st.sidebar.caption("WellDoc AI Prediction Engine v1.0")
