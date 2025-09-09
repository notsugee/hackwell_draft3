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
import heart_model  # Import the heart disease model module

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
page = st.sidebar.radio("Go to", ["Data Upload", "Data Visualization", "Heart Risk Prediction"])

if "patient_data" not in st.session_state:
    st.session_state.patient_data = None

if "prediction_results" not in st.session_state:
    st.session_state.prediction_results = None

if "selected_patient" not in st.session_state:
    st.session_state.selected_patient = None

if "heart_model" not in st.session_state:
    # Load the heart disease model
    st.session_state.heart_model = heart_model.load_heart_model()

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
    st.subheader("Model Visualizations")
    st.write("Below are the key model evaluation images:")
    col1, col2 = st.columns(2)
    with col1:
        st.image(load_image("confusion_matrix.png"), caption="Confusion Matrix", use_container_width=True)
    with col2:
        st.image(load_image("calibration_curve.png"), caption="Calibration Curve", use_container_width=True)

elif page == "Heart Risk Prediction":
    st.title("Heart Risk Prediction")
    
    if st.session_state.patient_data is None:
        st.warning("Please upload patient data first on the Data Upload page.")
        if st.button("Go to Data Upload"):
            st.session_state.page = "Data Upload"
    else:
        data = st.session_state.patient_data
        
        st.markdown("""
        This page uses a trained heart risk prediction model to assess cardiac risk.
        The model analyzes patient data to predict the probability of heart disease.
        """)
        
        # Select a patient for analysis
        st.subheader("Select Patient for Heart Risk Assessment")
        
        # Get list of unique patients
        unique_patients = sorted(data['patient_id'].unique())
        
        # Let user select a patient
        selected_patient = st.selectbox(
            "Choose a patient ID:",
            unique_patients,
            key="heart_patient_select"
        )
        
        # Show patient summary
        patient_data = data[data['patient_id'] == selected_patient].iloc[0].to_dict()
        
        # Display patient information
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Age", f"{patient_data['age']}")
        with col2:
            st.metric("Gender", f"{patient_data['gender']}")
        with col3:
            st.metric("Blood Glucose", f"{patient_data['blood_glucose']:.1f} mg/dL")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Systolic BP", f"{patient_data['systolic_bp']:.1f} mmHg")
        with col2:
            st.metric("Heart Rate", f"{patient_data['heart_rate']:.1f} bpm")
        with col3:
            conditions = []
            if patient_data['has_diabetes']:
                conditions.append("Diabetes")
            if patient_data['has_hypertension']:
                conditions.append("Hypertension")
            if patient_data['has_heart_disease']:
                conditions.append("Heart Disease")
            st.metric("Conditions", ", ".join(conditions) if conditions else "None")
        
        # Add optional inputs for model
        st.subheader("Additional Clinical Information")
        st.markdown("Provide additional information for more accurate predictions.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            cholesterol = st.number_input("Cholesterol (mg/dL)", 
                                         min_value=100, max_value=400, value=200)
            chest_pain = st.selectbox("Chest Pain Type", 
                                     ["Typical Angina", "Atypical Angina", 
                                      "Non-Anginal Pain", "Asymptomatic"],
                                     index=3)
        
        with col2:
            exercise_angina = st.checkbox("Exercise-Induced Angina", value=False)
            st_depression = st.number_input("ST Depression Induced by Exercise", 
                                           min_value=0.0, max_value=6.0, value=0.0, step=0.1)
        
        # Encode chest pain type
        cp_map = {
            "Typical Angina": 0,
            "Atypical Angina": 1,
            "Non-Anginal Pain": 2,
            "Asymptomatic": 3
        }
        
        # Add values to patient data
        patient_data['chol'] = cholesterol
        patient_data['cp'] = cp_map[chest_pain]
        patient_data['exang'] = 1 if exercise_angina else 0
        patient_data['oldpeak'] = st_depression
        
        # Button to run prediction
        if st.button("Run Heart Risk Prediction"):
            with st.spinner("Running heart risk prediction model..."):
                # Get prediction
                risk_prob, risk_class = heart_model.predict_heart_disease(
                    st.session_state.heart_model, patient_data)
                
                # Get explanation
                explanation = heart_model.generate_shap_explanation(
                    st.session_state.heart_model, patient_data)
                
                # Store in session state
                st.session_state.heart_prediction = {
                    'probability': risk_prob,
                    'prediction_class': risk_class,
                    'explanation': explanation
                }
            
            st.success("Heart risk prediction completed!")
        
        # Show prediction results if available
        if hasattr(st.session_state, 'heart_prediction'):
            heart_results = st.session_state.heart_prediction
            
            st.subheader("Heart Risk Prediction Results")
            
            tab1, tab2 = st.tabs(["Risk Assessment", "Model Explanation"])
            
            with tab1:
                st.write("### Heart Risk Assessment")
                
                # Risk score display
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Format risk level
                    risk_prob = heart_results['probability'] * 100
                    risk_level = "Low"
                    if risk_prob > 30:
                        risk_level = "Moderate"
                    if risk_prob > 60:
                        risk_level = "High"
                    
                    st.metric(
                        label="Heart Disease Risk", 
                        value=f"{risk_prob:.1f}%"
                    )
                    st.metric(label="Risk Level", value=risk_level)
                    st.metric(
                        label="Prediction", 
                        value="Positive" if heart_results['prediction_class'] == 1 else "Negative"
                    )
                
                with col2:
                    # Create risk gauge visualization
                    fig, ax = plt.subplots(figsize=(6, 4))
                    
                    # Create a simple gauge
                    ax.set_xlim(0, 100)
                    ax.set_ylim(0, 10)
                    ax.axvspan(0, 30, color='green', alpha=0.3)
                    ax.axvspan(30, 60, color='yellow', alpha=0.3)
                    ax.axvspan(60, 100, color='red', alpha=0.3)
                    
                    # Add risk marker
                    ax.scatter(risk_prob, 5, color='blue', s=300, zorder=3)
                    
                    # Add labels
                    ax.text(15, 2, "Low Risk", ha='center')
                    ax.text(45, 2, "Moderate Risk", ha='center')
                    ax.text(80, 2, "High Risk", ha='center')
                    ax.text(risk_prob, 7, f"{risk_prob:.1f}%", ha='center', fontweight='bold')
                    
                    # Remove axes
                    ax.set_axis_off()
                    
                    st.pyplot(fig)
                
                # Add clinical interpretation
                st.subheader("Clinical Interpretation")
                
                # Get or generate LLM explanation if available
                google_api_key = st.text_input(
                    "Google API Key for LLM explanation (optional)", 
                    type="password",
                    help="Enter your Google API key to generate a clinical explanation with LLM"
                )
                
                if st.button("Generate Clinical Explanation"):
                    with st.spinner("Generating clinical explanation..."):
                        explanation_text = heart_model.explain_prediction_with_llm(
                            heart_results['explanation'], 
                            api_key=google_api_key if google_api_key else None
                        )
                        st.session_state.heart_explanation = explanation_text
                
                if hasattr(st.session_state, 'heart_explanation'):
                    st.markdown(st.session_state.heart_explanation)
                else:
                    # Show a default explanation
                    if heart_results['prediction_class'] == 1:
                        st.info("""
                        **Clinical Interpretation:**
                        
                        This patient shows significant risk factors for heart disease. 
                        Recommend follow-up with cardiologist for a comprehensive evaluation.
                        Key contributing factors likely include age, blood pressure, and cholesterol levels.
                        """)
                    else:
                        st.info("""
                        **Clinical Interpretation:**
                        
                        This patient shows low risk for heart disease based on current data.
                        Recommend standard preventive care and regular monitoring of cardiac risk factors.
                        """)
            
            with tab2:
                st.write("### Model Explainability")
                
                # Create SHAP waterfall plot
                st.subheader("Feature Importance for This Prediction")
                
                # Generate waterfall plot
                fig = heart_model.plot_shap_waterfall(heart_results['explanation'])
                st.pyplot(fig)
                
                # Show model metrics
                st.subheader("Model Performance Metrics")
                
                metrics = heart_model.get_heart_model_metrics()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("AUROC", f"{metrics['auroc']:.2f}")
                    st.metric("Precision", f"{metrics['precision']:.2f}")
                with col2:
                    st.metric("AUPRC", f"{metrics['auprc']:.2f}")
                    st.metric("Recall", f"{metrics['recall']:.2f}")
                with col3:
                    st.metric("Accuracy", f"{metrics['accuracy']:.2f}")
                    st.metric("F1 Score", f"{metrics['f1']:.2f}")
                
                st.markdown("""
                ### About the Model
                
                This heart disease prediction model was trained on a comprehensive dataset of 
                cardiac patients. It uses a combination of demographic information, laboratory
                values, and clinical symptoms to predict the likelihood of heart disease.
                
                The model was evaluated using cross-validation and shows strong performance
                across multiple metrics including AUROC and precision-recall.
                """)

st.sidebar.markdown("---")
st.sidebar.caption("WellDoc AI Prediction Engine v1.0")
