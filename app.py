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
                # Dynamic Age Distribution
                fig, ax = plt.subplots(figsize=(10, 6))
                age_data = data.drop_duplicates('patient_id')['age']
                sns.histplot(age_data, bins=12, kde=True, color='#1f77b4', ax=ax)
                ax.set_title('Age Distribution of Patients')
                ax.set_xlabel('Age (years)')
                ax.set_ylabel('Number of Patients')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
            with col2:
                # Dynamic Gender Distribution
                fig, ax = plt.subplots(figsize=(8, 8))
                gender_counts = data.drop_duplicates('patient_id')['gender'].value_counts()
                ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', 
                       colors=['#1f77b4', '#ff7f0e'], startangle=90, explode=[0.05, 0])
                ax.set_title('Gender Distribution of Patients')
                st.pyplot(fig)
            
            # Add random region data for visualization purposes
            if 'region' not in data.columns:
                unique_patients = data['patient_id'].unique()
                region_map = {}
                regions = ['Northeast', 'Southeast', 'Midwest', 'West', 'Southwest']
                for patient in unique_patients:
                    region_map[patient] = np.random.choice(regions)
                data['region'] = data['patient_id'].map(region_map)
            
            # Dynamic Geographic Distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            region_counts = data.drop_duplicates('patient_id')['region'].value_counts()
            sns.barplot(x=region_counts.index, y=region_counts.values, palette='viridis', ax=ax)
            ax.set_title('Geographic Distribution of Patients')
            ax.set_xlabel('Region')
            ax.set_ylabel('Number of Patients')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        with tab2:
            st.write("### Clinical Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Dynamic Blood Glucose Levels
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(x='has_diabetes', y='blood_glucose', data=data, 
                           palette=['#2ca02c', '#d62728'], ax=ax)
                ax.set_title('Blood Glucose Levels by Diabetes Status')
                ax.set_xlabel('Has Diabetes')
                ax.set_ylabel('Blood Glucose (mg/dL)')
                ax.set_xticklabels(['No', 'Yes'])
                ax.axhline(y=125, color='red', linestyle='--', alpha=0.7, label='High Risk Threshold')
                ax.legend()
                st.pyplot(fig)
                
            with col2:
                # Dynamic Blood Pressure Readings
                fig, ax = plt.subplots(figsize=(10, 6))
                bp_data = data.groupby('has_hypertension').agg({
                    'systolic_bp': 'mean',
                    'diastolic_bp': 'mean'
                }).reset_index()

                x = np.arange(2)
                width = 0.35
                ax.bar(x - width/2, bp_data['systolic_bp'], width, label='Systolic', color='#1f77b4')
                ax.bar(x + width/2, bp_data['diastolic_bp'], width, label='Diastolic', color='#ff7f0e')
                ax.set_title('Average Blood Pressure by Hypertension Status')
                ax.set_xticks(x)
                ax.set_xticklabels(['Non-Hypertensive', 'Hypertensive'])
                ax.set_ylabel('Blood Pressure (mmHg)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            # Dynamic Clinical Correlation Heatmap
            fig, ax = plt.subplots(figsize=(12, 10))
            clinical_cols = ['blood_glucose', 'systolic_bp', 'diastolic_bp', 
                            'heart_rate', 'physical_activity', 'sleep_quality']
            correlation = data[clinical_cols].corr()
            mask = np.triu(np.ones_like(correlation, dtype=bool))
            sns.heatmap(correlation, mask=mask, annot=True, fmt='.2f', cmap='viridis',
                       linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax)
            ax.set_title('Correlation Between Clinical Metrics')
            st.pyplot(fig)
        
        with tab3:
            st.write("### Temporal Patterns")
            
            # Ensure date is in datetime format
            if data['date'].dtype != 'datetime64[ns]':
                data['date'] = pd.to_datetime(data['date'])
            
            # Dynamic Metric Changes Over Time
            fig, ax = plt.subplots(figsize=(12, 7))
            data['week'] = (data['date'].dt.to_period('W').astype(str)
                          .apply(lambda x: x.split('/')[-1]))
            
            # Calculate weekly averages for some patients
            sample_patients = data['patient_id'].unique()[:5]
            patient_subset = data[data['patient_id'].isin(sample_patients)]
            weekly_avg = patient_subset.groupby(['patient_id', 'week'])['blood_glucose'].mean().reset_index()
            
            # Plot the time series for each patient
            for patient_id, group in weekly_avg.groupby('patient_id'):
                ax.plot(group['week'], group['blood_glucose'], marker='o', linewidth=2, 
                       label=f"Patient {patient_id}")
            
            ax.set_title('Blood Glucose Trends Over Time for Sample Patients')
            ax.set_xlabel('Week')
            ax.set_ylabel('Average Blood Glucose (mg/dL)')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Dynamic Trend Analysis
                fig, ax = plt.subplots(figsize=(10, 6))
                date_groups = data.groupby('date')
                trend_data = date_groups.agg({
                    'blood_glucose': 'mean',
                    'systolic_bp': 'mean',
                    'physical_activity': 'mean'
                }).reset_index()
                trend_data = trend_data.sort_values('date')
                
                # Normalize the data for comparison
                def normalize(series):
                    if series.max() == series.min():
                        return series - series.min()
                    return (series - series.min()) / (series.max() - series.min())
                
                trend_data['glucose_norm'] = normalize(trend_data['blood_glucose'])
                trend_data['bp_norm'] = normalize(trend_data['systolic_bp'])
                trend_data['activity_norm'] = normalize(trend_data['physical_activity'])
                
                # Plot the trends
                ax.plot(trend_data['date'], trend_data['glucose_norm'], label='Blood Glucose', color='#1f77b4')
                ax.plot(trend_data['date'], trend_data['bp_norm'], label='Systolic BP', color='#ff7f0e')
                ax.plot(trend_data['date'], trend_data['activity_norm'], label='Physical Activity', color='#2ca02c')
                
                ax.set_title('Normalized Trend Analysis of Key Health Metrics')
                ax.set_xlabel('Date')
                ax.set_ylabel('Normalized Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
            with col2:
                # Dynamic Seasonal Patterns
                fig, ax = plt.subplots(figsize=(10, 6))
                data['month'] = data['date'].dt.month
                monthly_stats = data.groupby('month').agg({
                    'physical_activity': 'mean',
                    'blood_glucose': 'mean'
                }).reset_index()
                
                ax2 = ax.twinx()
                line1 = ax.plot(monthly_stats['month'], monthly_stats['physical_activity'], 
                               color='#1f77b4', marker='o', linewidth=2, label='Physical Activity')
                line2 = ax2.plot(monthly_stats['month'], monthly_stats['blood_glucose'], 
                                color='#ff7f0e', marker='s', linewidth=2, label='Blood Glucose')
                
                ax.set_xlabel('Month')
                ax.set_ylabel('Average Physical Activity (minutes)')
                ax2.set_ylabel('Average Blood Glucose (mg/dL)')
                ax.set_title('Seasonal Patterns in Activity and Blood Glucose')
                
                # Fix the month labels
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                ax.set_xticks(monthly_stats['month'])
                month_labels = [months[i-1] for i in monthly_stats['month']]
                ax.set_xticklabels(month_labels)
                
                ax.grid(True, alpha=0.3)
                
                # Combine legends
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax.legend(lines, labels, loc='upper left')
                st.pyplot(fig)

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
