import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

def generate_sample_data(num_patients=100, days=180):
    """
    Generate sample patient data for testing the dashboard
    
    Parameters:
    num_patients (int): Number of patients to generate data for
    days (int): Number of days of historical data to generate
    
    Returns:
    pd.DataFrame: Sample patient data
    """
    # Create patient IDs
    patient_ids = [f"P{i:04d}" for i in range(1, num_patients + 1)]
    
    # Create a list to store all patient records
    all_records = []
    
    # Generate data for each patient
    for patient_id in patient_ids:
        # Generate basic demographics
        age = np.random.randint(30, 85)
        gender = np.random.choice(['Male', 'Female'])
        weight = np.random.normal(70, 15)  # in kg
        height = np.random.normal(170, 10)  # in cm
        
        # Generate random medical conditions
        has_diabetes = np.random.choice([True, False], p=[0.6, 0.4])
        has_hypertension = np.random.choice([True, False], p=[0.7, 0.3])
        has_heart_disease = np.random.choice([True, False], p=[0.4, 0.6])
        
        # Generate data for each day
        end_date = datetime.now()
        
        for day in range(days):
            # Calculate the date for this record
            record_date = end_date - timedelta(days=days-day)
            
            # Generate daily measurements with some randomness and trends
            # Base values
            base_glucose = 130 if has_diabetes else 95
            base_systolic = 145 if has_hypertension else 120
            base_diastolic = 90 if has_hypertension else 80
            base_heart_rate = 85 if has_heart_disease else 70
            
            # Add trends (gradually worsening for some patients)
            trend_factor = 0.1 * day / days  # Increases over time
            if patient_id.endswith(('1', '3', '5', '7', '9')):  # Deteriorating patients
                trend_multiplier = 1.5
            else:  # Stable patients
                trend_multiplier = 0.2
                
            # Calculate values with trend and random noise
            glucose = base_glucose * (1 + trend_factor * trend_multiplier) + np.random.normal(0, 10)
            systolic = base_systolic * (1 + trend_factor * trend_multiplier) + np.random.normal(0, 5)
            diastolic = base_diastolic * (1 + trend_factor * trend_multiplier) + np.random.normal(0, 3)
            heart_rate = base_heart_rate * (1 + trend_factor * trend_multiplier) + np.random.normal(0, 4)
            
            # Medication adherence (boolean)
            medication_adherence = np.random.choice([1, 0], p=[0.9 - trend_factor, 0.1 + trend_factor])
            
            # Physical activity (minutes)
            physical_activity = max(0, np.random.normal(30, 10) * (1 - trend_factor * trend_multiplier))
            
            # Sleep quality (hours)
            sleep_quality = max(3, min(10, np.random.normal(7, 1) * (1 - trend_factor * 0.5)))
            
            # Create a record for this patient-day
            record = {
                'patient_id': patient_id,
                'date': record_date.strftime('%Y-%m-%d'),
                'age': age,
                'gender': gender,
                'weight': round(weight, 1),
                'height': round(height, 1),
                'has_diabetes': has_diabetes,
                'has_hypertension': has_hypertension,
                'has_heart_disease': has_heart_disease,
                'blood_glucose': round(glucose, 1),
                'systolic_bp': round(systolic, 1),
                'diastolic_bp': round(diastolic, 1),
                'heart_rate': round(heart_rate, 1),
                'medication_adherence': medication_adherence,
                'physical_activity': round(physical_activity, 1),
                'sleep_quality': round(sleep_quality, 1)
            }
            
            all_records.append(record)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_records)
    
    return df

def save_sample_data():
    """Generate and save sample data in CSV and JSON formats"""
    
    # Create data directory if it doesn't exist
    os.makedirs('assets', exist_ok=True)
    
    # Generate sample data
    sample_data = generate_sample_data(num_patients=50, days=90)
    
    # Save as CSV
    sample_data.to_csv('assets/sample_patient_data.csv', index=False)
    print(f"Sample CSV data saved to assets/sample_patient_data.csv")
    
    # Save as JSON
    sample_data_json = sample_data.to_dict(orient='records')
    with open('assets/sample_patient_data.json', 'w') as f:
        json.dump(sample_data_json, f)
    print(f"Sample JSON data saved to assets/sample_patient_data.json")
    
    return sample_data

if __name__ == "__main__":
    sample_data = save_sample_data()
    print(f"Generated data for {sample_data['patient_id'].nunique()} patients over {sample_data['date'].nunique()} days")
