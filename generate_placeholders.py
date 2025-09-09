import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from generate_sample_data import generate_sample_data

# Create directory if it doesn't exist
os.makedirs('assets', exist_ok=True)

# Load or generate the sample data
try:
    sample_data = pd.read_csv('assets/sample_patient_data.csv')
    print("Loaded existing sample data")
except FileNotFoundError:
    print("Generating new sample data")
    sample_data = generate_sample_data(num_patients=50, days=90)
    sample_data.to_csv('assets/sample_patient_data.csv', index=False)
    sample_data_json = sample_data.to_dict(orient='records')
    with open('assets/sample_patient_data.json', 'w') as f:
        import json
        json.dump(sample_data_json, f)

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
colors = sns.color_palette('viridis', 10)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Function to save figures with consistent styling
def save_figure(fig, name, dpi=150, bbox_inches='tight'):
    fig.tight_layout()
    fig.savefig(f"assets/{name}.png", dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)
    print(f"Created visualization: assets/{name}.png")

# DEMOGRAPHICS VISUALIZATIONS
print("Generating demographic visualizations...")

# Age Distribution
fig, ax = plt.subplots(figsize=(10, 6))
age_data = sample_data.drop_duplicates('patient_id')['age']
sns.histplot(age_data, bins=12, kde=True, color=colors[0], ax=ax)
ax.set_title('Age Distribution of Patients')
ax.set_xlabel('Age (years)')
ax.set_ylabel('Number of Patients')
ax.grid(True, alpha=0.3)
save_figure(fig, "age_distribution")

# Gender Distribution
fig, ax = plt.subplots(figsize=(8, 8))
gender_counts = sample_data.drop_duplicates('patient_id')['gender'].value_counts()
ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', 
       colors=[colors[1], colors[4]], startangle=90, explode=[0.05, 0])
ax.set_title('Gender Distribution of Patients')
save_figure(fig, "gender_distribution")

# Geographic Distribution (create a faux region column)
sample_data['region'] = np.random.choice(
    ['Northeast', 'Southeast', 'Midwest', 'West', 'Southwest'], 
    size=len(sample_data))
fig, ax = plt.subplots(figsize=(10, 6))
region_counts = sample_data.drop_duplicates('patient_id')['region'].value_counts()
sns.barplot(x=region_counts.index, y=region_counts.values, palette='viridis', ax=ax)
ax.set_title('Geographic Distribution of Patients')
ax.set_xlabel('Region')
ax.set_ylabel('Number of Patients')
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
save_figure(fig, "geographic_distribution")

# CLINICAL METRICS VISUALIZATIONS
print("Generating clinical metric visualizations...")

# Blood Glucose Levels
fig, ax = plt.subplots(figsize=(10, 6))
# Group patients by diabetes status
glucose_by_diabetes = sample_data.groupby('has_diabetes')['blood_glucose'].mean().reset_index()
# Create the visualization
sns.boxplot(x='has_diabetes', y='blood_glucose', data=sample_data, 
           palette=[colors[7], colors[2]], ax=ax)
ax.set_title('Blood Glucose Levels by Diabetes Status')
ax.set_xlabel('Has Diabetes')
ax.set_ylabel('Blood Glucose (mg/dL)')
ax.set_xticklabels(['No', 'Yes'])
ax.axhline(y=125, color='red', linestyle='--', alpha=0.7, label='High Risk Threshold')
ax.legend()
save_figure(fig, "clinical_metric1")

# Blood Pressure Readings
fig, ax = plt.subplots(figsize=(10, 6))
bp_data = sample_data.groupby('has_hypertension').agg({
    'systolic_bp': 'mean',
    'diastolic_bp': 'mean'
}).reset_index()

x = np.arange(2)
width = 0.35
ax.bar(x - width/2, bp_data['systolic_bp'], width, label='Systolic', color=colors[3])
ax.bar(x + width/2, bp_data['diastolic_bp'], width, label='Diastolic', color=colors[6])
ax.set_title('Average Blood Pressure by Hypertension Status')
ax.set_xticks(x)
ax.set_xticklabels(['Non-Hypertensive', 'Hypertensive'])
ax.set_ylabel('Blood Pressure (mmHg)')
ax.legend()
ax.grid(True, alpha=0.3)
save_figure(fig, "clinical_metric2")

# Clinical Correlation Heatmap
fig, ax = plt.subplots(figsize=(12, 10))
# Select relevant clinical metrics
clinical_cols = ['blood_glucose', 'systolic_bp', 'diastolic_bp', 
                'heart_rate', 'physical_activity', 'sleep_quality']
correlation = sample_data[clinical_cols].corr()
mask = np.triu(np.ones_like(correlation, dtype=bool))
sns.heatmap(correlation, mask=mask, annot=True, fmt='.2f', cmap='viridis',
           linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax)
ax.set_title('Correlation Between Clinical Metrics')
save_figure(fig, "clinical_correlation")

# TEMPORAL PATTERNS VISUALIZATIONS
print("Generating temporal pattern visualizations...")

# Metric Changes Over Time
fig, ax = plt.subplots(figsize=(12, 7))
# Convert date to datetime and calculate the week number
sample_data['date'] = pd.to_datetime(sample_data['date'])
sample_data['week'] = (sample_data['date'].dt.to_period('W').astype(str)
                      .apply(lambda x: x.split('/')[-1]))

# Calculate weekly averages for some patients
patient_subset = sample_data[sample_data['patient_id'].isin(sample_data['patient_id'].unique()[:5])]
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
save_figure(fig, "temporal_pattern")

# Trend Analysis
fig, ax = plt.subplots(figsize=(10, 6))
# Group by date and calculate average glucose
date_groups = sample_data.groupby('date')
trend_data = date_groups.agg({
    'blood_glucose': 'mean',
    'systolic_bp': 'mean',
    'physical_activity': 'mean'
}).reset_index()
trend_data = trend_data.sort_values('date')

# Normalize the data for comparison
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

trend_data['glucose_norm'] = normalize(trend_data['blood_glucose'])
trend_data['bp_norm'] = normalize(trend_data['systolic_bp'])
trend_data['activity_norm'] = normalize(trend_data['physical_activity'])

# Plot the trends
ax.plot(trend_data['date'], trend_data['glucose_norm'], label='Blood Glucose', color=colors[0])
ax.plot(trend_data['date'], trend_data['bp_norm'], label='Systolic BP', color=colors[2])
ax.plot(trend_data['date'], trend_data['activity_norm'], label='Physical Activity', color=colors[4])

ax.set_title('Normalized Trend Analysis of Key Health Metrics')
ax.set_xlabel('Date')
ax.set_ylabel('Normalized Value')
ax.legend()
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
save_figure(fig, "trend_analysis")

# Seasonal Patterns (using month as a proxy for seasonality)
fig, ax = plt.subplots(figsize=(10, 6))
sample_data['month'] = sample_data['date'].dt.month
monthly_stats = sample_data.groupby('month').agg({
    'physical_activity': 'mean',
    'blood_glucose': 'mean'
}).reset_index()

ax2 = ax.twinx()
line1 = ax.plot(monthly_stats['month'], monthly_stats['physical_activity'], 
               color=colors[5], marker='o', linewidth=2, label='Physical Activity')
line2 = ax2.plot(monthly_stats['month'], monthly_stats['blood_glucose'], 
                color=colors[2], marker='s', linewidth=2, label='Blood Glucose')

ax.set_xlabel('Month')
ax.set_ylabel('Average Physical Activity (minutes)')
ax2.set_ylabel('Average Blood Glucose (mg/dL)')
ax.set_title('Seasonal Patterns in Activity and Blood Glucose')

# Fix the month labels
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ax.set_xticks(monthly_stats['month'])
# Only use labels for months that exist in the data
month_labels = [months[i-1] for i in monthly_stats['month']]
ax.set_xticklabels(month_labels)

ax.grid(True, alpha=0.3)

# Combine legends
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc='upper left')
save_figure(fig, "seasonal_patterns")

# RISK PREDICTION AND MODEL EVALUATION
print("Generating risk prediction visualizations...")

# Create labels (deterioration within 90 days)
# Patients with increasing glucose trend are labeled as 1 (deterioration)
def prepare_prediction_data():
    # Get unique patients
    patients = sample_data['patient_id'].unique()
    
    # Prepare features and labels
    X = []
    y = []
    patient_features = {}
    
    for patient in patients:
        patient_data = sample_data[sample_data['patient_id'] == patient].sort_values('date')
        
        # Check if patient has enough data
        if len(patient_data) < 30:
            continue
            
        # Calculate deterioration based on glucose trend
        glucose_start = patient_data['blood_glucose'].iloc[:30].mean()
        glucose_end = patient_data['blood_glucose'].iloc[-30:].mean()
        glucose_change = (glucose_end - glucose_start) / glucose_start
        
        # Label as deterioration if glucose increased by more than 10%
        deterioration = 1 if glucose_change > 0.1 else 0
        
        # Extract features (averages from first 30 days)
        early_data = patient_data.iloc[:30]
        features = [
            early_data['blood_glucose'].mean(),
            early_data['systolic_bp'].mean(),
            early_data['diastolic_bp'].mean(),
            early_data['heart_rate'].mean(),
            early_data['medication_adherence'].mean(),
            early_data['physical_activity'].mean(),
            early_data['sleep_quality'].mean(),
            early_data['has_diabetes'].iloc[0],
            early_data['has_hypertension'].iloc[0],
            early_data['has_heart_disease'].iloc[0],
            early_data['age'].iloc[0]
        ]
        
        X.append(features)
        y.append(deterioration)
        patient_features[patient] = features
    
    feature_names = [
        'blood_glucose', 'systolic_bp', 'diastolic_bp', 'heart_rate',
        'medication_adherence', 'physical_activity', 'sleep_quality',
        'has_diabetes', 'has_hypertension', 'has_heart_disease', 'age'
    ]
    
    return np.array(X), np.array(y), patient_features, feature_names

# Prepare data for prediction model
X, y, patient_features, feature_names = prepare_prediction_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a simple model for demonstration
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Risk Gauge Visualization
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
# Choose a random test patient
test_idx = np.random.randint(0, len(y_test))
risk_score = y_pred_proba[test_idx] * 100

# Create the gauge chart
theta = np.linspace(0, 1.8 * np.pi, 100)
r = np.ones_like(theta)
ax.set_rlim(0, 1)
ax.set_theta_direction(-1)
ax.set_theta_offset(np.pi / 2.0)
ax.set_xticks([])
ax.set_yticks([])
ax.spines['polar'].set_visible(False)

# Add the gauge sectors
cmap = plt.cm.RdYlGn_r
norm = plt.Normalize(0, 100)
    
# Create colored sections
N = 100
for i in range(N):
    value = i
    color = cmap(norm(value))
    ax.bar(theta[i], r[i], width=0.01 * np.pi, bottom=0.7, 
          color=color, alpha=0.8, edgecolor='none', align='edge')

# Add the indicator needle
risk_theta = (1 - risk_score/100) * 1.8 * np.pi
ax.plot([risk_theta, risk_theta], [0, 0.7], 'k-', linewidth=3)

# Add text
ax.text(0.5, 0.5, f"{risk_score:.1f}%", transform=ax.transAxes,
       fontsize=42, ha='center', color='#333333')
ax.text(0.5, 0.35, 'Risk Score', transform=ax.transAxes,
       fontsize=20, ha='center', color='#333333')

# Add categories
ax.text(0.05 * np.pi, 0.82, 'Low Risk', fontsize=14, ha='left')
ax.text(0.9 * np.pi, 0.82, 'Moderate Risk', fontsize=14, ha='center')
ax.text(1.7 * np.pi, 0.82, 'High Risk', fontsize=14, ha='right')

# Save the gauge
save_figure(fig, "risk_gauge")

# Patient Risk Factors
fig, ax = plt.subplots(figsize=(10, 6))
# Get feature importance for the test patient
feature_importances = model.feature_importances_
# Scale feature values to show impact direction
test_patient_features = X_test[test_idx]
reference_values = X_train.mean(axis=0)
# Calculate the relative differences
rel_diff = (test_patient_features - reference_values) / reference_values
impact = rel_diff * feature_importances

# Sort by absolute impact
sorted_idx = np.argsort(np.abs(impact))[::-1][:7]  # Top 7 factors
colors = ['red' if val > 0 else 'green' for val in impact[sorted_idx]]

ax.barh(np.array(feature_names)[sorted_idx], impact[sorted_idx], color=colors)
ax.set_title('Top Risk Factors for Patient')
ax.set_xlabel('Risk Impact (red = higher risk, green = lower risk)')
ax.grid(True, alpha=0.3)
save_figure(fig, "patient_risk_factors")

# MODEL PERFORMANCE VISUALIZATIONS
print("Generating model performance visualizations...")

# ROC Curve
fig, ax = plt.subplots(figsize=(8, 8))
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

ax.plot(fpr, tpr, color=colors[0], lw=2,
       label=f'ROC curve (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=2)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic (ROC) Curve')
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
save_figure(fig, "roc_curve")

# Precision-Recall Curve
fig, ax = plt.subplots(figsize=(8, 8))
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = average_precision_score(y_test, y_pred_proba)

ax.plot(recall, precision, color=colors[2], lw=2,
       label=f'Precision-Recall curve (AUC = {pr_auc:.3f})')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve')
ax.legend(loc="lower left")
ax.grid(True, alpha=0.3)
save_figure(fig, "pr_curve")

# Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 8))
y_pred = (y_pred_proba > 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title('Confusion Matrix')
ax.set_xticklabels(['No Deterioration', 'Deterioration'])
ax.set_yticklabels(['No Deterioration', 'Deterioration'])
save_figure(fig, "confusion_matrix")

# Calibration Plot
fig, ax = plt.subplots(figsize=(8, 8))
prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
ax.plot(prob_pred, prob_true, marker='o', linewidth=2, color=colors[4])
ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Fraction of Positives')
ax.set_title('Calibration Plot')
ax.legend()
ax.grid(True, alpha=0.3)
save_figure(fig, "calibration_plot")

# MODEL EXPLAINABILITY VISUALIZATIONS
print("Generating model explainability visualizations...")

# Global Feature Importance
fig, ax = plt.subplots(figsize=(10, 6))
# Sort feature importances
sorted_idx = np.argsort(model.feature_importances_)
ax.barh(np.array(feature_names)[sorted_idx], model.feature_importances_[sorted_idx])
ax.set_title('Global Feature Importance')
ax.set_xlabel('Feature Importance')
ax.grid(True, alpha=0.3)
save_figure(fig, "global_feature_importance")

# Local Feature Importance for a specific patient
fig, ax = plt.subplots(figsize=(10, 6))

# For the example, we'll use the same test patient
test_features = X_test[test_idx]
# For simplicity, we'll use feature values * global importance as local importance
local_importance = test_features * model.feature_importances_
sorted_idx = np.argsort(np.abs(local_importance))

ax.barh(np.array(feature_names)[sorted_idx], local_importance[sorted_idx],
       color=[colors[i % len(colors)] for i in range(len(sorted_idx))])
ax.set_title('Patient-Specific Feature Importance')
ax.set_xlabel('Feature Impact')
ax.grid(True, alpha=0.3)
save_figure(fig, "local_feature_importance")

print("All visualizations have been created successfully!")
