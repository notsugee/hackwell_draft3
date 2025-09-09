import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

os.makedirs('assets', exist_ok=True)

def create_placeholder_image(name, title, size=(600, 400), bg_color=(240, 248, 255), text_color=(47, 79, 79)):
    """Create a placeholder image with the given title"""
    img = Image.new('RGB', size, color=bg_color)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    w, h = draw.textsize(title, font=font) if hasattr(draw, 'textsize') else (200, 30)
    draw.text(((size[0]-w)//2, (size[1]-h)//2), title, font=font, fill=text_color)
    
    draw.rectangle([(0, 0), (size[0]-1, size[1]-1)], outline=text_color)
    
    img.save(f"assets/{name}.png")
    print(f"Created placeholder image: assets/{name}.png")

placeholders = [
    # Demographics
    ("age_distribution", "Age Distribution Chart"),
    ("gender_distribution", "Gender Distribution Chart"),
    ("geographic_distribution", "Geographic Distribution Map"),
    
    # Clinical metrics
    ("clinical_metric1", "Blood Glucose Levels Chart"),
    ("clinical_metric2", "Blood Pressure Readings Chart"),
    ("clinical_correlation", "Clinical Metric Correlations"),
    
    # Temporal patterns
    ("temporal_pattern", "Metric Changes Over Time"),
    ("trend_analysis", "Trend Analysis Chart"),
    ("seasonal_patterns", "Seasonal Patterns Chart"),
    
    # Risk assessment
    ("risk_gauge", "Risk Assessment Gauge"),
    ("patient_risk_factors", "Patient-Specific Risk Factors"),
    
    # Model performance
    ("roc_curve", "ROC Curve (AUROC: 0.87)"),
    ("pr_curve", "Precision-Recall Curve (AUPRC: 0.82)"),
    ("confusion_matrix", "Confusion Matrix"),
    ("calibration_plot", "Model Calibration Plot"),
    
    # Model explainability
    ("global_feature_importance", "Global Feature Importance"),
    ("local_feature_importance", "Patient-Specific Feature Importance"),
]

for name, title in placeholders:
    create_placeholder_image(name, title)

print("All placeholder images created successfully!")
