"""
Chronic Care Risk Prediction Dashboard
=====================================

Streamlit dashboard for AI-driven risk prediction engine.
Provides cohort overview, individual patient analysis, and clinical insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

try:
    from dashboard.utils import (
        load_patient_data, generate_synthetic_patient_data, 
        calculate_risk_metrics, create_feature_explanation,
        generate_clinical_recommendations, format_clinical_value
    )
except ImportError:
    # Fallback if utils not available
    pass

# ---- Page Configuration ----
st.set_page_config(
    page_title="🏥 Chronic Care Risk Prediction Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Custom CSS ----
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    
    .high-risk {
        border-left-color: #dc3545;
        background-color: #fff5f5;
    }
    
    .medium-risk {
        border-left-color: #fd7e14;
        background-color: #fff8f0;
    }
    
    .low-risk {
        border-left-color: #198754;
        background-color: #f0fff4;
    }
    
    .patient-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin-bottom: 1rem;
    }
    
    .clinical-alert {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 4px;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    
    .sidebar-info {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_dashboard_data():
    """Load patient data and model for dashboard"""
    try:
        # Try to load processed data
        if os.path.exists("data/processed/chronic_care_data_processed.csv"):
            df = pd.read_csv("data/processed/chronic_care_data_processed.csv")
        else:
            st.warning("⚠️ No processed data found. Generating synthetic data...")
            df = generate_synthetic_patient_data(1000)
            os.makedirs("data/processed", exist_ok=True)
            df.to_csv("data/processed/chronic_care_data_processed.csv", index=False)
        
        # Try to load model
        model, scaler = None, None
        if os.path.exists("models/saved/risk_prediction_model.pkl"):
            model = joblib.load("models/saved/risk_prediction_model.pkl")
            scaler = joblib.load("models/saved/feature_scaler.pkl")
        else:
            st.warning("⚠️ No trained model found. Using simulated predictions...")
        
        # Generate or load predictions
        if model and scaler:
            feature_cols = [col for col in df.columns 
                          if col not in ['patient_id', 'deterioration_90d', 'risk_probability', 'risk_category']]
            X_scaled = scaler.transform(df[feature_cols])
            df['risk_probability'] = model.predict_proba(X_scaled)[:, 1]
        elif 'risk_probability' not in df.columns:
            # Generate synthetic risk scores
            np.random.seed(42)
            df['risk_probability'] = np.random.beta(2, 5, len(df))  # Skewed toward lower risk
        
        # Categorize risk levels
        df['risk_category'] = pd.cut(
            df['risk_probability'], 
            bins=[0, 0.3, 0.7, 1.0], 
            labels=['Low', 'Medium', 'High']
        )
        
        return df, model, scaler
        
    except Exception as e:
        st.error(f"❌ Error loading dashboard data: {str(e)}")
        return None, None, None

def render_cohort_overview(df: pd.DataFrame):
    """Render the cohort overview tab"""
    st.header("📊 Patient Cohort Overview")
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_patients = len(df)
        st.metric("👥 Total Patients", f"{total_patients:,}")
    
    with col2:
        high_risk_count = len(df[df['risk_category'] == 'High'])
        high_risk_pct = (high_risk_count / total_patients * 100) if total_patients > 0 else 0
        st.metric("🚨 High Risk", f"{high_risk_count}", delta=f"{high_risk_pct:.1f}%")
    
    with col3:
        avg_risk = df['risk_probability'].mean()
        st.metric("📈 Average Risk", f"{avg_risk:.3f}")
    
    with col4:
        urgent_cases = len(df[df['risk_probability'] > 0.8])
        st.metric("⚡ Urgent Cases", f"{urgent_cases}")
    
    # Risk Distribution and Categories
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Risk Score Distribution")
        
        fig_hist = px.histogram(
            df, x='risk_probability', nbins=25,
            title="Patient Risk Score Distribution",
            labels={'risk_probability': 'Risk Probability', 'count': 'Number of Patients'},
            color_discrete_sequence=['#3498db']
        )
        
        # Add threshold lines
        fig_hist.add_vline(x=0.3, line_dash="dash", line_color="orange", 
                          annotation_text="Medium Risk Threshold")
        fig_hist.add_vline(x=0.7, line_dash="dash", line_color="red", 
                          annotation_text="High Risk Threshold")
        
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.subheader("Risk Categories")
        
        risk_counts = df['risk_category'].value_counts()
        fig_pie = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Patients by Risk Level",
            color_discrete_map={
                'Low': '#2ecc71', 
                'Medium': '#f39c12', 
                'High': '#e74c3c'
            }
        )
        
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Patient List with Risk Sorting
    st.subheader("📋 Patient List")
    
    # Sorting options
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        sort_by = st.selectbox(
            "Sort by:", 
            ["Risk Probability ↓", "Age ↓", "Patient ID ↑"],
            index=0
        )
    
    with col2:
        show_only_high_risk = st.checkbox("High Risk Only", False)
    
    with col3:
        max_patients = st.number_input("Max Patients", 10, 500, 50)
    
    # Apply filters and sorting
    display_df = df.copy()
    if show_only_high_risk:
        display_df = display_df[display_df['risk_category'] == 'High']
    
    # Sort data
    if sort_by.startswith("Risk"):
        display_df = display_df.sort_values('risk_probability', ascending=False)
    elif sort_by.startswith("Age"):
        display_df = display_df.sort_values('age', ascending=False)
    else:
        display_df = display_df.sort_values('patient_id', ascending=True)
    
    # Prepare display columns
    display_columns = [
        'patient_id', 'age', 'risk_probability', 'risk_category',
        'diabetes_type2', 'heart_failure', 'obesity', 'hypertension'
    ]
    
    display_df = display_df[display_columns].head(max_patients).copy()
    display_df['risk_probability'] = display_df['risk_probability'].round(3)
    
    # Style the dataframe
    def style_risk_rows(row):
        if row['risk_category'] == 'High':
            return ['background-color: #ffebee'] * len(row)
        elif row['risk_category'] == 'Medium':
            return ['background-color: #fff8e1'] * len(row)
        else:
            return ['background-color: #e8f5e8'] * len(row)
    
    styled_df = display_df.style.apply(style_risk_rows, axis=1)
    st.dataframe(styled_df, use_container_width=True)

def render_patient_details(df: pd.DataFrame):
    """Render individual patient analysis tab"""
    st.header("👤 Individual Patient Analysis")
    
    # Patient Selection
    col1, col2 = st.columns([2, 1])
    with col1:
        patient_id = st.selectbox(
            "🔍 Select Patient:", 
            df['patient_id'].sort_values().tolist(),
            key="patient_selector"
        )
    
    with col2:
        if st.button("🎲 Random High Risk Patient"):
            high_risk_patients = df[df['risk_category'] == 'High']['patient_id'].tolist()
            if high_risk_patients:
                patient_id = np.random.choice(high_risk_patients)
                st.rerun()
    
    # Get patient data
    patient_data = df[df['patient_id'] == patient_id].iloc[0]
    
    # Patient Header Card
    risk_prob = patient_data['risk_probability']
    risk_cat = patient_data['risk_category']
    
    st.markdown(f"""
    <div class="patient-header">
        <h2>🩺 Patient {patient_id} - {risk_cat} Risk</h2>
        <p style="font-size: 1.2em; margin: 0;">
            Risk Score: <strong>{risk_prob:.3f}</strong> | 
            Age: <strong>{patient_data['age']}</strong> | 
            BMI: <strong>{patient_data['bmi']:.1f}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Risk Level Display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if risk_cat == 'High':
            st.markdown(f'''
            <div class="metric-card high-risk">
                <h3>🚨 HIGH RISK</h3>
                <h2>{risk_prob:.3f}</h2>
                <p>Immediate attention required</p>
            </div>''', unsafe_allow_html=True)
        elif risk_cat == 'Medium':
            st.markdown(f'''
            <div class="metric-card medium-risk">
                <h3>⚠️ MEDIUM RISK</h3>
                <h2>{risk_prob:.3f}</h2>
                <p>Enhanced monitoring recommended</p>
            </div>''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="metric-card low-risk">
                <h3>✅ LOW RISK</h3>
                <h2>{risk_prob:.3f}</h2>
                <p>Continue routine care</p>
            </div>''', unsafe_allow_html=True)
    
    with col2:
        st.metric("📊 Systolic BP", f"{patient_data['systolic_bp']:.0f} mmHg")
        st.metric("🩸 HbA1c", f"{patient_data['hba1c']:.1f}%")
    
    with col3:
        st.metric("💊 Medication Adherence", f"{patient_data['medication_adherence']:.0%}")
        st.metric("🏃 Exercise Frequency", f"{patient_data['exercise_frequency']:.0f}/week")
    
    # Clinical Conditions
    st.subheader("🏥 Chronic Conditions")
    conditions = ['diabetes_type2', 'heart_failure', 'obesity', 'hypertension', 'copd']
    active_conditions = []
    
    cols = st.columns(len(conditions))
    for i, condition in enumerate(conditions):
        with cols[i]:
            if patient_data.get(condition, 0) == 1:
                st.success(f"✓ {condition.replace('_', ' ').title()}")
                active_conditions.append(condition)
            else:
                st.info(f"○ {condition.replace('_', ' ').title()}")
    
    # Feature Importance for this patient
    st.subheader("🎯 Key Risk Factors")
    
    # Get all numeric features for analysis
    feature_cols = [col for col in df.columns 
                   if col not in ['patient_id', 'deterioration_90d', 'risk_probability', 'risk_category']
                   and pd.api.types.is_numeric_dtype(df[col])]
    
    # Calculate feature deviations from population mean
    feature_analysis = []
    population_stats = df[feature_cols].describe()
    
    for feature in feature_cols[:15]:  # Top 15 features
        patient_val = patient_data[feature]
        pop_mean = population_stats.loc['mean', feature]
        pop_std = population_stats.loc['std', feature]
        
        # Calculate z-score
        z_score = (patient_val - pop_mean) / pop_std if pop_std > 0 else 0
        
        feature_analysis.append({
            'feature': feature,
            'patient_value': patient_val,
            'population_mean': pop_mean,
            'z_score': z_score,
            'deviation': abs(z_score)
        })
    
    # Sort by deviation magnitude
    feature_analysis = sorted(feature_analysis, key=lambda x: x['deviation'], reverse=True)
    
    # Create feature impact visualization
    top_features = feature_analysis[:10]
    feature_names = [f['feature'].replace('_', ' ').title() for f in top_features]
    z_scores = [f['z_score'] for f in top_features]
    
    fig_features = px.bar(
        x=z_scores,
        y=feature_names,
        orientation='h',
        title="Patient vs Population Comparison (Z-Scores)",
        labels={'x': 'Standard Deviations from Mean', 'y': 'Clinical Features'},
        color=z_scores,
        color_continuous_scale='RdBu_r'
    )
    
    fig_features.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig_features, use_container_width=True)
    
    # Clinical Recommendations
    st.subheader("💡 Clinical Recommendations")
    
    recommendations = generate_patient_recommendations(patient_data, feature_analysis[:5])
    
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"""
        <div class="clinical-alert">
            <strong>{i}. {rec['title']}</strong><br>
            {rec['description']}<br>
            <em>Priority: {rec['priority']} | Timeline: {rec['timeline']}</em>
        </div>
        """, unsafe_allow_html=True)

def render_analytics_dashboard(df: pd.DataFrame):
    """Render population analytics tab"""
    st.header("📈 Population Analytics & Insights")
    
    # Population Overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Risk by Demographics")
        
        # Age group analysis
        df['age_group'] = pd.cut(
            df['age'], 
            bins=[0, 50, 65, 80, 120], 
            labels=['<50', '50-65', '65-80', '80+']
        )
        
        age_risk = df.groupby('age_group')['risk_probability'].agg(['mean', 'std', 'count']).round(3)
        
        fig_age = px.bar(
            x=age_risk.index, 
            y=age_risk['mean'],
            error_y=age_risk['std'],
            title="Average Risk by Age Group",
            labels={'x': 'Age Group', 'y': 'Average Risk Probability'},
            color=age_risk['mean'],
            color_continuous_scale='Reds'
        )
        
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        st.subheader("🏥 Risk by Chronic Conditions")
        
        conditions = ['diabetes_type2', 'heart_failure', 'obesity', 'hypertension']
        condition_risk = []
        
        for condition in conditions:
            condition_subset = df[df[condition] == 1]
            if len(condition_subset) > 0:
                avg_risk = condition_subset['risk_probability'].mean()
                count = len(condition_subset)
                condition_risk.append({
                    'condition': condition.replace('_', ' ').title(), 
                    'avg_risk': avg_risk,
                    'count': count
                })
        
        if condition_risk:
            condition_df = pd.DataFrame(condition_risk)
            
            fig_conditions = px.bar(
                condition_df, 
                x='condition', 
                y='avg_risk',
                title="Average Risk by Chronic Condition",
                labels={'avg_risk': 'Average Risk', 'condition': 'Condition'},
                color='avg_risk',
                color_continuous_scale='Oranges'
            )
            
            st.plotly_chart(fig_conditions, use_container_width=True)
    
    # Correlation Analysis
    st.subheader("🔗 Risk Factor Correlations")
    
    # Select numeric columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['patient_id']]
    
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            title="Feature Correlation Heatmap",
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        
        fig_corr.update_layout(height=600)
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Time Series Simulation
    st.subheader("📈 Risk Trend Simulation")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        selected_patient = st.selectbox(
            "Patient for Trend Analysis:",
            df['patient_id'].tolist()[:20],  # Limit for performance
            key="trend_patient"
        )
        
        trend_days = st.slider("Days to Show", 7, 90, 30)
    
    with col1:
        # Generate simulated time series
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=trend_days), 
            end=datetime.now(), 
            freq='D'
        )
        
        base_risk = df[df['patient_id'] == selected_patient]['risk_probability'].iloc[0]
        
        # Add realistic trend with noise
        np.random.seed(selected_patient)  # Consistent for same patient
        trend = np.linspace(base_risk * 0.9, base_risk * 1.1, len(dates))
        noise = np.random.normal(0, 0.02, len(dates))
        daily_risk = np.clip(trend + noise, 0, 1)
        
        # Create trend dataframe
        trend_df = pd.DataFrame({
            'date': dates,
            'risk': daily_risk,
            'patient_id': selected_patient
        })
        
        fig_trend = px.line(
            trend_df, 
            x='date', 
            y='risk',
            title=f"Risk Trend - Patient {selected_patient} ({trend_days} days)",
            labels={'risk': 'Risk Probability', 'date': 'Date'}
        )
        
        # Add threshold lines
        fig_trend.add_hline(y=0.7, line_dash="dash", line_color="red", 
                           annotation_text="High Risk")
        fig_trend.add_hline(y=0.3, line_dash="dash", line_color="orange", 
                           annotation_text="Medium Risk")
        
        fig_trend.update_layout(height=400)
        st.plotly_chart(fig_trend, use_container_width=True)

def render_sidebar():
    """Render sidebar with filters and controls"""
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-info">
            <h3>🏥 Dashboard Info</h3>
            <p>AI-driven chronic care risk prediction system</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.header("🎛️ Filters")
        
        # Load data for filter options
        df, _, _ = load_dashboard_data()
        if df is None:
            st.error("Cannot load data for filters")
            return {}
        
        # Risk level filter
        risk_levels = st.multiselect(
            "📊 Risk Levels",
            options=['Low', 'Medium', 'High'],
            default=['Medium', 'High'],
            help="Select risk levels to display in analysis"
        )
        
        # Age filter
        age_range = st.slider(
            "👴 Age Range",
            min_value=int(df['age'].min()),
            max_value=int(df['age'].max()),
            value=(40, 85),
            help="Filter patients by age range"
        )
        
        # Chronic conditions filter
        available_conditions = ['diabetes_type2', 'heart_failure', 'obesity', 'hypertension']
        conditions = st.multiselect(
            "🏥 Chronic Conditions",
            options=available_conditions,
            default=['diabetes_type2', 'heart_failure'],
            help="Show patients with selected conditions"
        )
        
        # Gender filter (if available)
        if 'gender' in df.columns:
            gender_filter = st.selectbox(
                "👤 Gender",
                options=['All', 'Male', 'Female'],
                index=0
            )
        else:
            gender_filter = 'All'
        
        st.markdown("---")
        
        # Dashboard controls
        st.header("⚙️ Controls")
        
        auto_refresh = st.checkbox("🔄 Auto Refresh", False)
        if auto_refresh:
            refresh_interval = st.slider("Refresh (seconds)", 10, 300, 60)
        
        export_data = st.button("💾 Export Data")
        if export_data:
            st.success("Export feature coming soon!")
        
        return {
            'risk_levels': risk_levels,
            'age_range': age_range,
            'conditions': conditions,
            'gender_filter': gender_filter,
            'auto_refresh': auto_refresh
        }

def generate_patient_recommendations(patient_data, top_risk_factors):
    """Generate clinical recommendations based on patient data"""
    recommendations = []
    
    # High glucose/HbA1c
    if patient_data.get('hba1c', 7) > 8.5:
        recommendations.append({
            'title': '🩸 Diabetes Management',
            'description': 'HbA1c >8.5% - Consider intensifying diabetes therapy, endocrine referral',
            'priority': 'High',
            'timeline': '1-2 weeks'
        })
    
    # High blood pressure
    if patient_data.get('systolic_bp', 120) > 160:
        recommendations.append({
            'title': '🩺 Hypertension Control',
            'description': 'Systolic BP >160 - Review antihypertensive therapy, 24-hour monitoring',
            'priority': 'High',
            'timeline': '1 week'
        })
    
    # Poor medication adherence
    if patient_data.get('medication_adherence', 1.0) < 0.7:
        recommendations.append({
            'title': '💊 Medication Adherence',
            'description': 'Poor adherence <70% - Pharmacy consultation, pill organizer, education',
            'priority': 'Medium',
            'timeline': '2-4 weeks'
        })
    
    # High BMI
    if patient_data.get('bmi', 25) > 35:
        recommendations.append({
            'title': '⚖️ Weight Management',
            'description': 'BMI >35 - Nutritional counseling, structured weight loss program',
            'priority': 'Medium',
            'timeline': '2-6 weeks'
        })
    
    # Default recommendation
    if not recommendations:
        recommendations.append({
            'title': '📋 Routine Monitoring',
            'description': 'Continue current care plan with regular monitoring',
            'priority': 'Low',
            'timeline': '3-6 months'
        })
    
    return recommendations[:4]  # Limit to top 4

def generate_synthetic_patient_data(n_patients=1000):
    """Generate synthetic patient data for demo purposes"""
    np.random.seed(42)
    
    data = {
        'patient_id': range(1, n_patients + 1),
        'age': np.random.normal(65, 15, n_patients).astype(int),
        'gender': np.random.choice([0, 1], n_patients, p=[0.45, 0.55]),
        'bmi': np.random.normal(28, 5, n_patients),
        'systolic_bp': np.random.normal(140, 20, n_patients),
        'diastolic_bp': np.random.normal(85, 15, n_patients),
        'heart_rate': np.random.normal(75, 10, n_patients),
        'glucose_level': np.random.normal(150, 40, n_patients),
        'hba1c': np.random.normal(7.5, 1.5, n_patients),
        'medication_adherence': np.random.uniform(0.3, 1.0, n_patients),
        'exercise_frequency': np.random.poisson(2, n_patients),
        'smoking_status': np.random.choice([0, 1], n_patients, p=[0.8, 0.2]),
        'diabetes_type2': np.random.choice([0, 1], n_patients, p=[0.6, 0.4]),
        'heart_failure': np.random.choice([0, 1], n_patients, p=[0.8, 0.2]),
        'obesity': np.random.choice([0, 1], n_patients, p=[0.7, 0.3]),
        'hypertension': np.random.choice([0, 1], n_patients, p=[0.5, 0.5]),
        'copd': np.random.choice([0, 1], n_patients, p=[0.9, 0.1]),
    }
    
    # Ensure realistic bounds
    data['age'] = np.clip(data['age'], 18, 95)
    data['bmi'] = np.clip(data['bmi'], 15, 50)
    data['systolic_bp'] = np.clip(data['systolic_bp'], 90, 200)
    data['hba1c'] = np.clip(data['hba1c'], 4.0, 15.0)
    
    # Create synthetic deterioration outcome
    risk_factors = (
        (data['age'] > 70) * 0.2 +
        (data['bmi'] > 30) * 0.15 +
        (data['systolic_bp'] > 160) * 0.2 +
        (data['hba1c'] > 9) * 0.25 +
        (data['medication_adherence'] < 0.7) * 0.2
    )
    
    data['deterioration_90d'] = (risk_factors + np.random.normal(0, 0.1, n_patients) > 0.4).astype(int)
    
    return pd.DataFrame(data)

def main():
    """Main dashboard application"""
    st.markdown("# 🏥 Chronic Care Risk Prediction Dashboard")
    st.markdown("*AI-driven deterioration risk assessment for chronic care patients*")
    
    # Load data
    with st.spinner("🔄 Loading patient data..."):
        df, model, scaler = load_dashboard_data()
    
    if df is None:
        st.error("❌ Failed to load dashboard data. Please check your data files.")
        st.stop()
    
    # Render sidebar filters
    filters = render_sidebar()
    
    # Apply filters to data
    filtered_df = df.copy()
    
    if filters.get('risk_levels'):
        filtered_df = filtered_df[filtered_df['risk_category'].isin(filters['risk_levels'])]
    
    if filters.get('age_range'):
        age_min, age_max = filters['age_range']
        filtered_df = filtered_df[filtered_df['age'].between(age_min, age_max)]
    
    if filters.get('conditions'):
        condition_filter = filtered_df[filters['conditions']].sum(axis=1) > 0
        filtered_df = filtered_df[condition_filter]
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Cohort Overview", 
        "👤 Patient Details", 
        "📈 Analytics",
        "ℹ️ Model Info"
    ])
    
    with tab1:
        render_cohort_overview(filtered_df)
    
    with tab2:
        render_patient_details(filtered_df)
    
    with tab3:
        render_analytics_dashboard(filtered_df)
    
    with tab4:
        st.header("🤖 Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Model Details")
            if model:
                st.success("✅ Trained model loaded successfully")
                st.info(f"Model Type: {type(model).__name__}")
                if hasattr(model, 'n_estimators'):
                    st.info(f"Estimators: {model.n_estimators}")
            else:
                st.warning("⚠️ Using synthetic predictions for demo")
            
            st.subheader("📊 Dataset Info")
            st.info(f"Total Patients: {len(df):,}")
            st.info(f"Features: {len([c for c in df.columns if c not in ['patient_id', 'deterioration_90d', 'risk_probability', 'risk_category']])}")
            
        with col2:
            st.subheader("🎯 Performance Metrics")
            if os.path.exists("outputs/reports"):
                st.info("📁 Check outputs/reports/ for detailed metrics")
            else:
                st.warning("No performance reports found")
            
            st.subheader("⚙️ System Status")
            st.success("✅ Dashboard Online")
            st.success("✅ Data Pipeline Active")
            if model:
                st.success("✅ Model Loaded")
            else:
                st.warning("⚠️ Demo Mode Active")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        🏥 <strong>Chronic Care Risk Prediction Engine v1.0</strong><br>
        Built with CrewAI • Streamlit • Machine Learning<br>
        <em>For clinical decision support • Not for diagnostic use</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
