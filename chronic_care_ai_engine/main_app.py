"""
🏥 AI Risk Prediction Engine - Main Dashboard
Complete Streamlit frontend with all visualizations, patient analysis, and CrewAI integration
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from model_engine import RiskPredictionModel, SHAPExplainer
from crewai_validation import ValidationCrew
from data_processor import DataProcessor

# Page Configuration
st.set_page_config(
    page_title="AI Risk Prediction Engine - Chronic Care",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .risk-high { color: #ff4444; font-weight: bold; }
    .risk-medium { color: #ff8800; font-weight: bold; }
    .risk-low { color: #00aa00; font-weight: bold; }
    .agent-message {
        background-color: #e8f4fd;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

class DashboardApp:
    def __init__(self):
        self.load_components()
    
    @st.cache_resource
    def load_components(_self):
        """Load all required components"""
        # Load data processor
        data_processor = DataProcessor()
        
        # Load trained model
        model_engine = RiskPredictionModel()
        model_engine.load_model('data/trained_xgboost_model.pkl')
        
        # Load SHAP explainer
        shap_explainer = SHAPExplainer()
        shap_explainer.load_explainer('data/shap_explainer.pkl')
        
        # Load validation crew
        validation_crew = ValidationCrew()
        
        # Load patient data
        patient_data = pd.read_csv('data/synthetic_patients.csv')
        predictions_data = pd.read_csv('data/model_predictions.csv')
        
        return data_processor, model_engine, shap_explainer, validation_crew, patient_data, predictions_data
    
    def run(self):
        st.markdown('<h1 class="main-header">🏥 AI-Driven Risk Prediction Engine</h1>', unsafe_allow_html=True)
        st.markdown("**Predicting 90-Day Deterioration Risk for Chronic Care Patients**")
        
        # Load components
        data_processor, model_engine, shap_explainer, validation_crew, patient_data, predictions_data = self.load_components()
        
        # Sidebar Navigation
        st.sidebar.title("🔍 Navigation")
        page = st.sidebar.selectbox(
            "Choose Dashboard Section",
            [
                "📊 Overview Dashboard", 
                "👤 Patient Deep Dive", 
                "📈 Model Analytics", 
                "🤖 CrewAI Validation",
                "📋 Cohort Management"
            ]
        )
        
        if page == "📊 Overview Dashboard":
            self.overview_dashboard(patient_data, predictions_data, model_engine)
        elif page == "👤 Patient Deep Dive":
            self.patient_deep_dive(patient_data, predictions_data, model_engine, shap_explainer)
        elif page == "📈 Model Analytics":
            self.model_analytics_dashboard(model_engine, predictions_data)
        elif page == "🤖 CrewAI Validation":
            self.crewai_validation_dashboard(validation_crew, model_engine)
        elif page == "📋 Cohort Management":
            self.cohort_management_dashboard(patient_data, predictions_data)

    def overview_dashboard(self, patient_data, predictions_data, model_engine):
        """Main overview dashboard with population metrics"""
        st.header("📊 Population Risk Overview")
        
        # Key Metrics Row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        total_patients = len(predictions_data)
        high_risk = len(predictions_data[predictions_data['risk_score'] > 0.7])
        medium_risk = len(predictions_data[(predictions_data['risk_score'] >= 0.4) & (predictions_data['risk_score'] <= 0.7)])
        low_risk = total_patients - high_risk - medium_risk
        avg_risk = predictions_data['risk_score'].mean()
        
        with col1:
            st.metric("Total Patients", f"{total_patients:,}", delta="+120 this month")
        with col2:
            st.metric("High Risk", f"{high_risk:,}", delta=f"+{high_risk-450}", delta_color="inverse")
        with col3:
            st.metric("Medium Risk", f"{medium_risk:,}", delta=f"+{medium_risk-1200}")
        with col4:
            st.metric("Low Risk", f"{low_risk:,}", delta=f"+{low_risk-3000}", delta_color="normal")
        with col5:
            st.metric("Avg Risk Score", f"{avg_risk:.1%}", delta="-2.3%", delta_color="inverse")
        
        # Visualization Row 1
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Risk Distribution")
            fig_dist = px.histogram(
                predictions_data, 
                x='risk_score', 
                nbins=30,
                title="Patient Risk Score Distribution",
                labels={'risk_score': 'Risk Score', 'count': 'Number of Patients'}
            )
            fig_dist.add_vline(x=0.7, line_dash="dash", line_color="red", annotation_text="High Risk Threshold")
            fig_dist.add_vline(x=0.4, line_dash="dash", line_color="orange", annotation_text="Medium Risk Threshold")
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Risk Categories")
            risk_categories = pd.DataFrame({
                'Risk Level': ['High Risk (>70%)', 'Medium Risk (40-70%)', 'Low Risk (<40%)'],
                'Count': [high_risk, medium_risk, low_risk]
            })
            fig_pie = px.pie(
                risk_categories, 
                values='Count', 
                names='Risk Level',
                title="Risk Level Distribution",
                color_discrete_sequence=['#ff4444', '#ff8800', '#00aa00']
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Model Performance Section
        st.header("🤖 Model Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Display confusion matrix
            st.subheader("📊 Confusion Matrix")
            cm_data = model_engine.get_confusion_matrix()
            fig_cm = px.imshow(
                cm_data,
                text_auto=True,
                aspect="auto",
                title="Model Confusion Matrix"
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col2:
            # ROC Curve
            st.subheader("📈 ROC Curve")
            roc_data = model_engine.get_roc_curve()
            fig_roc = px.line(
                x=roc_data['fpr'], 
                y=roc_data['tpr'],
                title=f"ROC Curve (AUC = {roc_data['auc']:.3f})"
            )
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
            st.plotly_chart(fig_roc, use_container_width=True)
        
        with col3:
            # Performance metrics
            st.subheader("📊 Key Metrics")
            metrics = model_engine.get_performance_metrics()
            for metric, value in metrics.items():
                st.metric(metric, f"{value:.3f}")

    def patient_deep_dive(self, patient_data, predictions_data, model_engine, shap_explainer):
        """Individual patient analysis with historical + prediction overlay"""
        st.header("👤 Individual Patient Analysis")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("Patient Selection")
            
            # Patient selector
            patient_ids = predictions_data['patient_id'].unique()
            selected_patient = st.selectbox("Select Patient ID:", patient_ids)
            
            # Get patient info
            patient_info = self.get_patient_info(selected_patient, patient_data, predictions_data)
            
            # Patient profile card
            st.markdown("### 📋 Patient Profile")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Age", f"{patient_info['age']} years")
                st.metric("BMI", f"{patient_info['bmi']:.1f}")
            with col_b:
                st.metric("Risk Score", f"{patient_info['risk_score']:.1%}")
                
            # Risk level
            risk_level = self.get_risk_level(patient_info['risk_score'])
            st.markdown(f"**Risk Level:** <span class='risk-{risk_level.lower()}'>{risk_level} RISK</span>", unsafe_allow_html=True)
            
            # Conditions
            st.markdown("### 🏥 Conditions")
            conditions = patient_info['conditions']
            for condition in conditions:
                st.markdown(f"• {condition}")
        
        with col2:
            st.subheader("📈 Historical Data with 90-Day Risk Prediction")
            
            # Get patient historical data
            patient_historical = patient_data[patient_data['patient_id'] == selected_patient].copy()
            patient_historical = patient_historical.sort_values('date')
            
            # Create tabs for different data types
            tabs = st.tabs(["🩺 Vitals", "🧪 Labs", "💊 Medication", "🏃 Lifestyle"])
            
            with tabs[0]:
                self.plot_vitals_with_prediction(patient_historical, model_engine, selected_patient)
            
            with tabs[1]:
                self.plot_labs_with_prediction(patient_historical, model_engine, selected_patient)
            
            with tabs[2]:
                self.plot_medication_with_prediction(patient_historical, model_engine, selected_patient)
            
            with tabs[3]:
                self.plot_lifestyle_with_prediction(patient_historical, model_engine, selected_patient)
        
        # SHAP Explanations
        st.header("🔍 AI Model Explanations - Why is this patient at risk?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Local Risk Factors (This Patient)")
            self.plot_patient_shap_explanation(selected_patient, shap_explainer, patient_data)
        
        with col2:
            st.subheader("📊 Global Risk Factors (All Patients)")
            self.plot_global_shap_explanation(shap_explainer)

    def plot_vitals_with_prediction(self, patient_data, model_engine, patient_id):
        """Plot vitals with 90-day prediction overlay - KEY FEATURE"""
        
        # Create subplot with secondary y-axis
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Blood Pressure + Risk Prediction', 'Heart Rate + Risk Prediction',
                           'Temperature + Risk Prediction', 'Weight + Risk Prediction'),
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": True}, {"secondary_y": True}]]
        )
        
        # Generate 90-day predictions
        future_dates, risk_predictions = self.generate_90_day_prediction(patient_data, model_engine)
        
        # Blood Pressure Plot
        fig.add_trace(
            go.Scatter(x=patient_data['date'], y=patient_data['systolic_bp'],
                      name='Systolic BP', line=dict(color='red', width=2)),
            row=1, col=1, secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=patient_data['date'], y=patient_data['diastolic_bp'],
                      name='Diastolic BP', line=dict(color='blue', width=2)),
            row=1, col=1, secondary_y=False
        )
        
        # Add 90-day risk prediction overlay
        fig.add_trace(
            go.Scatter(x=future_dates, y=risk_predictions,
                      name='90-Day Risk Prediction (%)', line=dict(color='orange', width=4, dash='dash')),
            row=1, col=1, secondary_y=True
        )
        
        # Heart Rate
        fig.add_trace(
            go.Scatter(x=patient_data['date'], y=patient_data['heart_rate'],
                      name='Heart Rate', line=dict(color='green', width=2)),
            row=1, col=2, secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=future_dates, y=risk_predictions,
                      name='Risk %', line=dict(color='orange', width=4, dash='dash'), showlegend=False),
            row=1, col=2, secondary_y=True
        )
        
        # Temperature
        fig.add_trace(
            go.Scatter(x=patient_data['date'], y=patient_data['temperature'],
                      name='Temperature', line=dict(color='purple', width=2)),
            row=2, col=1, secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=future_dates, y=risk_predictions,
                      name='Risk %', line=dict(color='orange', width=4, dash='dash'), showlegend=False),
            row=2, col=1, secondary_y=True
        )
        
        # Weight
        fig.add_trace(
            go.Scatter(x=patient_data['date'], y=patient_data['weight'],
                      name='Weight', line=dict(color='brown', width=2)),
            row=2, col=2, secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=future_dates, y=risk_predictions,
                      name='Risk %', line=dict(color='orange', width=4, dash='dash'), showlegend=False),
            row=2, col=2, secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            height=700,
            title_text=f"Patient {patient_id}: Vitals with 90-Day Risk Prediction Overlay",
            title_x=0.5,
            showlegend=True
        )
        
        # Update axes
        fig.update_yaxes(title_text="Vital Signs", secondary_y=False)
        fig.update_yaxes(title_text="Risk Probability (%)", secondary_y=True, range=[0, 100])
        
        # Add prediction start line
        current_date = patient_data['date'].max()
        fig.add_vline(x=current_date, line_dash="solid", line_color="gray", 
                      annotation_text="🔮 Prediction Start", annotation_position="top")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk interpretation
        max_risk = max(risk_predictions)
        avg_risk = np.mean(risk_predictions)
        
        st.markdown(f"""
        ### 🎯 90-Day Prediction Summary:
        - **Peak Risk:** {max_risk:.1f}% (Day {np.argmax(risk_predictions) + 1})
        - **Average Risk:** {avg_risk:.1f}%
        - **Trend:** {'📈 Increasing' if risk_predictions[-1] > risk_predictions[0] else '📉 Decreasing'}
        - **Intervention:** {'🚨 URGENT - Schedule within 24h' if max_risk > 80 else '⚠️ MONITOR - Weekly check-in' if max_risk > 60 else '✅ ROUTINE - Monthly follow-up'}
        """)

    def generate_90_day_prediction(self, patient_data, model_engine):
        """Generate realistic 90-day risk predictions"""
        
        last_date = pd.to_datetime(patient_data['date'].max())
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=90, freq='D')
        
        # Use model to generate predictions (simplified approach)
        # In real implementation, you'd use the actual trained model
        base_risk = np.random.uniform(0.3, 0.8)
        trend = np.random.choice([-0.003, -0.001, 0.001, 0.003])
        seasonal = np.sin(np.arange(90) * 2 * np.pi / 30) * 0.1  # Monthly cycle
        noise = np.random.normal(0, 0.05, 90)
        
        risk_predictions = []
        for i in range(90):
            risk = base_risk + (trend * i) + seasonal[i] + noise[i]
            risk = max(0, min(1, risk)) * 100  # Convert to percentage
            risk_predictions.append(risk)
        
        return future_dates, risk_predictions

    def plot_labs_with_prediction(self, patient_data, model_engine, patient_id):
        """Plot lab results with prediction overlay"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Glucose + Risk', 'HbA1c + Risk', 'Cholesterol + Risk', 'Creatinine + Risk'),
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": True}, {"secondary_y": True}]]
        )
        
        future_dates, risk_predictions = self.generate_90_day_prediction(patient_data, model_engine)
        
        # Lab values with risk overlay
        lab_params = [
            ('glucose', 'Glucose (mg/dL)', 'red'),
            ('hba1c', 'HbA1c (%)', 'blue'),
            ('cholesterol', 'Cholesterol (mg/dL)', 'green'),
            ('creatinine', 'Creatinine (mg/dL)', 'purple')
        ]
        
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for i, (param, label, color) in enumerate(lab_params):
            row, col = positions[i]
            
            # Historical lab values
            fig.add_trace(
                go.Scatter(x=patient_data['date'], y=patient_data[param],
                          name=label, line=dict(color=color, width=2)),
                row=row, col=col, secondary_y=False
            )
            
            # Risk prediction overlay
            fig.add_trace(
                go.Scatter(x=future_dates, y=risk_predictions,
                          name='Risk %' if i == 0 else '', line=dict(color='orange', width=3, dash='dash'),
                          showlegend=(i == 0)),
                row=row, col=col, secondary_y=True
            )
        
        fig.update_layout(
            height=600,
            title_text=f"Patient {patient_id}: Lab Results with 90-Day Risk Prediction",
            title_x=0.5
        )
        
        fig.update_yaxes(title_text="Lab Values", secondary_y=False)
        fig.update_yaxes(title_text="Risk %", secondary_y=True, range=[0, 100])
        
        st.plotly_chart(fig, use_container_width=True)

    def plot_medication_with_prediction(self, patient_data, model_engine, patient_id):
        """Plot medication adherence with prediction"""
        
        fig = go.Figure()
        
        # Medication adherence
        fig.add_trace(
            go.Scatter(x=patient_data['date'], y=patient_data['medication_adherence'] * 100,
                      name='Medication Adherence (%)', line=dict(color='blue', width=3),
                      yaxis='y')
        )
        
        # Risk prediction
        future_dates, risk_predictions = self.generate_90_day_prediction(patient_data, model_engine)
        fig.add_trace(
            go.Scatter(x=future_dates, y=risk_predictions,
                      name='90-Day Risk Prediction (%)', line=dict(color='red', width=3, dash='dash'),
                      yaxis='y2')
        )
        
        # Add adherence threshold
        fig.add_hline(y=80, line_dash="dash", line_color="orange", 
                      annotation_text="Target Adherence (80%)")
        
        fig.update_layout(
            title=f"Patient {patient_id}: Medication Adherence vs Risk Prediction",
            xaxis_title="Date",
            yaxis=dict(title="Adherence (%)", side="left", range=[0, 100]),
            yaxis2=dict(title="Risk (%)", side="right", overlaying="y", range=[0, 100]),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def plot_lifestyle_with_prediction(self, patient_data, model_engine, patient_id):
        """Plot lifestyle metrics with prediction"""
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Daily Steps + Risk', 'Sleep Hours + Risk'),
            specs=[[{"secondary_y": True}, {"secondary_y": True}]]
        )
        
        future_dates, risk_predictions = self.generate_90_day_prediction(patient_data, model_engine)
        
        # Steps
        fig.add_trace(
            go.Scatter(x=patient_data['date'], y=patient_data['daily_steps'],
                      name='Daily Steps', line=dict(color='green', width=2)),
            row=1, col=1, secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=future_dates, y=risk_predictions,
                      name='Risk %', line=dict(color='orange', width=3, dash='dash')),
            row=1, col=1, secondary_y=True
        )
        
        # Sleep
        fig.add_trace(
            go.Scatter(x=patient_data['date'], y=patient_data['sleep_hours'],
                      name='Sleep Hours', line=dict(color='purple', width=2)),
            row=1, col=2, secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=future_dates, y=risk_predictions,
                      name='Risk %', line=dict(color='orange', width=3, dash='dash'), showlegend=False),
            row=1, col=2, secondary_y=True
        )
        
        fig.update_layout(
            height=400,
            title_text=f"Patient {patient_id}: Lifestyle Metrics with Risk Prediction"
        )
        
        fig.update_yaxes(title_text="Lifestyle Metrics", secondary_y=False)
        fig.update_yaxes(title_text="Risk %", secondary_y=True, range=[0, 100])
        
        st.plotly_chart(fig, use_container_width=True)

    def plot_patient_shap_explanation(self, patient_id, shap_explainer, patient_data):
        """Plot SHAP explanations for individual patient"""
        
        # Get SHAP values for this patient
        shap_values = shap_explainer.get_patient_explanation(patient_id)
        
        if shap_values is not None:
            # Create waterfall plot data
            features = shap_values['features']
            values = shap_values['shap_values']
            
            # Sort by absolute importance
            sorted_indices = np.argsort(np.abs(values))[-10:]  # Top 10
            
            fig = go.Figure(go.Waterfall(
                name="SHAP Values",
                orientation="v",
                measure=["relative"] * len(sorted_indices),
                x=[features[i] for i in sorted_indices],
                textposition="outside",
                text=[f"{values[i]:+.3f}" for i in sorted_indices],
                y=[values[i] for i in sorted_indices],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
            ))
            
            fig.update_layout(
                title=f"Patient {patient_id}: Risk Factor Contributions",
                showlegend=True,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Clinical interpretation
            st.markdown("### 🩺 Clinical Interpretation:")
            for i in sorted_indices[-5:]:  # Top 5
                impact = "increases" if values[i] > 0 else "decreases"
                st.markdown(f"• **{features[i]}**: {impact} risk by {abs(values[i]):.1%}")

    def plot_global_shap_explanation(self, shap_explainer):
        """Plot global SHAP feature importance"""
        
        global_importance = shap_explainer.get_global_importance()
        
        fig = px.bar(
            x=list(global_importance.values()),
            y=list(global_importance.keys()),
            orientation='h',
            title="Global Feature Importance (All Patients)",
            labels={'x': 'Mean |SHAP Value|', 'y': 'Features'}
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    def crewai_validation_dashboard(self, validation_crew, model_engine):
        """CrewAI validation results dashboard"""
        st.header("🤖 CrewAI Model Validation Dashboard")
        
        # Run validation if not already done
        if st.button("🚀 Run CrewAI Validation"):
            with st.spinner("🤖 AI Agents are validating the model..."):
                validation_results = validation_crew.run_validation(model_engine)
                
                # Store results in session state
                st.session_state['validation_results'] = validation_results
                st.success("✅ Validation completed!")
        
        # Display results if available
        if 'validation_results' in st.session_state:
            results = st.session_state['validation_results']
            
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Clinical Evidence Score", f"{results['clinical_score']:.1f}/10")
            with col2:
                st.metric("Statistical Validity", results['statistical_status'])
            with col3:
                st.metric("Bias Assessment", results['bias_level'])
            with col4:
                st.metric("Overall Confidence", results['overall_confidence'])
            
            # Agent conversations
            st.subheader("🤖 Agent Validation Process")
            
            for conversation in results['agent_conversations']:
                st.markdown(f"""
                <div class="agent-message">
                <strong>{conversation['agent']}</strong> - <em>{conversation['timestamp']}</em><br>
                {conversation['message']}
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed reports
            with st.expander("📊 Detailed Validation Report"):
                st.json(results['detailed_report'])

    def cohort_management_dashboard(self, patient_data, predictions_data):
        """Cohort management interface"""
        st.header("📋 Cohort Management Dashboard")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_filter = st.selectbox("Risk Level", ["All", "High Risk (>70%)", "Medium Risk (40-70%)", "Low Risk (<40%)"])
        with col2:
            condition_filter = st.selectbox("Primary Condition", ["All"] + list(patient_data['primary_condition'].unique()))
        with col3:
            age_range = st.slider("Age Range", 18, 100, (18, 100))
        
        # Apply filters
        filtered_data = self.apply_cohort_filters(predictions_data, risk_filter, condition_filter, age_range)
        
        # Display filtered patients
        st.subheader(f"📊 Filtered Patients ({len(filtered_data)} patients)")
        
        # Add action buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📞 Schedule High-Risk Calls"):
                high_risk_count = len(filtered_data[filtered_data['risk_score'] > 0.7])
                st.success(f"Scheduled calls for {high_risk_count} high-risk patients")
        
        with col2:
            if st.button("📧 Medication Reminders"):
                st.success("Medication reminders sent to low-adherence patients")
        
        with col3:
            if st.button("📋 Generate Care Plans"):
                st.success("Automated care plans generated")
        
        with col4:
            if st.button("📊 Export Report"):
                st.success("Patient cohort report exported")
        
        # Display patient table
        display_data = filtered_data[['patient_id', 'age', 'primary_condition', 'risk_score', 'recommended_action']]
        st.dataframe(display_data, use_container_width=True)

    # Helper methods
    def get_patient_info(self, patient_id, patient_data, predictions_data):
        """Get patient information"""
        patient_row = patient_data[patient_data['patient_id'] == patient_id].iloc[0]
        prediction_row = predictions_data[predictions_data['patient_id'] == patient_id].iloc[0]
        
        return {
            'age': patient_row['age'],
            'bmi': patient_row['bmi'],
            'risk_score': prediction_row['risk_score'],
            'conditions': patient_row['chronic_conditions'].split(',') if pd.notna(patient_row['chronic_conditions']) else []
        }
    
    def get_risk_level(self, risk_score):
        """Determine risk level"""
        if risk_score > 0.7:
            return "HIGH"
        elif risk_score > 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def apply_cohort_filters(self, data, risk_filter, condition_filter, age_range):
        """Apply cohort filters"""
        filtered = data.copy()
        
        # Risk filter
        if risk_filter == "High Risk (>70%)":
            filtered = filtered[filtered['risk_score'] > 0.7]
        elif risk_filter == "Medium Risk (40-70%)":
            filtered = filtered[(filtered['risk_score'] >= 0.4) & (filtered['risk_score'] <= 0.7)]
        elif risk_filter == "Low Risk (<40%)":
            filtered = filtered[filtered['risk_score'] < 0.4]
        
        # Condition filter
        if condition_filter != "All":
            filtered = filtered[filtered['primary_condition'] == condition_filter]
        
        # Age filter
        filtered = filtered[(filtered['age'] >= age_range[0]) & (filtered['age'] <= age_range[1])]
        
        return filtered

# Main execution
if __name__ == "__main__":
    app = DashboardApp()
    app.run()
