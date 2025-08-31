"""
Cohort View for patient population management and filtering.
Displays patient cohorts with risk stratification and filtering capabilities.
"""

import json
import logging
from datetime import datetime, timedelta
from flask import Blueprint, render_template, request, jsonify, current_app
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Create blueprint
cohort_bp = Blueprint('cohort', __name__, template_folder='templates')

class CohortView:
    """Handles cohort visualization and management functionality."""
    
    def __init__(self, app=None):
        self.app = app
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize the cohort view with Flask app."""
        self.app = app

@cohort_bp.route('/')
@cohort_bp.route('/overview')
def cohort_overview():
    """Main cohort overview page with patient list and filtering."""
    try:
        # Get filter parameters
        filters = {
            'search': request.args.get('search', ''),
            'risk_level': request.args.get('risk_level', ''),
            'age_min': request.args.get('age_min', ''),
            'age_max': request.args.get('age_max', ''),
            'gender': request.args.get('gender', ''),
            'condition': request.args.get('condition', ''),
            'date_from': request.args.get('date_from', ''),
            'date_to': request.args.get('date_to', '')
        }
        
        # Pagination parameters
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 20))
        sort_by = request.args.get('sort_by', 'risk_score')
        sort_order = request.args.get('sort_order', 'desc')
        
        # Get patient data (in production, this would query the database)
        patients_data = _get_mock_patient_data()
        
        # Generate patient table
        table_generator = current_app.table_generator
        table_result = table_generator.generate_patient_cohort_table(
            patients=patients_data,
            filters=filters,
            sort_by=sort_by,
            sort_order=sort_order,
            page=page,
            page_size=page_size
        )
        
        # Generate summary cards
        summary_cards_html = table_generator.generate_summary_cards(
            table_result['summary_stats']
        )
        
        # Generate risk distribution chart
        chart_generator = current_app.chart_generator
        risk_chart = chart_generator.generate_risk_distribution_chart(patients_data)
        
        return render_template('cohort_overview.html',
                             table_html=table_result['table_html'],
                             pagination=table_result['pagination'],
                             summary_stats=table_result['summary_stats'],
                             summary_cards=summary_cards_html,
                             risk_chart=risk_chart,
                             filters=filters,
                             sort_by=sort_by,
                             sort_order=sort_order)
        
    except Exception as e:
        logger.error(f"Error in cohort overview: {str(e)}")
        return render_template('error.html', 
                             error_message="Unable to load cohort data"), 500

@cohort_bp.route('/api/export')
def export_cohort():
    """Export cohort data to CSV."""
    try:
        # Get filter parameters
        filters = {
            'search': request.args.get('search', ''),
            'risk_level': request.args.get('risk_level', ''),
            'condition': request.args.get('condition', '')
        }
        
        # Get filtered patient data
        patients_data = _get_mock_patient_data()
        # Apply filters (simplified for demo)
        
        # Generate CSV
        table_generator = current_app.table_generator
        csv_content = table_generator.export_table_to_csv(
            data=patients_data,
            filename=f"cohort_export_{datetime.now().strftime('%Y%m%d')}.csv"
        )
        
        from flask import Response
        return Response(
            csv_content,
            mimetype='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename=cohort_export_{datetime.now().strftime("%Y%m%d")}.csv'
            }
        )
        
    except Exception as e:
        logger.error(f"Error exporting cohort: {str(e)}")
        return jsonify({'error': 'Export failed'}), 500

@cohort_bp.route('/api/stats')
def cohort_statistics():
    """API endpoint for cohort statistics."""
    try:
        patients_data = _get_mock_patient_data()
        
        stats = {
            'total_patients': len(patients_data),
            'risk_distribution': _calculate_risk_distribution(patients_data),
            'age_distribution': _calculate_age_distribution(patients_data),
            'condition_distribution': _calculate_condition_distribution(patients_data),
            'gender_distribution': _calculate_gender_distribution(patients_data),
            'recent_hospitalizations': _calculate_recent_hospitalizations(patients_data)
        }
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error calculating cohort statistics: {str(e)}")
        return jsonify({'error': 'Unable to calculate statistics'}), 500

@cohort_bp.route('/risk-stratification')
def risk_stratification():
    """Risk stratification analysis page."""
    try:
        patients_data = _get_mock_patient_data()
        
        # Generate risk stratification charts
        chart_generator = current_app.chart_generator
        
        # Risk distribution
        risk_chart = chart_generator.generate_risk_distribution_chart(patients_data)
        
        # Feature importance chart
        feature_importance = _get_mock_feature_importance()
        importance_chart = chart_generator.generate_feature_importance_chart(feature_importance)
        
        # Population analytics
        population_data = _get_mock_population_data()
        analytics_chart = chart_generator.generate_population_analytics_chart(population_data)
        
        return render_template('risk_stratification.html',
                             risk_chart=risk_chart,
                             importance_chart=importance_chart,
                             analytics_chart=analytics_chart,
                             total_patients=len(patients_data))
        
    except Exception as e:
        logger.error(f"Error in risk stratification: {str(e)}")
        return render_template('error.html',
                             error_message="Unable to load risk analysis"), 500

@cohort_bp.route('/alerts')
def cohort_alerts():
    """Cohort-level alerts and notifications."""
    try:
        # Get active alerts
        alerts_manager = current_app.alerts_manager
        active_alerts = alerts_manager.get_active_alerts()
        
        # Get alert summary
        alert_summary = alerts_manager.get_alert_summary()
        
        return render_template('cohort_alerts.html',
                             alerts=active_alerts,
                             alert_summary=alert_summary)
        
    except Exception as e:
        logger.error(f"Error loading cohort alerts: {str(e)}")
        return render_template('error.html',
                             error_message="Unable to load alerts"), 500

def _get_mock_patient_data() -> List[Dict[str, Any]]:
    """Generate mock patient data for demonstration."""
    patients = []
    
    for i in range(100):
        patient = {
            'patient_id': f'SYNTH_{i+1:06d}',
            'name': f'Patient {i+1}',
            'age': 45 + (i % 45),
            'gender': 'Female' if i % 2 == 0 else 'Male',
            'primary_condition': ['Diabetes', 'Heart Failure', 'Obesity'][i % 3],
            'risk_score': min(0.95, max(0.05, 0.3 + (i % 7) * 0.1 + (i % 3) * 0.05)),
            'risk_level': _calculate_risk_level(0.3 + (i % 7) * 0.1 + (i % 3) * 0.05),
            'last_visit': (datetime.now() - timedelta(days=i % 90)).strftime('%Y-%m-%d'),
            'next_followup': (datetime.now() + timedelta(days=7 + i % 30)).strftime('%Y-%m-%d'),
            'recent_hospitalization': i % 15 == 0
        }
        patients.append(patient)
    
    return patients

def _calculate_risk_level(risk_score: float) -> str:
    """Calculate risk level from risk score."""
    if risk_score >= 0.7:
        return 'Critical' if risk_score >= 0.85 else 'High'
    elif risk_score >= 0.3:
        return 'Medium'
    else:
        return 'Low'

def _calculate_risk_distribution(patients: List[Dict[str, Any]]) -> Dict[str, int]:
    """Calculate risk level distribution."""
    distribution = {'Low': 0, 'Medium': 0, 'High': 0, 'Critical': 0}
    for patient in patients:
        risk_level = patient.get('risk_level', 'Low')
        distribution[risk_level] = distribution.get(risk_level, 0) + 1
    return distribution

def _calculate_age_distribution(patients: List[Dict[str, Any]]) -> Dict[str, int]:
    """Calculate age group distribution."""
    distribution = {'18-40': 0, '41-65': 0, '66-80': 0, '80+': 0}
    for patient in patients:
        age = patient.get('age', 0)
        if age <= 40:
            distribution['18-40'] += 1
        elif age <= 65:
            distribution['41-65'] += 1
        elif age <= 80:
            distribution['66-80'] += 1
        else:
            distribution['80+'] += 1
    return distribution

def _calculate_condition_distribution(patients: List[Dict[str, Any]]) -> Dict[str, int]:
    """Calculate primary condition distribution."""
    distribution = {}
    for patient in patients:
        condition = patient.get('primary_condition', 'Unknown')
        distribution[condition] = distribution.get(condition, 0) + 1
    return distribution

def _calculate_gender_distribution(patients: List[Dict[str, Any]]) -> Dict[str, int]:
    """Calculate gender distribution."""
    distribution = {'Male': 0, 'Female': 0}
    for patient in patients:
        gender = patient.get('gender', 'Unknown')
        if gender in distribution:
            distribution[gender] += 1
    return distribution

def _calculate_recent_hospitalizations(patients: List[Dict[str, Any]]) -> int:
    """Calculate number of patients with recent hospitalizations."""
    return sum(1 for patient in patients if patient.get('recent_hospitalization', False))

def _get_mock_feature_importance() -> Dict[str, float]:
    """Generate mock feature importance data."""
    return {
        'hba1c_trend_90d': 0.124,
        'medication_adherence_composite': 0.098,
        'hospitalization_recency_days': 0.087,
        'comorbidity_burden_weighted': 0.074,
        'systolic_bp_variability_30d': 0.069,
        'age_risk_category': 0.063,
        'egfr_decline_slope_180d': 0.059,
        'emergency_visits_6m': 0.055,
        'medication_count_active': 0.051,
        'diabetes_duration_years': 0.048
    }

def _get_mock_population_data() -> Dict[str, Any]:
    """Generate mock population analytics data."""
    return {
        'age_distribution': {'18-40': 15, '41-65': 45, '66-80': 30, '80+': 10},
        'gender_distribution': {'Male': 48, 'Female': 52},
        'condition_prevalence': {'Diabetes': 40, 'Heart Failure': 25, 'Obesity': 35},
        'risk_scores': [0.2 + i * 0.01 for i in range(60)],  # Mock risk score distribution
        'healthcare_utilization': {
            'ED Visits': 2.3,
            'Hospitalizations': 0.8,
            'Outpatient Visits': 8.5,
            'Specialist Visits': 3.2
        },
        'adherence_distribution': {'Good (>80%)': 60, 'Fair (60-80%)': 25, 'Poor (<60%)': 15},
        'outcome_trends': {
            'Jan': 12, 'Feb': 10, 'Mar': 8, 'Apr': 15, 'May': 11, 'Jun': 9
        }
    }
