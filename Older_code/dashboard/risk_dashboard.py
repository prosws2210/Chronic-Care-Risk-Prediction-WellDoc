"""
Risk Dashboard for population-level risk monitoring and analytics.
Provides comprehensive overview of patient population risk status and trends.
"""

import json
import logging
from datetime import datetime, timedelta
from flask import Blueprint, render_template, request, jsonify, current_app
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Create blueprint
risk_bp = Blueprint('risk_dashboard', __name__, template_folder='templates')

class RiskDashboard:
    """Handles population-level risk dashboard functionality."""
    
    def __init__(self, app=None):
        self.app = app
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize the risk dashboard with Flask app."""
        self.app = app

@risk_bp.route('/')
@risk_bp.route('/overview')
def overview():
    """Main risk dashboard overview with population metrics."""
    try:
        # Get dashboard data
        dashboard_data = _get_dashboard_data()
        
        # Generate charts
        chart_generator = current_app.chart_generator
        
        # Risk distribution pie chart
        patients_data = _get_mock_patients_summary()
        risk_distribution_chart = chart_generator.generate_risk_distribution_chart(patients_data)
        
        # Feature importance chart
        feature_importance = _get_feature_importance_data()
        importance_chart = chart_generator.generate_feature_importance_chart(feature_importance)
        
        # Population analytics
        population_data = _get_population_analytics_data()
        analytics_chart = chart_generator.generate_population_analytics_chart(population_data)
        
        # Generate summary tables
        table_generator = current_app.table_generator
        summary_cards = table_generator.generate_summary_cards(dashboard_data['summary_stats'])
        
        # High-risk patients table
        high_risk_patients = _get_high_risk_patients()
        high_risk_table = table_generator.generate_patient_cohort_table(
            patients=high_risk_patients,
            page_size=10
        )
        
        # Get active alerts
        alerts_manager = current_app.alerts_manager
        critical_alerts = alerts_manager.get_critical_alerts()
        alert_summary = alerts_manager.get_alert_summary()
        
        return render_template('risk_dashboard.html',
                             dashboard_data=dashboard_data,
                             risk_chart=risk_distribution_chart,
                             importance_chart=importance_chart,
                             analytics_chart=analytics_chart,
                             summary_cards=summary_cards,
                             high_risk_table=high_risk_table['table_html'],
                             critical_alerts=critical_alerts,
                             alert_summary=alert_summary)
        
    except Exception as e:
        logger.error(f"Error loading risk dashboard: {str(e)}")
        return render_template('error.html',
                             error_message="Unable to load risk dashboard"), 500

@risk_bp.route('/trends')
def risk_trends():
    """Risk trends analysis page."""
    try:
        # Get trend data
        trend_data = _get_risk_trend_data()
        
        # Generate trend charts
        chart_generator = current_app.chart_generator
        
        # Monthly risk trends
        monthly_trends = trend_data['monthly_trends']
        
        # Create chart configuration for JavaScript rendering
        trend_chart_config = chart_generator.create_interactive_chart_config(
            chart_type='line',
            data={
                'labels': [item['month'] for item in monthly_trends],
                'datasets': [{
                    'label': 'Average Risk Score',
                    'data': [item['avg_risk'] for item in monthly_trends],
                    'borderColor': 'rgb(75, 192, 192)',
                    'backgroundColor': 'rgba(75, 192, 192, 0.2)',
                    'tension': 0.4
                }, {
                    'label': 'High-Risk Patients (%)',
                    'data': [item['high_risk_pct'] for item in monthly_trends],
                    'borderColor': 'rgb(255, 99, 132)',
                    'backgroundColor': 'rgba(255, 99, 132, 0.2)',
                    'tension': 0.4
                }]
            }
        )
        
        return render_template('risk_trends.html',
                             trend_data=trend_data,
                             trend_chart_config=json.dumps(trend_chart_config))
        
    except Exception as e:
        logger.error(f"Error loading risk trends: {str(e)}")
        return render_template('error.html',
                             error_message="Unable to load risk trends"), 500

@risk_bp.route('/interventions')
def intervention_tracking():
    """Intervention tracking and outcomes page."""
    try:
        # Get intervention data
        intervention_data = _get_intervention_data()
        
        # Generate intervention charts
        chart_generator = current_app.chart_generator
        
        # Intervention effectiveness chart
        effectiveness_data = intervention_data['effectiveness']
        
        return render_template('intervention_tracking.html',
                             intervention_data=intervention_data)
        
    except Exception as e:
        logger.error(f"Error loading intervention tracking: {str(e)}")
        return render_template('error.html',
                             error_message="Unable to load intervention data"), 500

@risk_bp.route('/population-health')
def population_health():
    """Population health management page."""
    try:
        # Get population health data
        pop_health_data = _get_population_health_data()
        
        # Generate population health charts
        chart_generator = current_app.chart_generator
        
        population_chart = chart_generator.generate_population_analytics_chart(
            pop_health_data['analytics']
        )
        
        # Generate cohort comparison table
        table_generator = current_app.table_generator
        cohort_table = table_generator.generate_html_table(
            data=pop_health_data['cohort_comparison'],
            headers=['Cohort', 'Size', 'Avg Risk', 'Interventions', 'Outcomes'],
            title='Cohort Performance Comparison'
        )
        
        return render_template('population_health.html',
                             pop_health_data=pop_health_data,
                             population_chart=population_chart,
                             cohort_table=cohort_table)
        
    except Exception as e:
        logger.error(f"Error loading population health: {str(e)}")
        return render_template('error.html',
                             error_message="Unable to load population health data"), 500

@risk_bp.route('/quality-metrics')
def quality_metrics():
    """Quality metrics and performance indicators page."""
    try:
        # Get quality metrics
        quality_data = _get_quality_metrics_data()
        
        return render_template('quality_metrics.html',
                             quality_data=quality_data)
        
    except Exception as e:
        logger.error(f"Error loading quality metrics: {str(e)}")
        return render_template('error.html',
                             error_message="Unable to load quality metrics"), 500

@risk_bp.route('/api/dashboard-summary')
def api_dashboard_summary():
    """API endpoint for dashboard summary data."""
    try:
        summary = {
            'total_patients': 2847,
            'high_risk_patients': 412,
            'critical_alerts': 23,
            'avg_risk_score': 0.34,
            'trend_direction': 'stable',
            'last_updated': datetime.now().isoformat()
        }
        
        return jsonify(summary)
        
    except Exception as e:
        logger.error(f"Error getting dashboard summary: {str(e)}")
        return jsonify({'error': 'Unable to fetch summary'}), 500

@risk_bp.route('/api/risk-alerts', methods=['POST'])
def create_risk_alert():
    """API endpoint to create new risk alert."""
    try:
        data = request.get_json()
        patient_id = data.get('patient_id')
        risk_score = float(data.get('risk_score'))
        risk_factors = data.get('risk_factors', [])
        
        # Create alert using alerts manager
        alerts_manager = current_app.alerts_manager
        alert = alerts_manager.create_patient_risk_alert(
            patient_id=patient_id,
            risk_score=risk_score,
            risk_factors=risk_factors
        )
        
        alerts_manager.add_alert(alert)
        
        return jsonify({
            'success': True,
            'alert_id': alert.alert_id,
            'created_at': alert.created_at.isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error creating risk alert: {str(e)}")
        return jsonify({'error': 'Failed to create alert'}), 500

@risk_bp.route('/api/export-dashboard', methods=['POST'])
def export_dashboard_data():
    """Export dashboard data to various formats."""
    try:
        export_format = request.json.get('format', 'json')
        include_details = request.json.get('include_details', False)
        
        # Get comprehensive dashboard data
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'summary_statistics': _get_dashboard_data()['summary_stats'],
            'risk_distribution': _get_risk_distribution_data(),
            'high_risk_patients': _get_high_risk_patients() if include_details else [],
            'trend_data': _get_risk_trend_data()['monthly_trends'],
            'intervention_outcomes': _get_intervention_data()['outcomes']
        }
        
        if export_format == 'csv':
            # Convert to CSV format
            table_generator = current_app.table_generator
            csv_content = table_generator.export_table_to_csv(
                data=[export_data['summary_statistics']],
                filename=f"dashboard_export_{datetime.now().strftime('%Y%m%d')}.csv"
            )
            
            from flask import Response
            return Response(
                csv_content,
                mimetype='text/csv',
                headers={
                    'Content-Disposition': f'attachment; filename=dashboard_export_{datetime.now().strftime("%Y%m%d")}.csv'
                }
            )
        
        return jsonify(export_data)
        
    except Exception as e:
        logger.error(f"Error exporting dashboard data: {str(e)}")
        return jsonify({'error': 'Export failed'}), 500

def _get_dashboard_data() -> Dict[str, Any]:
    """Get comprehensive dashboard data."""
    return {
        'summary_stats': {
            'total_patients': 2847,
            'high_risk_count': 412,
            'critical_risk_count': 89,
            'average_risk_score': 34.2,
            'average_age': 64.5,
            'active_alerts': 47,
            'interventions_active': 156,
            'outcomes_improved': 78
        },
        'risk_breakdown': {
            'low_risk': {'count': 1624, 'percentage': 57.0},
            'medium_risk': {'count': 811, 'percentage': 28.5},
            'high_risk': {'count': 323, 'percentage': 11.3},
            'critical_risk': {'count': 89, 'percentage': 3.2}
        },
        'trending_metrics': {
            'risk_score_change': '+2.1%',
            'hospitalization_rate': '-8.3%',
            'medication_adherence': '+12.7%',
            'care_plan_completion': '+15.2%'
        },
        'condition_breakdown': {
            'diabetes': {'count': 1138, 'avg_risk': 0.38},
            'heart_failure': {'count': 712, 'avg_risk': 0.42},
            'obesity': {'count': 997, 'avg_risk': 0.29}
        }
    }

def _get_mock_patients_summary() -> List[Dict[str, Any]]:
    """Get mock patient data for charts."""
    return [
        {'risk_level': 'Low', 'count': 1624},
        {'risk_level': 'Medium', 'count': 811},
        {'risk_level': 'High', 'count': 323},
        {'risk_level': 'Critical', 'count': 89}
    ]

def _get_feature_importance_data() -> Dict[str, float]:
    """Get feature importance data for population."""
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
        'diabetes_duration_years': 0.048,
        'social_determinant_risk': 0.042,
        'lab_abnormality_frequency': 0.039,
        'specialist_visit_frequency': 0.036,
        'symptom_burden_score': 0.033,
        'insurance_stability_indicator': 0.030
    }

def _get_population_analytics_data() -> Dict[str, Any]:
    """Get population analytics data."""
    return {
        'age_distribution': {
            '18-40': 285,
            '41-65': 1282,
            '66-80': 1053,
            '80+': 227
        },
        'gender_distribution': {
            'Male': 1367,
            'Female': 1480
        },
        'condition_prevalence': {
            'Diabetes': 40.0,
            'Heart Failure': 25.0,
            'Obesity': 35.0,
            'Hypertension': 68.0,
            'Dyslipidemia': 52.0,
            'CKD': 18.0
        },
        'risk_scores': [0.2 + i * 0.01 for i in range(60)],  # Distribution
        'healthcare_utilization': {
            'ED Visits': 2.3,
            'Hospitalizations': 0.8,
            'Outpatient Visits': 8.5,
            'Specialist Visits': 3.2,
            'Telemedicine': 2.1
        },
        'adherence_distribution': {
            'Good (>80%)': 60,
            'Fair (60-80%)': 25,
            'Poor (<60%)': 15
        },
        'outcome_trends': {
            'Jan': 42,
            'Feb': 38,
            'Mar': 35,
            'Apr': 41,
            'May': 33,
            'Jun': 29,
            'Jul': 31,
            'Aug': 27
        }
    }

def _get_high_risk_patients() -> List[Dict[str, Any]]:
    """Get list of high-risk patients for dashboard table."""
    patients = []
    
    for i in range(20):  # Top 20 high-risk patients
        patient = {
            'patient_id': f'SYNTH_{1000 + i:06d}',
            'name': f'High Risk Patient {i+1}',
            'age': 65 + (i % 25),
            'gender': 'Female' if i % 2 == 0 else 'Male',
            'primary_condition': ['Diabetes', 'Heart Failure', 'Obesity'][i % 3],
            'risk_score': 0.75 + (i % 20) * 0.01,
            'risk_level': 'Critical' if 0.75 + (i % 20) * 0.01 >= 0.85 else 'High',
            'last_visit': (datetime.now() - timedelta(days=i * 3)).strftime('%Y-%m-%d'),
            'next_followup': (datetime.now() + timedelta(days=7 + i % 21)).strftime('%Y-%m-%d'),
            'recent_hospitalization': i % 5 == 0
        }
        patients.append(patient)
    
    return patients

def _get_risk_trend_data() -> Dict[str, Any]:
    """Get risk trend data for analytics."""
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug']
    
    monthly_trends = []
    for i, month in enumerate(months):
        monthly_trends.append({
            'month': month,
            'avg_risk': 0.32 + (i % 4) * 0.02,
            'high_risk_pct': 12.5 + (i % 3) * 1.5,
            'total_patients': 2800 + i * 15,
            'new_high_risk': 15 + (i % 5) * 3,
            'interventions_started': 25 + (i % 6) * 4
        })
    
    return {
        'monthly_trends': monthly_trends,
        'risk_velocity': {
            'increasing': 23.5,
            'stable': 64.2,
            'decreasing': 12.3
        },
        'seasonal_patterns': {
            'winter_increase': 8.3,
            'spring_stable': 2.1,
            'summer_decrease': -3.7,
            'fall_increase': 4.2
        }
    }

def _get_intervention_data() -> Dict[str, Any]:
    """Get intervention tracking data."""
    return {
        'active_interventions': {
            'diabetes_management': {
                'patients': 234,
                'completion_rate': 78.5,
                'avg_improvement': 15.2
            },
            'medication_adherence': {
                'patients': 189,
                'completion_rate': 82.1,
                'avg_improvement': 23.7
            },
            'lifestyle_modification': {
                'patients': 156,
                'completion_rate': 65.4,
                'avg_improvement': 12.8
            },
            'care_coordination': {
                'patients': 98,
                'completion_rate': 89.8,
                'avg_improvement': 18.3
            }
        },
        'outcomes': {
            'risk_reduction_achieved': 76,
            'hospitalizations_prevented': 23,
            'medication_adherence_improved': 127,
            'care_plan_completion': 89
        },
        'effectiveness': {
            'high_intensity': {
                'patients': 78,
                'success_rate': 84.6,
                'avg_risk_reduction': 28.5
            },
            'standard_care': {
                'patients': 234,
                'success_rate': 67.2,
                'avg_risk_reduction': 15.8
            },
            'self_management': {
                'patients': 156,
                'success_rate': 45.3,
                'avg_risk_reduction': 8.2
            }
        },
        'cost_effectiveness': {
            'cost_per_patient': 1247,
            'cost_per_qaly': 8934,
            'roi_percentage': 234,
            'savings_generated': 234567
        }
    }

def _get_population_health_data() -> Dict[str, Any]:
    """Get population health management data."""
    return {
        'analytics': _get_population_analytics_data(),
        'cohort_comparison': [
            ['High-Risk Diabetes', 234, '0.78', 156, 'Improved'],
            ['Heart Failure Mgmt', 189, '0.65', 134, 'Stable'],
            ['Complex Multi-Morbid', 98, '0.82', 89, 'Improved'],
            ['Social Risk', 67, '0.71', 45, 'Moderate'],
            ['Medication Adherence', 178, '0.59', 156, 'Significantly Improved']
        ],
        'quality_measures': {
            'diabetes_control': {
                'target': 70,
                'current': 68.5,
                'trend': 'improving'
            },
            'medication_adherence': {
                'target': 80,
                'current': 74.2,
                'trend': 'improving'
            },
            'care_coordination': {
                'target': 85,
                'current': 78.9,
                'trend': 'stable'
            }
        },
        'risk_stratification_effectiveness': {
            'sensitivity': 84.7,
            'specificity': 86.8,
            'ppv': 74.5,
            'npv': 92.1,
            'accuracy': 85.9
        }
    }

def _get_quality_metrics_data() -> Dict[str, Any]:
    """Get quality metrics and KPIs."""
    return {
        'clinical_quality': {
            'diabetes_hba1c_control': {
                'target': '>70%',
                'current': '68.5%',
                'trend': '+2.3%',
                'benchmark': '65%'
            },
            'heart_failure_guideline_adherence': {
                'target': '>85%',
                'current': '82.1%',
                'trend': '+4.7%',
                'benchmark': '78%'
            },
            'medication_reconciliation': {
                'target': '>95%',
                'current': '91.3%',
                'trend': '+1.8%',
                'benchmark': '89%'
            }
        },
        'operational_quality': {
            'risk_assessment_timeliness': {
                'target': '<24hrs',
                'current': '18.5hrs',
                'trend': '-2.3hrs',
                'benchmark': '22hrs'
            },
            'care_plan_updates': {
                'target': '>90%',
                'current': '87.2%',
                'trend': '+3.1%',
                'benchmark': '85%'
            },
            'patient_engagement': {
                'target': '>75%',
                'current': '79.6%',
                'trend': '+5.4%',
                'benchmark': '72%'
            }
        },
        'patient_outcomes': {
            'hospital_readmissions': {
                'target': '<15%',
                'current': '12.8%',
                'trend': '-2.1%',
                'benchmark': '16.2%'
            },
            'emergency_visits': {
                'target': '<2.5/year',
                'current': '2.1/year',
                'trend': '-0.3',
                'benchmark': '2.8/year'
            },
            'patient_satisfaction': {
                'target': '>4.0/5',
                'current': '4.2/5',
                'trend': '+0.1',
                'benchmark': '3.8/5'
            }
        },
        'financial_metrics': {
            'cost_per_member_per_month': {
                'current': '$234',
                'trend': '-$12',
                'benchmark': '$267'
            },
            'roi_on_interventions': {
                'current': '234%',
                'trend': '+23%',
                'benchmark': '180%'
            },
            'care_gap_closure_rate': {
                'current': '76.3%',
                'trend': '+8.7%',
                'benchmark': '68%'
            }
        }
    }

def _get_risk_distribution_data() -> Dict[str, Any]:
    """Get detailed risk distribution data."""
    return {
        'by_risk_level': {
            'low': 1624,
            'medium': 811,
            'high': 323,
            'critical': 89
        },
        'by_condition': {
            'diabetes': {'low': 650, 'medium': 325, 'high': 130, 'critical': 33},
            'heart_failure': {'low': 284, 'medium': 203, 'high': 162, 'critical': 63},
            'obesity': {'low': 690, 'medium': 283, 'high': 31, 'critical': 3}
        },
        'by_age_group': {
            '18-40': {'low': 200, 'medium': 65, 'high': 15, 'critical': 5},
            '41-65': {'low': 770, 'medium': 360, 'high': 115, 'critical': 37},
            '66-80': {'low': 550, 'medium': 320, 'high': 140, 'critical': 43},
            '80+': {'low': 104, 'medium': 66, 'high': 53, 'critical': 4}
        }
    }
