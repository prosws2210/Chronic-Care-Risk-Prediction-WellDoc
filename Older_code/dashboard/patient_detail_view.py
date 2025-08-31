"""
Patient Detail View for comprehensive individual patient information.
Displays detailed patient data, risk factors, and clinical timeline.
"""

import json
import logging
from datetime import datetime, timedelta
from flask import Blueprint, render_template, request, jsonify, current_app
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Create blueprint
patient_bp = Blueprint('patient', __name__, template_folder='templates')

class PatientDetailView:
    """Handles individual patient detail visualization and management."""
    
    def __init__(self, app=None):
        self.app = app
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize the patient detail view with Flask app."""
        self.app = app

@patient_bp.route('/<patient_id>')
def patient_detail(patient_id: str):
    """Main patient detail page with comprehensive information."""
    try:
        # Get patient data (in production, query from database)
        patient_data = _get_mock_patient_detail(patient_id)
        
        if not patient_data:
            return render_template('error.html',
                                 error_code=404,
                                 error_message=f"Patient {patient_id} not found"), 404
        
        # Generate charts
        chart_generator = current_app.chart_generator
        
        # Risk trend chart
        risk_history = patient_data.get('risk_history', [])
        risk_chart = chart_generator.generate_risk_trend_chart(patient_id, risk_history)
        
        # Vital signs chart
        vital_signs = patient_data.get('vital_signs', [])
        vitals_chart = chart_generator.generate_vital_signs_chart(patient_id, vital_signs)
        
        # Lab results chart
        lab_results = patient_data.get('lab_results', [])
        labs_chart = chart_generator.generate_lab_results_chart(patient_id, lab_results)
        
        # Medication adherence chart
        adherence_data = patient_data.get('medication_adherence', [])
        adherence_chart = chart_generator.generate_medication_adherence_chart(patient_id, adherence_data)
        
        # Generate tables
        table_generator = current_app.table_generator
        
        # Lab results table
        labs_table = table_generator.generate_lab_results_table(patient_id, lab_results)
        
        # Medications table
        medications = patient_data.get('medications', [])
        meds_table = table_generator.generate_medication_table(patient_id, medications)
        
        # Get patient-specific alerts
        alerts_manager = current_app.alerts_manager
        patient_alerts = alerts_manager.get_active_alerts(patient_id=patient_id)
        
        return render_template('patient_detail.html',
                             patient=patient_data,
                             risk_chart=risk_chart,
                             vitals_chart=vitals_chart,
                             labs_chart=labs_chart,
                             adherence_chart=adherence_chart,
                             labs_table=labs_table,
                             meds_table=meds_table,
                             alerts=patient_alerts)
        
    except Exception as e:
        logger.error(f"Error loading patient {patient_id}: {str(e)}")
        return render_template('error.html',
                             error_message=f"Unable to load patient {patient_id}"), 500

@patient_bp.route('/<patient_id>/timeline')
def patient_timeline(patient_id: str):
    """Patient clinical timeline view."""
    try:
        # Get patient timeline data
        timeline_data = _get_mock_patient_timeline(patient_id)
        patient_data = _get_mock_patient_detail(patient_id)
        
        return render_template('patient_timeline.html',
                             patient_id=patient_id,
                             patient=patient_data,
                             timeline=timeline_data)
        
    except Exception as e:
        logger.error(f"Error loading timeline for patient {patient_id}: {str(e)}")
        return render_template('error.html',
                             error_message="Unable to load patient timeline"), 500

@patient_bp.route('/<patient_id>/risk-explanation')
def risk_explanation(patient_id: str):
    """Patient risk explanation and SHAP analysis."""
    try:
        # Get risk explanation data
        explanation_data = _get_mock_risk_explanation(patient_id)
        patient_data = _get_mock_patient_detail(patient_id)
        
        # Generate SHAP waterfall chart (mock)
        chart_generator = current_app.chart_generator
        
        return render_template('risk_explanation.html',
                             patient_id=patient_id,
                             patient=patient_data,
                             explanation=explanation_data)
        
    except Exception as e:
        logger.error(f"Error loading risk explanation for patient {patient_id}: {str(e)}")
        return render_template('error.html',
                             error_message="Unable to load risk explanation"), 500

@patient_bp.route('/<patient_id>/interventions')
def patient_interventions(patient_id: str):
    """Patient intervention recommendations and care plan."""
    try:
        # Get intervention recommendations
        interventions = _get_mock_interventions(patient_id)
        patient_data = _get_mock_patient_detail(patient_id)
        
        return render_template('patient_interventions.html',
                             patient_id=patient_id,
                             patient=patient_data,
                             interventions=interventions)
        
    except Exception as e:
        logger.error(f"Error loading interventions for patient {patient_id}: {str(e)}")
        return render_template('error.html',
                             error_message="Unable to load interventions"), 500

@patient_bp.route('/<patient_id>/report')
def patient_report(patient_id: str):
    """Generate comprehensive patient report."""
    try:
        patient_data = _get_mock_patient_detail(patient_id)
        
        if not patient_data:
            return render_template('error.html',
                                 error_code=404,
                                 error_message=f"Patient {patient_id} not found"), 404
        
        # Generate comprehensive report data
        report_data = {
            'patient': patient_data,
            'risk_summary': _generate_risk_summary(patient_data),
            'clinical_summary': _generate_clinical_summary(patient_data),
            'intervention_recommendations': _get_mock_interventions(patient_id),
            'generated_date': datetime.now().strftime('%B %d, %Y'),
            'generated_by': 'AI Risk Assessment System'
        }
        
        return render_template('patient_report.html', **report_data)
        
    except Exception as e:
        logger.error(f"Error generating report for patient {patient_id}: {str(e)}")
        return render_template('error.html',
                             error_message="Unable to generate patient report"), 500

@patient_bp.route('/api/<patient_id>/update-risk', methods=['POST'])
def update_patient_risk(patient_id: str):
    """API endpoint to update patient risk assessment."""
    try:
        data = request.get_json()
        new_risk_score = data.get('risk_score')
        notes = data.get('notes', '')
        user_id = data.get('user_id', 'system')
        
        # In production, update database
        # For demo, create an alert if risk is high
        if float(new_risk_score) >= 0.7:
            alerts_manager = current_app.alerts_manager
            alert = alerts_manager.create_patient_risk_alert(
                patient_id=patient_id,
                risk_score=float(new_risk_score),
                risk_factors=['manual_update']
            )
            alerts_manager.add_alert(alert)
        
        return jsonify({
            'success': True,
            'patient_id': patient_id,
            'new_risk_score': new_risk_score,
            'updated_at': datetime.now().isoformat(),
            'notes': notes
        })
        
    except Exception as e:
        logger.error(f"Error updating risk for patient {patient_id}: {str(e)}")
        return jsonify({'error': 'Update failed'}), 500

@patient_bp.route('/api/<patient_id>/schedule-followup', methods=['POST'])
def schedule_followup(patient_id: str):
    """API endpoint to schedule patient follow-up."""
    try:
        data = request.get_json()
        followup_date = data.get('followup_date')
        followup_type = data.get('followup_type', 'routine')
        provider = data.get('provider')
        reason = data.get('reason', 'Routine follow-up')
        
        # In production, create calendar entry and update database
        followup_data = {
            'patient_id': patient_id,
            'followup_date': followup_date,
            'followup_type': followup_type,
            'provider': provider,
            'reason': reason,
            'status': 'scheduled',
            'scheduled_by': 'dashboard_user',
            'scheduled_at': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            **followup_data
        })
        
    except Exception as e:
        logger.error(f"Error scheduling followup for patient {patient_id}: {str(e)}")
        return jsonify({'error': 'Scheduling failed'}), 500

@patient_bp.route('/api/<patient_id>/add-note', methods=['POST'])
def add_patient_note(patient_id: str):
    """API endpoint to add clinical note."""
    try:
        data = request.get_json()
        note_text = data.get('note')
        note_type = data.get('type', 'clinical')
        author = data.get('author', 'dashboard_user')
        
        # In production, save to database
        note_data = {
            'patient_id': patient_id,
            'note': note_text,
            'type': note_type,
            'author': author,
            'created_at': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            'note_id': f"note_{datetime.now().timestamp()}",
            **note_data
        })
        
    except Exception as e:
        logger.error(f"Error adding note for patient {patient_id}: {str(e)}")
        return jsonify({'error': 'Failed to add note'}), 500

def _get_mock_patient_detail(patient_id: str) -> Optional[Dict[str, Any]]:
    """Generate mock detailed patient data."""
    if not patient_id.startswith('SYNTH_'):
        return None
    
    # Extract patient number from ID
    try:
        patient_num = int(patient_id.split('_')[1])
    except (IndexError, ValueError):
        return None
    
    # Generate consistent mock data based on patient number
    base_age = 45 + (patient_num % 45)
    risk_score = min(0.95, max(0.05, 0.3 + (patient_num % 7) * 0.1))
    
    patient = {
        'patient_id': patient_id,
        'name': f'Patient {patient_num}',
        'age': base_age,
        'gender': 'Female' if patient_num % 2 == 0 else 'Male',
        'date_of_birth': (datetime.now() - timedelta(days=base_age * 365)).strftime('%Y-%m-%d'),
        'primary_condition': ['Diabetes', 'Heart Failure', 'Obesity'][patient_num % 3],
        'risk_score': risk_score,
        'risk_level': _calculate_risk_level(risk_score),
        'last_visit': (datetime.now() - timedelta(days=patient_num % 90)).strftime('%Y-%m-%d'),
        'next_followup': (datetime.now() + timedelta(days=7 + patient_num % 30)).strftime('%Y-%m-%d'),
        
        # Demographics
        'demographics': {
            'race_ethnicity': ['White', 'Black/African American', 'Hispanic/Latino', 'Asian'][patient_num % 4],
            'marital_status': ['Single', 'Married', 'Divorced', 'Widowed'][patient_num % 4],
            'insurance_type': ['Commercial', 'Medicare', 'Medicaid'][patient_num % 3],
            'zip_code': f'{10001 + patient_num % 1000:05d}',
            'phone': f'({200 + patient_num % 800:03d}) {patient_num % 1000:03d}-{patient_num % 10000:04d}',
            'emergency_contact': f'Contact {patient_num % 100}',
            'preferred_language': 'English'
        },
        
        # Clinical data
        'conditions': _get_mock_conditions(patient_num),
        'vital_signs': _get_mock_vital_signs(patient_num),
        'lab_results': _get_mock_lab_results(patient_num),
        'medications': _get_mock_medications(patient_num),
        'medication_adherence': _get_mock_adherence_data(patient_num),
        'risk_history': _get_mock_risk_history(patient_num),
        'allergies': _get_mock_allergies(patient_num),
        'social_history': _get_mock_social_history(patient_num),
        
        # Care team
        'care_team': {
            'primary_care_provider': f'Dr. Smith {patient_num % 10}',
            'specialists': _get_mock_specialists(patient_num),
            'care_coordinator': f'Coordinator {patient_num % 5}' if patient_num % 3 == 0 else None,
            'pharmacy': f'Pharmacy {patient_num % 20}',
            'emergency_contact': f'Emergency Contact {patient_num % 100}'
        },
        
        # Recent activity
        'recent_activity': _get_mock_recent_activity(patient_num)
    }
    
    return patient

def _get_mock_patient_timeline(patient_id: str) -> List[Dict[str, Any]]:
    """Generate mock clinical timeline."""
    patient_num = int(patient_id.split('_')[1])
    timeline = []
    
    # Generate timeline events
    events = [
        {
            'date': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
            'type': 'lab_result',
            'title': 'Lab Results',
            'description': f'HbA1c: {7.2 + (patient_num % 10) * 0.2:.1f}%, eGFR: {85 - patient_num % 30}',
            'provider': f'Dr. Smith {patient_num % 10}',
            'severity': 'medium' if 7.2 + (patient_num % 10) * 0.2 > 8.0 else 'low'
        },
        {
            'date': (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d'),
            'type': 'visit',
            'title': 'Office Visit',
            'description': 'Routine diabetes management visit',
            'provider': f'Dr. Smith {patient_num % 10}',
            'severity': 'low'
        },
        {
            'date': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
            'type': 'medication',
            'title': 'Medication Change',
            'description': 'Metformin dosage increased to 1000mg BID',
            'provider': f'Dr. Smith {patient_num % 10}',
            'severity': 'low'
        }
    ]
    
    if patient_num % 10 == 0:
        events.insert(0, {
            'date': (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d'),
            'type': 'emergency',
            'title': 'ED Visit',
            'description': 'Hyperglycemic episode, discharged home',
            'provider': 'Emergency Department',
            'severity': 'high'
        })
    
    return sorted(events, key=lambda x: x['date'], reverse=True)

def _get_mock_risk_explanation(patient_id: str) -> Dict[str, Any]:
    """Generate mock SHAP explanation data."""
    patient_num = int(patient_id.split('_')[1])
    
    shap_values = {
        'hba1c_trend': 0.15 + (patient_num % 10) * 0.01,
        'medication_adherence': -0.10 - (patient_num % 5) * 0.02,
        'age': 0.08 + (patient_num % 20) * 0.002,
        'recent_hospitalization': 0.12 if patient_num % 15 == 0 else 0.0,
        'comorbidity_count': 0.06 + (patient_num % 4) * 0.02,
        'social_determinants': -0.03 + (patient_num % 6) * 0.01
    }
    
    explanation = {
        'risk_score': 0.3 + (patient_num % 7) * 0.1,
        'shap_values': shap_values,
        'top_risk_factors': [
            {'factor': 'HbA1c Worsening Trend', 'contribution': '+15.2%', 'modifiable': True},
            {'factor': 'Poor Medication Adherence', 'contribution': '+12.4%', 'modifiable': True},
            {'factor': 'Recent Hospitalization', 'contribution': '+8.7%', 'modifiable': False},
            {'factor': 'Advanced Age', 'contribution': '+6.1%', 'modifiable': False}
        ],
        'protective_factors': [
            {'factor': 'Regular Specialist Care', 'contribution': '-3.2%'},
            {'factor': 'Family Support', 'contribution': '-2.1%'}
        ],
        'what_if_scenarios': {
            'improved_adherence': {
                'current_risk': 0.73,
                'projected_risk': 0.52,
                'reduction': 0.21,
                'intervention': 'Medication adherence program'
            },
            'hba1c_control': {
                'current_risk': 0.73,
                'projected_risk': 0.48,
                'reduction': 0.25,
                'intervention': 'Intensive diabetes management'
            }
        }
    }
    
    return explanation

def _get_mock_interventions(patient_id: str) -> Dict[str, Any]:
    """Generate mock intervention recommendations."""
    patient_num = int(patient_id.split('_')[1])
    
    interventions = {
        'immediate_actions': [
            {
                'priority': 'high',
                'action': 'Endocrinology referral',
                'timeline': 'Within 48 hours',
                'reason': 'Uncontrolled diabetes with HbA1c >9%'
            },
            {
                'priority': 'medium',
                'action': 'Medication adherence counseling',
                'timeline': 'Within 1 week',
                'reason': 'Adherence rate below 70%'
            }
        ],
        'care_plan': {
            'goals': [
                'Achieve HbA1c <8% within 3 months',
                'Improve medication adherence to >80%',
                'Prevent hospitalizations'
            ],
            'interventions': [
                {
                    'type': 'clinical',
                    'description': 'Bi-weekly diabetes management visits',
                    'duration': '3 months'
                },
                {
                    'type': 'education',
                    'description': 'Diabetes self-management education',
                    'duration': 'One-time, 4-hour session'
                },
                {
                    'type': 'technology',
                    'description': 'Continuous glucose monitoring',
                    'duration': 'Ongoing'
                }
            ],
            'monitoring': [
                'Weekly glucose logs review',
                'Monthly HbA1c monitoring',
                'Quarterly comprehensive metabolic panel'
            ]
        },
        'predicted_outcomes': {
            'with_intervention': {
                'risk_reduction': '40%',
                'hba1c_target_achievement': '75%',
                'hospitalization_prevention': '60%'
            },
            'without_intervention': {
                'risk_increase': '25%',
                'complication_probability': '45%'
            }
        }
    }
    
    return interventions

def _calculate_risk_level(risk_score: float) -> str:
    """Calculate risk level from risk score."""
    if risk_score >= 0.85:
        return 'Critical'
    elif risk_score >= 0.7:
        return 'High'
    elif risk_score >= 0.3:
        return 'Medium'
    else:
        return 'Low'

def _get_mock_conditions(patient_num: int) -> List[Dict[str, Any]]:
    """Generate mock medical conditions."""
    primary = ['Diabetes', 'Heart Failure', 'Obesity'][patient_num % 3]
    conditions = [{'name': primary, 'primary': True, 'diagnosed_date': '2020-01-01', 'icd10': 'E11.9'}]
    
    # Add comorbidities
    if patient_num % 2 == 0:
        conditions.append({'name': 'Hypertension', 'primary': False, 'diagnosed_date': '2019-05-15', 'icd10': 'I10'})
    if patient_num % 3 == 0:
        conditions.append({'name': 'Dyslipidemia', 'primary': False, 'diagnosed_date': '2021-03-10', 'icd10': 'E78.5'})
    if patient_num % 5 == 0:
        conditions.append({'name': 'Chronic Kidney Disease', 'primary': False, 'diagnosed_date': '2022-08-20', 'icd10': 'N18.3'})
    
    return conditions

def _get_mock_vital_signs(patient_num: int) -> List[Dict[str, Any]]:
    """Generate mock vital signs data."""
    vitals = []
    base_date = datetime.now() - timedelta(days=90)
    
    for i in range(30):  # 30 measurements over 90 days
        date = base_date + timedelta(days=i * 3)
        
        # Generate realistic vital signs with some variation
        vital = {
            'timestamp': date.isoformat(),
            'systolic_bp': 120 + (patient_num % 40) + (i % 10) - 5,
            'diastolic_bp': 80 + (patient_num % 20) + (i % 5) - 2,
            'heart_rate': 70 + (patient_num % 30) + (i % 8) - 4,
            'temperature_f': 98.6 + (i % 3) * 0.2 - 0.2,
            'oxygen_saturation': 98 + (i % 3) - 1,
            'respiratory_rate': 16 + (i % 4) - 2,
            'weight_kg': 80 + (patient_num % 40) + (i % 5) - 2,
            'height_cm': 170 + (patient_num % 30)
        }
        vitals.append(vital)
    
    return vitals

def _get_mock_lab_results(patient_num: int) -> List[Dict[str, Any]]:
    """Generate mock laboratory results."""
    labs = []
    base_date = datetime.now() - timedelta(days=180)
    
    for i in range(6):  # 6 lab results over 180 days
        date = base_date + timedelta(days=i * 30)
        
        lab = {
            'test_date': date.strftime('%Y-%m-%d'),
            'hba1c': round(7.0 + (patient_num % 4) * 0.5 + (i * 0.1), 1),
            'glucose_fasting': 100 + (patient_num % 50) + (i * 5),
            'egfr': max(15, 90 - (patient_num % 30) - (i * 2)),
            'bnp': 50 + (patient_num % 200) + (i * 10),
            'total_cholesterol': 180 + (patient_num % 60) + (i * 5),
            'ldl_cholesterol': 100 + (patient_num % 40) + (i * 3),
            'hdl_cholesterol': max(30, 50 - (patient_num % 20) + (i % 3)),
            'triglycerides': 150 + (patient_num % 100) + (i * 8),
            'creatinine': round(0.8 + (patient_num % 10) * 0.1 + (i * 0.05), 2)
        }
        labs.append(lab)
    
    return labs

def _get_mock_medications(patient_num: int) -> List[Dict[str, Any]]:
    """Generate mock medication list."""
    meds = []
    
    # Primary condition medications
    primary_condition = ['Diabetes', 'Heart Failure', 'Obesity'][patient_num % 3]
    
    if primary_condition == 'Diabetes':
        meds.extend([
            {
                'medication_name': 'Metformin',
                'dosage': '1000 mg',
                'frequency': 'twice daily',
                'indication': 'Type 2 Diabetes',
                'adherence_rate': max(0.3, 0.85 - (patient_num % 10) * 0.05),
                'last_filled': (datetime.now() - timedelta(days=15)).strftime('%Y-%m-%d'),
                'next_due': (datetime.now() + timedelta(days=15)).strftime('%Y-%m-%d'),
                'prescriber': f'Dr. Smith {patient_num % 10}',
                'interactions': [],
                'side_effects': ['GI upset'] if patient_num % 4 == 0 else []
            },
            {
                'medication_name': 'Lisinopril',
                'dosage': '10 mg',
                'frequency': 'once daily',
                'indication': 'Hypertension',
                'adherence_rate': max(0.4, 0.90 - (patient_num % 8) * 0.05),
                'last_filled': (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d'),
                'next_due': (datetime.now() + timedelta(days=20)).strftime('%Y-%m-%d'),
                'prescriber': f'Dr. Smith {patient_num % 10}',
                'interactions': [],
                'side_effects': []
            }
        ])
    
    elif primary_condition == 'Heart Failure':
        meds.extend([
            {
                'medication_name': 'Lisinopril',
                'dosage': '20 mg',
                'frequency': 'once daily',
                'indication': 'Heart Failure',
                'adherence_rate': max(0.3, 0.80 - (patient_num % 12) * 0.03),
                'last_filled': (datetime.now() - timedelta(days=12)).strftime('%Y-%m-%d'),
                'next_due': (datetime.now() + timedelta(days=18)).strftime('%Y-%m-%d'),
                'prescriber': f'Dr. Cardio {patient_num % 5}',
                'interactions': [],
                'side_effects': []
            },
            {
                'medication_name': 'Metoprolol',
                'dosage': '50 mg',
                'frequency': 'twice daily',
                'indication': 'Heart Failure',
                'adherence_rate': max(0.4, 0.75 - (patient_num % 15) * 0.02),
                'last_filled': (datetime.now() - timedelta(days=8)).strftime('%Y-%m-%d'),
                'next_due': (datetime.now() + timedelta(days=22)).strftime('%Y-%m-%d'),
                'prescriber': f'Dr. Cardio {patient_num % 5}',
                'interactions': [],
                'side_effects': ['Fatigue'] if patient_num % 6 == 0 else []
            }
        ])
    
    elif primary_condition == 'Obesity':
        meds.extend([
            {
                'medication_name': 'Semaglutide',
                'dosage': '1.0 mg',
                'frequency': 'weekly',
                'indication': 'Weight Management',
                'adherence_rate': max(0.5, 0.85 - (patient_num % 8) * 0.04),
                'last_filled': (datetime.now() - timedelta(days=20)).strftime('%Y-%m-%d'),
                'next_due': (datetime.now() + timedelta(days=10)).strftime('%Y-%m-%d'),
                'prescriber': f'Dr. Endo {patient_num % 3}',
                'interactions': [],
                'side_effects': ['Nausea'] if patient_num % 5 == 0 else []
            }
        ])
    
    return meds

def _get_mock_adherence_data(patient_num: int) -> List[Dict[str, Any]]:
    """Generate mock medication adherence data."""
    adherence = []
    base_date = datetime.now() - timedelta(days=90)
    
    meds = ['Metformin', 'Lisinopril', 'Metoprolol']
    
    for i in range(30):  # 30 days of adherence data
        date = base_date + timedelta(days=i * 3)
        
        for med in meds[:2]:  # Use first 2 medications
            base_adherence = 0.85 - (patient_num % 10) * 0.05
            daily_variation = (i % 7) * 0.02 - 0.06
            adherence_rate = max(0.3, min(1.0, base_adherence + daily_variation))
            
            adherence.append({
                'date': date.strftime('%Y-%m-%d'),
                'medication_name': med,
                'adherence_rate': adherence_rate,
                'doses_prescribed': 2 if med == 'Metformin' else 1,
                'doses_taken': int(adherence_rate * (2 if med == 'Metformin' else 1))
            })
    
    return adherence

def _get_mock_risk_history(patient_num: int) -> List[Dict[str, Any]]:
    """Generate mock risk score history."""
    history = []
    base_date = datetime.now() - timedelta(days=180)
    base_risk = 0.3 + (patient_num % 7) * 0.1
    
    for i in range(12):  # 12 risk assessments over 180 days
        date = base_date + timedelta(days=i * 15)
        
        # Add some realistic variation to risk scores
        variation = (i % 4) * 0.02 - 0.03
        risk_score = max(0.05, min(0.95, base_risk + variation + (i * 0.01)))
        
        history.append({
            'date': date.isoformat(),
            'risk_score': round(risk_score, 3),
            'assessment_type': 'automated' if i % 3 != 0 else 'clinical_review'
        })
    
    return history

def _get_mock_allergies(patient_num: int) -> List[Dict[str, Any]]:
    """Generate mock allergy data."""
    allergies = []
    
    if patient_num % 8 == 0:
        allergies.append({
            'allergen': 'Penicillin',
            'reaction': 'Rash',
            'severity': 'Moderate'
        })
    
    if patient_num % 12 == 0:
        allergies.append({
            'allergen': 'Shellfish',
            'reaction': 'Anaphylaxis',
            'severity': 'Severe'
        })
    
    return allergies

def _get_mock_social_history(patient_num: int) -> Dict[str, Any]:
    """Generate mock social history."""
    return {
        'smoking_status': ['Never', 'Former', 'Current'][patient_num % 3],
        'alcohol_use': ['None', 'Social', 'Moderate'][patient_num % 3],
        'exercise_frequency': ['Sedentary', 'Light', 'Moderate', 'Active'][patient_num % 4],
        'diet_quality': ['Poor', 'Fair', 'Good'][patient_num % 3],
        'social_support': ['Limited', 'Moderate', 'Strong'][patient_num % 3],
        'employment_status': ['Employed', 'Retired', 'Unemployed', 'Disabled'][patient_num % 4],
        'education_level': ['High School', 'Some College', 'Bachelor\'s', 'Graduate'][patient_num % 4]
    }

def _get_mock_specialists(patient_num: int) -> List[Dict[str, str]]:
    """Generate mock specialist list."""
    specialists = []
    
    primary_condition = ['Diabetes', 'Heart Failure', 'Obesity'][patient_num % 3]
    
    if primary_condition == 'Diabetes':
        specialists.append({
            'specialty': 'Endocrinology',
            'provider': f'Dr. Endo {patient_num % 5}',
            'last_visit': (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        })
        if patient_num % 4 == 0:
            specialists.append({
                'specialty': 'Ophthalmology', 
                'provider': f'Dr. Eye {patient_num % 3}',
                'last_visit': (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
            })
    
    elif primary_condition == 'Heart Failure':
        specialists.append({
            'specialty': 'Cardiology',
            'provider': f'Dr. Cardio {patient_num % 4}',
            'last_visit': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        })
    
    return specialists

def _get_mock_recent_activity(patient_num: int) -> List[Dict[str, Any]]:
    """Generate mock recent activity."""
    activities = [
        {
            'date': (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d'),
            'activity': 'Risk score updated',
            'details': 'Automated risk assessment completed',
            'user': 'AI System'
        },
        {
            'date': (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'),
            'activity': 'Lab results reviewed',
            'details': 'HbA1c results flagged for attention',
            'user': f'Dr. Smith {patient_num % 10}'
        }
    ]
    
    if patient_num % 10 == 0:
        activities.insert(0, {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'activity': 'Alert generated',
            'details': 'High risk alert triggered',
            'user': 'AI System'
        })
    
    return activities

def _generate_risk_summary(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate risk summary for report."""
    return {
        'current_risk_score': patient_data['risk_score'],
        'risk_level': patient_data['risk_level'],
        'primary_risk_factors': [
            'Poor glycemic control (HbA1c trend)',
            'Suboptimal medication adherence',
            'Advanced age factors'
        ],
        'trend': 'Increasing' if patient_data['risk_score'] > 0.6 else 'Stable',
        'last_assessment': datetime.now().strftime('%Y-%m-%d')
    }

def _generate_clinical_summary(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate clinical summary for report."""
    return {
        'primary_diagnoses': [condition['name'] for condition in patient_data['conditions'] if condition['primary']],
        'active_medications': len(patient_data['medications']),
        'recent_hospitalizations': 1 if patient_data.get('patient_id', '').endswith(('0', '15', '30')) else 0,
        'care_gaps': ['Annual eye exam overdue', 'Nephrology referral pending'] if int(patient_data['patient_id'].split('_')[1]) % 5 == 0 else [],
        'vitals_status': 'Stable' if patient_data['risk_level'] in ['Low', 'Medium'] else 'Concerning'
    }
