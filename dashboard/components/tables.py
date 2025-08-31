"""
Table Generation System for structured clinical data display.
Creates sortable, filterable tables for patient cohorts and clinical data.
"""

import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
import pandas as pd

class TableGenerator:
    """Generates clinical tables with sorting, filtering, and pagination."""
    
    def __init__(self):
        self.default_page_size = 20
        self.risk_level_colors = {
            'Critical': '#DC3545',
            'High': '#FD7E14',
            'Medium': '#FFC107', 
            'Low': '#28A745'
        }
    
    def generate_patient_cohort_table(self, patients: List[Dict[str, Any]], 
                                    filters: Optional[Dict[str, Any]] = None,
                                    sort_by: str = 'risk_score',
                                    sort_order: str = 'desc',
                                    page: int = 1,
                                    page_size: int = None) -> Dict[str, Any]:
        """Generate comprehensive patient cohort table with filtering and pagination."""
        
        if page_size is None:
            page_size = self.default_page_size
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(patients)
        
        # Apply filters
        if filters:
            df = self._apply_filters(df, filters)
        
        # Sort data
        ascending = sort_order.lower() == 'asc'
        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=ascending)
        
        # Calculate pagination
        total_records = len(df)
        total_pages = (total_records + page_size - 1) // page_size
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        # Get page data
        page_data = df.iloc[start_idx:end_idx]
        
        # Generate HTML table
        table_html = self._generate_cohort_table_html(page_data)
        
        # Generate summary statistics
        summary_stats = self._calculate_cohort_summary(df)
        
        return {
            'table_html': table_html,
            'pagination': {
                'current_page': page,
                'total_pages': total_pages,
                'total_records': total_records,
                'page_size': page_size,
                'has_previous': page > 1,
                'has_next': page < total_pages
            },
            'summary_stats': summary_stats,
            'filters_applied': filters or {}
        }
    
    def _apply_filters(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters to patient DataFrame."""
        filtered_df = df.copy()
        
        # Text search filter
        if 'search' in filters and filters['search']:
            search_term = filters['search'].lower()
            mask = (
                filtered_df['patient_id'].str.lower().str.contains(search_term, na=False) |
                filtered_df['primary_condition'].str.lower().str.contains(search_term, na=False) |
                filtered_df['risk_level'].str.lower().str.contains(search_term, na=False)
            )
            filtered_df = filtered_df[mask]
        
        # Risk level filter
        if 'risk_level' in filters and filters['risk_level']:
            filtered_df = filtered_df[filtered_df['risk_level'] == filters['risk_level']]
        
        # Age range filter
        if 'age_min' in filters and filters['age_min']:
            filtered_df = filtered_df[filtered_df['age'] >= int(filters['age_min'])]
        if 'age_max' in filters and filters['age_max']:
            filtered_df = filtered_df[filtered_df['age'] <= int(filters['age_max'])]
        
        # Gender filter
        if 'gender' in filters and filters['gender']:
            filtered_df = filtered_df[filtered_df['gender'] == filters['gender']]
        
        # Condition filter
        if 'condition' in filters and filters['condition']:
            filtered_df = filtered_df[filtered_df['primary_condition'] == filters['condition']]
        
        # Date range filter
        if 'date_from' in filters and filters['date_from']:
            date_from = pd.to_datetime(filters['date_from'])
            filtered_df = filtered_df[pd.to_datetime(filtered_df['last_visit']) >= date_from]
        
        if 'date_to' in filters and filters['date_to']:
            date_to = pd.to_datetime(filters['date_to'])
            filtered_df = filtered_df[pd.to_datetime(filtered_df['last_visit']) <= date_to]
        
        return filtered_df
    
    def _generate_cohort_table_html(self, df: pd.DataFrame) -> str:
        """Generate HTML table for patient cohort data."""
        if df.empty:
            return '<div class="alert alert-info">No patients found matching the current filters.</div>'
        
        html = '''
        <div class="table-responsive">
            <table class="table table-striped table-hover" id="cohortTable">
                <thead class="table-dark">
                    <tr>
                        <th><input type="checkbox" id="selectAll"></th>
                        <th>Patient ID</th>
                        <th>Name</th>
                        <th>Age</th>
                        <th>Gender</th>
                        <th>Primary Condition</th>
                        <th>Risk Score</th>
                        <th>Risk Level</th>
                        <th>Last Visit</th>
                        <th>Next Follow-up</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
        '''
        
        for _, patient in df.iterrows():
            risk_color = self.risk_level_colors.get(patient.get('risk_level', 'Low'), '#6C757D')
            
            html += f'''
                <tr data-patient-id="{patient.get('patient_id', 'N/A')}">
                    <td><input type="checkbox" class="patient-checkbox" value="{patient.get('patient_id', '')}"></td>
                    <td><a href="/patient/{patient.get('patient_id', '')}" class="text-decoration-none">{patient.get('patient_id', 'N/A')}</a></td>
                    <td>{patient.get('name', 'N/A')}</td>
                    <td>{patient.get('age', 'N/A')}</td>
                    <td>{patient.get('gender', 'N/A')}</td>
                    <td>{patient.get('primary_condition', 'N/A')}</td>
                    <td>
                        <div class="progress" style="height: 20px;">
                            <div class="progress-bar" role="progressbar" 
                                 style="width: {patient.get('risk_score', 0) * 100:.1f}%; background-color: {risk_color};"
                                 aria-valuenow="{patient.get('risk_score', 0) * 100:.1f}" 
                                 aria-valuemin="0" aria-valuemax="100">
                                {patient.get('risk_score', 0) * 100:.1f}%
                            </div>
                        </div>
                    </td>
                    <td><span class="badge" style="background-color: {risk_color};">{patient.get('risk_level', 'N/A')}</span></td>
                    <td>{patient.get('last_visit', 'N/A')}</td>
                    <td>{patient.get('next_followup', 'N/A')}</td>
                    <td>
                        <div class="btn-group btn-group-sm" role="group">
                            <a href="/patient/{patient.get('patient_id', '')}" class="btn btn-outline-primary btn-sm">View</a>
                            <button type="button" class="btn btn-outline-secondary btn-sm" onclick="generateReport('{patient.get('patient_id', '')}')">Report</button>
                            <button type="button" class="btn btn-outline-warning btn-sm" onclick="scheduleFollowup('{patient.get('patient_id', '')}')">Schedule</button>
                        </div>
                    </td>
                </tr>
            '''
        
        html += '''
                </tbody>
            </table>
        </div>
        
        <script>
        document.getElementById('selectAll').addEventListener('change', function() {
            const checkboxes = document.querySelectorAll('.patient-checkbox');
            checkboxes.forEach(checkbox => checkbox.checked = this.checked);
        });
        
        function generateReport(patientId) {
            window.open('/patient/' + patientId + '/report', '_blank');
        }
        
        function scheduleFollowup(patientId) {
            // Implement scheduling logic
            alert('Scheduling follow-up for patient: ' + patientId);
        }
        </script>
        '''
        
        return html
    
    def _calculate_cohort_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics for the patient cohort."""
        if df.empty:
            return {}
        
        summary = {
            'total_patients': len(df),
            'average_age': round(df['age'].mean(), 1) if 'age' in df.columns else 0,
            'gender_distribution': df['gender'].value_counts().to_dict() if 'gender' in df.columns else {},
            'risk_level_distribution': df['risk_level'].value_counts().to_dict() if 'risk_level' in df.columns else {},
            'condition_distribution': df['primary_condition'].value_counts().to_dict() if 'primary_condition' in df.columns else {},
            'average_risk_score': round(df['risk_score'].mean() * 100, 1) if 'risk_score' in df.columns else 0,
            'high_risk_count': len(df[df['risk_level'].isin(['High', 'Critical'])]) if 'risk_level' in df.columns else 0
        }
        
        return summary
    
    def generate_lab_results_table(self, patient_id: str, 
                                 lab_results: List[Dict[str, Any]]) -> str:
        """Generate table for laboratory results with trend indicators."""
        if not lab_results:
            return '<div class="alert alert-info">No laboratory results available.</div>'
        
        # Sort by date (most recent first)
        sorted_results = sorted(lab_results, key=lambda x: x.get('test_date', ''), reverse=True)
        
        html = '''
        <div class="table-responsive">
            <table class="table table-striped table-sm">
                <thead class="table-dark">
                    <tr>
                        <th>Test Date</th>
                        <th>HbA1c (%)</th>
                        <th>Glucose (mg/dL)</th>
                        <th>eGFR (mL/min)</th>
                        <th>BNP (pg/mL)</th>
                        <th>Total Chol (mg/dL)</th>
                        <th>Trend</th>
                        <th>Alerts</th>
                    </tr>
                </thead>
                <tbody>
        '''
        
        for i, result in enumerate(sorted_results):
            # Determine trend arrows
            trend_indicators = self._calculate_lab_trends(sorted_results, i)
            
            # Check for critical values
            alerts = self._check_lab_alerts(result)
            
            html += f'''
                <tr>
                    <td>{result.get('test_date', 'N/A')}</td>
                    <td>{result.get('hba1c', 'N/A')} {trend_indicators.get('hba1c', '')}</td>
                    <td>{result.get('glucose_fasting', 'N/A')} {trend_indicators.get('glucose', '')}</td>
                    <td>{result.get('egfr', 'N/A')} {trend_indicators.get('egfr', '')}</td>
                    <td>{result.get('bnp', 'N/A')} {trend_indicators.get('bnp', '')}</td>
                    <td>{result.get('total_cholesterol', 'N/A')} {trend_indicators.get('cholesterol', '')}</td>
                    <td>{''.join(trend_indicators.values())}</td>
                    <td>{''.join(alerts)}</td>
                </tr>
            '''
        
        html += '''
                </tbody>
            </table>
        </div>
        '''
        
        return html
    
    def _calculate_lab_trends(self, results: List[Dict[str, Any]], 
                            current_index: int) -> Dict[str, str]:
        """Calculate trend indicators for lab values."""
        trends = {}
        
        if current_index >= len(results) - 1:
            return trends  # No previous result to compare
        
        current = results[current_index]
        previous = results[current_index + 1]
        
        lab_fields = ['hba1c', 'glucose_fasting', 'egfr', 'bnp', 'total_cholesterol']
        
        for field in lab_fields:
            if field in current and field in previous:
                try:
                    current_val = float(current[field])
                    previous_val = float(previous[field])
                    
                    if current_val > previous_val * 1.05:  # 5% increase
                        trends[field.replace('_fasting', '')] = '<i class="fas fa-arrow-up text-danger"></i>'
                    elif current_val < previous_val * 0.95:  # 5% decrease
                        if field == 'egfr':  # For eGFR, decrease is bad
                            trends[field] = '<i class="fas fa-arrow-down text-danger"></i>'
                        else:  # For others, decrease might be good
                            trends[field.replace('_fasting', '')] = '<i class="fas fa-arrow-down text-success"></i>'
                    else:
                        trends[field.replace('_fasting', '')] = '<i class="fas fa-minus text-muted"></i>'
                except (ValueError, TypeError):
                    trends[field.replace('_fasting', '')] = ''
        
        return trends
    
    def _check_lab_alerts(self, result: Dict[str, Any]) -> List[str]:
        """Check for critical lab values and generate alerts."""
        alerts = []
        
        # HbA1c alerts
        if 'hba1c' in result:
            try:
                hba1c = float(result['hba1c'])
                if hba1c >= 10.0:
                    alerts.append('<span class="badge bg-danger">Critical HbA1c</span>')
                elif hba1c >= 9.0:
                    alerts.append('<span class="badge bg-warning">High HbA1c</span>')
            except (ValueError, TypeError):
                pass
        
        # eGFR alerts
        if 'egfr' in result:
            try:
                egfr = float(result['egfr'])
                if egfr < 30:
                    alerts.append('<span class="badge bg-danger">Severe CKD</span>')
                elif egfr < 45:
                    alerts.append('<span class="badge bg-warning">Moderate CKD</span>')
            except (ValueError, TypeError):
                pass
        
        # BNP alerts
        if 'bnp' in result:
            try:
                bnp = float(result['bnp'])
                if bnp >= 400:
                    alerts.append('<span class="badge bg-danger">High BNP</span>')
            except (ValueError, TypeError):
                pass
        
        return alerts
    
    def generate_medication_table(self, patient_id: str,
                                medications: List[Dict[str, Any]]) -> str:
        """Generate table for patient medications with adherence tracking."""
        if not medications:
            return '<div class="alert alert-info">No medications on record.</div>'
        
        html = '''
        <div class="table-responsive">
            <table class="table table-striped">
                <thead class="table-dark">
                    <tr>
                        <th>Medication</th>
                        <th>Dosage</th>
                        <th>Frequency</th>
                        <th>Indication</th>
                        <th>Adherence</th>
                        <th>Last Filled</th>
                        <th>Next Due</th>
                        <th>Alerts</th>
                    </tr>
                </thead>
                <tbody>
        '''
        
        for med in medications:
            adherence_rate = med.get('adherence_rate', 0)
            adherence_color = self._get_adherence_color(adherence_rate)
            
            html += f'''
                <tr>
                    <td><strong>{med.get('medication_name', 'N/A')}</strong></td>
                    <td>{med.get('dosage', 'N/A')}</td>
                    <td>{med.get('frequency', 'N/A')}</td>
                    <td>{med.get('indication', 'N/A')}</td>
                    <td>
                        <div class="progress" style="height: 20px;">
                            <div class="progress-bar" role="progressbar" 
                                 style="width: {adherence_rate * 100:.0f}%; background-color: {adherence_color};"
                                 aria-valuenow="{adherence_rate * 100:.0f}" 
                                 aria-valuemin="0" aria-valuemax="100">
                                {adherence_rate * 100:.0f}%
                            </div>
                        </div>
                    </td>
                    <td>{med.get('last_filled', 'N/A')}</td>
                    <td>{med.get('next_due', 'N/A')}</td>
                    <td>{self._generate_medication_alerts(med)}</td>
                </tr>
            '''
        
        html += '''
                </tbody>
            </table>
        </div>
        '''
        
        return html
    
    def _get_adherence_color(self, adherence_rate: float) -> str:
        """Get color code based on adherence rate."""
        if adherence_rate >= 0.8:
            return '#28A745'  # Green
        elif adherence_rate >= 0.6:
            return '#FFC107'  # Yellow
        else:
            return '#DC3545'  # Red
    
    def _generate_medication_alerts(self, medication: Dict[str, Any]) -> str:
        """Generate alerts for medication issues."""
        alerts = []
        
        adherence_rate = medication.get('adherence_rate', 1.0)
        if adherence_rate < 0.6:
            alerts.append('<span class="badge bg-danger">Poor Adherence</span>')
        elif adherence_rate < 0.8:
            alerts.append('<span class="badge bg-warning">Suboptimal Adherence</span>')
        
        # Check for drug interactions (simplified)
        if medication.get('interactions', []):
            alerts.append('<span class="badge bg-info">Interactions</span>')
        
        # Check for side effects
        if medication.get('side_effects', []):
            alerts.append('<span class="badge bg-secondary">Side Effects</span>')
        
        return ' '.join(alerts)
    
    def export_table_to_csv(self, data: List[Dict[str, Any]], 
                          filename: str = None) -> str:
        """Export table data to CSV format."""
        if not filename:
            filename = f"clinical_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        df = pd.DataFrame(data)
        csv_content = df.to_csv(index=False)
        
        return csv_content
    
    def generate_summary_cards(self, summary_data: Dict[str, Any]) -> str:
        """Generate summary cards for dashboard overview."""
        html = '<div class="row mb-4">'
        
        cards = [
            {
                'title': 'Total Patients',
                'value': summary_data.get('total_patients', 0),
                'icon': 'fas fa-users',
                'color': 'primary'
            },
            {
                'title': 'High Risk',
                'value': summary_data.get('high_risk_count', 0),
                'icon': 'fas fa-exclamation-triangle',
                'color': 'danger'
            },
            {
                'title': 'Avg Risk Score',
                'value': f"{summary_data.get('average_risk_score', 0):.1f}%",
                'icon': 'fas fa-chart-line',
                'color': 'warning'
            },
            {
                'title': 'Avg Age',
                'value': f"{summary_data.get('average_age', 0):.0f} years",
                'icon': 'fas fa-birthday-cake',
                'color': 'info'
            }
        ]
        
        for card in cards:
            html += f'''
            <div class="col-md-3">
                <div class="card text-white bg-{card['color']} mb-3">
                    <div class="card-body">
                        <div class="d-flex justify-content-between">
                            <div>
                                <h4 class="card-title">{card['value']}</h4>
                                <p class="card-text">{card['title']}</p>
                            </div>
                            <div class="align-self-center">
                                <i class="{card['icon']} fa-2x"></i>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            '''
        
        html += '</div>'
        return html
