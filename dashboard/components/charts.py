"""
Chart Generation System for clinical data visualization.
Creates interactive charts for patient data, risk trends, and population analytics.
"""

import json
import base64
import io
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import pandas as pd

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ChartGenerator:
    """Generates clinical charts and visualizations for the dashboard."""
    
    def __init__(self):
        self.default_figsize = (12, 6)
        self.color_palette = {
            'critical': '#DC3545',
            'high': '#FD7E14', 
            'medium': '#FFC107',
            'low': '#28A745',
            'primary': '#007BFF',
            'secondary': '#6C757D'
        }
    
    def _save_plot_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        
        plot_url = base64.b64encode(plot_data).decode()
        return f"data:image/png;base64,{plot_url}"
    
    def generate_risk_distribution_chart(self, risk_data: List[Dict[str, Any]]) -> str:
        """Generate pie chart showing risk level distribution."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Count patients by risk level
        risk_counts = {}
        for patient in risk_data:
            risk_level = patient.get('risk_level', 'Unknown')
            risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
        
        labels = list(risk_counts.keys())
        sizes = list(risk_counts.values())
        colors = [self.color_palette.get(label.lower(), '#999999') for label in labels]
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                         autopct='%1.1f%%', startangle=90)
        
        ax.set_title('Patient Risk Distribution', fontsize=16, fontweight='bold')
        
        # Enhance text formatting
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        return self._save_plot_to_base64(fig)
    
    def generate_risk_trend_chart(self, patient_id: str, 
                                risk_history: List[Dict[str, Any]]) -> str:
        """Generate line chart showing patient risk over time."""
        fig, ax = plt.subplots(figsize=self.default_figsize)
        
        # Prepare data
        dates = [datetime.fromisoformat(entry['date']) for entry in risk_history]
        risk_scores = [entry['risk_score'] for entry in risk_history]
        
        # Plot risk trend
        ax.plot(dates, risk_scores, marker='o', linewidth=2, markersize=6,
                color=self.color_palette['primary'])
        
        # Add risk level zones
        ax.axhspan(0.7, 1.0, alpha=0.2, color=self.color_palette['critical'], label='Critical Risk')
        ax.axhspan(0.3, 0.7, alpha=0.2, color=self.color_palette['medium'], label='Moderate Risk')
        ax.axhspan(0.0, 0.3, alpha=0.2, color=self.color_palette['low'], label='Low Risk')
        
        ax.set_title(f'Risk Trend for Patient {patient_id}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Risk Score')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return self._save_plot_to_base64(fig)
    
    def generate_feature_importance_chart(self, 
                                        feature_importance: Dict[str, float]) -> str:
        """Generate horizontal bar chart for feature importance."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        features, importance = zip(*sorted_features[:15])  # Top 15 features
        
        # Create horizontal bar chart
        bars = ax.barh(range(len(features)), importance, 
                      color=self.color_palette['primary'], alpha=0.8)
        
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels([f.replace('_', ' ').title() for f in features])
        ax.set_xlabel('Importance Score')
        ax.set_title('Top Risk Factors - Feature Importance', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{width:.3f}', ha='left', va='center', fontweight='bold')
        
        ax.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        return self._save_plot_to_base64(fig)
    
    def generate_vital_signs_chart(self, patient_id: str,
                                 vital_signs_data: List[Dict[str, Any]]) -> str:
        """Generate multi-line chart for vital signs over time."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Convert data to DataFrame
        df = pd.DataFrame(vital_signs_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Blood Pressure
        ax1.plot(df['timestamp'], df['systolic_bp'], 'o-', label='Systolic', color='red')
        ax1.plot(df['timestamp'], df['diastolic_bp'], 'o-', label='Diastolic', color='blue')
        ax1.axhline(y=140, color='red', linestyle='--', alpha=0.5, label='High BP')
        ax1.set_title('Blood Pressure Trend')
        ax1.set_ylabel('mmHg')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Heart Rate
        ax2.plot(df['timestamp'], df['heart_rate'], 'o-', color='green')
        ax2.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Tachycardia')
        ax2.axhline(y=60, color='orange', linestyle='--', alpha=0.5, label='Bradycardia')
        ax2.set_title('Heart Rate Trend')
        ax2.set_ylabel('BPM')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Temperature
        ax3.plot(df['timestamp'], df['temperature_f'], 'o-', color='orange')
        ax3.axhline(y=100.4, color='red', linestyle='--', alpha=0.5, label='Fever')
        ax3.set_title('Temperature Trend')
        ax3.set_ylabel('°F')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Oxygen Saturation
        ax4.plot(df['timestamp'], df['oxygen_saturation'], 'o-', color='purple')
        ax4.axhline(y=95, color='red', linestyle='--', alpha=0.5, label='Low O2')
        ax4.set_title('Oxygen Saturation Trend')
        ax4.set_ylabel('%')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Format all x-axes
        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.tick_params(axis='x', rotation=45)
        
        plt.suptitle(f'Vital Signs Trends - Patient {patient_id}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        return self._save_plot_to_base64(fig)
    
    def generate_lab_results_chart(self, patient_id: str,
                                 lab_data: List[Dict[str, Any]]) -> str:
        """Generate chart for key laboratory results over time."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Convert to DataFrame
        df = pd.DataFrame(lab_data)
        df['test_date'] = pd.to_datetime(df['test_date'])
        df = df.sort_values('test_date')
        
        # HbA1c
        if 'hba1c' in df.columns:
            ax1.plot(df['test_date'], df['hba1c'], 'o-', color='red', linewidth=2)
            ax1.axhline(y=7.0, color='orange', linestyle='--', alpha=0.7, label='ADA Target')
            ax1.axhline(y=9.0, color='red', linestyle='--', alpha=0.7, label='Poor Control')
            ax1.set_title('HbA1c Trend')
            ax1.set_ylabel('HbA1c (%)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # eGFR (Kidney Function)
        if 'egfr' in df.columns:
            ax2.plot(df['test_date'], df['egfr'], 'o-', color='blue', linewidth=2)
            ax2.axhline(y=60, color='orange', linestyle='--', alpha=0.7, label='CKD Stage 3')
            ax2.axhline(y=30, color='red', linestyle='--', alpha=0.7, label='CKD Stage 4')
            ax2.set_title('Kidney Function (eGFR)')
            ax2.set_ylabel('eGFR (mL/min/1.73m²)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # BNP (Heart Failure Marker)
        if 'bnp' in df.columns:
            ax3.plot(df['test_date'], df['bnp'], 'o-', color='green', linewidth=2)
            ax3.axhline(y=100, color='orange', linestyle='--', alpha=0.7, label='Elevated')
            ax3.axhline(y=400, color='red', linestyle='--', alpha=0.7, label='High')
            ax3.set_title('BNP Trend')
            ax3.set_ylabel('BNP (pg/mL)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Cholesterol
        if 'total_cholesterol' in df.columns:
            ax4.plot(df['test_date'], df['total_cholesterol'], 'o-', color='purple', linewidth=2)
            ax4.axhline(y=200, color='orange', linestyle='--', alpha=0.7, label='Borderline')
            ax4.axhline(y=240, color='red', linestyle='--', alpha=0.7, label='High')
            ax4.set_title('Total Cholesterol')
            ax4.set_ylabel('mg/dL')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # Format x-axes
        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.tick_params(axis='x', rotation=45)
        
        plt.suptitle(f'Laboratory Results - Patient {patient_id}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        return self._save_plot_to_base64(fig)
    
    def generate_population_analytics_chart(self, 
                                          population_data: Dict[str, Any]) -> str:
        """Generate comprehensive population analytics dashboard."""
        fig = plt.figure(figsize=(16, 12))
        
        # Create subplot grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Age Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        age_groups = population_data.get('age_distribution', {})
        ax1.bar(age_groups.keys(), age_groups.values(), color=self.color_palette['primary'])
        ax1.set_title('Age Distribution')
        ax1.set_xlabel('Age Group')
        ax1.set_ylabel('Count')
        
        # Gender Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        gender_data = population_data.get('gender_distribution', {})
        ax2.pie(gender_data.values(), labels=gender_data.keys(), autopct='%1.1f%%')
        ax2.set_title('Gender Distribution')
        
        # Condition Prevalence
        ax3 = fig.add_subplot(gs[0, 2])
        conditions = population_data.get('condition_prevalence', {})
        ax3.barh(list(conditions.keys()), list(conditions.values()), 
                color=self.color_palette['secondary'])
        ax3.set_title('Condition Prevalence')
        ax3.set_xlabel('Percentage')
        
        # Risk Score Distribution
        ax4 = fig.add_subplot(gs[1, :])
        risk_scores = population_data.get('risk_scores', [])
        ax4.hist(risk_scores, bins=20, alpha=0.7, color=self.color_palette['primary'])
        ax4.axvline(x=0.3, color='green', linestyle='--', label='Low Risk Threshold')
        ax4.axvline(x=0.7, color='red', linestyle='--', label='High Risk Threshold')
        ax4.set_title('Population Risk Score Distribution')
        ax4.set_xlabel('Risk Score')
        ax4.set_ylabel('Number of Patients')
        ax4.legend()
        
        # Healthcare Utilization
        ax5 = fig.add_subplot(gs[2, 0])
        utilization = population_data.get('healthcare_utilization', {})
        ax5.bar(utilization.keys(), utilization.values(), color=self.color_palette['medium'])
        ax5.set_title('Healthcare Utilization')
        ax5.set_ylabel('Average per Patient')
        plt.setp(ax5.get_xticklabels(), rotation=45)
        
        # Medication Adherence
        ax6 = fig.add_subplot(gs[2, 1])
        adherence_data = population_data.get('adherence_distribution', {})
        ax6.pie(adherence_data.values(), labels=adherence_data.keys(), autopct='%1.1f%%')
        ax6.set_title('Medication Adherence')
        
        # Outcome Trends
        ax7 = fig.add_subplot(gs[2, 2])
        outcomes = population_data.get('outcome_trends', {})
        months = list(outcomes.keys())
        values = list(outcomes.values())
        ax7.plot(months, values, 'o-', linewidth=2, color=self.color_palette['primary'])
        ax7.set_title('Outcome Trends')
        ax7.set_xlabel('Month')
        ax7.set_ylabel('Adverse Events')
        plt.setp(ax7.get_xticklabels(), rotation=45)
        
        plt.suptitle('Population Health Analytics Dashboard', 
                    fontsize=18, fontweight='bold')
        plt.tight_layout()
        return self._save_plot_to_base64(fig)
    
    def generate_medication_adherence_chart(self, patient_id: str,
                                          adherence_data: List[Dict[str, Any]]) -> str:
        """Generate chart showing medication adherence patterns."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Overall adherence trend
        df = pd.DataFrame(adherence_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Calculate daily overall adherence
        daily_adherence = df.groupby('date')['adherence_rate'].mean()
        
        ax1.plot(daily_adherence.index, daily_adherence.values, 'o-', 
                linewidth=2, color=self.color_palette['primary'])
        ax1.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Target (80%)')
        ax1.axhline(y=0.6, color='orange', linestyle='--', alpha=0.7, label='Suboptimal (60%)')
        ax1.set_title(f'Overall Medication Adherence - Patient {patient_id}')
        ax1.set_ylabel('Adherence Rate')
        ax1.set_ylim(0, 1)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Individual medication adherence
        medications = df['medication_name'].unique()
        for i, med in enumerate(medications[:5]):  # Show top 5 medications
            med_data = df[df['medication_name'] == med]
            ax2.plot(med_data['date'], med_data['adherence_rate'], 
                    'o-', label=med, linewidth=2)
        
        ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.5)
        ax2.set_title('Individual Medication Adherence')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Adherence Rate')
        ax2.set_ylim(0, 1)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._save_plot_to_base64(fig)
    
    def create_interactive_chart_config(self, chart_type: str, 
                                      data: Dict[str, Any]) -> Dict[str, Any]:
        """Create configuration for interactive Chart.js charts."""
        base_config = {
            'type': chart_type,
            'options': {
                'responsive': True,
                'maintainAspectRatio': False,
                'plugins': {
                    'legend': {
                        'position': 'top'
                    }
                }
            }
        }
        
        if chart_type == 'line':
            base_config['options']['scales'] = {
                'x': {
                    'type': 'time',
                    'time': {
                        'unit': 'day'
                    }
                },
                'y': {
                    'beginAtZero': True
                }
            }
        
        base_config['data'] = data
        return base_config
