"""
Alert Management System for clinical notifications and risk warnings.
Handles real-time alerts for high-risk patients and system events.
"""

import json
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Any, Optional

class AlertSeverity(Enum):
    """Alert severity levels for clinical prioritization."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class AlertType(Enum):
    """Types of clinical alerts."""
    PATIENT_RISK = "patient_risk"
    CLINICAL_DETERIORATION = "clinical_deterioration"
    MEDICATION_ADHERENCE = "medication_adherence"
    LAB_CRITICAL = "lab_critical"
    SYSTEM_ERROR = "system_error"
    CARE_GAP = "care_gap"

class Alert:
    """Individual alert with clinical context."""
    
    def __init__(self, alert_id: str, alert_type: AlertType, severity: AlertSeverity,
                 title: str, message: str, patient_id: Optional[str] = None,
                 data: Optional[Dict[str, Any]] = None):
        self.alert_id = alert_id
        self.alert_type = alert_type
        self.severity = severity
        self.title = title
        self.message = message
        self.patient_id = patient_id
        self.data = data or {}
        self.created_at = datetime.now()
        self.acknowledged = False
        self.acknowledged_by = None
        self.acknowledged_at = None
        self.resolved = False
        self.resolved_at = None
    
    def acknowledge(self, user_id: str):
        """Mark alert as acknowledged by a healthcare provider."""
        self.acknowledged = True
        self.acknowledged_by = user_id
        self.acknowledged_at = datetime.now()
    
    def resolve(self):
        """Mark alert as resolved."""
        self.resolved = True
        self.resolved_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for JSON serialization."""
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type.value,
            'severity': self.severity.value,
            'title': self.title,
            'message': self.message,
            'patient_id': self.patient_id,
            'data': self.data,
            'created_at': self.created_at.isoformat(),
            'acknowledged': self.acknowledged,
            'acknowledged_by': self.acknowledged_by,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'resolved': self.resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None
        }

class AlertsManager:
    """Manages clinical alerts with filtering, prioritization, and notifications."""
    
    def __init__(self):
        self.active_alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
        self.alert_rules = self._initialize_alert_rules()
    
    def _initialize_alert_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize clinical alert rules and thresholds."""
        return {
            'high_risk_patient': {
                'threshold': 0.7,
                'severity': AlertSeverity.HIGH,
                'message_template': 'Patient {patient_id} has high deterioration risk ({risk_score:.1%})'
            },
            'critical_risk_patient': {
                'threshold': 0.85,
                'severity': AlertSeverity.CRITICAL,
                'message_template': 'URGENT: Patient {patient_id} has critical deterioration risk ({risk_score:.1%})'
            },
            'medication_non_adherence': {
                'threshold': 0.6,
                'severity': AlertSeverity.MEDIUM,
                'message_template': 'Patient {patient_id} has poor medication adherence ({adherence:.1%})'
            },
            'lab_critical_high': {
                'hba1c_threshold': 10.0,
                'severity': AlertSeverity.HIGH,
                'message_template': 'Critical HbA1c level for patient {patient_id}: {value}%'
            },
            'recent_hospitalization': {
                'days_threshold': 7,
                'severity': AlertSeverity.MEDIUM,
                'message_template': 'Patient {patient_id} had recent hospitalization ({days} days ago)'
            }
        }
    
    def add_alert(self, alert: Alert) -> None:
        """Add new alert to active alerts."""
        self.active_alerts.append(alert)
        # Sort by severity and creation time
        self.active_alerts.sort(key=lambda x: (x.severity.value, x.created_at), reverse=True)
    
    def create_patient_risk_alert(self, patient_id: str, risk_score: float,
                                risk_factors: List[str]) -> Alert:
        """Create alert for high-risk patient."""
        if risk_score >= self.alert_rules['critical_risk_patient']['threshold']:
            severity = AlertSeverity.CRITICAL
            title = "CRITICAL RISK ALERT"
            message = self.alert_rules['critical_risk_patient']['message_template'].format(
                patient_id=patient_id, risk_score=risk_score
            )
        elif risk_score >= self.alert_rules['high_risk_patient']['threshold']:
            severity = AlertSeverity.HIGH
            title = "High Risk Patient Alert"
            message = self.alert_rules['high_risk_patient']['message_template'].format(
                patient_id=patient_id, risk_score=risk_score
            )
        else:
            severity = AlertSeverity.MEDIUM
            title = "Risk Monitoring Alert"
            message = f"Patient {patient_id} risk score: {risk_score:.1%}"
        
        alert_id = f"risk_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return Alert(
            alert_id=alert_id,
            alert_type=AlertType.PATIENT_RISK,
            severity=severity,
            title=title,
            message=message,
            patient_id=patient_id,
            data={
                'risk_score': risk_score,
                'risk_factors': risk_factors,
                'requires_action': risk_score >= 0.7
            }
        )
    
    def create_medication_adherence_alert(self, patient_id: str, adherence_rate: float,
                                        medications: List[str]) -> Alert:
        """Create alert for poor medication adherence."""
        severity = AlertSeverity.HIGH if adherence_rate < 0.5 else AlertSeverity.MEDIUM
        title = "Medication Adherence Alert"
        message = self.alert_rules['medication_non_adherence']['message_template'].format(
            patient_id=patient_id, adherence=adherence_rate
        )
        
        alert_id = f"med_adherence_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return Alert(
            alert_id=alert_id,
            alert_type=AlertType.MEDICATION_ADHERENCE,
            severity=severity,
            title=title,
            message=message,
            patient_id=patient_id,
            data={
                'adherence_rate': adherence_rate,
                'medications': medications
            }
        )
    
    def create_lab_critical_alert(self, patient_id: str, lab_name: str, 
                                value: float, normal_range: str) -> Alert:
        """Create alert for critical lab values."""
        severity = AlertSeverity.HIGH
        title = f"Critical {lab_name} Alert"
        message = f"Critical {lab_name} level for patient {patient_id}: {value} (Normal: {normal_range})"
        
        alert_id = f"lab_critical_{patient_id}_{lab_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return Alert(
            alert_id=alert_id,
            alert_type=AlertType.LAB_CRITICAL,
            severity=severity,
            title=title,
            message=message,
            patient_id=patient_id,
            data={
                'lab_name': lab_name,
                'value': value,
                'normal_range': normal_range
            }
        )
    
    def get_active_alerts(self, severity_filter: Optional[AlertSeverity] = None,
                         patient_id: Optional[str] = None) -> List[Alert]:
        """Get active alerts with optional filtering."""
        alerts = [alert for alert in self.active_alerts if not alert.resolved]
        
        if severity_filter:
            alerts = [alert for alert in alerts if alert.severity == severity_filter]
        
        if patient_id:
            alerts = [alert for alert in alerts if alert.patient_id == patient_id]
        
        return alerts
    
    def get_critical_alerts(self) -> List[Alert]:
        """Get all critical alerts requiring immediate attention."""
        return self.get_active_alerts(severity_filter=AlertSeverity.CRITICAL)
    
    def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.active_alerts:
            if alert.alert_id == alert_id:
                alert.acknowledge(user_id)
                return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        for alert in self.active_alerts:
            if alert.alert_id == alert_id:
                alert.resolve()
                self.alert_history.append(alert)
                self.active_alerts.remove(alert)
                return True
        return False
    
    def get_alert_summary(self) -> Dict[str, int]:
        """Get summary of active alerts by severity."""
        summary = {severity.value: 0 for severity in AlertSeverity}
        
        for alert in self.get_active_alerts():
            summary[alert.severity.value] += 1
        
        return summary
    
    def cleanup_old_alerts(self, days: int = 30) -> int:
        """Remove resolved alerts older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=days)
        old_alerts = [alert for alert in self.alert_history 
                     if alert.resolved_at and alert.resolved_at < cutoff_date]
        
        for alert in old_alerts:
            self.alert_history.remove(alert)
        
        return len(old_alerts)
    
    def export_alerts_json(self, include_history: bool = False) -> str:
        """Export alerts to JSON format."""
        data = {
            'active_alerts': [alert.to_dict() for alert in self.get_active_alerts()],
            'alert_summary': self.get_alert_summary(),
            'export_timestamp': datetime.now().isoformat()
        }
        
        if include_history:
            data['alert_history'] = [alert.to_dict() for alert in self.alert_history]
        
        return json.dumps(data, indent=2)
