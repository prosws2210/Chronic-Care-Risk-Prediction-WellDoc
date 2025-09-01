"""
📊 Data Processing and Synthetic Dataset Generation
Generates realistic 5000-patient dataset with temporal features for chronic care prediction
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """Handles data generation, preprocessing, and feature engineering"""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.generated_data = None
        
    def generate_synthetic_dataset(self, n_patients=5000, save_path='data/synthetic_patients.csv'):
        """Generate realistic synthetic patient dataset"""
        
        print(f"Generating synthetic dataset for {n_patients} patients...")
        
        np.random.seed(42)  # For reproducibility
        
        all_patient_records = []
        
        for patient_id in range(n_patients):
            patient_records = self._generate_single_patient_data(patient_id)
            all_patient_records.extend(patient_records)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_patient_records)
        
        # Add outcome variable (deterioration within 90 days)
        df = self._add_outcome_variable(df)
        
        # Save dataset
        df.to_csv(save_path, index=False)
        print(f"Dataset saved to {save_path}")
        print(f"Total records: {len(df)}")
        print(f"Unique patients: {df['patient_id'].nunique()}")
        
        self.generated_data = df
        return df
    
    def _generate_single_patient_data(self, patient_id):
        """Generate data for a single patient over 30-180 days"""
        
        # Patient baseline characteristics
        age = np.random.normal(65, 15)
        age = max(18, min(95, age))  # Constrain age
        
        gender = np.random.choice(['M', 'F'], p=[0.48, 0.52])
        
        # BMI based on realistic distribution
        bmi = np.random.normal(28.5, 6)
        bmi = max(18, min(50, bmi))
        
        # Primary chronic condition
        primary_condition = np.random.choice([
            'Type 2 Diabetes', 'Hypertension', 'Heart Disease', 
            'Chronic Kidney Disease', 'COPD'
        ], p=[0.35, 0.25, 0.20, 0.12, 0.08])
        
        # Comorbidities
        comorbidities = self._generate_comorbidities(primary_condition)
        
        # Data collection period (30-180 days)
        days_of_data = np.random.randint(30, 181)
        end_date = datetime.now() - timedelta(days=np.random.randint(1, 30))
        start_date = end_date - timedelta(days=days_of_data)
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate baseline vital signs based on condition and age
        baseline_vitals = self._generate_baseline_vitals(primary_condition, age, bmi)
        
        patient_records = []
        
        for i, date in enumerate(date_range):
            # Add day-to-day variation and trends
            day_variation = self._generate_daily_variation(i, days_of_data, baseline_vitals, primary_condition)
            
            record = {
                'patient_id': f'PAT_{patient_id:04d}',
                'date': date.strftime('%Y-%m-%d'),
                'age': age,
                'gender': gender,
                'bmi': bmi + np.random.normal(0, 0.1),  # Slight daily variation
                'primary_condition': primary_condition,
                'chronic_conditions': ','.join(comorbidities),
                
                # Vital signs with realistic variation
                'systolic_bp': day_variation['systolic_bp'],
                'diastolic_bp': day_variation['diastolic_bp'],
                'heart_rate': day_variation['heart_rate'],
                'temperature': day_variation['temperature'],
                'weight': day_variation['weight'],
                'respiratory_rate': day_variation['respiratory_rate'],
                
                # Lab results (updated less frequently)
                'glucose': day_variation['glucose'],
                'hba1c': day_variation['hba1c'],
                'cholesterol': day_variation['cholesterol'],
                'ldl_cholesterol': day_variation['ldl_cholesterol'],
                'hdl_cholesterol': day_variation['hdl_cholesterol'],
                'triglycerides': day_variation['triglycerides'],
                'creatinine': day_variation['creatinine'],
                'bun': day_variation['bun'],
                'egfr': day_variation['egfr'],
                
                # Medication adherence
                'medication_adherence': day_variation['medication_adherence'],
                
                # Lifestyle factors
                'daily_steps': day_variation['daily_steps'],
                'sleep_hours': day_variation['sleep_hours'],
                'exercise_minutes': day_variation['exercise_minutes'],
            }
            
            patient_records.append(record)
        
        return patient_records
    
    def _generate_comorbidities(self, primary_condition):
        """Generate realistic comorbidities based on primary condition"""
        
        base_conditions = [primary_condition]
        
        # Common comorbidity patterns
        if primary_condition == 'Type 2 Diabetes':
            if np.random.random() < 0.7:
                base_conditions.append('Hypertension')
            if np.random.random() < 0.4:
                base_conditions.append('Dyslipidemia')
            if np.random.random() < 0.3:
                base_conditions.append('Chronic Kidney Disease')
                
        elif primary_condition == 'Hypertension':
            if np.random.random() < 0.5:
                base_conditions.append('Type 2 Diabetes')
            if np.random.random() < 0.4:
                base_conditions.append('Dyslipidemia')
                
        elif primary_condition == 'Heart Disease':
            if np.random.random() < 0.6:
                base_conditions.append('Hypertension')
            if np.random.random() < 0.5:
                base_conditions.append('Dyslipidemia')
            if np.random.random() < 0.3:
                base_conditions.append('Type 2 Diabetes')
        
        return list(set(base_conditions))  # Remove duplicates
    
    def _generate_baseline_vitals(self, primary_condition, age, bmi):
        """Generate baseline vital signs based on patient characteristics"""
        
        # Age and BMI adjustments
        age_factor = (age - 40) / 40  # Normalized age effect
        bmi_factor = (bmi - 25) / 10  # Normalized BMI effect
        
        baseline = {}
        
        # Blood pressure (condition-dependent)
        if primary_condition in ['Hypertension', 'Heart Disease']:
            baseline['systolic_bp'] = np.random.normal(145, 15) + age_factor * 10
            baseline['diastolic_bp'] = np.random.normal(90, 10) + age_factor * 5
        else:
            baseline['systolic_bp'] = np.random.normal(130, 12) + age_factor * 8
            baseline['diastolic_bp'] = np.random.normal(80, 8) + age_factor * 4
            
        # Heart rate
        baseline['heart_rate'] = np.random.normal(72, 8) + bmi_factor * 3
        
        # Temperature (mostly stable)
        baseline['temperature'] = np.random.normal(98.6, 0.3)
        
        # Weight (BMI-based)
        height = np.random.normal(68, 4)  # inches
        baseline['weight'] = (bmi * (height**2)) / 703  # Convert BMI to weight
        
        # Respiratory rate
        baseline['respiratory_rate'] = np.random.normal(16, 2)
        
        # Glucose (diabetes-dependent)
        if 'Diabetes' in primary_condition:
            baseline['glucose'] = np.random.normal(160, 30)
            baseline['hba1c'] = np.random.normal(8.2, 1.2)
        else:
            baseline['glucose'] = np.random.normal(95, 15)
            baseline['hba1c'] = np.random.normal(5.4, 0.4)
            
        # Cholesterol
        baseline['cholesterol'] = np.random.normal(200, 40) + age_factor * 20
        baseline['ldl_cholesterol'] = baseline['cholesterol'] * 0.6 + np.random.normal(0, 10)
        baseline['hdl_cholesterol'] = np.random.normal(45, 10) - bmi_factor * 5
        baseline['triglycerides'] = np.random.normal(150, 50) + bmi_factor * 30
        
        # Kidney function (age-dependent)
        baseline['creatinine'] = np.random.normal(1.0, 0.2) + age_factor * 0.3
        baseline['bun'] = np.random.normal(15, 5) + age_factor * 8
        baseline['egfr'] = max(15, 120 - age_factor * 30 + np.random.normal(0, 10))
        
        # Medication adherence (condition-dependent)
        if primary_condition in ['Type 2 Diabetes', 'Heart Disease']:
            baseline['medication_adherence'] = np.random.beta(8, 2)  # Higher adherence for serious conditions
        else:
            baseline['medication_adherence'] = np.random.beta(6, 3)
            
        # Lifestyle factors (age and condition dependent)
        baseline['daily_steps'] = max(500, np.random.normal(6000, 2000) - age_factor * 1500)
        baseline['sleep_hours'] = np.random.normal(7, 1)
        baseline['exercise_minutes'] = max(0, np.random.normal(30, 20) - age_factor * 15)
        
        return baseline
    
    def _generate_daily_variation(self, day_index, total_days, baseline_vitals, primary_condition):
        """Generate daily variations with realistic trends and noise"""
        
        # Progress through monitoring period (0 to 1)
        progress = day_index / total_days
        
        # Seasonal/weekly patterns
        weekly_cycle = np.sin(2 * np.pi * day_index / 7) * 0.05
        monthly_cycle = np.sin(2 * np.pi * day_index / 30) * 0.03
        
        # Random deterioration risk factor
        deterioration_risk = np.random.beta(2, 8)  # Most patients stable, some deteriorate
        
        # Generate trends (some patients improve, some deteriorate)
        trend_factor = (np.random.random() - 0.5) * 2 * progress  # -1 to +1 based on progress
        
        daily_values = {}
        
        for vital, baseline in baseline_vitals.items():
            # Base daily noise
            noise_std = {
                'systolic_bp': 8, 'diastolic_bp': 5, 'heart_rate': 6,
                'temperature': 0.4, 'weight': 0.5, 'respiratory_rate': 1,
                'glucose': 20, 'hba1c': 0.1, 'cholesterol': 10,
                'ldl_cholesterol': 8, 'hdl_cholesterol': 3, 'triglycerides': 15,
                'creatinine': 0.05, 'bun': 2, 'egfr': 3,
                'medication_adherence': 0.1, 'daily_steps': 1000,
                'sleep_hours': 0.8, 'exercise_minutes': 10
            }.get(vital, baseline * 0.05)
            
            # Daily noise
            daily_noise = np.random.normal(0, noise_std)
            
            # Trend component
            trend_component = trend_factor * baseline * 0.1
            
            # Cycles
            cycle_component = (weekly_cycle + monthly_cycle) * baseline
            
            # Deterioration effect (mainly affects key vitals)
            if vital in ['systolic_bp', 'glucose', 'hba1c'] and deterioration_risk > 0.7:
                deterioration_effect = deterioration_risk * baseline * 0.2
            else:
                deterioration_effect = 0
                
            # Combine all effects
            daily_value = baseline + daily_noise + trend_component + cycle_component + deterioration_effect
            
            # Apply realistic bounds
            daily_value = self._apply_vital_bounds(vital, daily_value)
            
            daily_values[vital] = daily_value
        
        return daily_values
    
    def _apply_vital_bounds(self, vital, value):
        """Apply realistic bounds to vital signs"""
        
        bounds = {
            'systolic_bp': (80, 250),
            'diastolic_bp': (40, 150), 
            'heart_rate': (40, 150),
            'temperature': (95, 104),
            'weight': (80, 400),
            'respiratory_rate': (8, 30),
            'glucose': (50, 400),
            'hba1c': (4, 15),
            'cholesterol': (100, 400),
            'ldl_cholesterol': (50, 300),
            'hdl_cholesterol': (20, 100),
            'triglycerides': (50, 500),
            'creatinine': (0.5, 8),
            'bun': (5, 100),
            'egfr': (5, 120),
            'medication_adherence': (0, 1),
            'daily_steps': (0, 25000),
            'sleep_hours': (3, 12),
            'exercise_minutes': (0, 300)
        }
        
        if vital in bounds:
            min_val, max_val = bounds[vital]
            return max(min_val, min(max_val, value))
        
        return value
    
    def _add_outcome_variable(self, df):
        """Add deterioration outcome based on realistic risk factors"""
        
        # Calculate risk score for each patient's final values
        patient_outcomes = []
        
        for patient_id in df['patient_id'].unique():
            patient_data = df[df['patient_id'] == patient_id].copy()
            
            # Get most recent values
            recent_data = patient_data.tail(30)  # Last 30 days
            
            # Risk factors
            risk_score = 0
            
            # High blood pressure
            if recent_data['systolic_bp'].mean() > 150:
                risk_score += 0.2
            if recent_data['systolic_bp'].std() > 15:  # High variability
                risk_score += 0.15
                
            # Poor glucose control
            if recent_data['glucose'].mean() > 180:
                risk_score += 0.25
            if recent_data['hba1c'].tail(1).iloc[0] > 9:
                risk_score += 0.2
                
            # Medication non-adherence
            if recent_data['medication_adherence'].mean() < 0.7:
                risk_score += 0.3
                
            # Age factor
            age = patient_data['age'].iloc[0]
            if age > 70:
                risk_score += 0.1
                
            # Multiple comorbidities
            comorbidity_count = len(patient_data['chronic_conditions'].iloc[0].split(','))
            risk_score += comorbidity_count * 0.05
            
            # Lifestyle factors
            if recent_data['daily_steps'].mean() < 2000:
                risk_score += 0.1
                
            # Random component
            risk_score += np.random.beta(2, 8) * 0.3
            
            # Convert to binary outcome
            deterioration = 1 if (risk_score > 0.5 and np.random.random() < risk_score) else 0
            
            patient_outcomes.append({
                'patient_id': patient_id,
                'deterioration_90_days': deterioration,
                'risk_score_calculated': min(1.0, risk_score)
            })
        
        # Merge outcomes back to main dataframe
        outcomes_df = pd.DataFrame(patient_outcomes)
        df = df.merge(outcomes_df, on='patient_id', how='left')
        
        # Print outcome statistics
        deterioration_rate = df['deterioration_90_days'].mean()
        print(f"Deterioration rate: {deterioration_rate:.1%}")
        
        return df
    
    def generate_predictions_dataset(self, patient_data_path='data/synthetic_patients.csv', 
                                   save_path='data/model_predictions.csv'):
        """Generate model predictions dataset"""
        
        # Load patient data
        if self.generated_data is not None:
            df = self.generated_data
        else:
            df = pd.read_csv(patient_data_path)
        
        # Get unique patients
        patients = df['patient_id'].unique()
        
        predictions = []
        
        for patient_id in patients:
            patient_data = df[df['patient_id'] == patient_id]
            
            # Calculate risk score (simplified model simulation)
            recent_data = patient_data.tail(30)
            
            # Simulate model prediction based on key features
            features = {
                'glucose_mean': recent_data['glucose'].mean(),
                'bp_systolic_mean': recent_data['systolic_bp'].mean(),
                'medication_adherence': recent_data['medication_adherence'].mean(),
                'age': patient_data['age'].iloc[0],
                'hba1c': recent_data['hba1c'].tail(1).iloc[0]
            }
            
            # Normalize features and calculate risk
            risk_score = self._calculate_simulated_risk(features)
            
            # Get patient info
            patient_info = {
                'patient_id': patient_id,
                'age': patient_data['age'].iloc[0],
                'primary_condition': patient_data['primary_condition'].iloc[0],
                'risk_score': risk_score,
                'recommended_action': self._get_recommended_action(risk_score)
            }
            
            predictions.append(patient_info)
        
        predictions_df = pd.DataFrame(predictions)
        predictions_df.to_csv(save_path, index=False)
        
        print(f"Predictions dataset saved to {save_path}")
        return predictions_df
    
    def _calculate_simulated_risk(self, features):
        """Simulate XGBoost model prediction"""
        
        risk = 0
        
        # Glucose effect
        if features['glucose_mean'] > 140:
            risk += 0.3
        elif features['glucose_mean'] > 110:
            risk += 0.15
            
        # Blood pressure effect  
        if features['bp_systolic_mean'] > 140:
            risk += 0.25
        elif features['bp_systolic_mean'] > 130:
            risk += 0.1
            
        # Medication adherence effect
        if features['medication_adherence'] < 0.7:
            risk += 0.4
        elif features['medication_adherence'] < 0.85:
            risk += 0.2
            
        # Age effect
        if features['age'] > 70:
            risk += 0.2
        elif features['age'] > 60:
            risk += 0.1
            
        # HbA1c effect
        if features['hba1c'] > 8:
            risk += 0.3
        elif features['hba1c'] > 7:
            risk += 0.15
            
        # Add some randomness
        risk += np.random.normal(0, 0.1)
        
        return max(0, min(1, risk))
    
    def _get_recommended_action(self, risk_score):
        """Get recommended action based on risk score"""
        
        if risk_score > 0.8:
            return "🚨 Urgent intervention - Schedule within 24h"
        elif risk_score > 0.6:
            return "📞 High priority - Call within 48h"
        elif risk_score > 0.4:
            return "📅 Medium priority - Schedule routine follow-up"
        else:
            return "✅ Low priority - Continue current care plan"
    
    def load_data(self, filepath):
        """Load existing dataset"""
        self.generated_data = pd.read_csv(filepath)
        return self.generated_data
    
    def get_data_summary(self):
        """Get summary statistics of the dataset"""
        if self.generated_data is None:
            return "No data loaded"
            
        df = self.generated_data
        
        summary = {
            'total_patients': df['patient_id'].nunique(),
            'total_records': len(df),
            'date_range': f"{df['date'].min()} to {df['date'].max()}",
            'conditions_distribution': df.groupby('primary_condition')['patient_id'].nunique().to_dict(),
            'deterioration_rate': f"{df['deterioration_90_days'].mean():.1%}" if 'deterioration_90_days' in df.columns else 'Not calculated',
            'avg_records_per_patient': f"{len(df) / df['patient_id'].nunique():.1f}"
        }
        
        return summary
