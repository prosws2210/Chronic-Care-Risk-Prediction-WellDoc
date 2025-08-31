"""
Main Flask application for the Chronic Care Risk Prediction Dashboard.
Handles routing, authentication, and application configuration.
"""

import os
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_login import LoginManager, login_required, current_user
from werkzeug.security import check_password_hash

# Import dashboard components
from .components.alerts import AlertsManager
from .components.charts import ChartGenerator
from .components.tables import TableGenerator

# Import view blueprints
from .cohort_view import cohort_bp
from .patient_detail_view import patient_bp
from .risk_dashboard import risk_bp

def create_app(config_name='development'):
    """Create and configure the Flask application."""
    
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')
    app.config['DEBUG'] = config_name == 'development'
    
    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s %(message)s'
    )
    
    # Initialize components
    alerts_manager = AlertsManager()
    chart_generator = ChartGenerator()
    table_generator = TableGenerator()
    
    # Store components in app context
    app.alerts_manager = alerts_manager
    app.chart_generator = chart_generator
    app.table_generator = table_generator
    
    # Initialize Flask-Login
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access the dashboard.'
    
    @login_manager.user_loader
    def load_user(user_id):
        # In a real application, load user from database
        # For demo purposes, return a mock user
        from types import SimpleNamespace
        if user_id == '1':
            return SimpleNamespace(
                id='1',
                username='clinician',
                email='clinician@hospital.com',
                role='clinician',
                is_authenticated=True,
                is_active=True,
                is_anonymous=False,
                get_id=lambda: '1'
            )
        return None
    
    # Register blueprints
    app.register_blueprint(cohort_bp, url_prefix='/cohort')
    app.register_blueprint(patient_bp, url_prefix='/patient')
    app.register_blueprint(risk_bp, url_prefix='/dashboard')
    
    # Main routes
    @app.route('/')
    def index():
        """Dashboard home page."""
        return redirect(url_for('risk_dashboard.overview'))
    
    @app.route('/api/alerts')
    def get_alerts():
        """API endpoint for fetching active alerts."""
        alerts = alerts_manager.get_active_alerts()
        return jsonify([alert.to_dict() for alert in alerts])
    
    @app.route('/api/alerts/<alert_id>/acknowledge', methods=['POST'])
    def acknowledge_alert(alert_id):
        """API endpoint for acknowledging alerts."""
        user_id = getattr(current_user, 'id', 'anonymous')
        success = alerts_manager.acknowledge_alert(alert_id, user_id)
        return jsonify({'success': success})
    
    @app.route('/api/alerts/<alert_id>/resolve', methods=['POST'])
    def resolve_alert(alert_id):
        """API endpoint for resolving alerts."""
        success = alerts_manager.resolve_alert(alert_id)
        return jsonify({'success': success})
    
    @app.route('/health')
    def health_check():
        """Health check endpoint for monitoring."""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        })
    
    # Error handlers
    @app.errorhandler(404)
    def not_found_error(error):
        return render_template('error.html', 
                             error_code=404,
                             error_message="Page not found"), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return render_template('error.html',
                             error_code=500,
                             error_message="Internal server error"), 500
    
    # Template filters
    @app.template_filter('datetime')
    def datetime_filter(value):
        """Format datetime for templates."""
        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value.replace('Z', '+00:00'))
            except ValueError:
                return value
        return value.strftime('%m/%d/%Y %I:%M %p')
    
    @app.template_filter('percentage')
    def percentage_filter(value):
        """Format decimal as percentage."""
        try:
            return f"{float(value) * 100:.1f}%"
        except (ValueError, TypeError):
            return "N/A"
    
    # Context processors
    @app.context_processor
    def inject_global_vars():
        """Inject global variables into all templates."""
        return {
            'current_time': datetime.now(),
            'app_name': 'Chronic Care Risk Dashboard',
            'version': '1.0.0'
        }
    
    return app

# Create the application instance
app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
