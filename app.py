import io, base64, csv, pickle, requests, os, socket
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import qrcode
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime, timedelta
from flask import (Flask, render_template, redirect, url_for, request,
                   flash, jsonify, send_file, Response, session, g)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (LoginManager, UserMixin, login_user, login_required,
                         logout_user, current_user)
from flask_bcrypt import Bcrypt
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect
from wtforms import (StringField, PasswordField, SubmitField, FloatField,
                     SelectField, DateTimeLocalField, TextAreaField, BooleanField)
from wtforms.validators import InputRequired, Length, Email, NumberRange, Optional
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

from translations import TRANSLATIONS, SUPPORTED_LANGUAGES, get_text

# ---------------- Version Info ----------------
APP_VERSION = "1.0.0"
APP_BUILD_DATE = "Week 8 — Final"
APP_AUTHOR = "Hypertension Risk Research Team"

# ---------------- Config ----------------
app = Flask(__name__)

# 🚀 POSTGRESQL & VERCEL FIX: Use Neon Database URL
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'fallback-secret-key-123')
# Connect to Neon PostgreSQL. The 'sslmode=require' is mandatory for Neon.
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:////tmp/hypertension_week8.db')
# Fix for SQLAlchemy 1.4+ if DATABASE_URL starts with postgres://
if app.config['SQLALCHEMY_DATABASE_URI'].startswith("postgres://"):
    app.config['SQLALCHEMY_DATABASE_URI'] = app.config['SQLALCHEMY_DATABASE_URI'].replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['BABEL_DEFAULT_LOCALE'] = 'en'

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
csrf = CSRFProtect(app)

# ---------------- ML ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(BASE_DIR, 'model_week8.pkl'), 'rb') as f:
        model = pickle.load(f)
    background = pd.read_csv(os.path.join(BASE_DIR, 'background_week8.csv'))
    explainer = shap.LinearExplainer(model, background)
    print("✓ Week 8 ML model + SHAP explainer loaded")
except FileNotFoundError:
    print("⚠ Run 'python model.py' first.")
    model, explainer = None, None

# ---------------- i18n ----------------
def get_current_language():
    url_lang = request.args.get('lang')
    if url_lang and url_lang in TRANSLATIONS:
        session['language'] = url_lang
        return url_lang
    return session.get('language', 'en')

@app.before_request
def before_request():
    g.lang = get_current_language()
    g.t = lambda key: get_text(g.lang, key)

@app.context_processor
def inject_globals():
    return {
        't': lambda key: get_text(g.lang, key),
        'current_lang': g.lang,
        'supported_languages': SUPPORTED_LANGUAGES,
        'app_version': APP_VERSION,
        'app_build': APP_BUILD_DATE,
    }

# ---------------- Models ----------------
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    preferred_language = db.Column(db.String(5), default='en')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    assessments = db.relationship('Assessment', backref='user', lazy=True)
    reminders = db.relationship('Reminder', backref='user', lazy=True)
    notifications = db.relationship('Notification', backref='user', lazy=True)
    emergency_contact = db.relationship('EmergencyContact', backref='user', uselist=False)
    preferences = db.relationship('NotificationPreference', backref='user', uselist=False)

class Assessment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    age = db.Column(db.Float); bmi = db.Column(db.Float)
    smoking = db.Column(db.Integer); alcohol = db.Column(db.Integer)
    physical_activity = db.Column(db.Integer); family_history = db.Column(db.Integer)
    probability = db.Column(db.Float); risk_level = db.Column(db.String(10))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class Reminder(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    reminder_type = db.Column(db.String(50)); frequency = db.Column(db.String(30))
    next_date = db.Column(db.DateTime); notes = db.Column(db.Text)
    active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class EmergencyContact(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, unique=True)
    name = db.Column(db.String(120), nullable=False)
    relationship = db.Column(db.String(60)); phone = db.Column(db.String(30), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Notification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    message = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(40), default='general')
    channel = db.Column(db.String(30), default='in-app')
    is_read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class NotificationPreference(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, unique=True)
    email_enabled = db.Column(db.Boolean, default=True)
    sms_enabled = db.Column(db.Boolean, default=False)
    in_app_enabled = db.Column(db.Boolean, default=True)
    risk_alerts = db.Column(db.Boolean, default=True)
    reminder_alerts = db.Column(db.Boolean, default=True)
    emergency_alerts = db.Column(db.Boolean, default=True)
    phone_number = db.Column(db.String(30))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ---------------- Forms ----------------
class RegisterForm(FlaskForm):
    email = StringField('Email', validators=[InputRequired(), Email()])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=6)])
    submit = SubmitField('Register')

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[InputRequired(), Email()])
    password = PasswordField('Password', validators=[InputRequired()])
    submit = SubmitField('Login')

class AssessmentForm(FlaskForm):
    age = FloatField('Age', validators=[InputRequired(), NumberRange(min=18, max=120)])
    bmi = FloatField('BMI', validators=[InputRequired(), NumberRange(min=10, max=60)])
    smoking = SelectField('Smoking', choices=[('0','Non-smoker'),('1','Current smoker')])
    alcohol = SelectField('Alcohol', choices=[('0','No'),('1','Yes')])
    physical_activity = SelectField('Physical Activity', choices=[('1','Active'),('0','Inactive')])
    family_history = SelectField('Family History', choices=[('0','No'),('1','Yes')])
    submit = SubmitField('Assess My Risk')

class ReminderForm(FlaskForm):
    reminder_type = SelectField('Type', choices=[('Blood Pressure Check','BP Check'),('Medication','Medication'),('Follow-up','Follow-up'),('Exercise','Exercise'),('Custom','Custom')])
    frequency = SelectField('Frequency', choices=[('Daily','Daily'),('Weekly','Weekly'),('Monthly','Monthly')])
    next_date = DateTimeLocalField('Next Date', format='%Y-%m-%dT%H:%M', validators=[InputRequired()])
    notes = TextAreaField('Notes', validators=[Optional(), Length(max=300)])
    submit = SubmitField('Save')

class EmergencyContactForm(FlaskForm):
    name = StringField('Name', validators=[InputRequired(), Length(min=2, max=120)])
    relationship = StringField('Relationship', validators=[Optional(), Length(max=60)])
    phone = StringField('Phone', validators=[InputRequired(), Length(min=6, max=30)])
    submit = SubmitField('Save')

class NotificationPreferencesForm(FlaskForm):
    email_enabled = BooleanField('Email')
    sms_enabled = BooleanField('SMS')
    in_app_enabled = BooleanField('In-App')
    risk_alerts = BooleanField('Risk Alerts')
    reminder_alerts = BooleanField('Reminder Alerts')
    emergency_alerts = BooleanField('Emergency Alerts')
    phone_number = StringField('Phone Number', validators=[Optional(), Length(max=30)])
    submit = SubmitField('Save Preferences')

# ---------------- Helpers ----------------
def get_risk_level(prob):
    if prob < 0.3: return 'Low'
    elif prob < 0.6: return 'Medium'
    return 'High'

def get_recommendation(risk_level):
    return {
        'Low': 'Great! Maintain your healthy habits. Check-up every 2 years.',
        'Medium': 'Moderate risk. Monitor BP every 6 months and discuss lifestyle changes with your doctor.',
        'High': '⚠️ High risk! Please schedule a screening appointment within 2-4 weeks.'
    }.get(risk_level, '')

def generate_shap_plot(features):
    if explainer is None: return None
    shap_vals = explainer.shap_values(features)
    exp = shap.Explanation(values=shap_vals[0], base_values=explainer.expected_value,
                           data=features[0],
                           feature_names=['age','bmi','smoking','alcohol','physical_activity','family_history'])
    fig = plt.figure(figsize=(10, 6))
    shap.waterfall_plot(exp, max_display=6, show=False)
    buf = io.BytesIO(); plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight'); plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def get_top_contributors(features):
    if explainer is None: return []
    shap_vals = explainer.shap_values(features)[0]
    names = ['age','bmi','smoking','alcohol','physical_activity','family_history']
    positive = [(n, v) for n, v in zip(names, shap_vals) if v > 0]
    positive.sort(key=lambda x: x[1], reverse=True)
    return [n for n, _ in positive[:3]]

EDUCATION_LIBRARY = {
    'smoking': {'title': 'Quit Smoking', 'content': 'Smoking damages artery walls.', 'tips': ['Set a quit date', 'Consider nicotine replacement', 'Avoid triggers', 'Join a program'], 'link': 'https://www.who.int/news-room/fact-sheets/detail/tobacco'},
    'bmi': {'title': 'Healthy Weight', 'content': 'Excess weight raises blood pressure.', 'tips': ['150 min exercise/week', 'Balanced diet', 'Reduce salt', 'Monitor BMI'], 'link': 'https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight'},
    'alcohol': {'title': 'Reduce Alcohol', 'content': 'Too much alcohol raises BP.', 'tips': ['Limit to 1-2 drinks/day', 'Have alcohol-free days', 'Avoid binge drinking'], 'link': 'https://www.who.int/news-room/fact-sheets/detail/alcohol'},
    'physical_activity': {'title': 'Increase Activity', 'content': 'Exercise strengthens the heart.', 'tips': ['30 min most days', 'Walk/cycle/swim', 'Take the stairs', 'Find enjoyable activities'], 'link': 'https://www.who.int/news-room/fact-sheets/detail/physical-activity'},
    'family_history': {'title': 'Genetic Risk', 'content': 'Family history increases your risk.', 'tips': ['Regular BP checks', 'Tell your doctor', 'Heart-healthy diet', 'Home monitoring'], 'link': 'https://www.cdc.gov/bloodpressure/family_history.htm'},
    'age': {'title': 'Age & BP', 'content': 'Vessels lose flexibility with age.', 'tips': ['BP check every 1-2 years', 'Healthy lifestyle', 'Discuss concerns with doctor'], 'link': 'https://www.nia.nih.gov/health/high-blood-pressure'}
}

def create_notification(user_id, title, message, category='general'):
    prefs = NotificationPreference.query.filter_by(user_id=user_id).first()
    if prefs:
        if category == 'risk' and not prefs.risk_alerts: return
        if category == 'reminder' and not prefs.reminder_alerts: return
        if category == 'emergency' and not prefs.emergency_alerts: return
    db.session.add(Notification(user_id=user_id, title=title, message=message, category=category, channel='in-app'))
    db.session.commit()

def generate_qr_code(data):
    qr = qrcode.QRCode(version=1, box_size=10, border=2, error_correction=qrcode.constants.ERROR_CORRECT_H)
    qr.add_data(data); qr.make(fit=True)
    img = qr.make_image(fill_color="#b3122e", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# ---------------- Routes ----------------
@app.route('/')
def index(): return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html', total_users=User.query.count(), total_assessments=Assessment.query.count(), app_version=APP_VERSION, app_build=APP_BUILD_DATE, app_author=APP_AUTHOR)

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'version': APP_VERSION, 'timestamp': datetime.utcnow().isoformat()}), 200

@app.route('/share/qr')
@login_required
def share_qr():
    share_url = url_for('index', _external=True)
    return render_template('share_qr.html', qr_img=generate_qr_code(share_url), share_url=share_url)

@app.route('/set-language/<lang>')
def set_language(lang):
    if lang in TRANSLATIONS:
        session['language'] = lang
        if current_user.is_authenticated: current_user.preferred_language = lang; db.session.commit()
    return redirect(request.referrer or url_for('index'))

@app.route('/register', methods=['GET','POST'])
def register():
    if current_user.is_authenticated: return redirect(url_for('dashboard'))
    form = RegisterForm()
    if form.validate_on_submit():
        if User.query.filter_by(email=form.email.data).first():
            flash('Email already registered.', 'danger'); return redirect(url_for('login'))
        hashed = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(email=form.email.data, password=hashed, preferred_language=g.lang)
        db.session.add(user); db.session.commit()
        db.session.add(NotificationPreference(user_id=user.id)); db.session.commit()
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET','POST'])
def login():
    if current_user.is_authenticated: return redirect(url_for('dashboard'))
    form = LoginForm()
    if form.validate_on_submit():
        try:
            user = User.query.filter_by(email=form.email.data).first()
            if user and user.password:
                if bcrypt.check_password_hash(user.password, form.password.data):
                    login_user(user)
                    session['language'] = user.preferred_language or 'en'
                    return redirect(url_for('dashboard'))
        except Exception as e:
            print(f"Login Error: {e}") # Prevents 500 error
        flash('Invalid credentials.', 'danger')
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout(): logout_user(); return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    recent_assessments = Assessment.query.filter_by(user_id=current_user.id).order_by(Assessment.timestamp.desc()).limit(5).all()
    active_reminders = Reminder.query.filter_by(user_id=current_user.id, active=True).order_by(Reminder.next_date.asc()).limit(5).all()
    unread_notifications = Notification.query.filter_by(user_id=current_user.id, is_read=False).order_by(Notification.created_at.desc()).limit(5).all()
    unread_count = Notification.query.filter_by(user_id=current_user.id, is_read=False).count()
    emergency = EmergencyContact.query.filter_by(user_id=current_user.id).first()
    
    return render_template('dashboard.html', user=current_user, recent_assessments=recent_assessments, active_reminders=active_reminders, unread_notifications=unread_notifications, unread_count=unread_count, emergency=emergency)

@app.route('/assessment', methods=['GET','POST'])
@login_required
def assessment():
    form = AssessmentForm()
    if form.validate_on_submit():
        features = np.array([[form.age.data, form.bmi.data, int(form.smoking.data), int(form.alcohol.data), int(form.physical_activity.data), int(form.family_history.data)]])
        probability = model.predict_proba(features)[0][1] if model else 0.5
        risk_level = get_risk_level(probability)
        
        rec = Assessment(user_id=current_user.id, age=form.age.data, bmi=form.bmi.data, smoking=int(form.smoking.data), alcohol=int(form.alcohol.data), physical_activity=int(form.physical_activity.data), family_history=int(form.family_history.data), probability=probability, risk_level=risk_level)
        db.session.add(rec); db.session.commit()
        create_notification(current_user.id, f'{risk_level} Risk Detected', f'Your probability is {round(probability*100,1)}%.', category='risk')
        
        return render_template('result.html', risk_level=risk_level, probability=round(probability*100,1), recommendation=get_recommendation(risk_level), shap_img=generate_shap_plot(features), education_cards=[EDUCATION_LIBRARY[f] for f in get_top_contributors(features) if f in EDUCATION_LIBRARY], emergency=EmergencyContact.query.filter_by(user_id=current_user.id).first())
    return render_template('assessment.html', form=form)

@app.route('/history')
@login_required
def history(): return render_template('history.html', assessments=Assessment.query.filter_by(user_id=current_user.id).order_by(Assessment.timestamp.desc()).all())

@app.route('/reminders', methods=['GET','POST'])
@login_required
def reminders():
    form = ReminderForm()
    if form.validate_on_submit():
        db.session.add(Reminder(user_id=current_user.id, reminder_type=form.reminder_type.data, frequency=form.frequency.data, next_date=form.next_date.data, notes=form.notes.data))
        db.session.commit(); flash('Reminder created!', 'success'); return redirect(url_for('reminders'))
    return render_template('reminders.html', form=form, reminders=Reminder.query.filter_by(user_id=current_user.id).all())

@app.route('/reminders/delete/<int:rem_id>', methods=['POST'])
@login_required
def delete_reminder(rem_id):
    rem = Reminder.query.get_or_404(rem_id)
    if rem.user_id == current_user.id: db.session.delete(rem); db.session.commit()
    return redirect(url_for('reminders'))

@app.route('/emergency', methods=['GET','POST'])
@login_required
def emergency():
    form = EmergencyContactForm()
    contact = EmergencyContact.query.filter_by(user_id=current_user.id).first()
    if form.validate_on_submit():
        if contact: contact.name=form.name.data; contact.relationship=form.relationship.data; contact.phone=form.phone.data
        else: db.session.add(EmergencyContact(user_id=current_user.id, name=form.name.data, relationship=form.relationship.data, phone=form.phone.data))
        db.session.commit(); flash('Saved!', 'success'); return redirect(url_for('emergency'))
    elif request.method == 'GET' and contact: form.name.data=contact.name; form.relationship.data=contact.relationship; form.phone.data=contact.phone
    return render_template('emergency.html', form=form, contact=contact)

@app.route('/notifications')
@login_required
def notifications():
    try:
        all_notifs = Notification.query.filter_by(user_id=current_user.id).order_by(Notification.created_at.desc()).all()
    except Exception as e:
        print(f"Notification DB Error: {e}")
        all_notifs = []
    return render_template('notifications.html', notifications=all_notifs)

@app.route('/notifications/mark_read/<int:n_id>', methods=['POST'])
@login_required
def mark_notification_read(n_id):
    n = Notification.query.get_or_404(n_id)
    if n.user_id == current_user.id: n.is_read = True; db.session.commit()
    return redirect(url_for('notifications'))

@app.route('/reports')
@login_required
def reports(): return render_template('reports.html', assessments=Assessment.query.filter_by(user_id=current_user.id).all(), reminders=Reminder.query.filter_by(user_id=current_user.id).all())

@app.route('/reports/pdf')
@login_required
def download_pdf():
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [Paragraph(f"Health Report: {current_user.email}", styles['Title']), Spacer(1, 20)]
    assessments = Assessment.query.filter_by(user_id=current_user.id).all()
    if assessments: story.append(Paragraph(f"Total Assessments: {len(assessments)}", styles['BodyText']))
    else: story.append(Paragraph("No assessments recorded.", styles['BodyText']))
    doc.build(story); buffer.seek(0)
    return send_file(buffer, mimetype='application/pdf', as_attachment=True, download_name='report.pdf')

@app.route('/reports/csv')
@login_required
def download_csv():
    output = io.StringIO(); writer = csv.writer(output)
    writer.writerow(['Date','Age','BMI','Risk Level'])
    for a in Assessment.query.filter_by(user_id=current_user.id).all():
        writer.writerow([a.timestamp.strftime('%Y-%m-%d'), a.age, a.bmi, a.risk_level])
    return Response(output.getvalue(), mimetype='text/csv', headers={'Content-Disposition': 'attachment; filename=data.csv'})

@app.route('/admin')
@login_required
def admin():
    if not current_user.is_admin:
        flash('Access denied.', 'danger')
        return redirect(url_for('dashboard'))
    try:
        assessments = Assessment.query.all()
        risk_counts = {'Low': 0, 'Medium': 0, 'High': 0}
        for a in assessments:
            if a.risk_level in risk_counts:
                risk_counts[a.risk_level] += 1
                
        return render_template('admin.html',
                               total_users=User.query.count(),
                               total_assess=len(assessments),
                               total_reminders=Reminder.query.count(),
                               total_notifications=Notification.query.count(),
                               risk_counts=risk_counts,
                               risk_chart="",
                               prob_chart="",
                               avg_prob=0)
    except Exception as e:
        print(f"Admin Error: {e}")
        flash('Error loading admin dashboard.', 'danger')
        return redirect(url_for('dashboard'))

@app.route('/api/facilities', methods=['POST'])
@csrf.exempt
@login_required
def nearby_facilities():
    data = request.get_json()
    lat, lon = data.get('lat'), data.get('lon')
    GEOAPIFY_API_KEY = os.environ.get('GEOAPIFY_API_KEY', 'dae4301a29c9463f908529798e9f79e1')
    url = f"https://api.geoapify.com/v2/places?categories=healthcare.hospital,healthcare.clinic_or_praxis&filter=circle:{lon},{lat},5000&bias=proximity:{lon},{lat}&limit=10&apiKey={GEOAPIFY_API_KEY}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200: return jsonify({'error': 'Service error.'}), 500
        features = resp.json().get('features', [])
        facilities = []
        for f in features:
            props = f.get('properties', {})
            facilities.append({'name': props.get('name', 'Unnamed'), 'lat': f['geometry']['coordinates'][1], 'lon': f['geometry']['coordinates'][0], 'distance': round(props.get('distance', 0) / 1000, 2), 'address': props.get('formatted', 'N/A')})
        return jsonify(facilities)
    except Exception as e: return jsonify({'error': str(e)}), 500

# 🚀 VERCEL DEPLOYMENT FIX: Run initialization at module level, not just in __main__
# This ensures the database tables and admin user are created on Vercel startup
with app.app_context():
    try:
        # WARNING: This drops all tables and recreates them cleanly
        db.drop_all() 
        db.create_all()
        
        # Recreate admin with a fresh, valid password hash
        admin = User(email='admin@example.com', password=bcrypt.generate_password_hash('admin123').decode('utf-8'), is_admin=True)
        db.session.add(admin)
        db.session.commit()
        print("✓ Database reset and Admin created successfully.")
    except Exception as e:
        print(f"⚠ Database initialization error: {e}")

if __name__ == '__main__':
    app.run(debug=True)