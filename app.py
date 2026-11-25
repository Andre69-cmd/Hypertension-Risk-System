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
APP_BUILD_DATE = " — Final"
APP_AUTHOR = "Hypertension Risk Research Team"

# ---------------- Config ----------------
app = Flask(__name__)

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'fallback-secret-key-123')

# 🚀 POSTGRESQL & VERCEL FIX: Get Neon Database URL
db_uri = os.environ.get('DATABASE_URL', 'sqlite:////tmp/hypertension_week8.db')

# 🛠️ CRITICAL FIX: Force SQLAlchemy to use psycopg2 instead of psycopg3
if db_uri and db_uri.startswith("postgres://"):
    db_uri = db_uri.replace("postgres://", "postgresql+psycopg2://", 1)
elif db_uri and db_uri.startswith("postgresql://"):
    db_uri = db_uri.replace("postgresql://", "postgresql+psycopg2://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = db_uri
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

# 🚀 NEW: Admin Chart Generator
def generate_admin_charts():
    try:
        assessments = Assessment.query.all()
        risk_counts = {'Low': 0, 'Medium': 0, 'High': 0}
        total_prob = 0.0
        
        for a in assessments:
            if a.risk_level in risk_counts:
                risk_counts[a.risk_level] += 1
            if a.probability is not None:
                total_prob += a.probability
                
        avg_prob = (total_prob / len(assessments)) if assessments else 0.0
        
        # 1. Risk Distribution Bar Chart
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        colors_chart = ['#28a745', '#ffc107', '#dc3545']
        ax1.bar(list(risk_counts.keys()), list(risk_counts.values()), color=colors_chart)
        ax1.set_title('Risk Level Distribution')
        ax1.set_ylabel('Number of Assessments')
        buf1 = io.BytesIO()
        fig1.savefig(buf1, format='png', bbox_inches='tight')
        plt.close(fig1)
        buf1.seek(0)
        risk_chart = base64.b64encode(buf1.read()).decode('utf-8')
        
        # 2. Probability Histogram
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        probs = [a.probability * 100 for a in assessments if a.probability is not None]
        if probs:
            ax2.hist(probs, bins=10, color='#007bff', edgecolor='black')
        ax2.set_title('Probability Histogram')
        ax2.set_xlabel('Probability (%)')
        ax2.set_ylabel('Frequency')
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format='png', bbox_inches='tight')
        plt.close(fig2)
        buf2.seek(0)
        prob_chart = base64.b64encode(buf2.read()).decode('utf-8')
        
        return risk_chart, prob_chart, avg_prob, risk_counts
    except Exception as e:
        print(f"⚠ Admin chart generation error: {e}")
        return "", "", 0.0, {'Low': 0, 'Medium': 0, 'High': 0}

# 🚀 NEW: Advanced Analytics Chart Generator
def generate_analytics_charts():
    try:
        assessments = Assessment.query.all()
        df = pd.DataFrame([{
            'age': a.age, 'bmi': a.bmi, 'smoking': a.smoking, 'alcohol': a.alcohol,
            'physical_activity': a.physical_activity, 'family_history': a.family_history,
            'probability': a.probability, 'risk_level': a.risk_level, 'timestamp': a.timestamp
        } for a in assessments])
        
        # 1. Generate Text Insights
        insights = []
        if len(df) > 0:
            high_risk_count = len(df[df['risk_level'] == 'High'])
            if high_risk_count > 0:
                insights.append(f"There are {high_risk_count} high-risk users requiring immediate attention.")
            avg_bmi = df['bmi'].mean()
            if avg_bmi > 25:
                insights.append(f"Average BMI ({avg_bmi:.1f}) is above the healthy range (18.5-24.9).")
            if len(df[df['smoking'] == 1]) > 0:
                insights.append("Smoking is a contributing factor for some users.")
        if not insights:
            insights.append("No significant risk patterns detected yet. Keep monitoring!")
            
        # 2. 6-Month Trend Data & Chart
        months_labels, monthly_counts, monthly_avg_prob = [], [], []
        trend_chart = ""
        
        if len(df) > 0 and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            end_date = datetime.utcnow()
            for i in range(5, -1, -1):
                month_date = end_date - pd.DateOffset(months=i)
                month_start = month_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                if month_date.month == 12:
                    month_end = month_date.replace(year=month_date.year+1, month=1, day=1) - pd.DateOffset(seconds=1)
                else:
                    month_end = month_date.replace(month=month_date.month+1, day=1) - pd.DateOffset(seconds=1)
                
                month_df = df[(df['timestamp'] >= month_start) & (df['timestamp'] <= month_end)]
                months_labels.append(month_start.strftime('%b %Y'))
                monthly_counts.append(len(month_df))
                avg_p = month_df['probability'].mean() * 100 if len(month_df) > 0 else 0
                monthly_avg_prob.append(round(avg_p, 1))
            
            fig_trend, ax_trend = plt.subplots(figsize=(8, 4))
            ax_trend.plot(months_labels, monthly_counts, marker='o', color='#007bff', linewidth=2)
            ax_trend.set_title('Assessments Over Last 6 Months')
            ax_trend.set_ylabel('Number of Assessments')
            plt.xticks(rotation=45)
            plt.tight_layout()
            buf_trend = io.BytesIO()
            fig_trend.savefig(buf_trend, format='png', bbox_inches='tight')
            plt.close(fig_trend)
            buf_trend.seek(0)
            trend_chart = base64.b64encode(buf_trend.read()).decode('utf-8')
            
        # 3. Risk Pie Chart
        risk_counts = {'Low': 0, 'Medium': 0, 'High': 0}
        if len(df) > 0:
            for r in df['risk_level']:
                if r in risk_counts: risk_counts[r] += 1
        
        fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
        if sum(risk_counts.values()) > 0:
            ax_pie.pie(list(risk_counts.values()), labels=list(risk_counts.keys()), colors=['#28a745', '#ffc107', '#dc3545'], autopct='%1.1f%%', startangle=90)
            ax_pie.set_title('Risk Distribution')
        else:
            ax_pie.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center')
        buf_pie = io.BytesIO()
        fig_pie.savefig(buf_pie, format='png', bbox_inches='tight')
        plt.close(fig_pie)
        buf_pie.seek(0)
        risk_pie = base64.b64encode(buf_pie.read()).decode('utf-8')
        
        # 4. Age vs Risk Scatter Plot
        age_scatter = ""
        if len(df) > 0:
            fig_scatter, ax_scatter = plt.subplots(figsize=(6, 4))
            color_map = {'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545'}
            for risk, color in color_map.items():
                subset = df[df['risk_level'] == risk]
                if len(subset) > 0:
                    ax_scatter.scatter(subset['age'], subset['probability']*100, c=color, label=risk, alpha=0.7)
            ax_scatter.set_title('Age vs Risk Probability')
            ax_scatter.set_xlabel('Age')
            ax_scatter.set_ylabel('Probability (%)')
            ax_scatter.legend()
            buf_scatter = io.BytesIO()
            fig_scatter.savefig(buf_scatter, format='png', bbox_inches='tight')
            plt.close(fig_scatter)
            buf_scatter.seek(0)
            age_scatter = base64.b64encode(buf_scatter.read()).decode('utf-8')
            
        # 5. Average Feature Values by Risk Level (Grouped Bar Chart)
        feature_chart = ""
        if len(df) > 0:
            features = ['age', 'bmi', 'smoking', 'alcohol', 'physical_activity', 'family_history']
            fig_feat, ax_feat = plt.subplots(figsize=(8, 4))
            x = np.arange(len(features))
            width = 0.25
            
            for i, risk in enumerate(['Low', 'Medium', 'High']):
                subset = df[df['risk_level'] == risk]
                if len(subset) > 0:
                    means = [subset[f].mean() for f in features]
                    ax_feat.bar(x + i*width, means, width, label=risk, color=['#28a745', '#ffc107', '#dc3545'][i])
            
            ax_feat.set_title('Average Feature Values by Risk Level')
            ax_feat.set_xticks(x + width)
            ax_feat.set_xticklabels(features, rotation=45)
            ax_feat.legend()
            plt.tight_layout()
            buf_feat = io.BytesIO()
            fig_feat.savefig(buf_feat, format='png', bbox_inches='tight')
            plt.close(fig_feat)
            buf_feat.seek(0)
            feature_chart = base64.b64encode(buf_feat.read()).decode('utf-8')
            
        return insights, trend_chart, months_labels, monthly_counts, monthly_avg_prob, risk_pie, age_scatter, feature_chart, risk_counts
    except Exception as e:
        print(f"⚠ Analytics chart generation error: {e}")
        return ["Error generating insights"], "", [], [], [], "", "", "", {'Low': 0, 'Medium': 0, 'High': 0}

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
            print(f"Login Error: {e}")
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
    
    risk_chart, prob_chart, avg_prob, risk_counts = generate_admin_charts()
                
    return render_template('admin.html',
                           total_users=User.query.count(),
                           total_assess=Assessment.query.count(),
                           total_reminders=Reminder.query.count(),
                           total_notifications=Notification.query.count(),
                           avg_prob=round(avg_prob * 100, 1),
                           risk_counts=risk_counts,
                           risk_chart=risk_chart,
                           prob_chart=prob_chart)

# 🚀 NEW: Analytics Route
@app.route('/analytics')
@login_required
def analytics():
    if not current_user.is_admin:
        flash('Access denied.', 'danger')
        return redirect(url_for('dashboard'))
    
    insights, trend_chart, months_labels, monthly_counts, monthly_avg_prob, risk_pie, age_scatter, feature_chart, risk_counts = generate_analytics_charts()
            
    return render_template('analytics.html', 
                           total_users=User.query.count(), 
                           total_assessments=Assessment.query.count(), 
                           risk_counts=risk_counts,
                           insights=insights,
                           trend_chart=trend_chart,
                           months_labels=months_labels,
                           monthly_counts=monthly_counts,
                           monthly_avg_prob=monthly_avg_prob,
                           risk_pie=risk_pie,
                           age_scatter=age_scatter,
                           feature_chart=feature_chart)

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

# 🚀 SAFE VERCEL DEPLOYMENT FIX: Initialize database safely on startup
with app.app_context():
    try:
        # Create tables safely (does NOT drop existing data, preventing data loss on Vercel cold starts)
        db.create_all()
        
        # Only create admin if it doesn't already exist
        admin = User.query.filter_by(email='admin@example.com').first()
        if not admin:
            new_admin = User(
                email='admin@example.com', 
                password=bcrypt.generate_password_hash('admin123').decode('utf-8'), 
                is_admin=True
            )
            db.session.add(new_admin)
            db.session.commit()
            print("✓ Admin user created successfully.")
        else:
            print("✓ Admin user already exists.")
    except Exception as e:
        print(f"⚠ Database initialization error: {e}")

if __name__ == '__main__':
    app.run(debug=True)