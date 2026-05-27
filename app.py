"""
=============================================================
  app.py — Smart Health Prediction System (Flask Backend)
=============================================================
  Run with:
      python app.py
=============================================================
"""

import os, json, io, base64, warnings
warnings.filterwarnings('ignore')

from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from flask import (Flask, render_template, request, redirect,
                   url_for, flash, session, jsonify, send_file, abort)
from flask_login import (LoginManager, login_user, logout_user,
                         login_required, current_user)
from werkzeug.security import generate_password_hash, check_password_hash

from database import db, User, Prediction
from report_generator import generate_pdf_report

# ─────────────────────────────────────────────
# App Configuration
# ─────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = 'health_predict_secret_2024_xK9m'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///health_system.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")

# ─────────────────────────────────────────────
# Helper — load a model bundle safely
# ─────────────────────────────────────────────
def load_bundle(prefix):
    """Returns (model, scaler, imputer, features) or (None,…) if missing."""
    try:
        model   = joblib.load(os.path.join(MODEL_DIR, f"{prefix}_model.pkl"))
        scaler  = joblib.load(os.path.join(MODEL_DIR, f"{prefix}_scaler.pkl"))
        imputer = joblib.load(os.path.join(MODEL_DIR, f"{prefix}_imputer.pkl"))
        feats   = joblib.load(os.path.join(MODEL_DIR, f"{prefix}_features.pkl"))
        return model, scaler, imputer, feats
    except FileNotFoundError:
        return None, None, None, None

# ─────────────────────────────────────────────
# Helper — compute risk level
# ─────────────────────────────────────────────
def risk_level(confidence, positive):
    if not positive:
        return "Low"
    if confidence < 60:
        return "Low"
    elif confidence < 80:
        return "Medium"
    return "High"

# ─────────────────────────────────────────────
# Helper — generate chart as base64 PNG
# ─────────────────────────────────────────────
def gauge_chart(confidence, result):
    fig, ax = plt.subplots(figsize=(5, 2.8), subplot_kw=dict(aspect="auto"))
    fig.patch.set_facecolor('#0d1b2a')
    ax.set_facecolor('#0d1b2a')

    color = '#ef4444' if result == "Positive" else '#22c55e'
    bg_color = '#1e3a5f'

    ax.barh(0, 100, height=0.5, color=bg_color, edgecolor='none')
    ax.barh(0, confidence, height=0.5, color=color, edgecolor='none')

    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(colors='#94a3b8', labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(50, -0.55, f"{confidence:.1f}% Confidence",
            ha='center', va='top', color='white', fontsize=13, fontweight='bold')

    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def history_chart(predictions):
    """Bar chart of disease-wise prediction counts for the current user."""
    from collections import Counter
    counts = Counter(p.disease for p in predictions)
    if not counts:
        return None

    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor('#0d1b2a')
    ax.set_facecolor('#0d1b2a')

    diseases = list(counts.keys())
    vals     = list(counts.values())
    colors   = ['#3b82f6', '#ef4444', '#f59e0b', '#22c55e'][:len(diseases)]

    bars = ax.bar(diseases, vals, color=colors, edgecolor='none', width=0.5)
    ax.set_ylabel("Predictions", color='#94a3b8')
    ax.set_title("Your Prediction History", color='white', fontsize=13)
    ax.tick_params(colors='#94a3b8')
    for spine in ax.spines.values():
        spine.set_color('#1e3a5f')

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                str(val), ha='center', color='white', fontsize=10)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ─────────────────────────────────────────────
# Flask-Login user loader
# ─────────────────────────────────────────────
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ═══════════════════════════════════════════════
#  AUTH ROUTES
# ═══════════════════════════════════════════════
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user, remember=True)
            return redirect(url_for('dashboard'))
        flash('Invalid username or password.', 'danger')
    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        username  = request.form.get('username', '').strip()
        email     = request.form.get('email', '').strip()
        password  = request.form.get('password', '')
        confirm   = request.form.get('confirm_password', '')
        age       = request.form.get('age', type=int)
        gender    = request.form.get('gender', '')

        if not all([full_name, username, email, password]):
            flash('All fields are required.', 'danger')
        elif password != confirm:
            flash('Passwords do not match.', 'danger')
        elif len(password) < 6:
            flash('Password must be at least 6 characters.', 'danger')
        elif User.query.filter_by(username=username).first():
            flash('Username already taken.', 'danger')
        elif User.query.filter_by(email=email).first():
            flash('Email already registered.', 'danger')
        else:
            user = User(
                full_name=full_name, username=username, email=email,
                password=generate_password_hash(password),
                age=age, gender=gender
            )
            db.session.add(user)
            db.session.commit()
            flash('Account created! Please log in.', 'success')
            return redirect(url_for('login'))
    return render_template('signup.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


# ═══════════════════════════════════════════════
#  DASHBOARD
# ═══════════════════════════════════════════════
@app.route('/dashboard')
@login_required
def dashboard():
    predictions = Prediction.query.filter_by(
        user_id=current_user.id).order_by(Prediction.created_at.desc()).limit(5).all()
    all_preds   = Prediction.query.filter_by(user_id=current_user.id).all()

    chart_b64 = history_chart(all_preds)

    stats = {
        'total'   : len(all_preds),
        'positive': sum(1 for p in all_preds if p.result == 'Positive'),
        'negative': sum(1 for p in all_preds if p.result == 'Negative'),
        'diseases': len(set(p.disease for p in all_preds)),
    }

    return render_template('dashboard.html',
                           recent_predictions=predictions,
                           chart_b64=chart_b64,
                           stats=stats)


# ═══════════════════════════════════════════════
#  PREDICTION ROUTES  (generic helper)
# ═══════════════════════════════════════════════
DISEASE_META = {
    'diabetes': {
        'title' : 'Diabetes Prediction',
        'prefix': 'diabetes',
        'fields': [
            {'name':'Pregnancies',              'label':'Pregnancies',              'type':'number','min':0,  'max':20,  'step':1,    'unit':'',     'info':'Number of times pregnant'},
            {'name':'Glucose',                  'label':'Glucose Level (mg/dL)',    'type':'number','min':0,  'max':300, 'step':1,    'unit':'mg/dL','info':'Plasma glucose concentration (2h oral test)'},
            {'name':'BloodPressure',            'label':'Blood Pressure (mmHg)',    'type':'number','min':0,  'max':200, 'step':1,    'unit':'mmHg', 'info':'Diastolic blood pressure'},
            {'name':'SkinThickness',            'label':'Skin Thickness (mm)',      'type':'number','min':0,  'max':100, 'step':1,    'unit':'mm',   'info':'Triceps skin fold thickness'},
            {'name':'Insulin',                  'label':'Insulin (µU/ml)',          'type':'number','min':0,  'max':900, 'step':1,    'unit':'µU/ml','info':'2-Hour serum insulin'},
            {'name':'BMI',                      'label':'BMI',                      'type':'number','min':0,  'max':70,  'step':0.1,  'unit':'',     'info':'Body mass index (weight/height²)'},
            {'name':'DiabetesPedigreeFunction', 'label':'Diabetes Pedigree Function','type':'number','min':0, 'max':3,   'step':0.001,'unit':'',     'info':'Diabetes hereditary risk score'},
            {'name':'Age',                      'label':'Age (years)',              'type':'number','min':1,  'max':120, 'step':1,    'unit':'yrs',  'info':'Patient age in years'},
        ],
        'suggestions': {
            True : ['Monitor blood sugar levels regularly.','Follow a low-carb diet.','Exercise for at least 30 min/day.','Consult an endocrinologist.','Stay hydrated and avoid sugary drinks.'],
            False: ['Maintain a healthy weight.','Continue regular exercise.','Eat balanced meals.','Schedule annual health checkups.','Avoid processed food.'],
        }
    },
    'heart': {
        'title' : 'Heart Disease Prediction',
        'prefix': 'heart',
        'fields': [
            {'name':'age',      'label':'Age',                      'type':'number','min':1,  'max':120,'step':1,  'unit':'yrs', 'info':'Patient age'},
            {'name':'sex',      'label':'Sex (1=Male, 0=Female)',   'type':'number','min':0,  'max':1,  'step':1,  'unit':'',    'info':'1 for male, 0 for female'},
            {'name':'cp',       'label':'Chest Pain Type (0-3)',    'type':'number','min':0,  'max':3,  'step':1,  'unit':'',    'info':'0=typical angina, 1=atypical, 2=non-anginal, 3=asymptomatic'},
            {'name':'trestbps', 'label':'Resting Blood Pressure',   'type':'number','min':80, 'max':220,'step':1,  'unit':'mmHg','info':'Resting blood pressure on admission'},
            {'name':'chol',     'label':'Serum Cholesterol',        'type':'number','min':100,'max':600,'step':1,  'unit':'mg/dl','info':'Serum cholesterol in mg/dl'},
            {'name':'fbs',      'label':'Fasting Blood Sugar >120', 'type':'number','min':0,  'max':1,  'step':1,  'unit':'',    'info':'1 = true (sugar > 120 mg/dl)'},
            {'name':'restecg',  'label':'Resting ECG Result (0-2)', 'type':'number','min':0,  'max':2,  'step':1,  'unit':'',    'info':'0=normal, 1=ST-T wave abnormality, 2=LV hypertrophy'},
            {'name':'thalach',  'label':'Max Heart Rate Achieved',  'type':'number','min':60, 'max':220,'step':1,  'unit':'bpm', 'info':'Maximum heart rate achieved'},
            {'name':'exang',    'label':'Exercise Induced Angina',  'type':'number','min':0,  'max':1,  'step':1,  'unit':'',    'info':'1 = yes, 0 = no'},
            {'name':'oldpeak',  'label':'ST Depression',            'type':'number','min':0,  'max':7,  'step':0.1,'unit':'',    'info':'ST depression induced by exercise relative to rest'},
            {'name':'slope',    'label':'Slope of Peak ST (0-2)',   'type':'number','min':0,  'max':2,  'step':1,  'unit':'',    'info':'0=upsloping, 1=flat, 2=downsloping'},
            {'name':'ca',       'label':'Major Vessels (0-3)',      'type':'number','min':0,  'max':3,  'step':1,  'unit':'',    'info':'Number of major vessels colored by fluoroscopy'},
            {'name':'thal',     'label':'Thalassemia (0-3)',        'type':'number','min':0,  'max':3,  'step':1,  'unit':'',    'info':'0=normal, 1=fixed defect, 2=reversable defect'},
        ],
        'suggestions': {
            True : ['Consult a cardiologist immediately.','Adopt a heart-healthy diet (less saturated fat).','Stop smoking and limit alcohol.','Monitor blood pressure daily.','Begin light cardio under medical supervision.'],
            False: ['Maintain healthy cholesterol levels.','Exercise regularly.','Keep a healthy weight.','Manage stress effectively.','Get annual heart checkups.'],
        }
    },
    'kidney': {
        'title' : 'Kidney Disease Prediction',
        'prefix': 'kidney',
        'fields': [
            {'name':'age',   'label':'Age',                  'type':'number','min':1,  'max':120,'step':1,  'unit':'yrs',   'info':'Patient age'},
            {'name':'bp',    'label':'Blood Pressure',       'type':'number','min':50, 'max':200,'step':1,  'unit':'mmHg',  'info':'Diastolic blood pressure'},
            {'name':'sg',    'label':'Specific Gravity',     'type':'number','min':1.0,'max':1.03,'step':0.001,'unit':'',   'info':'Urine specific gravity'},
            {'name':'al',    'label':'Albumin (0-5)',        'type':'number','min':0,  'max':5,  'step':1,  'unit':'',      'info':'Albumin in urine (0=absent to 5=heavy)'},
            {'name':'su',    'label':'Sugar (0-5)',          'type':'number','min':0,  'max':5,  'step':1,  'unit':'',      'info':'Sugar in urine (0=absent to 5=heavy)'},
            {'name':'bgr',   'label':'Blood Glucose (mg/dl)','type':'number','min':22, 'max':500,'step':1,  'unit':'mg/dl', 'info':'Random blood glucose'},
            {'name':'bu',    'label':'Blood Urea (mg/dl)',   'type':'number','min':1,  'max':400,'step':1,  'unit':'mg/dl', 'info':'Blood urea levels'},
            {'name':'sc',    'label':'Serum Creatinine',     'type':'number','min':0.4,'max':80, 'step':0.1,'unit':'mg/dl', 'info':'Serum creatinine'},
            {'name':'sod',   'label':'Sodium (mEq/L)',       'type':'number','min':111,'max':163,'step':1,  'unit':'mEq/L', 'info':'Serum sodium'},
            {'name':'pot',   'label':'Potassium (mEq/L)',    'type':'number','min':2.5,'max':47, 'step':0.1,'unit':'mEq/L', 'info':'Serum potassium'},
            {'name':'hemo',  'label':'Hemoglobin (g/dL)',    'type':'number','min':3.1,'max':17.8,'step':0.1,'unit':'g/dL','info':'Hemoglobin level'},
            {'name':'pcv',   'label':'Packed Cell Volume',   'type':'number','min':9,  'max':54, 'step':1,  'unit':'%',     'info':'Packed cell volume'},
            {'name':'wc',    'label':'WBC Count (/cmm)',      'type':'number','min':2200,'max':11000,'step':100,'unit':'/cmm','info':'White blood cell count'},
            {'name':'rc',    'label':'RBC Count (mill/cmm)', 'type':'number','min':2.1,'max':8,  'step':0.1,'unit':'mill/cmm','info':'Red blood cell count'},
        ],
        'suggestions': {
            True : ['Consult a nephrologist promptly.','Limit protein and potassium intake.','Control blood pressure strictly.','Stay well-hydrated (as advised by doctor).','Avoid NSAIDs and nephrotoxic drugs.'],
            False: ['Drink adequate water daily.','Avoid excessive salt intake.','Monitor blood pressure.','Get annual kidney function tests.','Exercise moderately.'],
        }
    },
    'liver': {
        'title' : 'Liver Disease Prediction',
        'prefix': 'liver',
        'fields': [
            {'name':'Age',                       'label':'Age',                            'type':'number','min':1,  'max':120,'step':1,  'unit':'yrs',  'info':'Patient age'},
            {'name':'Gender',                    'label':'Gender (1=Male, 0=Female)',      'type':'number','min':0,  'max':1,  'step':1,  'unit':'',     'info':'1 for male, 0 for female'},
            {'name':'Total_Bilirubin',           'label':'Total Bilirubin (mg/dL)',        'type':'number','min':0.1,'max':75, 'step':0.1,'unit':'mg/dL','info':'Total bilirubin level'},
            {'name':'Direct_Bilirubin',          'label':'Direct Bilirubin (mg/dL)',       'type':'number','min':0.1,'max':20, 'step':0.1,'unit':'mg/dL','info':'Direct bilirubin level'},
            {'name':'Alkaline_Phosphotase',      'label':'Alkaline Phosphotase (IU/L)',    'type':'number','min':63, 'max':2110,'step':1, 'unit':'IU/L', 'info':'Alkaline phosphotase enzyme'},
            {'name':'Alamine_Aminotransferase',  'label':'ALT / SGPT (IU/L)',              'type':'number','min':10, 'max':2000,'step':1, 'unit':'IU/L', 'info':'Alamine aminotransferase'},
            {'name':'Aspartate_Aminotransferase','label':'AST / SGOT (IU/L)',              'type':'number','min':10, 'max':5000,'step':1, 'unit':'IU/L', 'info':'Aspartate aminotransferase'},
            {'name':'Total_Protiens',            'label':'Total Proteins (g/dL)',           'type':'number','min':2.7,'max':9.6,'step':0.1,'unit':'g/dL','info':'Total protein level'},
            {'name':'Albumin',                   'label':'Albumin (g/dL)',                 'type':'number','min':0.9,'max':5.5,'step':0.1,'unit':'g/dL','info':'Albumin level'},
            {'name':'Albumin_and_Globulin_Ratio','label':'Albumin / Globulin Ratio',       'type':'number','min':0.3,'max':2.8,'step':0.01,'unit':'',   'info':'Ratio of albumin to globulin'},
        ],
        'suggestions': {
            True : ['See a hepatologist immediately.','Avoid alcohol completely.','Follow a liver-friendly diet (low-fat).','Avoid hepatotoxic medications.','Get a hepatitis B & C screening.'],
            False: ['Drink alcohol in moderation or abstain.','Eat a balanced diet rich in vegetables.','Exercise to maintain healthy weight.','Get regular liver function tests.','Avoid unnecessary medication overuse.'],
        }
    },
}


@app.route('/predict/<disease>', methods=['GET', 'POST'])
@login_required
def predict(disease):
    if disease not in DISEASE_META:
        abort(404)

    meta = DISEASE_META[disease]
    result_data = None

    if request.method == 'POST':
        model, scaler, imputer, features = load_bundle(meta['prefix'])

        if model is None:
            flash(f'Model not trained yet. Run: python models/train_models.py', 'warning')
            return render_template('predict.html', meta=meta, disease=disease, result=None)

        # ── Collect form values in feature order ──
        input_dict = {}
        for feat in features:
            # Try exact name, then lowercase, then case-insensitive
            val = request.form.get(feat) or request.form.get(feat.lower())
            if val is None:
                for k in request.form:
                    if k.lower() == feat.lower():
                        val = request.form[k]
                        break
            try:
                input_dict[feat] = float(val) if val not in (None, '') else np.nan
            except (ValueError, TypeError):
                input_dict[feat] = np.nan

        X_raw = pd.DataFrame([input_dict], columns=features)
        X_imp = imputer.transform(X_raw)
        X_sc  = scaler.transform(X_imp)

        pred        = model.predict(X_sc)[0]
        proba       = model.predict_proba(X_sc)[0]
        positive    = bool(pred == 1)
        confidence  = float(proba[1] if positive else proba[0]) * 100
        risk        = risk_level(confidence, positive)
        suggestions = meta['suggestions'][positive]
        chart_b64   = gauge_chart(confidence, "Positive" if positive else "Negative")

        # ── Save to DB ──
        record = Prediction(
            user_id    = current_user.id,
            disease    = meta['title'],
            result     = "Positive" if positive else "Negative",
            confidence = round(confidence, 2),
            risk_level = risk,
            input_data = json.dumps(input_dict)
        )
        db.session.add(record)
        db.session.commit()

        result_data = {
            'positive'   : positive,
            'confidence' : round(confidence, 2),
            'risk'       : risk,
            'suggestions': suggestions,
            'chart_b64'  : chart_b64,
            'pred_id'    : record.id,
        }

    return render_template('predict.html', meta=meta, disease=disease, result=result_data)


# ═══════════════════════════════════════════════
#  HISTORY
# ═══════════════════════════════════════════════
@app.route('/history')
@login_required
def history():
    predictions = Prediction.query.filter_by(
        user_id=current_user.id).order_by(Prediction.created_at.desc()).all()
    return render_template('history.html', predictions=predictions)


# ═══════════════════════════════════════════════
#  PDF REPORT
# ═══════════════════════════════════════════════
@app.route('/report/<int:pred_id>')
@login_required
def download_report(pred_id):
    pred = Prediction.query.get_or_404(pred_id)
    if pred.user_id != current_user.id:
        abort(403)

    pdf_bytes = generate_pdf_report(pred, current_user)
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f"health_report_{pred.disease.replace(' ','_')}_{pred.id}.pdf"
    )


# ═══════════════════════════════════════════════
#  TOOLS — BMI / CALORIE / CHATBOT
# ═══════════════════════════════════════════════
@app.route('/tools')
@login_required
def tools():
    return render_template('tools.html')


@app.route('/api/bmi', methods=['POST'])
@login_required
def calc_bmi():
    data   = request.json
    weight = float(data.get('weight', 0))
    height = float(data.get('height', 1)) / 100   # cm → m
    bmi    = round(weight / (height ** 2), 2) if height > 0 else 0

    if   bmi < 18.5: cat = "Underweight"
    elif bmi < 25:   cat = "Normal"
    elif bmi < 30:   cat = "Overweight"
    else:            cat = "Obese"

    return jsonify({'bmi': bmi, 'category': cat})


@app.route('/api/calorie', methods=['POST'])
@login_required
def calc_calorie():
    data     = request.json
    weight   = float(data.get('weight', 70))
    height   = float(data.get('height', 170))
    age      = int(data.get('age', 25))
    gender   = data.get('gender', 'male')
    activity = float(data.get('activity', 1.55))

    # Mifflin-St Jeor
    if gender == 'male':
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    tdee = round(bmr * activity)
    return jsonify({'bmr': round(bmr), 'tdee': tdee,
                    'lose': tdee - 500, 'gain': tdee + 500})


@app.route('/api/chatbot', methods=['POST'])
@login_required
def chatbot():
    """Rule-based symptom checker chatbot."""
    msg = request.json.get('message', '').lower()

    responses = {
        ('fever', 'temperature', 'hot'): "Fever can indicate infection. Drink fluids, rest, and monitor temperature. If above 103°F (39.4°C) for more than 3 days, see a doctor.",
        ('headache', 'head pain', 'migraine'): "Headaches can be caused by dehydration, stress, or tension. Try resting in a dark room and staying hydrated. Persistent severe headaches need medical attention.",
        ('chest pain', 'chest', 'heart'): "⚠️ Chest pain can be serious. If it is severe or radiates to your arm/jaw, call emergency services immediately. Do not ignore chest pain.",
        ('diabetes', 'blood sugar', 'glucose'): "Diabetes symptoms include increased thirst, frequent urination, and fatigue. Use our Diabetes Prediction tool for a risk assessment.",
        ('kidney', 'urine', 'urination'): "Kidney issues may cause changes in urination, swelling, or back pain. Try our Kidney Disease Prediction tool.",
        ('liver', 'jaundice', 'yellow'): "Liver problems can cause jaundice (yellow skin/eyes), fatigue, and abdominal pain. Try our Liver Disease Prediction tool.",
        ('bmi', 'weight', 'obese', 'overweight'): "Use our BMI Calculator under Health Tools to check your Body Mass Index and get personalized advice.",
        ('hello', 'hi', 'hey'): "Hello! I'm HealthBot 🤖. Ask me about symptoms, diseases, or use our prediction tools for a detailed health assessment.",
        ('help', 'what can you do'): "I can help with: symptom information, disease risk factors, directing you to our prediction tools, and general health tips.",
    }

    reply = "I'm not sure about that specific query. For accurate medical advice, please consult a healthcare professional. You can also try our disease prediction tools for a risk assessment."

    for keywords, response in responses.items():
        if any(kw in msg for kw in keywords):
            reply = response
            break

    return jsonify({'reply': reply})


# ═══════════════════════════════════════════════
#  PROFILE
# ═══════════════════════════════════════════════
@app.route('/profile')
@login_required
def profile():
    all_preds = Prediction.query.filter_by(user_id=current_user.id).all()
    chart     = history_chart(all_preds)
    return render_template('profile.html', chart_b64=chart, predictions=all_preds)


# ═══════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print("[SUCCESS] Database initialised")
    print("[SUCCESS] Smart Health Prediction System running at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)