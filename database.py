from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(db.Model, UserMixin):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100), nullable=False)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    age = db.Column(db.Integer, nullable=True)
    gender = db.Column(db.String(20), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    predictions = db.relationship('Prediction', backref='user', lazy=True, cascade="all, delete-orphan")


class Prediction(db.Model):
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    disease = db.Column(db.String(100), nullable=False)
    result = db.Column(db.String(50), nullable=False)  # "Positive" or "Negative"
    confidence = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(20), nullable=False)  # "Low", "Medium", "High"
    input_data = db.Column(db.Text, nullable=False)  # JSON-serialized input dict
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
