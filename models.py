"""
Earth Observation and Analysis — Database Models
=====================================
"""
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()


class User(db.Model):
    __tablename__ = 'users'

    id             = db.Column(db.Integer, primary_key=True)
    name           = db.Column(db.String(120), nullable=False)
    email          = db.Column(db.String(200), unique=True, nullable=False)
    password_hash  = db.Column(db.String(256), nullable=False)
    plan           = db.Column(db.String(20), default='basic')     # basic / pro / enterprise
    plan_status    = db.Column(db.String(20), default='trial')     # trial / active / cancelled / past_due
    trial_ends_at  = db.Column(db.DateTime)
    stripe_customer_id = db.Column(db.String(100))
    stripe_sub_id  = db.Column(db.String(100))
    created_at     = db.Column(db.DateTime, default=datetime.utcnow)
    active         = db.Column(db.Boolean, default=True)

    readings = db.relationship('Reading', backref='owner', lazy=True, cascade='all, delete-orphan')

    def has_access(self):
        if self.plan_status == 'active':
            return True
        if self.plan_status == 'trial' and self.trial_ends_at and self.trial_ends_at > datetime.utcnow():
            return True
        return False

    def to_dict(self):
        return {
            'id': self.id, 'name': self.name, 'email': self.email,
            'plan': self.plan, 'plan_status': self.plan_status,
            'trial_ends': self.trial_ends_at.isoformat() if self.trial_ends_at else None,
            'has_access': self.has_access(),
        }


class ReadingCache(db.Model):
    """
    Caches finished GEE results (stats + thumbnail URL) keyed by the
    exact family/index/dates/ROI combination. If two users (or the same
    user twice) request an identical reading, the second one returns
    instantly instead of re-running the full Earth Engine computation —
    this is the single biggest performance/cost win available, since
    every GEE call has real latency and counts against usage quota.
    """
    __tablename__ = 'reading_cache'

    id          = db.Column(db.Integer, primary_key=True)
    cache_key   = db.Column(db.String(64), unique=True, nullable=False, index=True)
    result_json = db.Column(db.Text, nullable=False)
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at  = db.Column(db.DateTime, nullable=False, index=True)


class Reading(db.Model):
    """One saved analysis result — the sounding log entry."""
    __tablename__ = 'readings'

    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    family      = db.Column(db.String(20))     # water / landsat / geo / veg
    index_name  = db.Column(db.String(30))
    start_date  = db.Column(db.String(20))
    end_date    = db.Column(db.String(20))
    roi_geojson = db.Column(db.Text)
    mean_value  = db.Column(db.Float)
    min_value   = db.Column(db.Float)
    max_value   = db.Column(db.Float)
    std_value   = db.Column(db.Float)
    image_count = db.Column(db.Integer)
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id, 'family': self.family, 'index': self.index_name,
            'start': self.start_date, 'end': self.end_date,
            'mean': self.mean_value, 'min': self.min_value, 'max': self.max_value,
            'std': self.std_value, 'images': self.image_count,
            'created_at': self.created_at.isoformat(),
        }
