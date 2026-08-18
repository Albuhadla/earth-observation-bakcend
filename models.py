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
    is_complimentary = db.Column(db.Boolean, default=False)  # permanent access, bypasses billing entirely — for the developer account and shared test/reviewer accounts
    totp_secret    = db.Column(db.String(64))   # base32 TOTP secret, only set once 2FA setup begins
    totp_enabled   = db.Column(db.Boolean, default=False)  # only true once the user has verified a code — a secret alone doesn't enable 2FA

    readings = db.relationship('Reading', backref='owner', lazy=True, cascade='all, delete-orphan')
    saved_locations = db.relationship('SavedLocation', backref='owner', lazy=True, cascade='all, delete-orphan')

    def has_access(self):
        if self.is_complimentary:
            return True  # permanent access — never expires, no subscription needed
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
            'is_complimentary': self.is_complimentary,
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


class SavedLocation(db.Model):
    """
    A region the user has chosen to monitor ongoing, rather than a
    one-off reading. This is the foundation for automatic recurring
    checks and anomaly detection — a scheduled job (built separately)
    re-runs the same family/index/ROI periodically and compares each
    new result against this location's own historical baseline.
    """
    __tablename__ = 'saved_locations'

    id              = db.Column(db.Integer, primary_key=True)
    user_id         = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    name            = db.Column(db.String(120), nullable=False)   # user-given label, e.g. "Al Ain Farm"
    family          = db.Column(db.String(20), nullable=False)
    index_name      = db.Column(db.String(30), nullable=False)
    roi_geojson     = db.Column(db.Text, nullable=False)
    check_frequency = db.Column(db.String(20), default='monthly')  # weekly / monthly
    active          = db.Column(db.Boolean, default=True)
    last_checked_at = db.Column(db.DateTime)
    last_mean_value = db.Column(db.Float)
    last_status     = db.Column(db.String(20))       # 'normal' / 'anomaly' — set by the check job
    created_at      = db.Column(db.DateTime, default=datetime.utcnow)

    history = db.relationship('LocationHistory', backref='location', lazy=True, cascade='all, delete-orphan')

    def to_dict(self):
        return {
            'id': self.id, 'name': self.name, 'family': self.family, 'index': self.index_name,
            'roi': self.roi_geojson, 'check_frequency': self.check_frequency, 'active': self.active,
            'last_checked_at': self.last_checked_at.isoformat() if self.last_checked_at else None,
            'last_mean_value': self.last_mean_value, 'last_status': self.last_status,
            'created_at': self.created_at.isoformat(),
        }


class LocationHistory(db.Model):
    """
    Every historical reading for a saved location, kept separately
    from the general Reading log — this is what a seasonal baseline
    ("the average August NDVI for THIS farm across past years") gets
    computed from, and what an anomaly gets compared against.
    """
    __tablename__ = 'location_history'

    id           = db.Column(db.Integer, primary_key=True)
    location_id  = db.Column(db.Integer, db.ForeignKey('saved_locations.id'), nullable=False)
    checked_at   = db.Column(db.DateTime, default=datetime.utcnow)
    mean_value   = db.Column(db.Float)
    status       = db.Column(db.String(20))            # 'normal' / 'anomaly'
    deviation_pct = db.Column(db.Float)                 # % vs. seasonal baseline, if one existed yet

    def to_dict(self):
        return {
            'id': self.id, 'checked_at': self.checked_at.isoformat(),
            'mean_value': self.mean_value, 'status': self.status, 'deviation_pct': self.deviation_pct,
        }


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
