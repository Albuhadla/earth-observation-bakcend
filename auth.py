"""
Earth Observation and Analysis — Authentication
=====================================
JWT register/login, bcrypt password hashing.
"""
import os, jwt, bcrypt
from datetime import datetime, timedelta
from functools import wraps
from flask import Blueprint, request, jsonify
from models import db, User

auth_bp = Blueprint('auth', __name__)
SECRET  = os.getenv('SECRET_KEY', 'change-me-in-production')
TOKEN_DAYS = int(os.getenv('TOKEN_EXPIRY_DAYS', 30))


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')
        token = auth_header.split(' ')[1] if auth_header.startswith('Bearer ') else None
        if not token:
            return jsonify({'error': 'Missing authentication token.'}), 401
        try:
            payload = jwt.decode(token, SECRET, algorithms=['HS256'])
            user = User.query.get(payload['user_id'])
            if not user or not user.active:
                return jsonify({'error': 'User not found.'}), 401
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Session expired. Sign in again.'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token.'}), 401
        return f(user, *args, **kwargs)
    return decorated


def optional_token(f):
    """Like token_required but proceeds with user=None if no/invalid token."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')
        token = auth_header.split(' ')[1] if auth_header.startswith('Bearer ') else None
        user = None
        if token:
            try:
                payload = jwt.decode(token, SECRET, algorithms=['HS256'])
                user = User.query.get(payload['user_id'])
            except Exception:
                user = None
        return f(user, *args, **kwargs)
    return decorated


def gen_token(user_id):
    return jwt.encode({'user_id': user_id, 'exp': datetime.utcnow()+timedelta(days=TOKEN_DAYS)}, SECRET, algorithm='HS256')

def hash_pw(pw): return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
def check_pw(pw, h): return bcrypt.checkpw(pw.encode(), h.encode())


@auth_bp.route('/register', methods=['POST'])
def register():
    d = request.get_json()
    name, email, pw = d.get('name','').strip(), (d.get('email','') or '').lower().strip(), d.get('password','')
    plan = d.get('plan', 'basic')
    if plan not in ('basic','pro','enterprise'): plan = 'basic'

    if not name or not email or len(pw) < 8:
        return jsonify({'error': 'Name, email, and an 8+ character password are required.'}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'An account with this email already exists.'}), 409

    user = User(name=name, email=email, password_hash=hash_pw(pw),
                plan=plan, plan_status='trial',
                trial_ends_at=datetime.utcnow()+timedelta(days=14))
    db.session.add(user); db.session.commit()

    try:
        from payments import create_stripe_customer
        cid = create_stripe_customer(user)
        if cid: user.stripe_customer_id = cid; db.session.commit()
    except Exception:
        pass

    return jsonify({'token': gen_token(user.id), 'user': user.to_dict()}), 201


@auth_bp.route('/login', methods=['POST'])
def login():
    d = request.get_json()
    email, pw = (d.get('email','') or '').lower().strip(), d.get('password','')
    user = User.query.filter_by(email=email).first()
    if not user or not check_pw(pw, user.password_hash):
        return jsonify({'error': 'Incorrect email or password.'}), 401
    return jsonify({'token': gen_token(user.id), 'user': user.to_dict()})


@auth_bp.route('/me', methods=['GET'])
@token_required
def me(user):
    return jsonify(user.to_dict())
