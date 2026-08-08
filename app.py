"""
Earth Observation and Analysis — Backend
==============================
  POST /api/auth/register        Register (+ 14-day trial)
  POST /api/auth/login           Login → JWT
  GET  /api/auth/me              Current user

  POST /api/payments/subscribe   Create/change Stripe subscription
  POST /api/payments/webhook     Stripe webhook

  POST /api/analysis/run         Run a reading (subscription required)
  POST /api/analysis/timeseries  Monthly trend (subscription required)
  GET  /api/analysis/history     This user's saved readings
"""
import os, logging
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv

from models import db, User, Reading
from auth import auth_bp, token_required
from payments import payments_bp
import gee_engine

load_dotenv()
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'change-me-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///razaza.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

CORS(app, origins=os.getenv('ALLOWED_ORIGINS', '*').split(','))
db.init_app(app)

app.register_blueprint(auth_bp, url_prefix='/api/auth')
app.register_blueprint(payments_bp, url_prefix='/api/payments')


def subscription_required(f):
    """Wrap a token_required view to also require an active plan/trial."""
    import functools
    @functools.wraps(f)
    def wrapper(user, *args, **kwargs):
        if not user.has_access():
            return jsonify({'error': 'Your trial or subscription has ended. Please subscribe to continue.',
                             'code': 'SUBSCRIPTION_REQUIRED'}), 402
        return f(user, *args, **kwargs)
    return wrapper


@app.route('/api/health')
def health():
    # Actually attempt initialisation here (not just check the package
    # imported) so this endpoint tells the truth about whether real GEE
    # calls will work, with the real error if not.
    ee_ready = gee_engine.EE_AVAILABLE and gee_engine.init_ee()
    return jsonify({
        'status': 'ok',
        'gee_package_installed': gee_engine.EE_AVAILABLE,
        'gee_available': ee_ready,
        'gee_init_error': gee_engine._last_ee_init_error if not ee_ready else None
    })


@app.route('/api/analysis/run', methods=['POST'])
@token_required
@subscription_required
def analysis_run(user):
    d = request.get_json()
    family, index_v = d.get('family'), d.get('index')
    start, end, coords = d.get('start'), d.get('end'), d.get('roi')

    if not all([family, index_v, start, end, coords]):
        return jsonify({'error': 'family, index, start, end and roi are all required.'}), 400
    if len(coords) < 3:
        return jsonify({'error': 'Region needs at least 3 points.'}), 400

    result = gee_engine.run_reading(family, index_v, start, end, coords)
    if 'error' in result:
        return jsonify(result), 422

    reading = Reading(
        user_id=user.id, family=family, index_name=index_v,
        start_date=start, end_date=end,
        mean_value=result['mean'], min_value=result['min'],
        max_value=result['max'], std_value=result['std'],
        image_count=result['images']
    )
    db.session.add(reading)
    db.session.commit()

    result['reading_id'] = reading.id
    return jsonify(result)


@app.route('/api/analysis/timeseries', methods=['POST'])
@token_required
@subscription_required
def analysis_timeseries(user):
    d = request.get_json()
    family, index_v = d.get('family'), d.get('index')
    start, end, coords = d.get('start'), d.get('end'), d.get('roi')
    if not all([family, index_v, start, end, coords]):
        return jsonify({'error': 'family, index, start, end and roi are all required.'}), 400

    result = gee_engine.run_timeseries(family, index_v, start, end, coords)
    return jsonify(result)


@app.route('/api/analysis/history', methods=['GET'])
@token_required
def analysis_history(user):
    readings = Reading.query.filter_by(user_id=user.id).order_by(Reading.created_at.desc()).limit(100).all()
    return jsonify([r.to_dict() for r in readings])


@app.errorhandler(404)
def not_found(e): return jsonify({'error': 'Not found.'}), 404

@app.errorhandler(500)
def server_error(e): return jsonify({'error': 'Internal server error.'}), 500


# Create tables at import time — this runs whether the app is started
# with `python app.py` (dev) or through gunicorn/Railway (production).
# Wrapped defensively so a transient DB hiccup on boot doesn't crash-loop
# the whole container.
try:
    with app.app_context():
        db.create_all()
        logging.info('Database tables ready.')
except Exception as e:
    logging.error(f'db.create_all() failed on startup: {e}')


if __name__ == '__main__':
    _port_raw = os.getenv('PORT', '5050')
    port = int(_port_raw) if _port_raw and _port_raw.isdigit() else 5050
    print(f'[Earth Observation and Analysis] Backend running at http://localhost:{port}')
    print(f'[Earth Observation and Analysis] Earth Engine available: {gee_engine.EE_AVAILABLE}')
    app.run(host='0.0.0.0', port=port, debug=os.getenv('FLASK_ENV')=='development')
