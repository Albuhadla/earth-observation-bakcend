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
import os, logging, json
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv

from models import db, User, Reading, SavedLocation, LocationHistory
from auth import auth_bp, token_required
from payments import payments_bp
import gee_engine
import cache as result_cache

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Error monitoring (Sentry) ───────────────────────────────────
# Optional — only activates if SENTRY_DSN is set. Without it, the app
# runs exactly as before; this never blocks startup if the package or
# DSN is missing.
SENTRY_DSN = os.getenv('SENTRY_DSN')
if SENTRY_DSN:
    try:
        import sentry_sdk
        from sentry_sdk.integrations.flask import FlaskIntegration
        sentry_sdk.init(
            dsn=SENTRY_DSN,
            integrations=[FlaskIntegration()],
            traces_sample_rate=0.2,   # 20% of requests traced for performance data
            environment=os.getenv('FLASK_ENV', 'production'),
        )
        logger.info('Sentry error monitoring active.')
    except Exception as e:
        logger.warning(f'Sentry init failed (continuing without it): {e}')

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'change-me-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///razaza.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# Connection pool tuning — matters once on PostgreSQL with real concurrent
# traffic; harmless no-ops on SQLite. pool_pre_ping avoids the classic
# "server closed the connection unexpectedly" error after idle periods.
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_recycle': 280,
}

_allowed_origins_raw = os.getenv('ALLOWED_ORIGINS', '*')
_allowed_origins = [o.strip() for o in _allowed_origins_raw.split(',') if o.strip()]
CORS(app, origins=_allowed_origins)
db.init_app(app)

# ── Rate limiting ────────────────────────────────────────────────
# Protects both your Earth Engine quota and your hosting bill from
# accidental (or malicious) request floods. Falls back to an in-memory
# limiter if no separate storage is configured — fine for a single
# backend instance, upgrade to Redis storage if you ever run multiple.
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    limiter = Limiter(
        app=app, key_func=get_remote_address,
        default_limits=['200 per hour'],
        storage_uri=os.getenv('RATELIMIT_STORAGE_URI', 'memory://'),
    )
    logger.info('Rate limiting active.')
except Exception as e:
    limiter = None
    logger.warning(f'Rate limiting unavailable (continuing without it): {e}')

def rate_limit(limit_string):
    """A limit decorator that's a no-op if flask-limiter didn't load —
    every route can use this unconditionally without extra if-checks."""
    def decorator(f):
        return limiter.limit(limit_string)(f) if limiter else f
    return decorator

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


def enterprise_required(f):
    """
    Like subscription_required, but specifically for the Enterprise
    tier's advanced analytics tools (tree counting, water body
    inventory, land classification) — these are meaningfully more
    computationally expensive than a normal reading, so they're kept
    as a distinct, higher-tier feature rather than bundled into every
    plan.
    """
    import functools
    @functools.wraps(f)
    def wrapper(user, *args, **kwargs):
        if not user.has_access():
            return jsonify({'error': 'Your trial or subscription has ended. Please subscribe to continue.',
                             'code': 'SUBSCRIPTION_REQUIRED'}), 402
        if user.plan != 'enterprise':
            return jsonify({'error': 'Advanced analytics (tree counting, water body inventory, land classification) require the Enterprise plan.',
                             'code': 'ENTERPRISE_REQUIRED'}), 402
        return f(user, *args, **kwargs)
    return wrapper


def pro_or_higher_required(f):
    """
    AI report generation makes a real, billed API call per request —
    kept off the Basic plan for cost-control reasons, available on
    Pro and Enterprise.
    """
    import functools
    @functools.wraps(f)
    def wrapper(user, *args, **kwargs):
        if not user.has_access():
            return jsonify({'error': 'Your trial or subscription has ended. Please subscribe to continue.',
                             'code': 'SUBSCRIPTION_REQUIRED'}), 402
        if user.plan not in ('pro', 'enterprise'):
            return jsonify({'error': 'AI-generated analysis requires the Pro or Enterprise plan.',
                             'code': 'PRO_REQUIRED'}), 402
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


@app.route('/api/test/watershed', methods=['POST'])
@token_required
def test_watershed(user):
    """
    ISOLATED test route — deliberately not part of the main analysis
    pipeline. Checks 5 proposed watershed-analysis capabilities
    (basin boundaries, stream network, rainfall, flood-risk zones,
    population at risk) against a real region, so we can see exactly
    which real Earth Engine datasets work before building any UI
    around them. Safe to remove once testing is done, or keep as a
    permanent internal diagnostic — doesn't affect anything else.
    """
    try:
        import watershed_test
    except Exception as e:
        return jsonify({'error': f'Test module failed to load: {e}'}), 500

    d = request.get_json()
    start, end, coords = d.get('start'), d.get('end'), d.get('roi')
    if not all([start, end, coords]):
        return jsonify({'error': 'start, end and roi are all required.'}), 400

    result = watershed_test.run_all_watershed_tests(start, end, coords)
    return jsonify(result)


@app.route('/api/analysis/changemap', methods=['POST'])
@token_required
@subscription_required
@rate_limit('15 per hour')
def analysis_changemap(user):
    d = request.get_json()
    family, index_v = d.get('family'), d.get('index')
    start1, end1 = d.get('start1'), d.get('end1')
    start2, end2 = d.get('start2'), d.get('end2')
    coords = d.get('roi')

    if not all([family, index_v, start1, end1, start2, end2, coords]):
        return jsonify({'error': 'family, index, start1, end1, start2, end2 and roi are all required.'}), 400
    if len(coords) < 3:
        return jsonify({'error': 'Region needs at least 3 points.'}), 400

    result = gee_engine.run_change_map(family, index_v, start1, end1, start2, end2, coords)
    if 'error' in result:
        return jsonify(result), 422
    return jsonify(result)


@app.route('/api/analysis/waterlevel', methods=['POST'])
@token_required
@subscription_required
@rate_limit('15 per hour')
def analysis_waterlevel(user):
    d = request.get_json()
    start1, end1 = d.get('start1'), d.get('end1')
    start2, end2 = d.get('start2'), d.get('end2')
    coords = d.get('roi')

    if not all([start1, end1, start2, end2, coords]):
        return jsonify({'error': 'start1, end1, start2, end2 and roi are all required.'}), 400
    if len(coords) < 3:
        return jsonify({'error': 'Region needs at least 3 points.'}), 400

    result = gee_engine.run_water_level(start1, end1, start2, end2, coords)
    if 'error' in result:
        return jsonify(result), 422
    return jsonify(result)


@app.route('/api/ai/report', methods=['POST'])
@token_required
@pro_or_higher_required
@rate_limit('20 per hour')
def ai_report(user):
    """
    Generates a detailed AI description of each calculation plus a
    holistic synthesis across all of them. Isolated import so a
    problem here (missing package, bad API key) can never break the
    rest of the app — it just returns a clear error for this one route.
    """
    try:
        import ai_engine
    except Exception as e:
        return jsonify({'error': f'AI module failed to load: {e}'}), 500

    d = request.get_json()
    readings = d.get('readings', [])
    location = d.get('location')
    change_map = d.get('change_map')
    trend = d.get('trend')
    water_level = d.get('water_level')
    language = d.get('language', 'en')

    if not readings and not water_level:
        return jsonify({'error': 'No calculations to describe — take at least one reading first.'}), 400

    result = ai_engine.generate_ai_report(readings, location, change_map, trend, water_level, language)
    if 'error' in result:
        return jsonify(result), 422
    return jsonify(result)


# ══════════════════════════════════════════════════════════════
# SAVED LOCATIONS — the foundation for recurring monitoring.
# A saved location is just persisted region+family+index config;
# the actual periodic re-checking and anomaly detection are a
# separate scheduled job, built on top of this.
# ══════════════════════════════════════════════════════════════

# Locations included per plan — mirrors the same spirit as the
# family/quota gating already in place for readings. None of this
# blocks a user from taking normal one-off readings; it only limits
# how many locations can be under ongoing automatic monitoring.
PLAN_LOCATION_LIMIT = {'basic': 0, 'pro': 5, 'enterprise': 25}


@app.route('/api/locations', methods=['GET'])
@token_required
@subscription_required
def list_locations(user):
    locations = SavedLocation.query.filter_by(user_id=user.id).order_by(SavedLocation.created_at.desc()).all()
    return jsonify({'locations': [l.to_dict() for l in locations]})


@app.route('/api/locations', methods=['POST'])
@token_required
@subscription_required
@rate_limit('20 per hour')
def create_location(user):
    d = request.get_json()
    name, family, index_v, roi = d.get('name'), d.get('family'), d.get('index'), d.get('roi')

    if not all([name, family, index_v, roi]):
        return jsonify({'error': 'name, family, index and roi are all required.'}), 400
    if len(roi) < 3:
        return jsonify({'error': 'Region needs at least 3 points.'}), 400

    limit = PLAN_LOCATION_LIMIT.get(user.plan, 0)
    current_count = SavedLocation.query.filter_by(user_id=user.id, active=True).count()
    if current_count >= limit:
        return jsonify({
            'error': f'Your {user.plan.title()} plan includes {limit} monitored location{"s" if limit != 1 else ""}. Upgrade for more, or remove an existing one.',
            'code': 'LOCATION_LIMIT_REACHED'
        }), 402

    loc = SavedLocation(
        user_id=user.id, name=name, family=family, index_name=index_v,
        roi_geojson=json.dumps(roi), check_frequency=d.get('check_frequency', 'monthly')
    )
    db.session.add(loc)
    db.session.commit()
    return jsonify(loc.to_dict())


@app.route('/api/locations/<int:location_id>', methods=['DELETE'])
@token_required
def delete_location(user, location_id):
    loc = SavedLocation.query.filter_by(id=location_id, user_id=user.id).first()
    if not loc:
        return jsonify({'error': 'Location not found.'}), 404
    db.session.delete(loc)
    db.session.commit()
    return jsonify({'deleted': True})


@app.route('/api/locations/<int:location_id>/history', methods=['GET'])
@token_required
def location_history(user, location_id):
    loc = SavedLocation.query.filter_by(id=location_id, user_id=user.id).first()
    if not loc:
        return jsonify({'error': 'Location not found.'}), 404
    history = LocationHistory.query.filter_by(location_id=loc.id).order_by(LocationHistory.checked_at.asc()).all()
    return jsonify({'location': loc.to_dict(), 'history': [h.to_dict() for h in history]})


@app.route('/api/analysis/advanced', methods=['POST'])
@token_required
@enterprise_required
@rate_limit('10 per hour')
def analysis_advanced(user):
    d = request.get_json()
    tool = d.get('tool')
    start, end, coords = d.get('start'), d.get('end'), d.get('roi')

    if not all([tool, start, end, coords]):
        return jsonify({'error': 'tool, start, end and roi are all required.'}), 400
    if len(coords) < 3:
        return jsonify({'error': 'Region needs at least 3 points.'}), 400

    if tool == 'treeCount':
        result = gee_engine.run_tree_count(start, end, coords)
    elif tool == 'waterBodies':
        result = gee_engine.run_water_bodies(start, end, coords)
    elif tool == 'landClassify':
        n_clusters = int(d.get('n_clusters', 5))
        result = gee_engine.run_land_classify(start, end, coords, n_clusters=n_clusters)
    else:
        return jsonify({'error': f'Unknown advanced tool: {tool}'}), 400

    if 'error' in result:
        return jsonify(result), 422
    return jsonify(result)


PLAN_FAMILY_ACCESS = {
    'basic':      {'water', 'veg'},
    'pro':        {'water', 'veg', 'landsat', 'geo', 'urban', 'fire', 'heat', 'pollution'},
    'enterprise': {'water', 'veg', 'landsat', 'geo', 'urban', 'archaeology', 'fire', 'heat', 'pollution'},
}
PLAN_MONTHLY_LIMIT = {'basic': 50, 'pro': 300, 'enterprise': None}  # None = unlimited


@app.route('/api/analysis/run', methods=['POST'])
@token_required
@subscription_required
@rate_limit('30 per hour')
def analysis_run(user):
    d = request.get_json()
    family, index_v = d.get('family'), d.get('index')
    start, end, coords = d.get('start'), d.get('end'), d.get('roi')

    if not all([family, index_v, start, end, coords]):
        return jsonify({'error': 'family, index, start, end and roi are all required.'}), 400
    if len(coords) < 3:
        return jsonify({'error': 'Region needs at least 3 points.'}), 400

    # Plan-based family access — e.g. Basic can't reach Geology/Urban/
    # Archaeology, Pro can't reach Archaeology & Terrain specifically.
    allowed_families = PLAN_FAMILY_ACCESS.get(user.plan, PLAN_FAMILY_ACCESS['basic'])
    if family not in allowed_families:
        return jsonify({
            'error': f'The "{family}" family isn\'t included in your {user.plan.title()} plan. Upgrade to unlock it.',
            'code': 'PLAN_UPGRADE_REQUIRED'
        }), 402

    # Plan-based monthly reading quota
    limit = PLAN_MONTHLY_LIMIT.get(user.plan)
    if limit is not None:
        month_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        used_this_month = Reading.query.filter(
            Reading.user_id == user.id, Reading.created_at >= month_start
        ).count()
        if used_this_month >= limit:
            return jsonify({
                'error': f'You\'ve used all {limit} readings included in your {user.plan.title()} plan this month. Upgrade for more.',
                'code': 'QUOTA_REACHED'
            }), 402

    # Serve from cache when possible — identical region/dates/index
    # returns instantly instead of re-running Earth Engine.
    result = result_cache.get(family, index_v, start, end, coords)
    if result is None:
        result = gee_engine.run_reading(family, index_v, start, end, coords)
        if 'error' in result:
            return jsonify(result), 422
        result_cache.set(family, index_v, start, end, coords, result)
    else:
        logger.info(f'Cache hit: {family}/{index_v} {start}->{end}')

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
@rate_limit('15 per hour')
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
