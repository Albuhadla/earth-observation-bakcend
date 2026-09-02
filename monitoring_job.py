"""
Weekly monitoring check job — runs as a separate Railway Cron service,
NOT part of the main Flask app. Set its Cron Schedule to something like
"0 6 * * 1" (every Monday at 6am UTC) in Railway's service settings.

For every active saved location belonging to an Enterprise user, this:
  1. Re-runs the same family/index/ROI reading via the same GEE engine
     the rest of the platform uses (no separate calculation logic).
  2. Compares the new reading against the location's last check —
     a simple week-over-week % change, not the full frontend
     good/warn/bad threshold table (which lives in app.js and would
     be real duplicated logic to maintain here for a background job).
  3. On the first-ever check for a location, there's nothing to
     compare against yet — it just records the baseline, no alert.
  4. If the change crosses ANOMALY_THRESHOLD, sends an email via
     Resend and marks the check as an anomaly.
  5. Records this check in LocationHistory regardless of outcome, so
     the location's own history keeps growing either way.

This script runs to completion and exits — required for a Railway
Cron job (a service that stays running would be skipped every
subsequent scheduled run).
"""
import os
import json
import logging
from datetime import datetime

from flask import Flask
from models import db, User, SavedLocation, LocationHistory
import gee_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('monitoring_job')

# A week-over-week relative change bigger than this counts as worth
# alerting on. Simple and generic across every index, rather than
# duplicating the full per-index threshold table that only exists in
# the frontend today.
ANOMALY_THRESHOLD = 0.25  # 25% relative change

RESEND_API_KEY = os.getenv('RESEND_API_KEY')
ALERT_FROM_EMAIL = os.getenv('ALERT_FROM_EMAIL', 'alerts@earthobservation.land')


def send_alert_email(to_email, location_name, index_label, old_mean, new_mean, pct_change):
    """
    Sends via Resend's simple HTTP API. Non-fatal on failure — a
    missing/misconfigured email key should never crash the whole
    check job partway through everyone else's locations.
    """
    if not RESEND_API_KEY:
        logger.warning('RESEND_API_KEY not set — skipping email, but the anomaly is still recorded in LocationHistory.')
        return False
    try:
        import requests
        direction = 'increased' if new_mean >= old_mean else 'decreased'
        resp = requests.post(
            'https://api.resend.com/emails',
            headers={'Authorization': f'Bearer {RESEND_API_KEY}', 'Content-Type': 'application/json'},
            json={
                'from': f'Earth Observation and Analysis <{ALERT_FROM_EMAIL}>',
                'to': [to_email],
                'subject': f'⚠️ Change detected — {location_name}',
                'html': f"""
                    <div style="font-family:Arial,sans-serif; max-width:520px;">
                        <h2 style="color:#0e3a4a;">Change detected at {location_name}</h2>
                        <p><b>{index_label}</b> has {direction} by <b>{abs(pct_change):.1f}%</b> since your last weekly check.</p>
                        <table style="width:100%; border-collapse:collapse; margin:16px 0;">
                            <tr style="background:#eef5f5;"><td style="padding:8px;">Previous reading</td><td style="padding:8px;"><b>{old_mean:.4f}</b></td></tr>
                            <tr><td style="padding:8px;">Current reading</td><td style="padding:8px;"><b>{new_mean:.4f}</b></td></tr>
                        </table>
                        <p>Sign in to Earth Observation and Analysis to view the full details and generate a report.</p>
                        <p style="color:#999; font-size:12px; margin-top:24px;">You're receiving this because automatic monitoring is enabled for this location on your Enterprise plan.</p>
                    </div>
                """
            },
            timeout=15
        )
        if resp.status_code >= 300:
            logger.warning(f'Resend returned {resp.status_code}: {resp.text[:300]}')
            return False
        return True
    except Exception as e:
        logger.warning(f'Email send failed (non-fatal): {e}')
        return False


def run_weekly_checks():
    app = Flask(__name__)
    db_url = os.getenv('DATABASE_URL', 'sqlite:///razaza.db')
    if db_url.startswith('postgres://'):
        db_url = db_url.replace('postgres://', 'postgresql://', 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = db_url
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)

    with app.app_context():
        if not (gee_engine.EE_AVAILABLE and gee_engine.init_ee()):
            logger.error('Earth Engine not available — aborting this run entirely rather than recording false/simulated checks.')
            return

        locations = (SavedLocation.query
                     .join(User, SavedLocation.user_id == User.id)
                     .filter(SavedLocation.active == True, User.plan == 'enterprise')
                     .all())
        logger.info(f'{len(locations)} active Enterprise-monitored location(s) to check.')

        checked, anomalies, failed = 0, 0, 0
        for loc in locations:
            try:
                coords = json.loads(loc.roi_geojson)
                end = datetime.utcnow().strftime('%Y-%m-%d')
                # 90-day trailing window keeps each check reasonably
                # fast and current, rather than always re-scanning the
                # location's entire history on every run.
                from datetime import timedelta
                start = (datetime.utcnow() - timedelta(days=90)).strftime('%Y-%m-%d')

                result = gee_engine.run_reading(loc.family, loc.index_name, start, end, coords)
                if 'error' in result or result.get('source') == 'simulated':
                    logger.warning(f'Location {loc.id} ({loc.name}): reading failed or simulated, skipping this check — not recorded as a false data point.')
                    failed += 1
                    continue

                new_mean = result['mean']
                prior = (LocationHistory.query
                         .filter_by(location_id=loc.id)
                         .order_by(LocationHistory.checked_at.desc())
                         .first())

                is_anomaly = False
                pct_change = None
                if prior and prior.mean_value is not None:
                    denom = max(abs(prior.mean_value), 0.01)
                    pct_change = ((new_mean - prior.mean_value) / denom) * 100
                    is_anomaly = abs(pct_change) / 100 > ANOMALY_THRESHOLD

                entry = LocationHistory(
                    location_id=loc.id, mean_value=new_mean,
                    status='anomaly' if is_anomaly else 'normal',
                    deviation_pct=round(pct_change, 2) if pct_change is not None else None
                )
                db.session.add(entry)

                loc.last_checked_at = datetime.utcnow()
                loc.last_mean_value = new_mean
                loc.last_status = 'anomaly' if is_anomaly else 'normal'
                db.session.commit()
                checked += 1

                if is_anomaly:
                    anomalies += 1
                    user = User.query.get(loc.user_id)
                    if user and user.email:
                        send_alert_email(user.email, loc.name, f'{loc.family}/{loc.index_name}',
                                          prior.mean_value, new_mean, pct_change)
                    logger.info(f'Location {loc.id} ({loc.name}): ANOMALY — {pct_change:+.1f}% change.')
                else:
                    logger.info(f'Location {loc.id} ({loc.name}): checked, no anomaly.')

            except Exception as e:
                db.session.rollback()
                failed += 1
                logger.error(f'Location {loc.id} ({loc.name}) failed: {e}')
                continue

        logger.info(f'Weekly monitoring run complete: {checked} checked, {anomalies} anomalies, {failed} failed/skipped.')


if __name__ == '__main__':
    run_weekly_checks()
