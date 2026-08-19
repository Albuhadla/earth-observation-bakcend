"""
Earth Observation and Analysis — Result Caching
==================================================
Database-backed cache for finished GEE readings. Works with either
SQLite (dev) or PostgreSQL (production) since it's just a table via
SQLAlchemy — no separate cache server (Redis) required to get most
of the benefit. If usage later grows large enough that a DB-backed
cache becomes a bottleneck itself, swapping this module's storage
for Redis is a contained change — nothing else in the app needs to
know the difference, since callers only ever use get()/set() here.
"""
import hashlib
import json
import os
import logging
from datetime import datetime, timedelta

from models import db, ReadingCache

logger = logging.getLogger(__name__)

CACHE_TTL_HOURS = int(os.getenv('CACHE_TTL_HOURS', 24))


def _make_key(kind, params, coords=None):
    """
    Deterministic cache key from the exact request. `kind` separates
    different endpoint types (reading/changemap/trend/waterlevel/
    advanced) so they can never collide with each other even if their
    parameter dicts happen to look similar. Coordinates (if present)
    are rounded to ~1 metre precision so trivially-different polygon
    drawings of "the same" region still hit the cache, while genuinely
    different regions don't collide.
    """
    payload_dict = dict(params)
    if coords:
        payload_dict['_coords'] = [[round(lat, 5), round(lng, 5)] for lat, lng in coords]
    payload = json.dumps({'kind': kind, 'params': payload_dict}, sort_keys=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def get(kind, params, coords=None):
    """Return a cached result dict, or None if not cached / expired."""
    try:
        key = _make_key(kind, params, coords)
        row = ReadingCache.query.filter_by(cache_key=key).first()
        if not row:
            return None
        if row.expires_at < datetime.utcnow():
            db.session.delete(row)
            db.session.commit()
            return None
        result = json.loads(row.result_json)
        result['cached'] = True
        return result
    except Exception as e:
        logger.warning(f'Cache read failed (non-fatal): {e}')
        return None


def set(kind, params, result, coords=None):
    """Store a finished result. Never lets a cache-write failure break the request."""
    try:
        # Don't cache errors or simulated fallbacks — only real results
        # are worth serving to a second request without recomputation.
        if not result or 'error' in result or result.get('source') != 'gee_live':
            return
        key = _make_key(kind, params, coords)
        existing = ReadingCache.query.filter_by(cache_key=key).first()
        expires = datetime.utcnow() + timedelta(hours=CACHE_TTL_HOURS)
        payload = json.dumps({k: v for k, v in result.items() if k != 'cached'})

        if existing:
            existing.result_json = payload
            existing.expires_at = expires
            existing.created_at = datetime.utcnow()
        else:
            db.session.add(ReadingCache(cache_key=key, result_json=payload, expires_at=expires))
        db.session.commit()
    except Exception as e:
        logger.warning(f'Cache write failed (non-fatal): {e}')
        db.session.rollback()


def cleanup_expired():
    """Housekeeping — call periodically (or on each cache miss) to keep the table small."""
    try:
        deleted = ReadingCache.query.filter(ReadingCache.expires_at < datetime.utcnow()).delete()
        db.session.commit()
        if deleted:
            logger.info(f'Cache cleanup: removed {deleted} expired entries.')
    except Exception as e:
        logger.warning(f'Cache cleanup failed (non-fatal): {e}')
        db.session.rollback()
