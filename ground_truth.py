"""
Earth Observation and Analysis — Ground-Truth Data
==================================================================
Real sensor data from official networks, used to validate our
satellite-derived proxies against actual measured ground readings —
genuinely different data, not a second view of the same thing.

  - OpenAQ: real government-grade air quality ground stations,
    worldwide. Validates the Pollution family (NO2, SO2).
  - USGS Water Services: real river/lake gauge height readings,
    US only. Validates the Water Level estimate tool.

Both isolated from the main pipeline (same pattern as ai_engine.py,
watershed_test.py) — a failure here never breaks the underlying
satellite reading, it just means no ground-truth comparison is shown.
"""
import os
import logging
import requests

logger = logging.getLogger(__name__)

OPENAQ_BASE = 'https://api.openaq.org/v3'
USGS_BASE = 'https://waterservices.usgs.gov/nwis/iv/'

# Our internal pollution index name -> OpenAQ's parameter name
OPENAQ_PARAM_MAP = {'no2': 'no2', 'so2': 'so2'}
# 'aerosol' (UV Aerosol Index) has no direct ground-station equivalent —
# PM2.5/PM10 are the closest general air-quality context available,
# shown as a bonus rather than a direct validation for that one index.


def get_openaq_ground_truth(lat, lng, index_v):
    """
    Finds the nearest real air-quality ground station (within 25km,
    OpenAQ's maximum radius) and returns its latest actual measured
    reading for the matching pollutant, if one exists. Returns None on
    any failure or if no matching station is nearby — never raises.
    """
    api_key = os.getenv('OPENAQ_API_KEY')
    if not api_key:
        return None
    try:
        headers = {'X-API-Key': api_key}
        resp = requests.get(
            f'{OPENAQ_BASE}/locations',
            params={'coordinates': f'{lat},{lng}', 'radius': 25000, 'limit': 5},
            headers=headers, timeout=10
        )
        if resp.status_code != 200:
            logger.warning(f'OpenAQ locations lookup failed: {resp.status_code}')
            return None
        locations = resp.json().get('results', [])
        if not locations:
            return None

        target_param = OPENAQ_PARAM_MAP.get(index_v)

        for loc in locations:
            loc_id = loc.get('id')
            loc_name = loc.get('name', 'Unknown station')
            loc_coords = loc.get('coordinates', {})
            latest_resp = requests.get(
                f'{OPENAQ_BASE}/locations/{loc_id}/latest',
                headers=headers, timeout=10
            )
            if latest_resp.status_code != 200:
                continue
            readings = latest_resp.json().get('results', [])
            for r in readings:
                param_name = (r.get('parameter') or {}).get('name', '').lower()
                # Direct match for NO2/SO2; otherwise fall back to PM2.5 as
                # general context so the user still sees something real
                # even when there's no ground equivalent for their exact index.
                if (target_param and param_name == target_param) or \
                   (not target_param and param_name == 'pm25'):
                    return {
                        'station_name': loc_name,
                        'station_lat': loc_coords.get('latitude'),
                        'station_lon': loc_coords.get('longitude'),
                        'parameter': param_name,
                        'value': r.get('value'),
                        'unit': (r.get('parameter') or {}).get('units'),
                        'measured_at': (r.get('datetime') or {}).get('utc'),
                        'is_direct_match': bool(target_param),
                    }
        return None
    except Exception as e:
        logger.warning(f'OpenAQ ground-truth lookup failed (non-fatal): {e}')
        return None


def get_usgs_water_level(bbox):
    """
    Finds real USGS river/lake gauge height readings within a bounding
    box — US coverage only, but genuinely real sensor data, and
    requires no API key at all. bbox is (west, south, east, north) in
    decimal degrees. Returns None on any failure or if no gauge is in
    range — never raises.
    """
    try:
        west, south, east, north = bbox
        resp = requests.get(
            USGS_BASE,
            params={
                'bBox': f'{west},{south},{east},{north}',
                'parameterCd': '00065',  # gage height
                'siteStatus': 'active',
                'format': 'json',
            },
            timeout=10
        )
        if resp.status_code != 200:
            logger.warning(f'USGS water services lookup failed: {resp.status_code}')
            return None
        series = resp.json().get('value', {}).get('timeSeries', [])
        if not series:
            return None

        ts = series[0]
        site_name = ts.get('sourceInfo', {}).get('siteName', 'Unknown gauge')
        site_coords = ts.get('sourceInfo', {}).get('geoLocation', {}).get('geogLocation', {})
        values = ts.get('values', [{}])[0].get('value', [])
        if not values:
            return None
        latest = values[-1]

        return {
            'site_name': site_name,
            'site_lat': site_coords.get('latitude'),
            'site_lon': site_coords.get('longitude'),
            'gage_height_ft': latest.get('value'),
            'measured_at': latest.get('dateTime'),
        }
    except Exception as e:
        logger.warning(f'USGS ground-truth lookup failed (non-fatal): {e}')
        return None
