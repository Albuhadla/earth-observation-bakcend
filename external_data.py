"""
Earth Observation and Analysis — External Ground-Truth Sources
==================================================================
Real, independent measurements from ground sensors — used alongside
(never instead of) our satellite-derived readings, to give genuine
validation rather than just another view of the same kind of proxy
data. Deliberately isolated: any failure here (missing API key,
network issue, no station nearby) degrades gracefully and never
breaks the actual satellite reading it's attached to.

  - OpenAQ (openaq.org): real government-grade ground-station air
    quality measurements, worldwide. Used alongside Pollution family
    readings (NO2, SO2).
  - USGS Water Services (waterservices.usgs.gov): real, free,
    government stream/lake gage height measurements — U.S. water
    bodies only. Used alongside the Water Level estimate tool.
"""
import os
import logging
import requests

logger = logging.getLogger(__name__)

OPENAQ_BASE = 'https://api.openaq.org/v3'
USGS_SITE_BASE = 'https://waterservices.usgs.gov/nwis/site/'
USGS_IV_BASE = 'https://waterservices.usgs.gov/nwis/iv/'

# OpenAQ parameter name -> our pollution index name
OPENAQ_PARAM_MAP = {'no2': 'no2', 'so2': 'so2'}


def get_openaq_ground_truth(lat, lng, index_v, radius_m=25000):
    """
    Finds the nearest real OpenAQ ground station to (lat, lng) with a
    recent reading for the given pollutant, and returns its latest
    measured value. Returns None (never raises) if no key is
    configured, no station is nearby, or the request fails for any
    reason — this is a nice-to-have enrichment, never a hard
    dependency for the actual satellite reading.
    """
    api_key = os.getenv('OPENAQ_API_KEY')
    if not api_key or index_v not in OPENAQ_PARAM_MAP:
        return None
    try:
        headers = {'X-API-Key': api_key}
        # Step 1: find nearby stations, closest first (API already sorts by distance)
        resp = requests.get(f'{OPENAQ_BASE}/locations', headers=headers, timeout=8, params={
            'coordinates': f'{lat},{lng}', 'radius': min(radius_m, 25000), 'limit': 10
        })
        if resp.status_code != 200:
            return None
        locations = resp.json().get('results', [])
        if not locations:
            return None

        target_param = OPENAQ_PARAM_MAP[index_v]
        for loc in locations:
            # Only consider stations that actually measure this pollutant
            sensors = loc.get('sensors', [])
            matching_sensor = next(
                (s for s in sensors if s.get('parameter', {}).get('name') == target_param), None)
            if not matching_sensor:
                continue

            loc_id = loc.get('id')
            latest_resp = requests.get(f'{OPENAQ_BASE}/locations/{loc_id}/latest', headers=headers, timeout=8)
            if latest_resp.status_code != 200:
                continue
            readings = latest_resp.json().get('results', [])
            sensor_id = matching_sensor.get('id')
            match = next((r for r in readings if r.get('sensorsId') == sensor_id), None)
            if not match:
                continue

            return {
                'station_name': loc.get('name'),
                'distance_note': f"nearest station within {radius_m/1000:.0f}km",
                'value': match.get('value'),
                'unit': matching_sensor.get('parameter', {}).get('units'),
                'measured_at': match.get('datetime', {}).get('utc') if isinstance(match.get('datetime'), dict) else match.get('datetime'),
                'source': 'OpenAQ (real ground-station measurement)',
            }
        return None
    except Exception as e:
        logger.warning(f'OpenAQ ground-truth lookup failed (non-fatal): {e}')
        return None


def get_usgs_water_level(lat, lng, radius_deg=0.3):
    """
    Finds the nearest real USGS stream/lake gage station to (lat, lng)
    and returns its current real gage height reading. US water bodies
    only — returns None gracefully everywhere else, or on any failure.
    """
    try:
        bbox = f'{lng-radius_deg},{lat-radius_deg},{lng+radius_deg},{lat+radius_deg}'
        site_resp = requests.get(USGS_SITE_BASE, timeout=8, params={
            'format': 'rdb', 'bBox': bbox, 'siteType': 'ST,LK', 'siteStatus': 'active'
        })
        if site_resp.status_code != 200 or not site_resp.text.strip():
            return None
        # RDB is tab-delimited with comment lines starting with '#'
        lines = [l for l in site_resp.text.splitlines() if l and not l.startswith('#')]
        if len(lines) < 3:
            return None  # header + type-row only, no actual sites
        header = lines[0].split('\t')
        first_site = lines[2].split('\t')  # line 1 is the RDB type-declaration row
        site_no_idx = header.index('site_no') if 'site_no' in header else 1
        site_name_idx = header.index('station_nm') if 'station_nm' in header else 2
        site_no = first_site[site_no_idx]
        site_name = first_site[site_name_idx]

        iv_resp = requests.get(USGS_IV_BASE, timeout=8, params={
            'sites': site_no, 'parameterCd': '00065', 'siteStatus': 'all', 'format': 'json'
        })
        if iv_resp.status_code != 200:
            return None
        series = iv_resp.json().get('value', {}).get('timeSeries', [])
        if not series:
            return None
        values = series[0].get('values', [{}])[0].get('value', [])
        if not values:
            return None
        latest = values[-1]

        return {
            'station_name': site_name, 'site_no': site_no,
            'gage_height_ft': float(latest.get('value')),
            'measured_at': latest.get('dateTime'),
            'source': 'USGS Water Services (real gage measurement)',
        }
    except Exception as e:
        logger.warning(f'USGS water level lookup failed (non-fatal): {e}')
        return None
