"""
Area Pre-Study — "what is around this land?"
=================================================
For a drawn region, reports roads, transit, malls, shops, schools, health,
worship, parks and building density from OpenStreetMap (via the public
Overpass API — free, no key), scores accessibility and services 0-100, and
returns POI markers for the map and the printed report.

Honesty rules (kept in the output, not just here):
  * OpenStreetMap is volunteer-edited — coverage differs by area. Counts are
    "mapped features", not a census.
  * Distances are straight-line from the region's centre, not drive time.
  * Nothing here covers zoning, ownership, prices or utility pipelines.

Isolated from the satellite pipeline (like ai_engine / cache) — a failure here
never affects normal readings.
"""
import math
import json
import logging
import requests

logger = logging.getLogger(__name__)

OVERPASS_SERVERS = [
    'https://overpass-api.de/api/interpreter',
    'https://overpass.kumi.systems/api/interpreter',
    'https://overpass.private.coffee/api/interpreter',
]
HEADERS = {'User-Agent': 'EarthObservationAnalysis/1.0 (earthobservation.land)'}
RADIUS_M = 5000

# key, English label, Arabic label, group, (good_m, far_m) distance scoring, weight
ITEMS = [
    ('metro',      'Metro / tram / light rail', 'المترو / الترام / القطار الخفيف', 'access',    (800, 5000),  2.0),
    ('train',      'Train station',            'محطة قطار',                        'access',    (1500, 5000), 0.5),
    ('mall',       'Shopping mall',            'مركز تسوق',                         'retail',    (2000, 5000), 1.5),
    ('supermarket','Supermarket',              'سوبرماركت',                         'retail',    (500, 3000),  1.5),
    ('bank',       'Bank',                     'بنك',                               'retail',    (800, 3000),  0.5),
    ('fuel',       'Fuel station',             'محطة وقود',                         'retail',    (1000, 4000), 0.5),
    ('school',     'School / kindergarten',    'مدرسة / روضة',                      'education', (800, 3000),  1.5),
    ('university', 'University / college',     'جامعة / كلية',                      'education', (2000, 5000), 0.5),
    ('hospital',   'Hospital / clinic',        'مستشفى / عيادة',                    'health',    (1500, 5000), 1.5),
    ('mosque',     'Mosque',                   'مسجد',                              'community', (500, 2000),  1.0),
    ('park',       'Park',                     'حديقة',                             'community', (800, 3000),  1.0),
    ('hotel',      'Hotel',                    'فندق',                              'community', (2000, 5000), 0.5),
]
# bus stops and dining are scored by how many are nearby, not nearest distance
COUNT_ITEMS = [
    ('bus',    'Bus stops (within 1 km)',      'محطات حافلات (ضمن 1 كم)',  'access', 1000, 20),   # score = count * 20
    ('dining', 'Restaurants & cafés (within 1.5 km)', 'مطاعم ومقاهٍ (ضمن 1.5 كم)', 'retail', 1500, 8),
]
GROUPS = [
    ('access',    'Transport & access',     'النقل وسهولة الوصول',  0.30),
    ('retail',    'Shopping & daily needs', 'التسوق والاحتياجات اليومية', 0.25),
    ('education', 'Education',              'التعليم',               0.15),
    ('health',    'Health',                 'الصحة',                 0.15),
    ('community', 'Community & leisure',    'المجتمع والترفيه',      0.15),
]


# ── geometry ──────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def region_centre(coords):
    """coords = [[lat, lng], ...]. Mean of the bounding box — stable for any shape."""
    lats = [c[0] for c in coords]
    lngs = [c[1] for c in coords]
    return (min(lats) + max(lats)) / 2, (min(lngs) + max(lngs)) / 2


def dist_to_polyline(lat, lng, geom):
    """Min metres from a point to a polyline [{lat,lon},...] (local flat projection)."""
    if not geom:
        return None
    kx = 111320.0 * math.cos(math.radians(lat))
    ky = 110540.0
    pts = [((g['lon'] - lng) * kx, (g['lat'] - lat) * ky) for g in geom]
    best = None
    if len(pts) == 1:
        return math.hypot(*pts[0])
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        dx, dy = x2 - x1, y2 - y1
        L2 = dx * dx + dy * dy
        t = 0 if L2 == 0 else max(0, min(1, -(x1 * dx + y1 * dy) / L2))
        d = math.hypot(x1 + t * dx, y1 + t * dy)
        best = d if best is None or d < best else best
    return best


# ── Overpass ──────────────────────────────────────────────────
def _overpass(query, timeout=45):
    last = None
    for url in OVERPASS_SERVERS:
        try:
            r = requests.post(url, data={'data': query}, headers=HEADERS, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            last = f'{url} -> HTTP {r.status_code}'
        except Exception as e:
            last = f'{url} -> {e}'
        logger.warning(f'Overpass attempt failed: {last}')
    raise RuntimeError(f'All map-data servers failed ({last})')


def _poi_query(lat, lng):
    a = f'(around:{RADIUS_M},{lat},{lng})'
    a1 = f'(around:1000,{lat},{lng})'
    a15 = f'(around:1500,{lat},{lng})'
    return f"""[out:json][timeout:40];
(
 nwr{a}["railway"="station"]["station"="subway"];
 nwr{a}["station"="subway"];
 nwr{a}["railway"~"^(tram_stop|subway_entrance)$"];
 nwr{a}["railway"="station"]["light_rail"="yes"];
 nwr{a}["railway"="station"];
 nwr{a}["shop"="mall"];
 nwr{a}["shop"="supermarket"];
 nwr{a}["amenity"="bank"];
 nwr{a}["amenity"="fuel"];
 nwr{a}["amenity"~"^(school|kindergarten|university|college|hospital|clinic)$"];
 nwr{a}["amenity"="place_of_worship"]["religion"="muslim"];
 nwr{a}["leisure"="park"];
 nwr{a}["tourism"="hotel"];
 node{a1}["highway"="bus_stop"];
 nwr{a15}["amenity"~"^(restaurant|cafe|fast_food)$"];
);
out center tags 4000;"""


def _classify(tags):
    """Returns the item key a feature belongs to (or None)."""
    am, rw, sh = tags.get('amenity'), tags.get('railway'), tags.get('shop')
    if tags.get('highway') == 'bus_stop':
        return 'bus'
    if rw in ('tram_stop', 'subway_entrance') or tags.get('station') == 'subway' \
            or tags.get('light_rail') == 'yes':
        return 'metro'
    if rw == 'station':
        return 'train'
    if sh == 'mall':
        return 'mall'
    if sh == 'supermarket':
        return 'supermarket'
    if am == 'bank':
        return 'bank'
    if am == 'fuel':
        return 'fuel'
    if am in ('school', 'kindergarten'):
        return 'school'
    if am in ('university', 'college'):
        return 'university'
    if am in ('hospital', 'clinic'):
        return 'hospital'
    if am == 'place_of_worship' and tags.get('religion') == 'muslim':
        return 'mosque'
    if tags.get('leisure') == 'park':
        return 'park'
    if tags.get('tourism') == 'hotel':
        return 'hotel'
    if am in ('restaurant', 'cafe', 'fast_food'):
        return 'dining'
    return None


def _score_dist(d, good, far):
    if d is None:
        return 0
    if d <= good:
        return 100
    if d >= far:
        return 0
    return round(100 * (far - d) / (far - good))


def _verdict(score):
    if score >= 70:
        return ('Well served', 'خدمات ممتازة')
    if score >= 45:
        return ('Moderately served', 'خدمات متوسطة')
    return ('Under-served', 'خدمات ضعيفة')


def _context(b1):
    if b1 is None:
        return ('Unknown', 'غير معروف')
    if b1 >= 800:
        return ('Established urban area', 'منطقة حضرية قائمة')
    if b1 >= 150:
        return ('Developing / mixed area', 'منطقة نامية / مختلطة')
    return ('Sparse or undeveloped area', 'منطقة قليلة العمران أو غير مطورة')


# ── main entry ────────────────────────────────────────────────
def run_prestudy(coords):
    if not coords or len(coords) < 3:
        return {'error': 'Draw a region first.'}
    lat, lng = region_centre(coords)
    warnings = []

    # 1) points of interest
    try:
        poi_json = _overpass(_poi_query(lat, lng))
    except Exception as e:
        return {'error': f'Map data is temporarily unavailable — please try again in a minute. ({e})'}

    by_key = {}
    for el in poi_json.get('elements', []):
        tags = el.get('tags') or {}
        key = _classify(tags)
        if not key:
            continue
        if el.get('type') == 'node':
            elat, elon = el.get('lat'), el.get('lon')
        else:
            c = el.get('center') or {}
            elat, elon = c.get('lat'), c.get('lon')
        if elat is None or elon is None:
            continue
        d = haversine(lat, lng, elat, elon)
        by_key.setdefault(key, []).append({
            'lat': elat, 'lng': elon, 'dist_m': round(d),
            'name': tags.get('name:en') or tags.get('name'),
            'name_ar': tags.get('name:ar'),
        })
    for k in by_key:
        by_key[k].sort(key=lambda x: x['dist_m'])

    # 2) roads (geometry of major roads only) and building counts
    roads = {'nearest_major_m': None, 'nearest_major_name': None, 'nearest_major_type': None, 'ways_within_1km': None}
    buildings = {'within_1km': None, 'within_3km': None, 'within_5km': None}
    try:
        rd = _overpass(f"""[out:json][timeout:40];
way(around:{RADIUS_M},{lat},{lng})["highway"~"^(motorway|trunk|primary|secondary)$"];
out geom tags 600;""")
        best = None
        for w in rd.get('elements', []):
            d = dist_to_polyline(lat, lng, w.get('geometry'))
            if d is not None and (best is None or d < best[0]):
                t = w.get('tags') or {}
                best = (d, t.get('name:en') or t.get('name') or t.get('ref'), t.get('highway'))
        if best:
            roads['nearest_major_m'] = round(best[0])
            roads['nearest_major_name'] = best[1]
            roads['nearest_major_type'] = best[2]
    except Exception as e:
        warnings.append('Road data could not be loaded.')
        logger.warning(f'prestudy roads failed: {e}')
    try:
        cj = _overpass(f"""[out:json][timeout:40];
way(around:1000,{lat},{lng})["building"]; out count;
way(around:3000,{lat},{lng})["building"]; out count;
way(around:5000,{lat},{lng})["building"]; out count;
way(around:1000,{lat},{lng})["highway"]; out count;""")
        counts = [int((e.get('tags') or {}).get('total', 0)) for e in cj.get('elements', []) if e.get('type') == 'count']
        if len(counts) >= 4:
            buildings = {'within_1km': counts[0], 'within_3km': counts[1], 'within_5km': counts[2]}
            roads['ways_within_1km'] = counts[3]
    except Exception as e:
        warnings.append('Building counts could not be loaded.')
        logger.warning(f'prestudy counts failed: {e}')

    # 3) assemble items, scores, markers
    group_items = {g[0]: [] for g in GROUPS}
    markers = []

    # roads join the access group as an item
    road_score = _score_dist(roads['nearest_major_m'], 500, 4000)
    group_items['access'].append({
        'key': 'road', 'label': 'Major road (motorway to secondary)', 'label_ar': 'طريق رئيسي',
        'nearest_m': roads['nearest_major_m'], 'nearest_name': roads['nearest_major_name'], 'nearest_name_ar': None,
        'count_1km': roads['ways_within_1km'], 'count_3km': None, 'count_5km': None,
        'count_note': 'road segments within 1 km (all types)', 'score': road_score, 'weight': 2.0,
    })

    for key, label, label_ar, group, (good, far), weight in ITEMS:
        pois = by_key.get(key, [])
        nearest = pois[0] if pois else None
        group_items[group].append({
            'key': key, 'label': label, 'label_ar': label_ar,
            'nearest_m': nearest['dist_m'] if nearest else None,
            'nearest_name': nearest['name'] if nearest else None,
            'nearest_name_ar': nearest['name_ar'] if nearest else None,
            'count_1km': sum(1 for p in pois if p['dist_m'] <= 1000),
            'count_3km': sum(1 for p in pois if p['dist_m'] <= 3000),
            'count_5km': len(pois),
            'score': _score_dist(nearest['dist_m'] if nearest else None, good, far), 'weight': weight,
        })
        for p in pois[:8]:
            markers.append({'lat': p['lat'], 'lng': p['lng'], 'type': key, 'group': group, 'name': p['name'], 'name_ar': p['name_ar'], 'dist_m': p['dist_m']})

    for key, label, label_ar, group, radius, per in COUNT_ITEMS:
        pois = [p for p in by_key.get(key, []) if p['dist_m'] <= radius]
        group_items[group].append({
            'key': key, 'label': label, 'label_ar': label_ar,
            'nearest_m': pois[0]['dist_m'] if pois else None,
            'nearest_name': pois[0]['name'] if pois else None, 'nearest_name_ar': pois[0]['name_ar'] if pois else None,
            'count_1km': sum(1 for p in pois if p['dist_m'] <= 1000), 'count_3km': None, 'count_5km': None,
            'count_within': len(pois), 'score': min(100, len(pois) * per), 'weight': 1.0 if key == 'bus' else 0.5,
        })
        for p in pois[:6]:
            markers.append({'lat': p['lat'], 'lng': p['lng'], 'type': key, 'group': group, 'name': p['name'], 'name_ar': p['name_ar'], 'dist_m': p['dist_m']})

    groups_out, overall = [], 0.0
    for gkey, glabel, glabel_ar, gweight in GROUPS:
        items = group_items[gkey]
        tw = sum(i['weight'] for i in items) or 1
        gscore = round(sum(i['score'] * i['weight'] for i in items) / tw)
        overall += gscore * gweight
        for i in items:
            i.pop('weight', None)
        groups_out.append({'key': gkey, 'label': glabel, 'label_ar': glabel_ar, 'score': gscore, 'items': items})
    overall = round(overall)
    v_en, v_ar = _verdict(overall)
    c_en, c_ar = _context(buildings['within_1km'])

    if not by_key and buildings['within_5km'] in (None, 0):
        warnings.append('Very little is mapped around this location in OpenStreetMap — the area may be undeveloped, or simply under-mapped.')

    return {
        'center': {'lat': lat, 'lng': lng}, 'radius_m': RADIUS_M, 'source': 'openstreetmap',
        'overall_score': overall, 'verdict': v_en, 'verdict_ar': v_ar,
        'groups': groups_out, 'roads': roads, 'buildings': buildings,
        'area_context': c_en, 'area_context_ar': c_ar,
        'markers': markers, 'warnings': warnings,
        'data_note': 'Source: OpenStreetMap contributors. Counts are mapped features and coverage varies by area. '
                     'Distances are straight-line from the centre of your region. Zoning, ownership, prices and '
                     'utility networks are not included.',
    }
