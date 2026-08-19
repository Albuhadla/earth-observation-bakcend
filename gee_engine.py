"""
Earth Observation and Analysis — Earth Engine computation layer
======================================================
Python port of the index math from the GEE JavaScript app
(Al_Razaza_WaterQuality_FINAL.js). Falls back to a seeded
simulation if the `earthengine-api` package or credentials
are not available, so the API always returns something.
"""
import os, hashlib, random, logging, math, base64
import requests
from io import BytesIO

logger = logging.getLogger(__name__)

try:
    import ee
    EE_AVAILABLE = True
    _initialized = False
    _last_ee_init_error = None
except Exception as _ee_import_err:
    # Catch ANY failure here, not just ImportError — version conflicts,
    # missing system libs, etc. on the host can raise other exception
    # types, and letting those escape crashes the whole app on every boot.
    EE_AVAILABLE = False
    _initialized = False
    _last_ee_init_error = str(_ee_import_err)
    logger.warning(f'earthengine-api unavailable, running in simulation mode: {_ee_import_err}')


def init_ee():
    global _initialized, _last_ee_init_error
    if _initialized or not EE_AVAILABLE:
        return EE_AVAILABLE
    try:
        service_account = os.getenv('GEE_SERVICE_ACCOUNT')
        project = os.getenv('GEE_PROJECT', '')

        # Preferred for hosted deployments (Railway, Render, etc.) with a
        # PUBLIC repo: base64-encode the entire key JSON file and paste
        # that into a Railway variable called GEE_KEY_JSON_B64. This is
        # far more reliable than pasting raw JSON, since copy/pasting a
        # multi-line private key into a text box often corrupts its
        # internal line breaks (a very common "MalformedFraming" error).
        key_json_b64 = os.getenv('GEE_KEY_JSON_B64')

        # Older/plain option: paste the raw JSON directly. Kept for
        # compatibility, but prefer the base64 variable above.
        key_json = os.getenv('GEE_KEY_JSON')

        # Fallback for local development: a gee_key.json file sitting
        # next to this script (never commit this file — see .gitignore).
        key_file = os.getenv('GEE_KEY_FILE', 'gee_key.json')

        if service_account and key_json_b64:
            import base64
            decoded = base64.b64decode(key_json_b64).decode('utf-8')
            creds = ee.ServiceAccountCredentials(service_account, key_data=decoded)
            ee.Initialize(creds, project=project)
        elif service_account and key_json:
            creds = ee.ServiceAccountCredentials(service_account, key_data=key_json)
            ee.Initialize(creds, project=project)
        elif service_account and os.path.exists(key_file):
            creds = ee.ServiceAccountCredentials(service_account, key_file)
            ee.Initialize(creds, project=project)
        else:
            ee.Initialize(project=project)
        _initialized = True
        logger.info('Earth Engine initialised.')
        return True
    except Exception as e:
        _last_ee_init_error = str(e)
        logger.error(f'EE init failed: {e}')
        return False


# ── index metadata (mirrors FAMILIES in app.js) ────────────────
RANGES = {
    'NDTI':(-0.3,0.3), 'NDCI':(-0.2,0.5), 'TSS':(-0.5,1), 'NDWI':(-0.3,0.5),
    'SABI':(-0.1,0.2), 'FAI':(-0.05,0.08), 'Secchi':(0,4),
    'ndvi':(-0.2,0.8), 'mndwi':(-0.5,0.5), 'ndbi':(-0.5,0.3),
    'iron':(0.8,3), 'clay':(0.8,2.5), 'carbonate':(0.5,2), 'evaporite':(0,0.5), 'allmin':(0.8,2.5),
    'reeds':(0,0.8), 'riparian':(0.5,4), 'halophyte':(0.5,4), 'scrub':(0.05,0.35), 'mud':(0.3,3), 'sav':(-0.2,0.5),
    'nightlights': (0, 40),  # radiance, nW/cm2/sr — urban cores often 20-60+
    'ubndbi':      (-0.5, 0.3),  # same NDBI math, exposed under the Urban family too
    'slope':       (0, 45),      # degrees — most tell/mound/ditch edges fall in this range
    'vegAnomaly':  (-0.15, 0.15),# NDVI deviation from local neighbourhood average
    'sar':         (-25, 0),     # Sentinel-1 VV backscatter, dB
    # Fire
    'nbr':         (-0.2, 0.8),  # Normalized Burn Ratio — healthy veg high, burned scar low/negative
    'activefire':  (0, 1),       # binary-ish hotspot confidence mask
    # Heat
    'lst':         (10, 55),     # Land Surface Temperature, °C — wide to cover desert extremes
    'sst':         (5, 35),      # Sea Surface Temperature, °C
    # Pollution (Sentinel-5P units)
    'no2':         (0, 0.0002),  # mol/m² tropospheric column
    'so2':         (0, 0.001),   # mol/m² column
    'aerosol':     (-1, 2),      # UV Aerosol Index, unitless
}

# Palettes — mirror app.js's FAMILIES[*].items[*].palette exactly, so the
# real GEE-rendered thumbnail uses the identical colour ramp the frontend
# legend shows.
PALETTES = {
    'NDTI':      ['081d58','225ea8','41b6c4','c7e9b4','ffffcc','fdae61','d73027'],
    'NDCI':      ['440154','414487','2a788e','22a884','7ad151','fde725'],
    'TSS':       ['313695','74add1','ffffbf','fdae61','a50026'],
    'NDWI':      ['d7191c','fdae61','ffffbf','abd9e9','2c7bb6'],
    'SABI':      ['00429d','73a2c6','d3eacd','f4777f','93003a'],
    'FAI':       ['08306b','6baed6','ffffcc','fd8d3c','e31a1c'],
    'Secchi':    ['a50026','fdae61','abd9e9','313695'],
    'ndvi':      ['d73027','fee08b','ffffbf','66bd63','1a9850'],
    'mndwi':     ['8b4513','f5deb3','a8ddb5','084081'],
    'ndbi':      ['313695','abd9e9','fdae61','a50026'],
    'iron':      ['ffffff','f4a460','8b4513','5c1a00'],
    'clay':      ['ffffff','a8d5a2','2d6e2a','0d3d0a'],
    'carbonate': ['ffffff','b0b0dd','404099','101077'],
    'evaporite': ['1a3a5c','5ba4cf','deebf7','ffffff'],
    'allmin':    ['313695','74add1','ffffbf','f46d43','a50026'],
    'reeds':     ['ffffff','80c080','105510'],
    'riparian':  ['ffffff','c0e080','1a4000'],
    'halophyte': ['ffffff','e8c880','402000'],
    'scrub':     ['ffffff','d4c870','303000'],
    'mud':       ['004080','80b8d0','ffffff'],
    'sav':       ['002040','0060a0','80d0f0'],
    'nightlights': ['000000','1a0033','4d0080','b30086','ff8c00','ffeb3b','ffffff'],
    'ubndbi':    ['313695','abd9e9','fdae61','a50026'],
    'slope':     ['1a9850','a6d96a','fee08b','fc8d59','d73027'],
    'vegAnomaly':['8c510a','d8b365','f5f5f5','5ab4ac','01665e'],
    'sar':       ['000000','404040','808080','c0c0c0','ffffff'],
    'nbr':       ['1a9850','a6d96a','ffffbf','fdae61','a50026','4d0000'],  # green→healthy, dark red→severely burned
    'activefire':['00000000','ffeb3b','ff8c00','d7191c'],
    'lst':       ['313695','74add1','ffffbf','fdae61','a50026','4d0000'],
    'sst':       ['08306b','2171b5','6baed6','c6dbef','fee0d2','fc9272','de2d26'],
    'no2':       ['ffffff','ffffbf','fdae61','d73027','7f0000'],
    'so2':       ['ffffff','fee8c8','fdbb84','e34a33','990000'],
    'aerosol':   ['313695','abd9e9','ffffbf','fdae61','a50026'],
}

# True/false-colour Landsat combos have no single "value" band — these show
# the actual composite bands directly instead of a palette ramp.
RGB_VIS = {
    'natural':  {'bands':['red','green','blue'], 'min':0, 'max':0.3, 'gamma':1.2},
    'falseveg': {'bands':['nir','red','green'], 'min':0, 'max':0.4},
    'urban':    {'bands':['swir2','swir1','red'], 'min':0, 'max':0.4},
    'agri':     {'bands':['swir1','nir','blue'], 'min':0, 'max':0.4},
    # Sentinel-2 true colour — 10m native resolution vs Landsat's 30m,
    # roughly 3x sharper for spotting individual streets/buildings.
    's2rgb':    {'bands':['B4','B3','B2'], 'min':0, 'max':0.3, 'gamma':1.2},
}


# Native ground resolution (metres/pixel) for every index — used to size
# thumbnails dynamically so a large ROI doesn't get squeezed into the same
# fixed pixel count as a small one (which is what caused blocky/low-res
# results on bigger regions).
NATIVE_SCALE = {
    'NDTI':10, 'NDCI':10, 'TSS':10, 'NDWI':10, 'SABI':10, 'FAI':10, 'Secchi':10,
    'ndvi':30, 'mndwi':30, 'ndbi':30,
    'iron':30, 'clay':30, 'carbonate':30, 'evaporite':30, 'allmin':30,
    'reeds':10, 'riparian':10, 'halophyte':10, 'scrub':10, 'mud':10, 'sav':10,
    'nightlights':500, 's2rgb':10, 'ubndbi':30,
    'natural':30, 'falseveg':30, 'urban':30, 'agri':30,
    'elevation':30, 'slope':30, 'vegAnomaly':10, 'sar':10,
    'nbr':10, 'activefire':1000, 'lst':30, 'sst':1000,
    'no2':1113, 'so2':1113, 'aerosol':1113,  # Sentinel-5P native ~7km, oversampled to ~1.1km grid
}


def compute_optimal_dimensions(roi, native_scale, min_dim=350, max_dim=1024):
    """
    Pick a thumbnail pixel size that roughly matches the sensor's real
    ground resolution across the drawn ROI, instead of always using the
    same fixed size — which blurs large regions and wastes detail on
    small ones. Capped both ways: never too small to be useless, never
    so large it makes the request slow or hits Earth Engine's limits.
    """
    try:
        coords = roi.bounds().coordinates().get(0).getInfo()
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        mid_lat = sum(lats) / len(lats)
        m_per_deg_lat = 111320
        m_per_deg_lon = 111320 * math.cos(math.radians(mid_lat))
        width_m  = (max(lons) - min(lons)) * m_per_deg_lon
        height_m = (max(lats) - min(lats)) * m_per_deg_lat
        px = max(width_m, height_m) / native_scale
        return int(max(min_dim, min(max_dim, px)))
    except Exception as e:
        logger.warning(f'Dimension calc failed, using default: {e}')
        return 640


def compute_adaptive_scale(roi_area_m2, native_scale, target_pixels=2_000_000):
    """
    A huge region computed at native 10m resolution can mean tens of
    millions of pixels for a single reduceRegion() call — that's what
    was actually causing "large regions always come back simulated"
    (the request took longer than the server's timeout, not a real
    Earth Engine failure, and no GEE tier fixes a client-side timeout).
    This keeps total pixel count roughly bounded by using a coarser
    scale for bigger regions, the same trade-off real GIS tools make
    automatically — full native detail on a farm, sensible fast
    aggregation on a whole country.
    """
    if roi_area_m2 <= 0:
        return native_scale
    ideal_scale = math.sqrt(roi_area_m2 / target_pixels)
    return max(native_scale, round(ideal_scale))


def crop_thumb_to_content(thumb_url, padding_frac=0.06):
    """
    Earth Engine's getThumbURL() always returns a rectangular image
    matching the ROI's bounding box — for a thin, diagonal, or oddly-
    angled region, most of that rectangle falls outside the actual
    drawn polygon and comes back fully transparent. The visible real
    content ends up occupying only a small fraction of the image,
    which is what was actually being reported as "only a fraction of
    the image showing."

    This fetches the thumbnail, finds the real bounding box of the
    non-transparent pixels, and crops to that — with a small padding
    margin so the region doesn't look uncomfortably tight against the
    edges. Returns a data: URI (so no extra hosting/URL needed), or
    the original URL unchanged if anything about this fails, since a
    slightly-too-wide image is a much smaller problem than a broken one.
    """
    try:
        from PIL import Image
        resp = requests.get(thumb_url, timeout=15)
        if resp.status_code != 200:
            return thumb_url
        img = Image.open(BytesIO(resp.content)).convert('RGBA')

        alpha = img.split()[-1]
        bbox = alpha.getbbox()
        if bbox is None:
            return thumb_url  # fully transparent — nothing to crop to, leave as-is

        left, top, right, bottom = bbox
        w, h = img.size
        pad_x = int((right - left) * padding_frac)
        pad_y = int((bottom - top) * padding_frac)
        left = max(0, left - pad_x); top = max(0, top - pad_y)
        right = min(w, right + pad_x); bottom = min(h, bottom + pad_y)

        # If the real content is already most of the image, cropping
        # would barely change anything — skip the extra work entirely.
        if (right - left) >= w * 0.92 and (bottom - top) >= h * 0.92:
            return thumb_url

        cropped = img.crop((left, top, right, bottom))
        buf = BytesIO()
        cropped.save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f'data:image/png;base64,{b64}'
    except Exception as e:
        logger.warning(f'Thumbnail crop-to-content skipped (non-fatal): {e}')
        return thumb_url


def crop_thumb_pair_to_shared_content(url_a, url_b, padding_frac=0.06):
    """
    Same idea as crop_thumb_to_content(), but for a before/after PAIR
    that needs to stay visually comparable — crops both images to the
    UNION of their content bounds (not each one's own tightest crop
    independently), so a real side-by-side comparison isn't looking at
    two different zoom levels. Returns (url_a, url_b), each either a
    cropped data: URI or its original URL unchanged if anything fails.
    """
    try:
        from PIL import Image
        resp_a = requests.get(url_a, timeout=15)
        resp_b = requests.get(url_b, timeout=15)
        if resp_a.status_code != 200 or resp_b.status_code != 200:
            return url_a, url_b

        img_a = Image.open(BytesIO(resp_a.content)).convert('RGBA')
        img_b = Image.open(BytesIO(resp_b.content)).convert('RGBA')
        if img_a.size != img_b.size:
            return url_a, url_b  # shouldn't happen (same ROI/dimensions), but don't risk a mismatched union

        bbox_a = img_a.split()[-1].getbbox()
        bbox_b = img_b.split()[-1].getbbox()
        if bbox_a is None or bbox_b is None:
            return url_a, url_b

        left = min(bbox_a[0], bbox_b[0]); top = min(bbox_a[1], bbox_b[1])
        right = max(bbox_a[2], bbox_b[2]); bottom = max(bbox_a[3], bbox_b[3])
        w, h = img_a.size
        pad_x = int((right - left) * padding_frac); pad_y = int((bottom - top) * padding_frac)
        left = max(0, left - pad_x); top = max(0, top - pad_y)
        right = min(w, right + pad_x); bottom = min(h, bottom + pad_y)

        if (right - left) >= w * 0.92 and (bottom - top) >= h * 0.92:
            return url_a, url_b  # already mostly full — not worth cropping

        def _encode(img):
            cropped = img.crop((left, top, right, bottom))
            buf = BytesIO()
            cropped.save(buf, format='PNG')
            return f'data:image/png;base64,{base64.b64encode(buf.getvalue()).decode("utf-8")}'

        return _encode(img_a), _encode(img_b)
    except Exception as e:
        logger.warning(f'Paired thumbnail crop skipped (non-fatal): {e}')
        return url_a, url_b


def safe_thumb_url(index_v, roi, img=None, composite=None):
    """
    Generate a real, GEE-rendered PNG of the actual computed layer,
    clipped to the drawn ROI — this is what makes the frontend show
    real satellite-derived pixels instead of a stylised placeholder.
    Returns None on any failure so a thumbnail issue never breaks
    the numeric reading itself.
    """
    native_scale = NATIVE_SCALE.get(index_v, 30)
    dims = compute_optimal_dimensions(roi, native_scale)
    raw_url = None
    try:
        if index_v == 'elevation' and composite is not None:
            # Raised-relief hillshade — reveals subtle mounds/depressions
            # that raw elevation colouring would hide, since absolute
            # elevation varies wildly by region but relief shading is
            # self-normalising (always renders 0-255 regardless of the
            # actual metres involved).
            hillshade = ee.Terrain.hillshade(composite, 315, 45)  # standard NW light, 45° sun angle
            raw_url = hillshade.clip(roi).getThumbURL({
                'region': roi, 'dimensions': dims, 'format': 'png', 'min': 0, 'max': 255
            })
        elif index_v in RGB_VIS and composite is not None:
            vis = RGB_VIS[index_v]
            raw_url = composite.clip(roi).getThumbURL({
                'region': roi, 'dimensions': dims, 'format': 'png', **vis
            })
        elif img is not None:
            lo, hi = RANGES.get(index_v, (0, 1))
            palette = PALETTES.get(index_v, ['0a3040', '0e5468', '00c2d1'])
            raw_url = img.select('value').clip(roi).getThumbURL({
                'region': roi, 'dimensions': dims, 'format': 'png',
                'min': lo, 'max': hi, 'palette': palette
            })
    except Exception as e:
        logger.warning(f'Thumbnail generation failed for {index_v}: {e}')
        return None

    if raw_url is None:
        return None
    # A thin, diagonal, or oddly-angled ROI leaves most of this
    # rectangular thumbnail transparent — crop to the real visible
    # content so the region isn't lost in a mostly-empty image.
    return crop_thumb_to_content(raw_url)



def mask_s2(image):
    scl = image.select('SCL')
    clear = (scl.neq(1).And(scl.neq(3)).And(scl.neq(8))
             .And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11)))
    optical = image.select(['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12'])
    return optical.updateMask(clear).divide(10000).copyProperties(image, ['system:time_start'])


def mask_l9(image):
    qa = image.select('QA_PIXEL')
    clear = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
    return image.updateMask(clear).multiply(0.0000275).add(-0.2).copyProperties(image, ['system:time_start'])


# ══════════════════════════════════════════════════════════════
# MULTI-MISSION LANDSAT ARCHIVE — 1984 to present
# ══════════════════════════════════════════════════════════════
# Landsat 5, 7, 8, and 9 use DIFFERENT band numbers for the same
# physical wavelength — e.g. Landsat 5/7's "SR_B4" is near-infrared,
# but Landsat 8/9's "SR_B4" is red. Naively swapping in older Landsat
# missions without remapping bands would silently corrupt every index
# computed from them (this is the exact class of bug behind the
# impossible NDVI/TSS values found in an earlier academic dissertation
# on this exact lake — a missing/incorrect band or scale mapping
# across sensor generations). Every image below is remapped to a
# common logical naming (blue/green/red/nir/swir1/swir2) and scaled
# with the same Collection 2 Level 2 formula BEFORE any index formula
# ever touches it, so the same formula code is always correct
# regardless of which satellite the image came from.
LANDSAT_MISSIONS = [
    # (collection id, mission start, mission end or None, band map)
    ('LANDSAT/LT05/C02/T1_L2', '1984-03-01', '2013-01-05',
     {'blue': 'SR_B1', 'green': 'SR_B2', 'red': 'SR_B3', 'nir': 'SR_B4', 'swir1': 'SR_B5', 'swir2': 'SR_B7', 'qa': 'QA_PIXEL'}),
    ('LANDSAT/LE07/C02/T1_L2', '1999-04-15', None,
     {'blue': 'SR_B1', 'green': 'SR_B2', 'red': 'SR_B3', 'nir': 'SR_B4', 'swir1': 'SR_B5', 'swir2': 'SR_B7', 'qa': 'QA_PIXEL'}),
    ('LANDSAT/LC08/C02/T1_L2', '2013-04-11', None,
     {'blue': 'SR_B2', 'green': 'SR_B3', 'red': 'SR_B4', 'nir': 'SR_B5', 'swir1': 'SR_B6', 'swir2': 'SR_B7', 'qa': 'QA_PIXEL'}),
    ('LANDSAT/LC09/C02/T1_L2', '2021-10-31', None,
     {'blue': 'SR_B2', 'green': 'SR_B3', 'red': 'SR_B4', 'nir': 'SR_B5', 'swir1': 'SR_B6', 'swir2': 'SR_B7', 'qa': 'QA_PIXEL'}),
]


def mask_landsat_generic(image, band_map):
    """
    Cloud-masks, applies the standard Collection 2 Level-2 scale
    factor (identical formula across every Landsat mission, so this
    part alone was never the risk), and — the actual fix — renames
    each sensor's differently-numbered bands to common logical names
    so every index formula downstream is automatically correct no
    matter which satellite generation the pixel came from.
    """
    qa = image.select(band_map['qa'])
    clear = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
    renamed = image.select(
        [band_map['blue'], band_map['green'], band_map['red'], band_map['nir'], band_map['swir1'], band_map['swir2']],
        ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
    )
    scaled = renamed.updateMask(clear).multiply(0.0000275).add(-0.2)
    return scaled.copyProperties(image, ['system:time_start'])


def get_landsat_collection(roi, start_date, end_date):
    """
    Merges every Landsat mission whose operational lifetime overlaps
    the requested date range into one collection, bands already
    remapped to common logical names. This is what actually extends
    the usable archive back to 1984 instead of being capped at
    Landsat 9's 2021 launch — the same date range that used to return
    "no images" for anything before Sep 2021 now pulls from whichever
    real satellite was actually operating at that time.
    """
    merged = None
    for coll_id, mission_start, mission_end, band_map in LANDSAT_MISSIONS:
        if mission_end and start_date > mission_end:
            continue
        if end_date < mission_start:
            continue
        coll = (ee.ImageCollection(coll_id)
                .filterBounds(roi).filterDate(start_date, end_date)
                .filter(ee.Filter.lt('CLOUD_COVER', 30))
                .map(lambda img, bm=band_map: mask_landsat_generic(img, bm)))
        merged = coll if merged is None else merged.merge(coll)
    return merged


def roi_from_coords(coords):
    """coords: [[lat,lng], ...] from the frontend."""
    ring = [[c[1], c[0]] for c in coords]  # to [lng,lat]
    return ee.Geometry.Polygon([ring])


def water_index_image(s2, index_v):
    wm = s2.normalizedDifference(['B3', 'B8']).gt(-0.05)
    if index_v == 'NDTI':
        return s2.normalizedDifference(['B4', 'B3']).rename('value').updateMask(wm)
    if index_v == 'NDCI':
        return s2.normalizedDifference(['B5', 'B4']).rename('value').updateMask(wm)
    if index_v == 'TSS':
        return s2.select('B4').divide(s2.select('B3').max(0.0001)).subtract(1).rename('value').updateMask(wm)
    if index_v == 'NDWI':
        return s2.normalizedDifference(['B3', 'B8']).rename('value')
    if index_v == 'SABI':
        return (s2.select('B8').subtract(s2.select('B4'))
                .divide(s2.select('B2').add(s2.select('B3')).max(0.0001)).rename('value').updateMask(wm))
    if index_v == 'FAI':
        base = s2.select('B4').add(s2.select('B11').subtract(s2.select('B4')).multiply(0.1012))
        return s2.select('B8').subtract(base).rename('value').updateMask(wm)
    if index_v == 'Secchi':
        return (s2.select('B2').divide(s2.select('B4').add(s2.select('B3')).max(0.0001))
                .multiply(2.5).rename('value').updateMask(wm))
    return s2.normalizedDifference(['B3', 'B8']).rename('value')


def landsat_combo_image(image, index_v):
    # Uses logical band names (blue/green/red/nir/swir1/swir2) — correct
    # automatically regardless of which Landsat mission the composite
    # came from, since get_landsat_collection() already remapped them.
    if index_v == 'ndvi':
        return image.normalizedDifference(['nir', 'red']).rename('value')
    if index_v == 'mndwi':
        return image.normalizedDifference(['green', 'swir1']).rename('value')
    if index_v == 'ndbi':
        return image.normalizedDifference(['swir1', 'nir']).rename('value')
    # RGB-style combos have no single "value" band — return NDVI as the stat proxy
    return image.normalizedDifference(['nir', 'red']).rename('value')


def geology_image(image, index_v):
    ndwi = image.normalizedDifference(['green', 'nir'])
    land_mask = ndwi.lt(0.0)
    blue, green, red, nir, swir1, swir2 = [image.select(b) for b in ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']]
    if index_v == 'iron':
        img = red.divide(blue.max(0.0001))
    elif index_v == 'clay':
        img = swir1.divide(swir2.max(0.0001))
    elif index_v == 'carbonate':
        img = swir1.divide(swir2.add(red).max(0.0001))
    elif index_v == 'evaporite':
        img = blue.add(green).add(red).divide(3).multiply(swir1.divide(swir2.max(0.0001)))
    else:  # allmin
        iron, clay, carb, sil = red.divide(blue.max(0.0001)), swir1.divide(swir2.max(0.0001)), \
                                 swir1.divide(swir2.add(red).max(0.0001)), nir.divide(swir1.max(0.0001))
        img = iron.multiply(0.3).add(clay.multiply(0.25)).add(carb.multiply(0.25)).add(sil.multiply(0.2))
    return img.rename('value').updateMask(land_mask)


def vegetation_image(s2, index_v):
    B2,B3,B4,B5,B8,B11 = [s2.select(b) for b in ['B2','B3','B4','B5','B8','B11']]
    ndwi = s2.normalizedDifference(['B3','B8'])
    shoreline_mask = ndwi.gt(-0.3).And(ndwi.lt(0.3))
    if index_v == 'reeds':
        ndvi = s2.normalizedDifference(['B8','B4']); ndi = s2.normalizedDifference(['B8','B11'])
        img = ndvi.add(ndi).divide(2)
    elif index_v == 'riparian':
        evi = B8.subtract(B4).divide(B8.add(B4.multiply(6)).subtract(B2.multiply(7.5)).add(1).max(0.0001)).multiply(2.5)
        re  = B5.divide(B4.max(0.0001))
        img = evi.multiply(0.6).add(re.multiply(0.4))
    elif index_v == 'halophyte':
        img = B3.divide(B4.max(0.0001)).multiply(B8.divide(B11.max(0.0001)))
    elif index_v == 'scrub':
        img = s2.normalizedDifference(['B8','B4'])
    elif index_v == 'mud':
        img = B3.add(B4).divide(2).divide(B8.add(0.001))
    elif index_v == 'sav':
        img = B8.subtract(B2).divide(B8.add(B2).max(0.0001))
        return img.rename('value').updateMask(ndwi.gt(0.0))
    else:
        img = s2.normalizedDifference(['B8','B4'])
    return img.rename('value').updateMask(shoreline_mask)


# ══════════════════════════════════════════════════════════════
# FIRE, HEAT, AND POLLUTION — three new families, each backed by a
# real, standard satellite data product rather than a derived proxy.
# ══════════════════════════════════════════════════════════════

def fire_index_image(s2, index_v):
    """
    Normalized Burn Ratio (NBR) — a standard, published technique for
    mapping burned vegetation: healthy vegetation reflects strongly in
    NIR and weakly in SWIR2, while a burn scar does the opposite. This
    is the same formula used in official post-fire severity mapping
    (e.g. USGS/USFS dNBR products), just applied to a single period
    here rather than a before/after difference.
    """
    nir, swir2 = s2.select('B8'), s2.select('B12')
    img = nir.subtract(swir2).divide(nir.add(swir2).max(0.0001))
    return img.rename('value')


def active_fire_mask(coords_roi, start_date, end_date):
    """
    Real active-fire hotspot detections from NASA FIRMS (Fire
    Information for Resource Management System) — this is observed
    thermal-anomaly hotspot data, not a derived index. Returns a
    0/1-ish confidence image, masked to hotspot pixels only.
    """
    coll = ee.ImageCollection('FIRMS').filterBounds(coords_roi).filterDate(start_date, end_date)
    img = coll.select('T21').max()  # brightness temperature of the hottest detection per pixel
    # Normalise into a roughly 0-1 confidence-style value for consistent legend rendering
    normalized = img.subtract(300).divide(500).clamp(0, 1)
    return normalized.rename('value')


def heat_index_image(composite, index_v, is_landsat):
    """
    Land Surface Temperature from Landsat's thermal band (ST_B10,
    already scaled to Kelvin in Collection 2 Level-2 products — a
    standard, published product used in real urban-heat-island
    studies), converted to Celsius.
    """
    if is_landsat:
        kelvin = composite.select('ST_B10')
        celsius = kelvin.multiply(0.00341802).add(149.0).subtract(273.15)
        return celsius.rename('value')
    return composite.rename('value')


def sst_image(coords_roi, start_date, end_date):
    """
    Real Sea Surface Temperature from NOAA's optimum-interpolation SST
    product — a genuine, published ocean temperature dataset, not
    derived from a land-focused sensor.
    """
    coll = (ee.ImageCollection('NOAA/CDR/OISST/V2_1')
            .filterBounds(coords_roi).filterDate(start_date, end_date))
    img = coll.select('sst').mean().multiply(0.01)  # product is scaled by 100
    return img.rename('value'), coll.size()


def pollution_index_image(coords_roi, start_date, end_date, index_v):
    """
    Real Sentinel-5P TROPOMI air-quality columns — genuine satellite
    measurements of atmospheric trace gases, not estimated from
    surface reflectance. NO2 (traffic/industrial), SO2 (industrial/
    volcanic), and UV Aerosol Index (smoke, dust, pollution haze) are
    all standard, published air-quality products.
    """
    band_map = {
        'no2':     ('COPERNICUS/S5P/OFFL/L3_NO2', 'tropospheric_NO2_column_number_density'),
        'so2':     ('COPERNICUS/S5P/OFFL/L3_SO2', 'SO2_column_number_density'),
        'aerosol': ('COPERNICUS/S5P/OFFL/L3_AER_AI', 'absorbing_aerosol_index'),
    }
    coll_id, band = band_map[index_v]
    coll = (ee.ImageCollection(coll_id).select(band)
            .filterBounds(coords_roi).filterDate(start_date, end_date))
    img = coll.mean()
    return img.rename('value'), coll.size()


# ══════════════════════════════════════════════════════════════
# CHANGE MAP & HOTSPOT DETECTION
# Two periods of the same index, compared pixel-by-pixel — classifies
# every pixel as increased/stable/decreased, computes area per class,
# and automatically ranks the zones of largest change as prioritised
# hotspots. Reuses the exact connected-component technique already
# proven in Water Body Inventory.
# ══════════════════════════════════════════════════════════════

def get_composite_and_index(family, index_v, roi, start_date, end_date):
    """
    Shared helper — builds the composite and computes the single-band
    'value' image for a given family/index/date range. Returns
    (composite, img, image_count), or (None, None, 0) if that specific
    index has no single comparable value (true-colour composites) or
    no imagery was found.
    """
    try:
        if family == 'water':
            coll = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                    .filterBounds(roi).filterDate(start_date, end_date)
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
                    .sort('CLOUDY_PIXEL_PERCENTAGE').limit(30))
            count = coll.size().getInfo()
            if count == 0: return None, None, 0
            composite = coll.map(mask_s2).median().clip(roi)
            return composite, water_index_image(composite, index_v), count

        elif family == 'landsat':
            if index_v not in ('ndvi', 'mndwi', 'ndbi'):
                return None, None, 0  # RGB combos have no single comparable value
            coll = get_landsat_collection(roi, start_date, end_date)
            count = coll.size().getInfo() if coll is not None else 0
            if count == 0: return None, None, 0
            composite = coll.median().clip(roi)
            return composite, landsat_combo_image(composite, index_v), count

        elif family == 'geo':
            coll = get_landsat_collection(roi, start_date, end_date)
            count = coll.size().getInfo() if coll is not None else 0
            if count == 0: return None, None, 0
            composite = coll.median().clip(roi)
            return composite, geology_image(composite, index_v), count

        elif family == 'veg':
            coll = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                    .filterBounds(roi).filterDate(start_date, end_date)
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
                    .sort('CLOUDY_PIXEL_PERCENTAGE').limit(30))
            count = coll.size().getInfo()
            if count == 0: return None, None, 0
            composite = coll.map(mask_s2).median().clip(roi)
            return composite, vegetation_image(composite, index_v), count

        elif family == 'urban':
            if index_v == 'nightlights':
                coll = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG').filterBounds(roi).filterDate(start_date, end_date)
                count = coll.size().getInfo()
                if count == 0: return None, None, 0
                composite = coll.select('avg_rad').median().clip(roi)
                return composite, composite.max(0).rename('value'), count
            elif index_v == 'ubndbi':
                coll = get_landsat_collection(roi, start_date, end_date)
                count = coll.size().getInfo() if coll is not None else 0
                if count == 0: return None, None, 0
                composite = coll.median().clip(roi)
                return composite, composite.normalizedDifference(['swir1', 'nir']).rename('value'), count
            else:
                return None, None, 0  # s2rgb — true colour, no single value
        else:
            return None, None, 0
    except Exception as e:
        logger.warning(f'get_composite_and_index failed for {family}/{index_v}: {e}')
        return None, None, 0


def true_color_vis(composite, family):
    """
    Renders a true-colour RGB view of whichever composite came back —
    Sentinel-2 uses its raw band names, the merged multi-mission
    Landsat archive uses the logical names (red/green/blue) we
    remapped everything to. Works regardless of which sensor actually
    produced the image.
    """
    if family in ('water', 'veg'):
        return composite.visualize(bands=['B4', 'B3', 'B2'], min=0, max=0.3, gamma=1.2)
    else:  # landsat, geo, urban/ubndbi — merged Landsat archive, logical band names
        return composite.visualize(bands=['red', 'green', 'blue'], min=0, max=0.3, gamma=1.2)


def run_water_level(start1, end1, start2, end2, coords):
    """
    Estimates water surface elevation change using a genuine,
    established remote sensing technique: since optical satellites
    can't measure water depth directly, this reads the Copernicus DEM
    elevation at the water's EDGE (the boundary of the NDWI water
    mask) — the shoreline sits exactly at the water surface elevation
    by definition, so the average boundary elevation is a real proxy
    for water level. Doing this for two periods and comparing gives an
    estimated water level change in real metres, not just a relative
    index value. This is the same method used in published lake/
    reservoir monitoring studies (e.g. Lake Mead).

    Honest limitation, stated here and again in the response: the DEM
    is a static, one-time dataset — if the shoreline terrain itself
    changed between the two periods (erosion, sedimentation), that
    introduces some error. Absolute elevation values also carry the
    DEM's own vertical uncertainty (~3-10m), though comparing the same
    location at two times cancels out much of that systematic error,
    making the RELATIVE change more trustworthy than either reading
    alone — still a PROXY MEASUREMENT, not certified altimetry.
    """
    if not (EE_AVAILABLE and init_ee()):
        return {'error': 'GEE not available on this server right now — check /api/health first.'}
    try:
        roi = roi_from_coords(coords)
        dem = ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM').mosaic().clip(roi)

        def boundary_elevation(start_date, end_date):
            coll = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                    .filterBounds(roi).filterDate(start_date, end_date)
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
                    .sort('CLOUDY_PIXEL_PERCENTAGE').limit(30))
            count = coll.size().getInfo()
            if count == 0:
                return None, None, count
            composite = coll.map(mask_s2).median().clip(roi)
            water_mask = composite.normalizedDifference(['B3', 'B8']).gt(0.0)
            # Boundary = water pixels adjacent to non-water pixels — an erosion
            # trick: the mask minus its own erosion leaves just the edge ring.
            eroded = water_mask.reduceNeighborhood(reducer=ee.Reducer.min(), kernel=ee.Kernel.square(1))
            boundary = water_mask.And(eroded.Not()).selfMask()
            elev_at_boundary = dem.updateMask(boundary)
            stats = elev_at_boundary.reduceRegion(
                reducer=ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True),
                geometry=roi, scale=30, bestEffort=True, maxPixels=1e9, tileScale=4
            ).getInfo()
            return stats.get('DEM_mean'), stats.get('DEM_count'), count

        level1, pixels1, count1 = boundary_elevation(start1, end1)
        if level1 is None:
            return {'error': 'No imagery found for the first period.'}
        level2, pixels2, count2 = boundary_elevation(start2, end2)
        if level2 is None:
            return {'error': 'No imagery found for the second period.'}
        if pixels1 is None or pixels2 is None or pixels1 < 5 or pixels2 < 5:
            return {'error': 'Not enough water-boundary pixels detected to estimate a level — try a larger region or a lake/reservoir with a clearer shoreline.'}

        change_m = round(level2 - level1, 2)
        result = {
            'source': 'gee_live',
            'images_period1': count1, 'images_period2': count2,
            'level1_m': round(level1, 2), 'level2_m': round(level2, 2),
            'change_m': change_m,
            'boundary_pixels1': int(pixels1), 'boundary_pixels2': int(pixels2),
        }

        # Real USGS gauge validation, US coverage only — a genuinely
        # measured water level, not another view of the DEM estimate.
        # Never blocks the main result if it fails or no gauge is nearby.
        try:
            import ground_truth
            lats = [c[0] for c in coords]; lngs = [c[1] for c in coords]
            # Pad the ROI's bounding box slightly — a gauge just outside
            # your exact drawn shape is still relevant for the same water body.
            pad = 0.05  # ~5km at most latitudes
            bbox = (min(lngs)-pad, min(lats)-pad, max(lngs)+pad, max(lats)+pad)
            gt = ground_truth.get_usgs_water_level(bbox)
            if gt:
                result['usgs_ground_truth'] = gt
        except Exception as e:
            logger.warning(f'USGS ground-truth lookup skipped (non-fatal): {e}')

        return result
    except Exception as e:
        logger.error(f'Water level estimation error: {e}')
        return {'error': str(e)}


def run_change_map(family, index_v, start1, end1, start2, end2, coords):
    """
    Compares the same index across two periods pixel-by-pixel. Returns
    area breakdown (increased/stable/decreased) plus automatically
    detected, ranked hotspot zones — the same kind of "before/after/
    change" map and priority table a professional EO report would
    include, rather than just two separate readings.
    """
    if not (EE_AVAILABLE and init_ee()):
        return {'error': 'GEE not available on this server right now — check /api/health first.'}
    try:
        roi = roi_from_coords(coords)

        composite1, img1, count1 = get_composite_and_index(family, index_v, roi, start1, end1)
        if img1 is None:
            return {'error': 'No imagery found for the first period, or this index doesn\'t support change-map analysis (true-colour composites have no single comparable value).'}

        composite2, img2, count2 = get_composite_and_index(family, index_v, roi, start2, end2)
        if img2 is None:
            return {'error': 'No imagery found for the second period.'}

        val1 = img1.select('value')
        val2 = img2.select('value')
        diff = val2.subtract(val1).rename('diff')

        lo, hi = RANGES.get(index_v, (0, 1))
        span = hi - lo
        sig_threshold = span * 0.12  # 12% of the index's typical range = "significant" change

        roi_area_m2 = roi.area(1).getInfo()
        native_scale = NATIVE_SCALE.get(index_v, 30)
        effective_scale = compute_adaptive_scale(roi_area_m2, native_scale, target_pixels=1_500_000)

        classified = (ee.Image(0)
                      .where(diff.gt(sig_threshold), 1)
                      .where(diff.lt(-sig_threshold), -1)
                      .rename('cls').updateMask(diff.mask()))

        area_img = ee.Image.pixelArea().addBands(classified)
        grouped = area_img.reduceRegion(
            reducer=ee.Reducer.sum().group(groupField=1, groupName='cls'),
            geometry=roi, scale=effective_scale, bestEffort=True, maxPixels=1e10, tileScale=4
        ).getInfo()
        class_ha = {-1: 0.0, 0: 0.0, 1: 0.0}
        for g in grouped.get('groups', []):
            class_ha[int(g['cls'])] = (g.get('sum') or 0) / 10000
        total_ha = sum(class_ha.values()) or 1

        mean_stats = ee.Image.cat([val1.rename('start'), val2.rename('end')]).reduceRegion(
            reducer=ee.Reducer.mean(), geometry=roi, scale=effective_scale, bestEffort=True, maxPixels=1e9, tileScale=4
        ).getInfo()
        start_mean = mean_stats.get('start')
        end_mean = mean_stats.get('end')
        overall_change_pct = None
        if start_mean not in (None, 0):
            overall_change_pct = round((end_mean - start_mean) / abs(start_mean) * 100, 1)

        # ── Hotspot detection — zones of largest significant change ──
        # connectedComponents + reduceToVectors are the two most memory-
        # hungry operations Earth Engine offers ("User memory limit
        # exceeded" was happening right here). Fixed with a deliberately
        # coarser scale just for this step (hotspot zones don't need
        # pixel-perfect boundaries) plus tileScale, which tells Earth
        # Engine to split the work into smaller pieces instead of
        # holding it all in memory at once.
        hotspots = []
        try:
            hotspot_scale = max(effective_scale * 3, native_scale * 3)
            sig_mask = diff.abs().gt(sig_threshold).selfMask()
            labeled = sig_mask.connectedComponents(connectedness=ee.Kernel.plus(1), maxSize=256)
            zones_fc = labeled.select('labels').reduceToVectors(
                geometry=roi, scale=hotspot_scale, geometryType='polygon',
                maxPixels=1e9, bestEffort=True, labelProperty='zone_id', tileScale=8
            )
            zones_fc = zones_fc.map(lambda f: f.set('area_ha', f.geometry().area(1).divide(10000)))
            zones_fc = zones_fc.sort('area_ha', False).limit(8)
            zone_stats_fc = diff.reduceRegions(collection=zones_fc, reducer=ee.Reducer.mean(), scale=hotspot_scale, tileScale=4)

            letters = 'ABCDEFGH'
            raw_zones = []
            for i, f in enumerate(zone_stats_fc.getInfo().get('features', [])):
                props = f.get('properties', {})
                area_ha = round(props.get('area_ha', 0) or 0, 2)
                mean_change = props.get('mean')
                if area_ha < 0.5 or mean_change is None:
                    continue  # skip noise-sized zones

                # Extract the actual zone boundary so the frontend can draw it
                # on the map — this was being computed by Earth Engine and then
                # silently discarded before, which is why hotspot zones were
                # never actually visible anywhere except as a text table.
                geom = f.get('geometry') or {}
                ring_lnglat = geom.get('coordinates', [[]])[0] if geom.get('type') == 'Polygon' else []
                ring = [[pt[1], pt[0]] for pt in ring_lnglat]  # GeoJSON [lng,lat] -> our [lat,lng] convention
                centroid = None
                if ring:
                    centroid = [sum(p[0] for p in ring) / len(ring), sum(p[1] for p in ring) / len(ring)]

                raw_zones.append({
                    'area_ha': area_ha, 'mean_change': round(mean_change, 4),
                    'ring': ring, 'centroid': centroid,
                })

            raw_zones.sort(key=lambda h: -(h['area_ha'] * abs(h['mean_change'])))
            for i, z in enumerate(raw_zones[:5]):
                hotspots.append({
                    'zone': letters[i] if i < len(letters) else str(i + 1),
                    'area_ha': z['area_ha'],
                    'mean_change': z['mean_change'],
                    'direction': 'increase' if z['mean_change'] > 0 else 'decrease',
                    'priority': 'High' if i == 0 else 'Medium' if i == 1 else 'Low',
                    'ring': z['ring'], 'centroid': z['centroid'],
                })
        except Exception as e:
            logger.warning(f'Hotspot detection failed (non-fatal, continuing without it): {e}')

        # ── Visualisation: diverging palette, red=decrease, grey=stable, green=increase ──
        dims = compute_optimal_dimensions(roi, native_scale)
        thumb_url = None
        try:
            vis = classified.visualize(min=-1, max=1, palette=['d73027', 'bdbdbd', '1a9850'])
            thumb_url = vis.clip(roi).getThumbURL({'region': roi, 'dimensions': dims, 'format': 'png'})
            thumb_url = crop_thumb_to_content(thumb_url)
        except Exception as e:
            logger.warning(f'Change map thumbnail failed: {e}')

        # Real before/after true-colour photos — this is what actually
        # answers "show me the exact thing that changed" (bare land in
        # 2020, buildings in 2025), rather than only the abstract
        # increase/stable/decrease classification map above, which
        # reads as a diffuse, unclear "fog" on its own without the two
        # real reference photos alongside it.
        before_thumb_url = None
        after_thumb_url = None
        try:
            before_thumb_url = true_color_vis(composite1, family).clip(roi).getThumbURL(
                {'region': roi, 'dimensions': dims, 'format': 'png'})
            after_thumb_url = true_color_vis(composite2, family).clip(roi).getThumbURL(
                {'region': roi, 'dimensions': dims, 'format': 'png'})
            # Cropped together to a SHARED bounding box (not each own
            # tightest crop independently) so before/after stay visually
            # comparable at the same zoom level — see the two-image
            # variant of the helper below.
            before_thumb_url, after_thumb_url = crop_thumb_pair_to_shared_content(before_thumb_url, after_thumb_url)
        except Exception as e:
            logger.warning(f'Change map before/after thumbnail failed: {e}')

        return {
            'source': 'gee_live',
            'images_period1': count1, 'images_period2': count2,
            'before_thumb_url': before_thumb_url, 'after_thumb_url': after_thumb_url,
            'start_mean': round(start_mean, 4) if start_mean is not None else None,
            'end_mean': round(end_mean, 4) if end_mean is not None else None,
            'overall_change_pct': overall_change_pct,
            'increased_ha': round(class_ha[1], 2), 'increased_pct': round(class_ha[1] / total_ha * 100, 1),
            'stable_ha': round(class_ha[0], 2), 'stable_pct': round(class_ha[0] / total_ha * 100, 1),
            'decreased_ha': round(class_ha[-1], 2), 'decreased_pct': round(class_ha[-1] / total_ha * 100, 1),
            'hotspots': hotspots,
            'thumb_url': thumb_url,
        }
    except Exception as e:
        logger.error(f'Change map error: {e}')
        return {'error': str(e)}


# ── main entry point ────────────────────────────────────────────

def run_reading(family, index_v, start_date, end_date, coords):
    """Returns dict with mean/min/max/std/images or {'error': ...}"""
    if not EE_AVAILABLE:
        return simulate(family, index_v, start_date, end_date,
                         real_error='earthengine-api package not installed on the server')
    if not init_ee():
        return simulate(family, index_v, start_date, end_date,
                         real_error=_last_ee_init_error or 'Earth Engine initialisation failed (see server logs)')

    try:
        roi = roi_from_coords(coords)

        if family in ('water',):
            # Capped to the 30 clearest images — without this, a long
            # date range means Earth Engine has to merge potentially
            # dozens of images every single time, which is what was
            # actually causing "preview takes over a minute" (this
            # affects every Sentinel-2 family equally, not just water).
            # Sorting by cloud cover first means we lose nothing in
            # quality — if anything the composite gets cleaner.
            coll = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                    .filterBounds(roi).filterDate(start_date, end_date)
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
                    .sort('CLOUDY_PIXEL_PERCENTAGE').limit(30))
            count = coll.size().getInfo()
            if count == 0: return {'error': 'No Sentinel-2 images for this period/region.'}
            composite = coll.map(mask_s2).median().clip(roi)
            img = water_index_image(composite, index_v)
            scale = 30

        elif family == 'landsat':
            coll = get_landsat_collection(roi, start_date, end_date)
            count = coll.size().getInfo() if coll is not None else 0
            if count == 0: return {'error': 'No Landsat imagery for this period/region (Landsat archive covers 1984–present).'}
            composite = coll.median().clip(roi)
            img = landsat_combo_image(composite, index_v)
            scale = 30

        elif family == 'geo':
            coll = get_landsat_collection(roi, start_date, end_date)
            count = coll.size().getInfo() if coll is not None else 0
            if count == 0: return {'error': 'No Landsat imagery for this period/region (Landsat archive covers 1984–present).'}
            composite = coll.median().clip(roi)
            img = geology_image(composite, index_v)
            scale = 30

        elif family == 'veg':
            coll = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                    .filterBounds(roi).filterDate(start_date, end_date)
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
                    .sort('CLOUDY_PIXEL_PERCENTAGE').limit(30))
            count = coll.size().getInfo()
            if count == 0: return {'error': 'No Sentinel-2 images for this period/region.'}
            composite = coll.map(mask_s2).median().clip(roi)
            img = vegetation_image(composite, index_v)
            scale = 10

        elif family == 'urban':
            if index_v == 'nightlights':
                # VIIRS monthly nighttime lights — global, no cloud masking needed.
                # City growth shows up directly as brightening over time.
                coll = (ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')
                        .filterBounds(roi).filterDate(start_date, end_date))
                count = coll.size().getInfo()
                if count == 0:
                    return {'error': 'No VIIRS nighttime lights data for this period/region (data starts April 2012).'}
                composite = coll.select('avg_rad').median().clip(roi)
                img = composite.max(0).rename('value')  # clip stray negative radiance noise
                scale = 500  # VIIRS native resolution is ~500m

            elif index_v == 's2rgb':
                # Sentinel-2 true colour — 10m resolution, ~3x sharper than
                # Landsat 9's 30m, best option for spotting individual
                # streets/buildings/new construction by eye.
                # Filter aligned to 40% (same as every other Sentinel-2
                # branch) — this used to be a stricter 20%, which made the
                # "Original imagery" companion photo fail far more often
                # than the main analysis it's paired with.
                coll = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                        .filterBounds(roi).filterDate(start_date, end_date)
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
                        .sort('CLOUDY_PIXEL_PERCENTAGE').limit(30))
                count = coll.size().getInfo()
                if count == 0: return {'error': 'No Sentinel-2 images for this period/region.'}
                composite = coll.map(mask_s2).median().clip(roi)
                img = composite.normalizedDifference(['B8','B4']).rename('value')  # NDVI as stat proxy only
                scale = 10

            else:  # 'ubndbi' — built-up index, same math as Landsat NDBI
                coll = get_landsat_collection(roi, start_date, end_date)
                count = coll.size().getInfo() if coll is not None else 0
                if count == 0: return {'error': 'No Landsat imagery for this period/region (Landsat archive covers 1984–present).'}
                composite = coll.median().clip(roi)
                img = composite.normalizedDifference(['swir1', 'nir']).rename('value')
                scale = 30

        elif family == 'archaeology':
            if index_v == 'elevation':
                # Copernicus GLO-30 — free global 30m DEM. It's registered
                # in Earth Engine's catalog as a tiled ImageCollection, not
                # a single Image, so it needs mosaicking into one
                # continuous surface before use. Static dataset, not
                # date-dependent, so "count" is a placeholder.
                dem = ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM').mosaic().clip(roi)
                composite = dem  # used by safe_thumb_url for hillshade rendering
                img = dem.rename('value')  # stats report real elevation in metres
                count = 1
                scale = 30

            elif index_v == 'slope':
                dem = ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM').mosaic().clip(roi)
                composite = dem
                img = ee.Terrain.slope(dem).rename('value')  # degrees
                count = 1
                scale = 30

            elif index_v == 'vegAnomaly':
                # The classic "crop mark" technique — buried walls stress
                # vegetation above them (locally lower NDVI than the
                # surrounding area), buried ditches/canals hold moisture
                # better (locally higher NDVI). Subtracting a smoothed
                # local average turns raw NDVI into an anomaly map that
                # highlights these deviations instead of overall greenness.
                coll = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                        .filterBounds(roi).filterDate(start_date, end_date)
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
                        .sort('CLOUDY_PIXEL_PERCENTAGE').limit(30))
                count = coll.size().getInfo()
                if count == 0: return {'error': 'No Sentinel-2 images for this period/region.'}
                composite = coll.map(mask_s2).median().clip(roi)
                ndvi = composite.normalizedDifference(['B8', 'B4'])
                local_avg = ndvi.focalMean(radius=30, units='meters')
                img = ndvi.subtract(local_avg).rename('value')
                scale = 10

            else:  # 'sar' — Sentinel-1 radar, can reveal buried features under dry soil/sand
                coll = (ee.ImageCollection('COPERNICUS/S1_GRD')
                        .filterBounds(roi).filterDate(start_date, end_date)
                        .filter(ee.Filter.eq('instrumentMode', 'IW'))
                        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')))
                count = coll.size().getInfo()
                if count == 0: return {'error': 'No Sentinel-1 SAR images for this period/region.'}
                composite = coll.select('VV').median().clip(roi)
                img = composite.rename('value')
                scale = 10

        elif family == 'fire':
            if index_v == 'activefire':
                count = ee.ImageCollection('FIRMS').filterBounds(roi).filterDate(start_date, end_date).size().getInfo()
                if count == 0: return {'error': 'No FIRMS active-fire detections for this period/region.'}
                img = active_fire_mask(roi, start_date, end_date).clip(roi)
                composite = None  # no true-colour composite for this data type — safe_thumb_url falls through to its generic single-band branch correctly
                scale = 1000
            else:  # 'nbr' — burn severity from Sentinel-2
                coll = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                        .filterBounds(roi).filterDate(start_date, end_date)
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
                        .sort('CLOUDY_PIXEL_PERCENTAGE').limit(30))
                count = coll.size().getInfo()
                if count == 0: return {'error': 'No Sentinel-2 images for this period/region.'}
                composite = coll.map(mask_s2).median().clip(roi)
                img = fire_index_image(composite, index_v)
                scale = 10

        elif family == 'heat':
            if index_v == 'sst':
                img_raw, coll_size = sst_image(roi, start_date, end_date)
                count = coll_size.getInfo()
                if count == 0: return {'error': 'No sea surface temperature data for this period/region — this dataset only covers ocean areas.'}
                img = img_raw.clip(roi)
                composite = None  # no true-colour composite for this data type — safe_thumb_url falls through to its generic single-band branch correctly
                scale = 1000
            else:  # 'lst' — Landsat thermal band
                coll = get_landsat_collection(roi, start_date, end_date)
                count = coll.size().getInfo() if coll is not None else 0
                if count == 0: return {'error': 'No Landsat imagery for this period/region.'}
                # Thermal band isn't remapped by get_landsat_collection (that
                # only handles the optical bands) — pull ST_B10 separately
                # from the same collection window.
                thermal_coll = (ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
                                .merge(ee.ImageCollection('LANDSAT/LC08/C02/T1_L2'))
                                .filterBounds(roi).filterDate(start_date, end_date)
                                .filter(ee.Filter.lt('CLOUD_COVER', 30)))
                thermal_count = thermal_coll.size().getInfo()
                if thermal_count == 0: return {'error': 'No Landsat thermal imagery for this period/region (thermal band only on Landsat 8/9).'}
                composite = thermal_coll.median().clip(roi)
                img = heat_index_image(composite, index_v, True)
                scale = 30

        elif family == 'pollution':
            img_raw, coll_size = pollution_index_image(roi, start_date, end_date, index_v)
            count = coll_size.getInfo()
            if count == 0: return {'error': 'No Sentinel-5P data for this period/region.'}
            img = img_raw.clip(roi)
            composite = None  # no true-colour composite for this data type — safe_thumb_url falls through to its generic single-band branch correctly
            scale = 1113

        else:
            return {'error': f'Unknown family: {family}'}

        # Adapt resolution to region size — this is the actual fix for
        # "huge regions always time out and fall back to simulated data".
        # bestEffort alone doesn't prevent a request from simply taking
        # too long; this keeps the pixel count (and therefore compute
        # time) roughly bounded regardless of how large the drawn area is.
        roi_area_m2 = roi.area(1).getInfo()
        effective_scale = compute_adaptive_scale(roi_area_m2, scale)

        stats = img.select('value').reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.minMax(), sharedInputs=True)
                                     .combine(ee.Reducer.stdDev(), sharedInputs=True),
            geometry=roi, scale=effective_scale, bestEffort=True, maxPixels=1e9, tileScale=4
        ).getInfo()

        mean = stats.get('value_mean')
        if mean is None:
            return {'error': 'No valid pixels inside the drawn region for this index.'}

        # Real GEE-rendered thumbnail of the actual computed layer,
        # clipped to the exact ROI — shown on the frontend map/panel
        # instead of a synthetic placeholder.
        thumb_url = safe_thumb_url(index_v, roi, img=img, composite=composite)

        result = {
            'mean': round(mean, 4),
            'min':  round(stats.get('value_min', mean), 4),
            'max':  round(stats.get('value_max', mean), 4),
            'std':  round(stats.get('value_stdDev', 0) or 0, 4),
            'images': count,
            'source': 'gee_live',
            'thumb_url': thumb_url
        }

        # Real ground-station validation for Pollution — a genuinely
        # different, independently-measured data point, not another
        # view of the same satellite estimate. Never blocks the main
        # reading if it fails or no station is nearby.
        if family == 'pollution':
            try:
                import ground_truth
                lat_c = sum(c[0] for c in coords) / len(coords)
                lng_c = sum(c[1] for c in coords) / len(coords)
                gt = ground_truth.get_openaq_ground_truth(lat_c, lng_c, index_v)
                if gt:
                    result['ground_truth'] = gt
            except Exception as e:
                logger.warning(f'Ground-truth lookup skipped (non-fatal): {e}')

        return result
    except Exception as e:
        logger.error(f'GEE reading error: {e}')
        return simulate(family, index_v, start_date, end_date, real_error=str(e))


# ══════════════════════════════════════════════════════════════
# ADVANCED ANALYTICS — Enterprise-only
# Real object counting and classification, not just index readings.
# ══════════════════════════════════════════════════════════════

def run_tree_count(start_date, end_date, coords):
    """
    Counts individual tree/palm crowns within a farm boundary using
    multi-pass local-maxima detection on a vegetation index — each
    isolated peak in greenness corresponds to one tree canopy centre.

    A single fixed search radius systematically misses smaller/younger
    trees near a larger one, since Earth Engine correctly identifies
    only the single tallest peak within that radius — smaller
    neighbours get suppressed, not missed by accident. This runs a
    second pass: detect large/mature trees first, mask out their own
    footprint, then re-detect on what's left — freeing up smaller
    trees that were previously being overshadowed by a bigger
    neighbour to become local maxima of their own right.
    """
    if not (EE_AVAILABLE and init_ee()):
        return _simulate_advanced('treeCount', coords)
    try:
        roi = roi_from_coords(coords)
        coll = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                .filterBounds(roi).filterDate(start_date, end_date)
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
                .sort('CLOUDY_PIXEL_PERCENTAGE').limit(30))
        count_imgs = coll.size().getInfo()
        if count_imgs == 0:
            return {'error': 'No Sentinel-2 images for this period/region.'}

        composite = coll.map(mask_s2).median().clip(roi)
        ndvi = composite.normalizedDifference(['B8', 'B4'])

        # Pass 1 — large/mature trees, wider search radius.
        kernel_large = ee.Kernel.circle(radius=1, units='pixels')  # ~10m
        large_local_max = ndvi.focalMax(kernel=kernel_large)
        large_peaks = ndvi.eq(large_local_max).And(ndvi.gt(0.45))

        # Mask out the footprint immediately around each large-tree
        # detection before searching for smaller trees, so a mature
        # tree's own crown can't get double-counted as several small
        # ones, and so a genuinely separate small tree right next to
        # it is no longer shadowed by the bigger tree's peak.
        exclusion = large_peaks.focalMax(kernel=ee.Kernel.circle(radius=1, units='pixels'))
        remaining_ndvi = ndvi.updateMask(exclusion.Not())

        # Pass 2 — small/young trees, on whatever's left. Slightly
        # relaxed vegetation threshold, since younger canopies read
        # less densely green than mature ones.
        small_local_max = remaining_ndvi.focalMax(kernel=ee.Kernel.circle(radius=1, units='pixels'))
        small_peaks = remaining_ndvi.eq(small_local_max).And(remaining_ndvi.gt(0.30)).unmask(0)

        peaks = large_peaks.Or(small_peaks).selfMask()
        # Keep the underlying NDVI value at each detected peak — this is
        # what the size/vigour categorisation below groups by.
        peak_ndvi = ndvi.updateMask(peaks)

        area_ha = ee.Number(roi.area(1)).divide(10000)
        tree_count = peaks.reduceRegion(
            reducer=ee.Reducer.count(), geometry=roi, scale=10,
            bestEffort=True, maxPixels=1e9, tileScale=4
        ).getInfo()
        n_trees = int(tree_count.get('nd', 0) or 0)
        area_ha_val = area_ha.getInfo()
        density = round(n_trees / area_ha_val, 1) if area_ha_val > 0 else 0

        # Honest categorisation by canopy size/vigour (from NDVI peak
        # intensity) — NOT species identification. True species-level
        # classification needs hyperspectral imagery or region-specific
        # labelled training samples, neither of which free 10m
        # Sentinel-2 data can provide. This groups detected trees into
        # Large/mature, Medium, and Small/young canopy categories,
        # which is a genuinely different and useful signal (e.g.
        # spotting a young replanting block vs. an established grove)
        # without overclaiming what the data can actually support.
        size_categories = {}
        try:
            for label, lo, hi in [('large_mature', 0.6, 2), ('medium', 0.45, 0.6), ('small_young', 0.0, 0.45)]:
                bucket_mask = peak_ndvi.gte(lo).And(peak_ndvi.lt(hi))
                bucket_count = bucket_mask.selfMask().reduceRegion(
                    reducer=ee.Reducer.count(), geometry=roi, scale=10,
                    bestEffort=True, maxPixels=1e9, tileScale=4
                ).getInfo()
                size_categories[label] = int(bucket_count.get('nd', 0) or 0)
        except Exception as e:
            logger.warning(f'Tree size categorisation failed (non-fatal): {e}')

        # Visual: true-colour base with detected tree centres highlighted
        dims = compute_optimal_dimensions(roi, 10)
        thumb_url = None
        try:
            highlight = composite.visualize(bands=['B4','B3','B2'], min=0, max=0.3, gamma=1.2)
            dots = peaks.visualize(palette=['ff3b30'], forceRgbOutput=True)
            combined = ee.ImageCollection([highlight, dots.updateMask(peaks)]).mosaic()
            thumb_url = combined.clip(roi).getThumbURL({'region': roi, 'dimensions': dims, 'format': 'png'})
        except Exception as e:
            logger.warning(f'Tree count thumbnail failed: {e}')

        return {
            'tool': 'treeCount', 'source': 'gee_live', 'images': count_imgs,
            'count': n_trees, 'area_ha': round(area_ha_val, 2), 'density_per_ha': density,
            'size_categories': size_categories,
            'thumb_url': thumb_url,
        }
    except Exception as e:
        logger.error(f'Tree count error: {e}')
        return _simulate_advanced('treeCount', coords, real_error=str(e))


def run_water_bodies(start_date, end_date, coords):
    """
    Counts and sizes distinct water bodies (lakes, reservoirs, ponds)
    within a region using connected-component labelling on an NDWI
    water mask — each isolated cluster of water pixels is one body.
    Suitable for regional/national-scale inventories, not just a
    single lake.
    """
    if not (EE_AVAILABLE and init_ee()):
        return _simulate_advanced('waterBodies', coords)
    try:
        roi = roi_from_coords(coords)
        coll = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                .filterBounds(roi).filterDate(start_date, end_date)
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
                .sort('CLOUDY_PIXEL_PERCENTAGE').limit(30))
        count_imgs = coll.size().getInfo()
        if count_imgs == 0:
            return {'error': 'No Sentinel-2 images for this period/region.'}

        composite = coll.map(mask_s2).median().clip(roi)
        water_mask = composite.normalizedDifference(['B3', 'B8']).gt(0.0).selfMask()

        # Adapt scale to region size — this is what actually keeps a
        # whole-country query from timing out, rather than the fixed 30m
        # that was still too fine (too many pixels) for very large areas.
        roi_area_m2 = roi.area(1).getInfo()
        effective_scale = compute_adaptive_scale(roi_area_m2, 30, target_pixels=3_000_000)

        # Label each connected group of water pixels as one distinct body.
        # maxSize capped lower + tileScale added — the same fix that
        # solved "User memory limit exceeded" on the Change Map feature;
        # connectedComponents is one of Earth Engine's heaviest operations.
        labeled = water_mask.connectedComponents(connectedness=ee.Kernel.plus(1), maxSize=256)

        n_bodies = labeled.select('labels').reduceRegion(
            reducer=ee.Reducer.countDistinct(), geometry=roi, scale=effective_scale,
            bestEffort=True, maxPixels=1e10, tileScale=8
        ).getInfo()
        body_count = int(n_bodies.get('labels', 0) or 0)

        area_stats = water_mask.multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=roi, scale=effective_scale, bestEffort=True, maxPixels=1e10, tileScale=4
        ).getInfo()
        total_water_ha = round((area_stats.get('nd', 0) or 0) / 10000, 2)

        dims = compute_optimal_dimensions(roi, 30)
        thumb_url = None
        try:
            vis = labeled.select('labels').randomVisualizer()
            thumb_url = vis.clip(roi).getThumbURL({'region': roi, 'dimensions': dims, 'format': 'png'})
        except Exception as e:
            logger.warning(f'Water body thumbnail failed: {e}')

        return {
            'tool': 'waterBodies', 'source': 'gee_live', 'images': count_imgs,
            'count': body_count, 'total_water_ha': total_water_ha,
            'thumb_url': thumb_url,
        }
    except Exception as e:
        logger.error(f'Water body count error: {e}')
        return _simulate_advanced('waterBodies', coords, real_error=str(e))


def run_land_classify(start_date, end_date, coords, n_clusters=5):
    """
    Unsupervised K-Means classification, followed by automatic class
    identification — each cluster's real spectral signature (mean
    NDVI, NDWI, NDBI, and brightness) is compared against standard
    remote-sensing thresholds for water, vegetation, built-up, sand,
    and bare soil, so results read as "Water" and "Vegetation / Trees"
    instead of generic "Class 1" / "Class 2". This is a well-
    established post-classification labelling technique, not a
    certified classification — it's a best-guess automatic label,
    worth a visual sanity-check against the true-colour image
    alongside it, same honesty standard as the rest of the platform.
    """
    if not (EE_AVAILABLE and init_ee()):
        return _simulate_advanced('landClassify', coords)
    try:
        roi = roi_from_coords(coords)
        coll = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                .filterBounds(roi).filterDate(start_date, end_date)
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
                .sort('CLOUDY_PIXEL_PERCENTAGE').limit(30))
        count_imgs = coll.size().getInfo()
        if count_imgs == 0:
            return {'error': 'No Sentinel-2 images for this period/region.'}

        composite = coll.map(mask_s2).median().clip(roi)
        bands = ['B2', 'B3', 'B4', 'B8', 'B11']
        training = composite.select(bands).sample(
            region=roi, scale=10, numPixels=5000, seed=42, geometries=False, tileScale=4
        )
        clusterer = ee.Clusterer.wekaKMeans(n_clusters).train(training)
        classified = composite.select(bands).cluster(clusterer)
        cluster_band = classified.rename('cluster')

        roi_area_m2 = roi.area(1).getInfo()
        effective_scale = compute_adaptive_scale(roi_area_m2, 10, target_pixels=3_000_000)

        area_img = ee.Image.pixelArea().addBands(cluster_band)
        grouped = area_img.reduceRegion(
            reducer=ee.Reducer.sum().group(groupField=1, groupName='cluster'),
            geometry=roi, scale=effective_scale, bestEffort=True, maxPixels=1e10, tileScale=4
        ).getInfo()
        groups = grouped.get('groups', [])
        total_area = sum(g['sum'] for g in groups) or 1
        area_by_cluster = {int(g['cluster']): g['sum'] for g in groups}

        # Real spectral signature per cluster — this is what actually
        # identifies what each cluster represents, rather than leaving
        # it as an arbitrary numbered label.
        ndvi = composite.normalizedDifference(['B8', 'B4'])
        ndwi = composite.normalizedDifference(['B3', 'B8'])
        ndbi = composite.normalizedDifference(['B11', 'B8'])
        brightness = composite.select(['B2', 'B3', 'B4']).reduce(ee.Reducer.mean())

        def grouped_mean(value_img):
            stacked = value_img.rename('val').addBands(cluster_band)
            result = stacked.reduceRegion(
                reducer=ee.Reducer.mean().group(groupField=1, groupName='cluster'),
                geometry=roi, scale=effective_scale, bestEffort=True, maxPixels=1e10, tileScale=4
            ).getInfo()
            return {int(g['cluster']): g['mean'] for g in result.get('groups', [])}

        ndvi_by_cluster = grouped_mean(ndvi)
        ndwi_by_cluster = grouped_mean(ndwi)
        ndbi_by_cluster = grouped_mean(ndbi)
        bright_by_cluster = grouped_mean(brightness)

        def identify_class(cid):
            v_ndvi = ndvi_by_cluster.get(cid, 0) or 0
            v_ndwi = ndwi_by_cluster.get(cid, 0) or 0
            v_ndbi = ndbi_by_cluster.get(cid, 0) or 0
            v_bright = bright_by_cluster.get(cid, 0) or 0
            if v_ndwi > 0.1:
                return 'Water', '3182bd'
            if v_ndvi > 0.35:
                return 'Vegetation / Trees', '31a354'
            if v_bright > 0.28 and v_ndvi < 0.15:
                return 'Sand / Bare desert', 'e6d8ad'
            if v_ndbi > 0.0 and v_bright > 0.15:
                return 'Buildings / Urban', '969696'
            return 'Bare land / Soil', 'a87c4f'

        classes = []
        for cid, area_m2 in area_by_cluster.items():
            label, color = identify_class(cid)
            classes.append({
                'id': cid, 'label': label, 'color': color,
                'area_ha': round(area_m2 / 10000, 2),
                'pct': round(area_m2 / total_area * 100, 1),
            })
        classes.sort(key=lambda c: -c['area_ha'])

        # Visualisation uses each cluster's IDENTIFIED colour, so the
        # rendered map and the legend/table actually agree with each
        # other — instead of an arbitrary rainbow palette by cluster
        # index that has no relationship to what's on the ground.
        dims = compute_optimal_dimensions(roi, 10)
        thumb_url = None
        try:
            palette_by_id = {c['id']: c['color'] for c in classes}
            ordered_palette = [palette_by_id.get(i, 'cccccc') for i in range(n_clusters)]
            vis = classified.visualize(min=0, max=n_clusters - 1, palette=ordered_palette)
            thumb_url = vis.clip(roi).getThumbURL({'region': roi, 'dimensions': dims, 'format': 'png'})
        except Exception as e:
            logger.warning(f'Land classify thumbnail failed: {e}')

        return {
            'tool': 'landClassify', 'source': 'gee_live', 'images': count_imgs,
            'num_classes': len(classes), 'classes': classes,
            'thumb_url': thumb_url,
        }
    except Exception as e:
        logger.error(f'Land classification error: {e}')
        return _simulate_advanced('landClassify', coords, real_error=str(e))


def _simulate_advanced(tool, coords, real_error=None):
    """Fallback for the 3 advanced tools when GEE is unavailable."""
    rng = random.Random(_seed('adv', tool, str(len(coords))))
    note = (f'GEE call failed, showing simulated data. Real error: {real_error}'
            if real_error else 'GEE credentials not configured on the server — showing simulated data.')
    if tool == 'treeCount':
        count = rng.randint(80, 900)
        return {'tool':'treeCount','source':'simulated','images':rng.randint(10,40),
                'count':count, 'area_ha':round(count/rng.uniform(80,150),2),
                'density_per_ha':round(rng.uniform(80,150),1), 'note':note}
    if tool == 'waterBodies':
        count = rng.randint(5, 400)
        return {'tool':'waterBodies','source':'simulated','images':rng.randint(10,40),
                'count':count, 'total_water_ha':round(count*rng.uniform(2,40),2), 'note':note}
    demo_classes = [
        ('Vegetation / Trees', '31a354'), ('Water', '3182bd'), ('Buildings / Urban', '969696'),
        ('Bare land / Soil', 'a87c4f'), ('Sand / Bare desert', 'e6d8ad'),
    ]
    classes = [{'id':i, 'label':lbl, 'color':clr, 'area_ha':round(rng.uniform(50,2000),2), 'pct':0}
               for i,(lbl,clr) in enumerate(demo_classes)]
    total = sum(c['area_ha'] for c in classes) or 1
    for c in classes: c['pct'] = round(c['area_ha']/total*100,1)
    return {'tool':'landClassify','source':'simulated','images':rng.randint(10,40),
            'num_classes':5, 'classes':classes, 'note':note}


def run_timeseries(family, index_v, start_date, end_date, coords, max_points=None):
    # Elevation/slope are static terrain data, not a time series — a
    # monthly trend chart doesn't apply to them the way it does to an
    # actual satellite index. Fail clearly rather than returning a
    # meaningless flat line.
    if family == 'archaeology' and index_v in ('elevation', 'slope'):
        return {'error': 'Elevation and slope are static terrain data — trend charts apply to time-varying indices like vegetation anomaly or SAR instead.'}

    # Heat and Pollution use entirely different collections (Landsat
    # thermal/NOAA SST, Sentinel-5P) that the generic dispatch below
    # doesn't know how to build — better to say so plainly than to
    # silently fall through to the Sentinel-2 default and return
    # wrong data under a real-looking chart.
    if family in ('heat', 'pollution', 'fire'):
        return {'error': f'Monthly trend charts for {family.title()} aren\'t built yet — single readings already work for this family.'}

    if not (EE_AVAILABLE and init_ee()):
        return simulate_timeseries(index_v, start_date, end_date)
    try:
        roi = roi_from_coords(coords)

        # Urban and archaeology families span multiple different
        # collections depending on which index was picked. Landsat-based
        # families (landsat, geo, urban/ubndbi) now use the merged
        # multi-mission archive so a trend chart can span decades, not
        # just since Landsat 9 launched in 2021.
        is_multi_landsat = (family in ('landsat', 'geo')) or (family == 'urban' and index_v == 'ubndbi')

        if is_multi_landsat:
            coll = get_landsat_collection(roi, start_date, end_date)
        elif family == 'urban':
            if index_v == 'nightlights':
                coll_id = 'NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG'
            else:  # s2rgb
                coll_id = 'COPERNICUS/S2_SR_HARMONIZED'
            coll = ee.ImageCollection(coll_id).filterBounds(roi).filterDate(start_date, end_date)
            if 'S2' in coll_id:
                coll = coll.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
        elif family == 'archaeology':
            coll_id = 'COPERNICUS/S1_GRD' if index_v == 'sar' else 'COPERNICUS/S2_SR_HARMONIZED'
            coll = ee.ImageCollection(coll_id).filterBounds(roi).filterDate(start_date, end_date)
            if coll_id == 'COPERNICUS/S1_GRD':
                coll = (coll.filter(ee.Filter.eq('instrumentMode', 'IW'))
                            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')))
            else:
                coll = coll.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
        else:  # water, veg — Sentinel-2 only
            coll = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                    .filterBounds(roi).filterDate(start_date, end_date)
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40)))

        start = ee.Date(start_date); end = ee.Date(end_date)
        n_months = end.difference(start, 'month').round()
        n_months_val = int(n_months.getInfo())
        if n_months_val < 1:
            n_months_val = 1

        # max_points gives a deliberately lighter-weight trend — evenly
        # spaced samples across the full range instead of every single
        # month. Used when a full per-month series for many indices at
        # once (e.g. Auto-Analyze running an entire family) would be
        # prohibitively slow; a normal single "Plot Trend" click still
        # gets the full monthly resolution.
        if max_points and n_months_val > max_points:
            step = (n_months_val - 1) / (max_points - 1) if max_points > 1 else 0
            month_indices = sorted(set(round(i * step) for i in range(max_points)))
        else:
            month_indices = list(range(n_months_val))

        def month_val(n):
            n = ee.Number(n)
            m0 = start.advance(n, 'month'); m1 = m0.advance(1, 'month')
            imgs = coll.filterDate(m0, m1)

            if family == 'urban' and index_v == 'nightlights':
                idx = imgs.select('avg_rad').median().max(0).rename('value')
            elif family == 'archaeology' and index_v == 'sar':
                idx = imgs.select('VV').median().rename('value')
            elif family == 'archaeology' and index_v == 'vegAnomaly':
                composite = imgs.map(mask_s2).median()
                ndvi = composite.normalizedDifference(['B8','B4'])
                idx = ndvi.subtract(ndvi.focalMean(radius=30, units='meters')).rename('value')
            elif is_multi_landsat:
                # Already cloud-masked, scaled, and band-remapped by
                # get_landsat_collection() — no further masking needed here.
                composite = imgs.median()
                if family == 'landsat':
                    idx = landsat_combo_image(composite, index_v)
                elif family == 'geo':
                    idx = geology_image(composite, index_v)
                else:  # urban / ubndbi
                    idx = composite.normalizedDifference(['swir1', 'nir']).rename('value')
            else:
                masked = imgs.map(mask_s2)
                composite = masked.median()
                if family == 'water':
                    idx = water_index_image(composite, index_v)
                elif family == 'veg':
                    idx = vegetation_image(composite, index_v)
                else:  # urban / s2rgb
                    idx = composite.normalizedDifference(['B8','B4']).rename('value')

            scale = 500 if (family=='urban' and index_v=='nightlights') else 100
            val = idx.select('value').reduceRegion(ee.Reducer.mean(), roi, scale, bestEffort=True).get('value')
            return ee.Feature(None, {'m': m0.format('YYYY-MM'), 'v': val})

        fc = ee.FeatureCollection(ee.List(month_indices).map(month_val))
        info = fc.getInfo()
        labels, values = [], []
        for f in info['features']:
            v = f['properties'].get('v')
            if v is not None:
                labels.append(f['properties']['m']); values.append(round(v, 4))
        return {'labels': labels, 'values': values, 'source': 'gee_live'}
    except Exception as e:
        logger.error(f'GEE timeseries error: {e}')
        return simulate_timeseries(index_v, start_date, end_date)


# ── simulation fallback (mirrors app.js simulateReading) ────────

def _seed(*parts):
    h = hashlib.md5('|'.join(parts).encode()).hexdigest()
    return int(h[:8], 16)

def simulate(family, index_v, start_date, end_date, real_error=None):
    rng = random.Random(_seed(family, index_v, start_date, end_date))
    lo, hi = RANGES.get(index_v, (0, 1))
    span = hi - lo
    mean = lo + span * (0.35 + rng.random()*0.3)
    std  = span * 0.08 * (0.6 + rng.random()*0.8)
    # Surface the REAL underlying error when we have one, instead of a
    # generic message — this is what makes the actual cause visible in
    # the browser's Network tab without needing to dig through server logs.
    note = (f'GEE call failed, showing simulated data. Real error: {real_error}'
            if real_error else
            'GEE credentials not configured on the server — showing simulated data.')
    return {
        'mean': round(mean,4), 'std': round(std,4),
        'min': round(max(lo, mean-std*2.2),4), 'max': round(min(hi, mean+std*2.2),4),
        'images': 18 + rng.randint(0,55), 'source': 'simulated',
        'note': note
    }

def simulate_timeseries(index_v, start_date, end_date):
    import datetime
    rng = random.Random(_seed('ts', index_v, start_date, end_date))
    lo, hi = RANGES.get(index_v, (0,1))
    d0 = datetime.date.fromisoformat(start_date); d1 = datetime.date.fromisoformat(end_date)
    labels, values = [], []
    base = lo + (hi-lo)*0.45; drift = 0
    d = d0
    while d <= d1 and len(labels) < 36:
        drift += (rng.random()-0.5)*(hi-lo)*0.04
        values.append(round(base+drift+(rng.random()-0.5)*(hi-lo)*0.1, 4))
        labels.append(d.strftime('%b %y'))
        month = d.month + 1; year = d.year + (month>12); month = month if month<=12 else 1
        d = d.replace(year=year, month=month, day=1)
    return {'labels': labels, 'values': values, 'source': 'simulated'}
