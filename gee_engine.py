"""
Earth Observation and Analysis — Earth Engine computation layer
======================================================
Python port of the index math from the GEE JavaScript app
(Al_Razaza_WaterQuality_FINAL.js). Falls back to a seeded
simulation if the `earthengine-api` package or credentials
are not available, so the API always returns something.
"""
import os, hashlib, random, logging, math

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
}

# True/false-colour Landsat combos have no single "value" band — these show
# the actual composite bands directly instead of a palette ramp.
RGB_VIS = {
    'natural':  {'bands':['SR_B4','SR_B3','SR_B2'], 'min':0, 'max':0.3, 'gamma':1.2},
    'falseveg': {'bands':['SR_B5','SR_B4','SR_B3'], 'min':0, 'max':0.4},
    'urban':    {'bands':['SR_B7','SR_B6','SR_B4'], 'min':0, 'max':0.4},
    'agri':     {'bands':['SR_B6','SR_B5','SR_B2'], 'min':0, 'max':0.4},
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
    try:
        if index_v == 'elevation' and composite is not None:
            # Raised-relief hillshade — reveals subtle mounds/depressions
            # that raw elevation colouring would hide, since absolute
            # elevation varies wildly by region but relief shading is
            # self-normalising (always renders 0-255 regardless of the
            # actual metres involved).
            hillshade = ee.Terrain.hillshade(composite, 315, 45)  # standard NW light, 45° sun angle
            return hillshade.clip(roi).getThumbURL({
                'region': roi, 'dimensions': dims, 'format': 'png', 'min': 0, 'max': 255
            })
        if index_v in RGB_VIS and composite is not None:
            vis = RGB_VIS[index_v]
            return composite.clip(roi).getThumbURL({
                'region': roi, 'dimensions': dims, 'format': 'png', **vis
            })
        if img is not None:
            lo, hi = RANGES.get(index_v, (0, 1))
            palette = PALETTES.get(index_v, ['0a3040', '0e5468', '00c2d1'])
            return img.select('value').clip(roi).getThumbURL({
                'region': roi, 'dimensions': dims, 'format': 'png',
                'min': lo, 'max': hi, 'palette': palette
            })
    except Exception as e:
        logger.warning(f'Thumbnail generation failed for {index_v}: {e}')
    return None



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


def landsat_combo_image(l9, index_v):
    if index_v == 'ndvi':
        return l9.normalizedDifference(['SR_B5', 'SR_B4']).rename('value')
    if index_v == 'mndwi':
        return l9.normalizedDifference(['SR_B3', 'SR_B6']).rename('value')
    if index_v == 'ndbi':
        return l9.normalizedDifference(['SR_B6', 'SR_B5']).rename('value')
    # RGB-style combos have no single "value" band — return NDVI as the stat proxy
    return l9.normalizedDifference(['SR_B5', 'SR_B4']).rename('value')


def geology_image(l9, index_v):
    ndwi = l9.normalizedDifference(['SR_B3', 'SR_B5'])
    land_mask = ndwi.lt(0.0)
    B2,B3,B4,B5,B6,B7 = [l9.select(b) for b in ['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7']]
    if index_v == 'iron':
        img = B4.divide(B2.max(0.0001))
    elif index_v == 'clay':
        img = B6.divide(B7.max(0.0001))
    elif index_v == 'carbonate':
        img = B6.divide(B7.add(B4).max(0.0001))
    elif index_v == 'evaporite':
        img = B2.add(B3).add(B4).divide(3).multiply(B6.divide(B7.max(0.0001)))
    else:  # allmin
        iron, clay, carb, sil = B4.divide(B2.max(0.0001)), B6.divide(B7.max(0.0001)), \
                                 B6.divide(B7.add(B4).max(0.0001)), B5.divide(B6.max(0.0001))
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
            coll = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                    .filterBounds(roi).filterDate(start_date, end_date)
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40)))
            count = coll.size().getInfo()
            if count == 0: return {'error': 'No Sentinel-2 images for this period/region.'}
            composite = coll.map(mask_s2).median().clip(roi)
            img = water_index_image(composite, index_v)
            scale = 30

        elif family == 'landsat':
            coll = (ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
                    .filterBounds(roi).filterDate(start_date, end_date)
                    .filter(ee.Filter.lt('CLOUD_COVER', 30)))
            count = coll.size().getInfo()
            if count == 0: return {'error': 'No Landsat 9 images for this period/region (launched Sep 2021).'}
            composite = coll.map(mask_l9).median().clip(roi)
            img = landsat_combo_image(composite, index_v)
            scale = 30

        elif family == 'geo':
            coll = (ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
                    .filterBounds(roi).filterDate(start_date, end_date)
                    .filter(ee.Filter.lt('CLOUD_COVER', 30)))
            count = coll.size().getInfo()
            if count == 0: return {'error': 'No Landsat 9 images for this period/region.'}
            composite = coll.map(mask_l9).median().clip(roi)
            img = geology_image(composite, index_v)
            scale = 30

        elif family == 'veg':
            coll = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                    .filterBounds(roi).filterDate(start_date, end_date)
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40)))
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
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40)))
                count = coll.size().getInfo()
                if count == 0: return {'error': 'No Sentinel-2 images for this period/region.'}
                composite = coll.map(mask_s2).median().clip(roi)
                img = composite.normalizedDifference(['B8','B4']).rename('value')  # NDVI as stat proxy only
                scale = 10

            else:  # 'ubndbi' — built-up index, same math as Landsat NDBI
                coll = (ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
                        .filterBounds(roi).filterDate(start_date, end_date)
                        .filter(ee.Filter.lt('CLOUD_COVER', 30)))
                count = coll.size().getInfo()
                if count == 0: return {'error': 'No Landsat 9 images for this period/region (launched Sep 2021).'}
                composite = coll.map(mask_l9).median().clip(roi)
                img = composite.normalizedDifference(['SR_B6','SR_B5']).rename('value')
                scale = 30

        elif family == 'archaeology':
            if index_v == 'elevation':
                # Copernicus GLO-30 — free global 30m DEM. Static dataset,
                # not date-dependent, so "count" is a placeholder (there's
                # no revisit/cloud filtering concept for elevation data).
                dem = ee.Image('COPERNICUS/DEM/GLO30').select('DEM').clip(roi)
                composite = dem  # used by safe_thumb_url for hillshade rendering
                img = dem.rename('value')  # stats report real elevation in metres
                count = 1
                scale = 30

            elif index_v == 'slope':
                dem = ee.Image('COPERNICUS/DEM/GLO30').select('DEM').clip(roi)
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
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40)))
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
        else:
            return {'error': f'Unknown family: {family}'}

        stats = img.select('value').reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.minMax(), sharedInputs=True)
                                     .combine(ee.Reducer.stdDev(), sharedInputs=True),
            geometry=roi, scale=scale, bestEffort=True, maxPixels=1e9, tileScale=4
        ).getInfo()

        mean = stats.get('value_mean')
        if mean is None:
            return {'error': 'No valid pixels inside the drawn region for this index.'}

        # Real GEE-rendered thumbnail of the actual computed layer,
        # clipped to the exact ROI — shown on the frontend map/panel
        # instead of a synthetic placeholder.
        thumb_url = safe_thumb_url(index_v, roi, img=img, composite=composite)

        return {
            'mean': round(mean, 4),
            'min':  round(stats.get('value_min', mean), 4),
            'max':  round(stats.get('value_max', mean), 4),
            'std':  round(stats.get('value_stdDev', 0) or 0, 4),
            'images': count,
            'source': 'gee_live',
            'thumb_url': thumb_url
        }
    except Exception as e:
        logger.error(f'GEE reading error: {e}')
        return simulate(family, index_v, start_date, end_date, real_error=str(e))


def run_timeseries(family, index_v, start_date, end_date, coords):
    # Elevation/slope are static terrain data, not a time series — a
    # monthly trend chart doesn't apply to them the way it does to an
    # actual satellite index. Fail clearly rather than returning a
    # meaningless flat line.
    if family == 'archaeology' and index_v in ('elevation', 'slope'):
        return {'error': 'Elevation and slope are static terrain data — trend charts apply to time-varying indices like vegetation anomaly or SAR instead.'}

    if not (EE_AVAILABLE and init_ee()):
        return simulate_timeseries(index_v, start_date, end_date)
    try:
        roi = roi_from_coords(coords)

        # Urban and archaeology families span multiple different
        # collections depending on which index was picked.
        if family == 'urban':
            if index_v == 'nightlights':
                coll_id = 'NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG'
            elif index_v == 's2rgb':
                coll_id = 'COPERNICUS/S2_SR_HARMONIZED'
            else:
                coll_id = 'LANDSAT/LC09/C02/T1_L2'
        elif family == 'archaeology':
            coll_id = 'COPERNICUS/S1_GRD' if index_v == 'sar' else 'COPERNICUS/S2_SR_HARMONIZED'
        else:
            coll_id = 'COPERNICUS/S2_SR_HARMONIZED' if family in ('water','veg') else 'LANDSAT/LC09/C02/T1_L2'

        coll = ee.ImageCollection(coll_id).filterBounds(roi).filterDate(start_date, end_date)
        if coll_id == 'COPERNICUS/S1_GRD':
            coll = (coll.filter(ee.Filter.eq('instrumentMode', 'IW'))
                        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')))
        elif 'S2' in coll_id:
            coll = coll.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
        elif 'LANDSAT' in coll_id:
            coll = coll.filter(ee.Filter.lt('CLOUD_COVER', 30))
        # VIIRS needs no cloud filter — it's a pre-composited monthly product

        start = ee.Date(start_date); end = ee.Date(end_date)
        n_months = end.difference(start, 'month').round()

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
            else:
                masked = imgs.map(mask_s2 if 'S2' in coll_id else mask_l9)
                composite = masked.median()
                if family == 'water':
                    idx = water_index_image(composite, index_v)
                elif family == 'landsat':
                    idx = landsat_combo_image(composite, index_v)
                elif family == 'geo':
                    idx = geology_image(composite, index_v)
                elif family == 'veg':
                    idx = vegetation_image(composite, index_v)
                elif family == 'urban' and index_v == 's2rgb':
                    idx = composite.normalizedDifference(['B8','B4']).rename('value')
                else:  # urban / ubndbi
                    idx = composite.normalizedDifference(['SR_B6','SR_B5']).rename('value')

            scale = 500 if (family=='urban' and index_v=='nightlights') else 100
            val = idx.select('value').reduceRegion(ee.Reducer.mean(), roi, scale, bestEffort=True).get('value')
            return ee.Feature(None, {'m': m0.format('YYYY-MM'), 'v': val})

        fc = ee.FeatureCollection(ee.List.sequence(0, n_months.subtract(1)).map(month_val))
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
