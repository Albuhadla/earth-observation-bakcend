"""
Earth Observation and Analysis — Watershed Feature TEST module
==================================================================
A standalone, isolated test for the "watershed analysis" capabilities
proposed after reviewing a reference dashboard (GeoReplica.ai style).

This file is deliberately NOT wired into the main /api/analysis/run
pipeline — it has its own route so we can verify each real Earth
Engine dataset actually returns valid data for a real region BEFORE
investing time building the full family/UI around it.

Tests, one at a time, in order of how likely they are to need
adjustment (dataset IDs in Earth Engine's catalog do shift over time,
same lesson as the Copernicus DEM mosaic bug earlier):

  1. Watershed/basin boundary  — HydroSHEDS
  2. Stream/river network      — HydroSHEDS
  3. Rainfall trend            — CHIRPS (already know this one works —
                                  same dataset family used elsewhere)
  4. Derived flood-risk zone   — our own DEM + water-proximity logic
  5. Population at risk        — WorldPop
"""
import logging
from gee_engine import EE_AVAILABLE, init_ee, roi_from_coords, mask_s2

logger = logging.getLogger(__name__)

try:
    import ee
except Exception:
    pass


def test_watershed_boundary(roi):
    """Test 1: can we retrieve a real watershed/basin boundary for this region?"""
    try:
        # HydroSHEDS hierarchical basins — level 7 gives sub-basin-scale
        # polygons, a reasonable middle ground between a whole continent
        # and a tiny catchment.
        basins = ee.FeatureCollection('WWF/HydroSHEDS/v1/Basins/hybas_7')
        intersecting = basins.filterBounds(roi)
        count = intersecting.size().getInfo()
        if count == 0:
            return {'ok': False, 'note': 'No HydroSHEDS basin found intersecting this region.'}
        first = intersecting.first().getInfo()
        return {
            'ok': True,
            'basins_found': count,
            'sample_basin_properties': first.get('properties', {}),
        }
    except Exception as e:
        return {'ok': False, 'error': str(e)}


def test_stream_network(roi):
    """Test 2: can we retrieve real river/stream geometry for this region?"""
    try:
        rivers = ee.FeatureCollection('WWF/HydroSHEDS/v1/FreeFlowingRivers')
        intersecting = rivers.filterBounds(roi)
        count = intersecting.size().getInfo()
        return {'ok': count > 0, 'stream_segments_found': count}
    except Exception as e:
        return {'ok': False, 'error': str(e)}


def test_rainfall_trend(roi, start_date, end_date):
    """Test 3: CHIRPS precipitation — the same dataset family we already
    use elsewhere in the platform, so this one should be the most reliable."""
    try:
        coll = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').filterBounds(roi).filterDate(start_date, end_date)
        count = coll.size().getInfo()
        if count == 0:
            return {'ok': False, 'note': 'No CHIRPS rainfall data for this period/region.'}
        total = coll.sum().reduceRegion(
            reducer=ee.Reducer.mean(), geometry=roi, scale=5000, bestEffort=True, maxPixels=1e9
        ).getInfo()
        return {'ok': True, 'days_of_data': count, 'total_rainfall_mm': round(total.get('precipitation', 0) or 0, 1)}
    except Exception as e:
        return {'ok': False, 'error': str(e)}


def test_flood_risk_zones(roi):
    """Test 4: our own derived risk layer — low elevation + close to water,
    using the same Copernicus DEM we already use for Archaeology & Terrain."""
    try:
        dem = ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM').mosaic().clip(roi)
        water = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').gt(50).selfMask()
        distance_to_water = water.fastDistanceTransform(30).sqrt().multiply(30)  # metres to nearest water pixel

        stats = dem.reduceRegion(reducer=ee.Reducer.percentile([10, 50]), geometry=roi, scale=90, bestEffort=True, maxPixels=1e9).getInfo()
        low_elev_threshold = stats.get('DEM_p10')  # bottom 10% elevation = "low-lying"

        risk = dem.lt(low_elev_threshold if low_elev_threshold is not None else 9999).And(distance_to_water.lt(1000))
        risk_area = risk.selfMask().multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=roi, scale=90, bestEffort=True, maxPixels=1e9
        ).getInfo()
        risk_ha = round((risk_area.get('DEM', 0) or 0) / 10000, 2)
        return {'ok': True, 'estimated_risk_area_ha': risk_ha, 'low_elevation_threshold_m': low_elev_threshold}
    except Exception as e:
        return {'ok': False, 'error': str(e)}


def test_population_at_risk(roi):
    """Test 5: WorldPop population density, to see if we could estimate
    people living inside a derived risk zone."""
    try:
        pop = ee.ImageCollection('WorldPop/GP/100m/pop').filterBounds(roi).filterDate('2020-01-01', '2021-01-01').mosaic()
        total_pop = pop.reduceRegion(
            reducer=ee.Reducer.sum(), geometry=roi, scale=100, bestEffort=True, maxPixels=1e9
        ).getInfo()
        return {'ok': True, 'total_population_in_roi': round(total_pop.get('population', 0) or 0)}
    except Exception as e:
        return {'ok': False, 'error': str(e)}


def run_all_watershed_tests(start_date, end_date, coords):
    """Runs all 5 tests against one real region and returns a combined
    report — this is what the test route calls."""
    if not (EE_AVAILABLE and init_ee()):
        return {'error': 'GEE not available on this server right now — check /api/health first.'}

    roi = roi_from_coords(coords)

    results = {
        '1_watershed_boundary': test_watershed_boundary(roi),
        '2_stream_network':     test_stream_network(roi),
        '3_rainfall_trend':     test_rainfall_trend(roi, start_date, end_date),
        '4_flood_risk_zones':   test_flood_risk_zones(roi),
        '5_population_at_risk': test_population_at_risk(roi),
    }
    passed = sum(1 for r in results.values() if r.get('ok'))
    results['summary'] = f'{passed} of 5 capabilities returned valid real data.'
    return results
