"""
Earth Observation and Analysis — AI Synthesis Engine
==================================================================
Two things, both using Claude:
  1. A detailed, plain-language description of what's happening in
     EACH individual calculation the user has run this session.
  2. An overall synthesis connecting multiple calculations together
     into one holistic narrative — e.g. "turbidity + chlorophyll +
     vegetation anomaly together suggest early-stage eutrophication",
     something the fixed template system genuinely cannot do since it
     only ever explains one index at a time.

Hard guardrail, stated explicitly in the system prompt and re-checked
in code: the model may ONLY reference numbers, dates, and locations
actually present in the structured data it's given. It must not invent
causes, historical events, or figures not present in the input — this
matches the platform's whole "honest about confidence" positioning
established on the Methodology page. An index's HIGH CONFIDENCE vs
PROXY MEASUREMENT classification is passed in explicitly so the model
uses appropriately careful language for each.

Deliberately isolated from the main analysis pipeline (same pattern as
cache.py, watershed_test.py) — a failure here never breaks a normal
reading.
"""
import os
import json
import logging
import base64
import requests

logger = logging.getLogger(__name__)

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except Exception:
    ANTHROPIC_AVAILABLE = False

ANTHROPIC_MODEL = 'claude-sonnet-5'

HIGH_CONFIDENCE_INDICES = {'ndvi', 'NDWI', 'mndwi', 'nightlights', 'ndbi', 'ubndbi'}

SYSTEM_PROMPT = """You are a remote sensing analyst writing sections of a professional \
Earth observation report. You are given structured JSON data describing one or more \
satellite-derived readings (index values, statistics, dates, location, and — when \
available — change-over-time and trend data) for a single region.

STRICT RULES — do not break these under any circumstance:
1. Only reference numbers, dates, locations, and facts that are literally present in \
the JSON you are given. Never invent a cause, historical event, external factor, or \
figure that isn't in the data.
2. Every index is labelled "confidence": "high" or "confidence": "proxy" in the input. \
For "high" confidence indices, you may state relative differences plainly. For "proxy" \
indices, you must use hedged, correlational language (e.g. "consistent with", \
"suggests", "a proxy signal for") rather than definitive claims, and should note that \
field verification would be needed to confirm.
3. If asked to explain WHY a value changed and the data doesn't contain a stated cause, \
say plainly that the data shows the pattern but not a confirmed cause, and suggest what \
kind of field investigation would help — never guess at a specific cause.
4. Write in plain, professional English. No bullet-point-only answers — write real \
prose paragraphs, the way a human analyst would write a report section.
5. Your ENTIRE reply must be a single valid JSON object and nothing else. The very \
first character of your reply must be "{" and the very last character must be "}". \
Do not write any sentence, greeting, acknowledgement, or explanation before or after \
the JSON. Do not wrap it in markdown code fences (no ```). If you find yourself about \
to write anything other than the JSON object itself, stop and output only the JSON.
"""


def _classify_confidence(index_v):
    return 'high' if index_v in HIGH_CONFIDENCE_INDICES else 'proxy'


def _build_payload(readings, location, change_map, trend, water_level=None):
    """Strips each reading down to only the fields the model should see —
    deliberately not including internal IDs, raw thumbnail URLs, etc."""
    def _reading_dict(r):
        d = {
            'index': r.get('title') or r.get('indexV'),
            'family': r.get('fam'),
            'confidence': _classify_confidence(r.get('indexV')),
            'mean': r.get('mean'), 'min': r.get('min'), 'max': r.get('max'), 'std': r.get('std'),
            'date_range': f"{r.get('start')} to {r.get('end')}",
            'images_used': r.get('images'),
        }
        gt = r.get('groundTruth')
        if gt:
            d['real_ground_station_reading'] = {
                'note': 'A genuine measured reading from a real ground station, not a satellite estimate — treat this as the more trustworthy of the two values where they can be directly compared.',
                'station_name': gt.get('station_name'),
                'parameter': gt.get('parameter'), 'value': gt.get('value'), 'unit': gt.get('unit'),
                'is_direct_match': gt.get('is_direct_match'),
            }
        return d

    payload = {
        'location': location or None,
        'readings': [_reading_dict(r) for r in readings],
    }
    if change_map:
        payload['change_analysis'] = {
            'index': change_map.get('indexV'),
            'period_1': f"{change_map.get('start1')} to {change_map.get('end1')}",
            'period_2': f"{change_map.get('start2')} to {change_map.get('end2')}",
            'overall_change_pct': change_map.get('overall_change_pct'),
            'increased_ha': change_map.get('increased_ha'), 'stable_ha': change_map.get('stable_ha'),
            'decreased_ha': change_map.get('decreased_ha'),
            'hotspots': change_map.get('hotspots'),
        }
    if trend:
        payload['trend'] = {
            'index': trend.get('indexV'),
            'labels': trend.get('labels'),
            'values': trend.get('data'),
            'stated_slope': trend.get('slopeText'),
        }
    if water_level:
        payload['water_level_estimate'] = {
            'confidence': 'proxy',
            'method': 'Estimated from Copernicus DEM elevation sampled at the water body\'s shoreline (boundary of the detected water extent) for each period — not direct altimetry.',
            'period_1': water_level.get('period1'),
            'period_2': water_level.get('period2'),
            'level_1_m': water_level.get('level1_m'),
            'level_2_m': water_level.get('level2_m'),
            'change_m': water_level.get('change_m'),
        }
        usgs = water_level.get('usgs_ground_truth')
        if usgs:
            payload['water_level_estimate']['real_usgs_gauge_reading'] = {
                'note': 'A genuine measured USGS river/lake gauge reading near this region — the actual current water level in feet, not an estimate. Mention this explicitly as real validating evidence.',
                'site_name': usgs.get('site_name'),
                'gage_height_ft': usgs.get('gage_height_ft'),
            }
    return payload


def generate_ai_report(readings, location, change_map, trend, water_level=None, language='en'):
    """
    Returns {'per_calculation': [{'index':..., 'description':...}, ...],
    'synthesis': '...'} or {'error': '...'} on failure. Never raises —
    always returns a dict so the route can respond cleanly either way.
    language: 'en' (default) or 'ar' — writes the actual descriptions
    and synthesis in Arabic when requested, not a translated-afterward
    version, so the phrasing reads naturally rather than machine-translated.
    """
    if not ANTHROPIC_AVAILABLE:
        return {'error': 'AI module not installed on the server.'}

    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        return {'error': 'AI feature not configured on the server (missing API key).'}

    if not readings and not water_level:
        return {'error': 'No calculations to describe — take at least one reading first.'}

    # Cap the number of readings actually sent — an accumulated session
    # history (multiple Auto-Analyze runs across different families
    # without clearing) can otherwise grow large, and every single
    # reading needs its own written description, which scales up both
    # response time and the real risk of hitting the request timeout.
    # Most recent readings are kept, since they're the most likely to
    # be what the person actually wants analyzed right now.
    MAX_READINGS_FOR_AI = 20
    if len(readings) > MAX_READINGS_FOR_AI:
        readings = readings[-MAX_READINGS_FOR_AI:]

    try:
        payload = _build_payload(readings, location, change_map, trend, water_level)

        language_instruction = (
            "\n\nWrite your entire response — every description and the synthesis — in "
            "Modern Standard Arabic. Keep index names, technical abbreviations (e.g. NDVI, "
            "NDTI, NDBI), units, and numeric values exactly as given, un-translated, since "
            "these are standard scientific notation even in Arabic-language reports."
            if language == 'ar' else ""
        )

        user_prompt = f"""Here is the structured data for this session:

{json.dumps(payload, indent=2)}

Call the generate_report tool with your analysis.{language_instruction}"""

        # Tool use (structured output) instead of asking Claude to format
        # JSON in a plain text reply — this is the actual, reliable fix.
        # Text-based JSON formatting depends on the model following
        # prompt instructions exactly every single time, which is what
        # was failing intermittently. Tool use enforces the schema at
        # the API level, so the response is guaranteed to already be a
        # correctly-shaped dict — no text parsing, no preamble risk,
        # no markdown-fence guessing.
        report_tool = {
            "name": "generate_report",
            "description": "Submit the per-calculation descriptions and holistic synthesis for this Earth observation report.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "per_calculation": {
                        "type": "array",
                        "description": "One entry per reading in the input data.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "index": {"type": "string", "description": "The index name, exactly as given in the input."},
                                "description": {"type": "string", "description": "2-4 sentence plain-language paragraph describing what this specific reading shows."}
                            },
                            "required": ["index", "description"]
                        }
                    },
                    "synthesis": {
                        "type": "string",
                        "description": "2-4 paragraph holistic narrative connecting the readings (or a deeper single paragraph if there's only one reading). If water_level_estimate is present in the input, explicitly weave it in — state the estimated change in metres, note it's a PROXY MEASUREMENT (not direct altimetry), and connect it to the other readings where relevant (e.g. does a water level drop align with a water extent decrease?)."
                    }
                },
                "required": ["per_calculation", "synthesis"]
            }
        }

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            tools=[report_tool],
            tool_choice={"type": "tool", "name": "generate_report"}
        )

        tool_use_block = next((b for b in response.content if getattr(b, 'type', None) == 'tool_use'), None)
        if tool_use_block is None:
            # Fallback: in case a future model variant still replies in
            # plain text despite tool_choice, try to salvage it from the
            # raw text rather than failing outright.
            raw_text = "".join(getattr(b, 'text', '') for b in response.content).strip()
            parsed = _parse_ai_json(raw_text)
            if parsed is None:
                snippet = raw_text[:300].replace('\n', ' ')
                logger.error(f'AI report: no tool_use block and text fallback failed. Raw response started with: {snippet}')
                return {'error': f'AI did not return structured output. Raw output started with: "{snippet}..." — please try again.'}
        else:
            parsed = tool_use_block.input  # already a parsed dict — no JSON string parsing needed

        if 'per_calculation' not in parsed or 'synthesis' not in parsed:
            # If the response was cut off by the token limit, say so
            # directly instead of a generic "missing fields" message —
            # this is what actually happened here, since per_calculation
            # is generated before synthesis and a long session (or a
            # language like Arabic that needs more tokens per sentence)
            # can hit the ceiling before synthesis is written.
            if getattr(response, 'stop_reason', None) == 'max_tokens':
                return {'error': f'AI response was cut off by the length limit before finishing (got: {list(parsed.keys())}) — please try again; if this keeps happening, it may help to run this with fewer readings in the session at once.'}
            return {'error': f'AI response was missing expected fields — got keys: {list(parsed.keys())}. Please try again.'}
        return parsed

    except Exception as e:
        logger.error(f'AI report generation failed: {e}')
        return {'error': f'AI request failed: {e}'}


def _parse_ai_json(raw_text):
    """
    Tries several increasingly-forgiving strategies to extract valid
    JSON from a model response, since even an explicit "JSON only, no
    markdown fences" instruction doesn't always get followed exactly —
    a stray preamble sentence or a fence wrapper is common enough to
    be worth handling directly rather than failing outright.
    Returns the parsed dict, or None if every strategy fails.
    """
    text = raw_text.strip()

    # Strategy 1: direct parse — the model followed instructions exactly.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip a markdown code fence, with or without a "json" tag.
    if '```' in text:
        parts = text.split('```')
        for part in parts:
            candidate = part.strip()
            if candidate.startswith('json'):
                candidate = candidate[4:].strip()
            if candidate.startswith('{'):
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue

    # Strategy 3: extract from the first '{' to the last '}' — handles a
    # stray sentence before or after the actual JSON object.
    first = text.find('{')
    last = text.rfind('}')
    if first != -1 and last != -1 and last > first:
        try:
            return json.loads(text[first:last+1])
        except json.JSONDecodeError:
            pass

    return None


# ══════════════════════════════════════════════════════════════
# TERRAIN PATTERN ANALYSIS — Archaeology & Terrain family only.
# The only place in this whole platform where the AI actually looks
# at the image itself, not just numbers. Deliberately scoped to
# shape/pattern classification, never object identification — at
# 10-30m per pixel, satellite imagery physically cannot resolve
# individual objects the way a normal photo can, so pretending
# otherwise would be a real honesty violation, not just an exaggeration.
# ══════════════════════════════════════════════════════════════

TERRAIN_PATTERN_SYSTEM_PROMPT = """You are a remote-sensing archaeologist describing a \
satellite-derived terrain anomaly image. CRITICAL CONSTRAINT: satellite imagery at this \
resolution (10-30 metres per pixel) CANNOT resolve individual objects the way a normal \
photograph can — one pixel covers an area the size of a building or larger. You are doing \
PATTERN-BASED SHAPE CLASSIFICATION, not object identification, and must never imply \
otherwise.

Describe only the geometric SHAPE of any anomaly visible (circular, linear, rectangular, \
irregular, or none), then state which of these established archaeological remote-sensing \
categories it is MOST CONSISTENT WITH, if any:
- Circular/mound-shaped, raised, isolated from surroundings -> possible settlement mound (tell)
- Linear, roughly consistent width, extends across the landscape -> possible buried wall, \
ancient road, or canal
- Rectangular/grid pattern, sharp right angles, regular repeated spacing -> possible \
building foundations
- Irregular, follows natural terrain contours with no consistent geometry -> most likely a \
natural formation, not human-made — this is the correct answer for most images

Rules, no exceptions:
1. Use hedged language throughout: "consistent with", "could indicate", "worth investigating" \
— never a confirmed identification.
2. Never invent historical context, dates, cultures, or civilizations not visible in the \
image itself.
3. If the image shows no clear anomaly or pattern, say so plainly rather than inventing one \
— an honest "nothing conclusive here" is a correct, useful answer.
4. Always end with a one-sentence recommendation for field or ground verification.
5. Keep the whole response to 2-4 sentences — a focused observation, not a report."""


def analyze_terrain_pattern(thumb_url, index_v, index_label):
    """
    Sends the actual rendered terrain image to Claude for pattern-based
    shape classification. Returns a short description string, or None
    on any failure — never blocks or breaks the reading itself.
    """
    if not ANTHROPIC_AVAILABLE:
        return None
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key or not thumb_url:
        return None
    try:
        img_resp = requests.get(thumb_url, timeout=15)
        if img_resp.status_code != 200:
            logger.warning(f'Terrain pattern: could not fetch image, status {img_resp.status_code}')
            return None
        img_b64 = base64.b64encode(img_resp.content).decode('utf-8')
        media_type = img_resp.headers.get('Content-Type', 'image/png').split(';')[0]
        if media_type not in ('image/png', 'image/jpeg', 'image/webp', 'image/gif'):
            media_type = 'image/png'

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=400,
            system=TERRAIN_PATTERN_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": img_b64}},
                    {"type": "text", "text": f"This is a {index_label} terrain anomaly map. Describe the shape and suggest the most likely category, per your instructions."}
                ]
            }]
        )
        text = "".join(getattr(b, 'text', '') for b in response.content).strip()
        return text if text else None
    except Exception as e:
        logger.warning(f'Terrain pattern analysis failed (non-fatal): {e}')
        return None
