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
5. Return ONLY valid JSON matching the exact schema you're asked for, no markdown code \
fences, no preamble, no explanation outside the JSON.
"""


def _classify_confidence(index_v):
    return 'high' if index_v in HIGH_CONFIDENCE_INDICES else 'proxy'


def _build_payload(readings, location, change_map, trend):
    """Strips each reading down to only the fields the model should see —
    deliberately not including internal IDs, raw thumbnail URLs, etc."""
    payload = {
        'location': location or None,
        'readings': [
            {
                'index': r.get('title') or r.get('indexV'),
                'family': r.get('fam'),
                'confidence': _classify_confidence(r.get('indexV')),
                'mean': r.get('mean'), 'min': r.get('min'), 'max': r.get('max'), 'std': r.get('std'),
                'date_range': f"{r.get('start')} to {r.get('end')}",
                'images_used': r.get('images'),
            }
            for r in readings
        ],
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
    return payload


def generate_ai_report(readings, location, change_map, trend):
    """
    Returns {'per_calculation': [{'index':..., 'description':...}, ...],
    'synthesis': '...'} or {'error': '...'} on failure. Never raises —
    always returns a dict so the route can respond cleanly either way.
    """
    if not ANTHROPIC_AVAILABLE:
        return {'error': 'AI module not installed on the server.'}

    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        return {'error': 'AI feature not configured on the server (missing API key).'}

    if not readings:
        return {'error': 'No calculations to describe — take at least one reading first.'}

    try:
        payload = _build_payload(readings, location, change_map, trend)

        user_prompt = f"""Here is the structured data for this session:

{json.dumps(payload, indent=2)}

Return JSON with exactly this schema:
{{
  "per_calculation": [
    {{"index": "<the index name>", "description": "<2-4 sentence plain-language paragraph describing what this specific reading shows, using its confidence level appropriately>"}}
  ],
  "synthesis": "<if there is more than one reading, a 2-4 paragraph synthesis connecting them into one holistic narrative. If there's only one reading, still provide a slightly deeper single-reading synthesis paragraph. If change_analysis or trend data is present, weave it in explicitly.>"
}}"""

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}]
        )

        raw_text = "".join(block.text for block in response.content if hasattr(block, 'text'))
        raw_text = raw_text.strip()
        # Guard against the model wrapping in a markdown fence despite instructions
        if raw_text.startswith('```'):
            raw_text = raw_text.split('```')[1]
            if raw_text.startswith('json'):
                raw_text = raw_text[4:]

        parsed = json.loads(raw_text)
        if 'per_calculation' not in parsed or 'synthesis' not in parsed:
            return {'error': 'AI response was missing expected fields — please try again.'}
        return parsed

    except json.JSONDecodeError as e:
        logger.error(f'AI report: model returned invalid JSON: {e}')
        return {'error': 'AI response could not be parsed — please try again.'}
    except Exception as e:
        logger.error(f'AI report generation failed: {e}')
        return {'error': f'AI request failed: {e}'}
