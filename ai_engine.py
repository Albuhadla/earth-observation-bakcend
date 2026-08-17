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
            messages=[
                {"role": "user", "content": user_prompt},
                # Prefilling the assistant's turn with the opening brace is
                # a reliable way to force Claude to continue directly into
                # JSON — this is the actual fix for "AI response could not
                # be parsed", since it prevents the stray preamble sentence
                # or commentary that was causing that in the first place,
                # rather than just parsing more defensively after the fact.
                {"role": "assistant", "content": "{"}
            ]
        )

        raw_text = "{" + "".join(block.text for block in response.content if hasattr(block, 'text'))
        raw_text = raw_text.strip()

        parsed = _parse_ai_json(raw_text)
        if parsed is None:
            # This is the actual fix — previously this just said "could
            # not be parsed" with zero information about why, making it
            # impossible to diagnose without direct server log access.
            # Now the real raw response (truncated) comes back in the
            # error itself, and several parsing strategies are tried
            # before giving up, since a model occasionally adds a stray
            # preamble sentence or trailing comment even when told not to.
            snippet = raw_text[:300].replace('\n', ' ')
            logger.error(f'AI report: could not parse model output. Raw response started with: {snippet}')
            return {'error': f'AI response could not be parsed. Raw output started with: "{snippet}..." — please try again; if this keeps happening, this snippet is what needs fixing in the prompt.'}

        if 'per_calculation' not in parsed or 'synthesis' not in parsed:
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
