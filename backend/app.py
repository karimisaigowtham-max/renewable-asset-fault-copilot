import re
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend import detector

app = FastAPI(title='Renewable Asset Fault & Anomaly Detection Copilot')
BASE = Path(__file__).resolve().parents[1]
CSV = BASE / 'data' / 'asset_telemetry.csv'
LIMITS = [
    "Only 5 days per asset, so the 'usual for this hour' baseline is based on very few days.",
    "No labelled faults exist; accuracy is measured on injected synthetic faults, not real failures.",
    "Explanations are rule-based suggestions from sensor patterns, not confirmed root causes.",
]


class Query(BaseModel):
    text: str = ''


@lru_cache(maxsize=1)
def results():
    df = detector.load(CSV)
    f = detector.detect(df)
    return df, f, detector.build_events(f), detector.evaluate(df)


@app.get('/api/demo')
def demo():
    _, f, events, _ = results()
    return {
        'case_study': 'Renewable Asset Fault & Anomaly Detection Copilot',
        'status': 'running', 'rows': len(f), 'assets': sorted(f.asset_id.unique().tolist()),
        'events_found': len(events),
    }


@app.get('/api/anomalies')
def anomalies(asset_id: str = ''):
    _, _, events, _ = results()
    return [e for e in events if not asset_id or e['asset_id'] == asset_id]


@app.get('/api/metrics')
def metrics():
    return {**results()[3], 'limitations': LIMITS}


@app.get('/api/series/{asset_id}')
def series(asset_id: str):
    f = results()[1]
    g = f[f.asset_id == asset_id]
    cols = ['timestamp_id', 'power_mw', 'expected_power_mw', 'vibration_mm_s', 'temperature_c', 'severity']
    return g[cols].round(3).to_dict(orient='records')


@app.post('/api/run')
def run_case(query: Query):
    f, events = results()[1], results()[2]
    text = query.text.lower()
    assets = [a for a in f.asset_id.unique() if a.lower() in text]
    sel = [e for e in events if not assets or e['asset_id'] in assets]
    for word, sig in (('vibration', 'vibration'), ('temperature', 'temperature'), ('power', 'power')):
        if word in text:
            sel = [e for e in sel if sig in e['signals']]
    if re.search(r'\b(high|urgent|critical)\b', text):
        sel = [e for e in sel if e['severity'] == 'high']
    if sel:
        summary = f"Found {len(sel)} abnormal event(s). Most important: {sel[0]['explanation']}"
    else:
        summary = "No matching abnormal events were found for that question."
    return {'input': query.text, 'assets_considered': assets or 'all', 'summary': summary,
            'events': sel, 'metrics': metrics(), 'limitations': LIMITS}


@app.get('/')
def index():
    return FileResponse(BASE / 'frontend' / 'index.html')
