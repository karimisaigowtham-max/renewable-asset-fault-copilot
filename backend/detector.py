"""Anomaly detection for solar/wind telemetry.

Two independent methods are combined:
  1. Statistical rules  - robust z-scores (median/MAD) on temperature, vibration, power.
  2. IsolationForest    - unsupervised ML model run on the same z-score features.
Rule hit + IsolationForest hit  -> HIGH severity
Rule hit only                   -> MEDIUM
IsolationForest only (z >= 2.5) -> WATCH
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

Z_THRESH = 3.5          # robust z-score above which a rule fires
WATCH_Z = 2.5           # IsolationForest-only points need at least this z to be shown
CONTAMINATION = 0.03    # expected share of abnormal rows (IsolationForest setting)
NUM = ['irradiance_wm2', 'wind_speed_ms', 'temperature_c', 'vibration_mm_s', 'power_mw']


def _rz(x, floor):
    """Robust z-score: (x - median) / (1.4826 * MAD), with a floor so tiny MADs don't explode."""
    x = pd.Series(x)
    med = x.median()
    mad = (x - med).abs().median()
    return (x - med) / max(1.4826 * mad, floor)


def load(path):
    df = pd.read_csv(path)
    df[NUM] = df[NUM].apply(pd.to_numeric, errors='coerce')          # bad values -> NaN
    df = df.drop_duplicates().sort_values(['asset_id', 'timestamp_id']).reset_index(drop=True)
    df[NUM] = df.groupby('asset_id')[NUM].transform(lambda s: s.interpolate(limit_direction='both'))
    df['day'] = df.groupby('asset_id').cumcount() // 24 + 1
    df['asset_type'] = np.where(df.asset_id.str.upper().str.startswith('SOLAR'), 'solar', 'wind')
    return df


def add_features(df):
    parts = []
    for _, g in df.groupby('asset_id'):
        g = g.copy()
        # Temperature has a daily cycle -> compare each reading with the typical value for that hour.
        g['temp_resid'] = g.temperature_c - g.groupby('hour').temperature_c.transform('median')
        g['temp_z'] = _rz(g.temp_resid, 1.0).values
        g['vib_z'] = _rz(g.vibration_mm_s, 0.05).values
        # Expected power from the driver (sun for solar, wind for turbines): cubic fit, refit without outliers.
        drv = 'irradiance_wm2' if g.asset_type.iloc[0] == 'solar' else 'wind_speed_ms'
        coef = np.polyfit(g[drv], g.power_mw, 3)
        res = g.power_mw - np.polyval(coef, g[drv])
        keep = res.abs() <= 3 * max(1.4826 * res.abs().median(), 0.03)
        if keep.sum() > 10:
            coef = np.polyfit(g.loc[keep, drv], g.loc[keep, 'power_mw'], 3)
        g['expected_power_mw'] = np.polyval(coef, g[drv]).clip(min=0)
        g['power_resid'] = g.power_mw - g.expected_power_mw
        g['power_z'] = _rz(g.power_resid, 0.03).values
        parts.append(g)
    return pd.concat(parts).sort_index()


def detect(df):
    f = add_features(df)
    f['rule_vib'] = f.vib_z > Z_THRESH
    f['rule_temp'] = f.temp_z > Z_THRESH
    f['rule_power'] = f.power_z < -Z_THRESH          # only under-performance is a fault
    f['rule_flag'] = f[['rule_vib', 'rule_temp', 'rule_power']].any(axis=1)
    X = f[['temp_z', 'vib_z', 'power_z']]
    iso = IsolationForest(n_estimators=200, contamination=CONTAMINATION, random_state=42)
    f['if_flag'] = iso.fit_predict(X) == -1
    f['if_score'] = -iso.score_samples(X)            # higher = more abnormal
    strong = X.abs().max(axis=1) >= WATCH_Z
    f['flagged'] = f.rule_flag | (f.if_flag & strong)
    f['severity'] = np.select(
        [f.rule_flag & f.if_flag, f.rule_flag, f.flagged], ['high', 'medium', 'watch'], default='normal')
    return f


CHECKS = {
    'vib+temp': ["Inspect bearings and lubrication", "Check mounting bolts / looseness", "Check cooling fan and vents"],
    'temp': ["Check cooling fan, vents and filters", "Look for blocked airflow or high load", "Verify the temperature sensor"],
    'vib': ["Inspect bearings and drive-train alignment", "Check mounting and balance", "Verify the vibration sensor"],
    'power_wind': ["Check pitch/yaw alignment and any curtailment order", "Check for derating or a controller fault", "Compare with neighbouring turbines"],
    'power_solar': ["Check inverter status and derating alarms", "Look for shading, soiling or string faults", "Compare with neighbouring inverters"],
}


def _event(aid, e, f):
    base = f[f.asset_id == aid]
    atype = e.asset_type.iloc[0]
    vib, tmp, pw = e.rule_vib.any(), e.rule_temp.any(), e.rule_power.any()
    if not (vib or tmp or pw):          # watch-only event: describe the strongest z-score
        col = e[['temp_z', 'vib_z', 'power_z']].abs().max().idxmax()
        vib, tmp, pw = col == 'vib_z', col == 'temp_z', col == 'power_z'
    parts, key = [], None
    if vib:
     parts.append(f"vibration peaked at {e.vibration_mm_s.max():.2f} mm/s (normal is about {base.vibration_mm_s.median():.2f})")
    if tmp:
     parts.append(f"temperature ran {e.temp_resid.max():.1f} °C above the usual for this hour (peak {e.temperature_c.max():.1f} °C)")
    if pw:
      parts.append(f"power was {abs(e.power_resid.min()):.2f} MW below what the "
                 f"{'sunlight' if atype == 'solar' else 'wind speed'} should give")
    key = 'vib+temp' if vib and tmp else 'temp' if tmp else 'vib' if vib else f'power_{atype}'
    meaning = {
        'vib+temp': "Vibration and heat rising together usually points to a mechanical problem such as a worn bearing or loose part.",
        'temp': "Heat without extra vibration usually points to a cooling or airflow problem.",
        'vib': "Extra vibration on its own suggests a mechanical imbalance or sensor issue.",
        'power_wind': "Low output at normal wind suggests derating, a pitch/yaw issue or curtailment.",
        'power_solar': "Low output in normal sunlight suggests inverter derating, shading or soiling.",
    }[key]
    n = len(e)
    when = f"day {int(e.day.iloc[0])}, {int(e.hour.iloc[0]):02d}:00" + (f" for {n} hours" if n > 1 else "")
    sev = 'high' if (e.severity == 'high').any() else 'medium' if (e.severity == 'medium').any() else 'watch'
    return {
        'asset_id': aid, 'when': when, 'hours': n, 'severity': sev,
        'start_index': int(e.timestamp_id.iloc[0]), 'end_index': int(e.timestamp_id.iloc[-1]),
        'signals': [s for s, on in (('vibration', vib), ('temperature', tmp), ('power', pw)) if on],
        'explanation': f"On {aid}, {' and '.join(parts)}. {meaning}",
        'recommended_checks': CHECKS[key],
        'score': float(e.if_score.max()),
    }


def build_events(f):
    events = []
    for aid, g in f[f.flagged].groupby('asset_id'):
        grp = (g.timestamp_id.diff().fillna(1) > 1).cumsum()
        events += [_event(aid, e, f) for _, e in g.groupby(grp)]
    order = {'high': 0, 'medium': 1, 'watch': 2}
    return sorted(events, key=lambda e: (order[e['severity']], -e['score']))


def evaluate(df, n=24, seed=0):
    """No labelled faults exist, so we inject known synthetic faults and measure recall/precision."""
    base = detect(df)
    known = set(base.index[base.flagged])            # real anomalies already in the data: not scored
    rng = np.random.default_rng(seed)
    kinds = ['bearing', 'overheat', 'underperf']
    d, injected = df.copy(), {}
    for i in range(n):
        kind = kinds[i % 3]
        pool = [j for j in d.index if j not in known and j not in injected
                and (kind != 'underperf' or d.at[j, 'power_mw'] > 0.3)]
        j = int(rng.choice(pool))
        if kind == 'bearing':
            d.at[j, 'vibration_mm_s'] *= 2.7
            d.at[j, 'temperature_c'] += 12
        elif kind == 'overheat':
            d.at[j, 'temperature_c'] += 18
        else:
            d.at[j, 'power_mw'] *= 0.4
        injected[j] = kind
    f = detect(d)
    hit = set(f.index[f.flagged]) - known
    tp = len(hit & set(injected))
    fp = len(hit - set(injected))
    recall = {k: round(sum(1 for j, t in injected.items() if t == k and j in hit) / (n // 3), 2) for k in kinds}
    return {
        'method': 'synthetic fault injection (24 faults: bearing, overheating, under-performance)',
        'injected': n, 'detected': tp, 'false_alarms': fp,
        'recall': round(tp / n, 2), 'precision': round(tp / (tp + fp), 2) if tp + fp else None,
        'recall_by_fault': recall,
        'real_anomalies_in_data': len(known),
    }
