
"""Anomaly detection for solar and wind telemetry.

Detection methods:
1. Statistical rules using robust Z-scores (median and MAD).
2. Isolation Forest for unsupervised anomaly detection.

Severity:
- HIGH: Statistical rule and Isolation Forest both detect an anomaly.
- MEDIUM: Statistical rule detects an anomaly.
- WATCH: Isolation Forest detects an anomaly with a strong Z-score.
- NORMAL: No anomaly detected.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

Z_THRESH = 3.5
WATCH_Z = 2.5
CONTAMINATION = 0.03

NUM = [
    'irradiance_wm2',
    'wind_speed_ms',
    'temperature_c',
    'vibration_mm_s',
    'power_mw'
]


# --------------------------------------------------
# ROBUST Z-SCORE
# --------------------------------------------------

def _rz(x, floor):
    """Calculate robust Z-scores using median and MAD."""

    x = pd.Series(x)

    med = x.median()
    mad = (x - med).abs().median()

    denominator = max(1.4826 * mad, floor)

    return (x - med) / denominator


# --------------------------------------------------
# LOAD AND PREPROCESS DATA
# --------------------------------------------------

def load(path):
    """Load, clean and prepare telemetry data."""

    df = pd.read_csv(path)

    # Convert numerical columns and handle invalid values.
    df[NUM] = df[NUM].apply(
        pd.to_numeric,
        errors='coerce'
    )

    # Remove duplicates and sort readings.
    df = (
        df.drop_duplicates()
        .sort_values(['asset_id', 'timestamp_id'])
        .reset_index(drop=True)
    )

    # Fill missing numerical values within each asset.
    df[NUM] = df.groupby('asset_id')[NUM].transform(
        lambda s: s.interpolate(limit_direction='both')
    )

    # Derive day number from sequential hourly readings.
    df['day'] = df.groupby('asset_id').cumcount() // 24 + 1

    # Identify solar and wind assets.
    df['asset_type'] = np.where(
        df.asset_id.str.upper().str.startswith('SOLAR'),
        'solar',
        'wind'
    )

    return df


# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------

def add_features(df):
    """Create features used by the anomaly detection models."""

    parts = []

    for _, g in df.groupby('asset_id'):

        g = g.copy()

        # Compare temperature with the typical temperature
        # recorded during the same hour.
        g['temp_resid'] = (
            g.temperature_c
            - g.groupby('hour').temperature_c.transform('median')
        )

        g['temp_z'] = _rz(g.temp_resid, 1.0).values

        # Vibration anomaly score.
        g['vib_z'] = _rz(g.vibration_mm_s, 0.05).values

        # Select the relevant power driver.
        driver = (
            'irradiance_wm2'
            if g.asset_type.iloc[0] == 'solar'
            else 'wind_speed_ms'
        )

        # Estimate expected power using a cubic polynomial.
        coefficients = np.polyfit(
            g[driver],
            g.power_mw,
            3
        )

        predicted = np.polyval(coefficients, g[driver])

        residual = g.power_mw - predicted

        # Remove extreme residuals and refit the model.
        keep = residual.abs() <= (
            3 * max(1.4826 * residual.abs().median(), 0.03)
        )

        if keep.sum() > 10:
            coefficients = np.polyfit(
                g.loc[keep, driver],
                g.loc[keep, 'power_mw'],
                3
            )

        # Expected power output.
        g['expected_power_mw'] = np.polyval(
            coefficients,
            g[driver]
        ).clip(min=0)

        # Difference between actual and expected power.
        g['power_resid'] = (
            g.power_mw - g.expected_power_mw
        )

        g['power_z'] = _rz(
            g.power_resid,
            0.03
        ).values

        parts.append(g)

    return pd.concat(parts).sort_index()


# --------------------------------------------------
# ANOMALY DETECTION
# --------------------------------------------------

def detect(df):
    """Detect abnormal telemetry using rules and Isolation Forest."""

    f = add_features(df)

    # Statistical anomaly rules.
    f['rule_vib'] = f.vib_z > Z_THRESH

    f['rule_temp'] = f.temp_z > Z_THRESH

    # Negative power residual indicates under-performance.
    f['rule_power'] = f.power_z < -Z_THRESH

    f['rule_flag'] = f[
        ['rule_vib', 'rule_temp', 'rule_power']
    ].any(axis=1)

    # Features supplied to Isolation Forest.
    X = f[['temp_z', 'vib_z', 'power_z']]

    # Train unsupervised anomaly detection model.
    iso = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=42
    )

    f['if_flag'] = iso.fit_predict(X) == -1

    # Higher score means more abnormal.
    f['if_score'] = -iso.score_samples(X)

    # Require a sufficiently strong deviation for WATCH events.
    strong = X.abs().max(axis=1) >= WATCH_Z

    f['flagged'] = (
        f.rule_flag | (f.if_flag & strong)
    )

    # Assign severity.
    f['severity'] = np.select(
        [
            f.rule_flag & f.if_flag,
            f.rule_flag,
            f.flagged
        ],
        [
            'high',
            'medium',
            'watch'
        ],
        default='normal'
    )

    return f


# --------------------------------------------------
# OPERATOR RECOMMENDATIONS
# --------------------------------------------------

CHECKS = {

    'vib+temp': [
        "Inspect bearings and lubrication",
        "Check mounting bolts and looseness",
        "Check cooling fan and vents"
    ],

    'temp': [
        "Check cooling fan, vents and filters",
        "Look for blocked airflow or high load",
        "Verify the temperature sensor"
    ],

    'vib': [
        "Inspect bearings and drive-train alignment",
        "Check mounting and balance",
        "Verify the vibration sensor"
    ],

    'power_wind': [
        "Check pitch/yaw alignment and any curtailment order",
        "Check for derating or a controller fault",
        "Compare with neighbouring turbines"
    ],

    'power_solar': [
        "Check inverter status and derating alarms",
        "Look for shading, soiling or string faults",
        "Compare with neighbouring inverters"
    ],

    'review': [
        "Verify the sensor readings",
        "Review current operating conditions",
        "Compare with neighbouring assets"
    ]
}


# --------------------------------------------------
# EVENT EXPLANATION
# --------------------------------------------------

def _event(aid, e, f):
    """Convert abnormal readings into an operator-friendly event."""

    base = f[f.asset_id == aid]

    atype = e.asset_type.iloc[0]

    vib = e.rule_vib.any()
    tmp = e.rule_temp.any()
    pw = e.rule_power.any()

    # For WATCH-only events, select the strongest signal.
    if not (vib or tmp or pw):

        col = (
            e[['temp_z', 'vib_z', 'power_z']]
            .abs()
            .max()
            .idxmax()
        )

        vib = col == 'vib_z'
        tmp = col == 'temp_z'
        pw = col == 'power_z'

    parts = []

    # Vibration explanation.
    if vib:

        peak_vib = e.vibration_mm_s.max()

        normal_vib = base.vibration_mm_s.median()

        parts.append(
            f"vibration peaked at {peak_vib:.2f} mm/s "
            f"(normal is about {normal_vib:.2f})"
        )

    # Temperature explanation.
    if tmp:

        idx = e.temp_resid.abs().idxmax()

        temp_delta = e.loc[idx, 'temp_resid']

        peak_temp = e.loc[idx, 'temperature_c']

        direction = (
            "above" if temp_delta >= 0 else "below"
        )

        parts.append(
            f"temperature was {abs(temp_delta):.1f} °C "
            f"{direction} the usual for this hour "
            f"(reading {peak_temp:.1f} °C)"
        )

    # Power explanation.
    if pw:

        idx = e.power_resid.abs().idxmax()

        power_delta = e.loc[idx, 'power_resid']

        direction = (
            "below" if power_delta < 0 else "above"
        )

        driver_name = (
            "sunlight" if atype == 'solar' else "wind speed"
        )

        parts.append(
            f"power was {abs(power_delta):.2f} MW "
            f"{direction} the expected output based on "
            f"{driver_name}"
        )

    # Select an appropriate explanation and checks.
    if vib and tmp and e.temp_resid.max() > 0:

        key = 'vib+temp'

    elif tmp and e.temp_resid.max() > 0:

        key = 'temp'

    elif vib:

        key = 'vib'

    elif pw and e.power_resid.min() < 0:

        key = f'power_{atype}'

    else:

        # Do not describe low temperature as overheating
        # or higher-than-expected power as under-performance.
        key = 'review'

    meanings = {

        'vib+temp': (
            "Vibration and heat rising together may indicate "
            "a mechanical problem such as a worn bearing or loose part."
        ),

        'temp': (
            "Elevated temperature without extra vibration may indicate "
            "a cooling or airflow problem."
        ),

        'vib': (
            "Extra vibration may indicate a mechanical imbalance, "
            "loose mounting or sensor issue."
        ),

        'power_wind': (
            "Low output at normal wind speed may indicate derating, "
            "a pitch/yaw issue or curtailment."
        ),

        'power_solar': (
            "Low output in normal sunlight may indicate inverter derating, "
            "shading or soiling."
        ),

        'review': (
            "This reading differs from the usual operating pattern. "
            "Verify the sensor readings and review operating conditions "
            "before drawing a conclusion."
        )
    }

    # Event duration and timing.
    n = len(e)

    when = (
        f"day {int(e.day.iloc[0])}, "
        f"{int(e.hour.iloc[0]):02d}:00"
        + (f" for {n} hours" if n > 1 else "")
    )

    # Event severity.
    sev = (
        'high'
        if (e.severity == 'high').any()
        else 'medium'
        if (e.severity == 'medium').any()
        else 'watch'
    )

    return {

        'asset_id': aid,

        'when': when,

        'hours': n,

        'severity': sev,

        'start_index': int(e.timestamp_id.iloc[0]),

        'end_index': int(e.timestamp_id.iloc[-1]),

        'signals': [
            s for s, on in (
                ('vibration', vib),
                ('temperature', tmp),
                ('power', pw)
            )
            if on
        ],

        'explanation': (
            f"On {aid}, {' and '.join(parts)}. "
            f"{meanings[key]}"
        ),

        'recommended_checks': CHECKS[key],

        'score': float(e.if_score.max())
    }


# --------------------------------------------------
# GROUP READINGS INTO EVENTS
# --------------------------------------------------

def build_events(f):
    """Group consecutive abnormal readings into events."""

    events = []

    for aid, g in f[f.flagged].groupby('asset_id'):

        grp = (
            g.timestamp_id.diff()
            .fillna(1) > 1
        ).cumsum()

        events += [
            _event(aid, e, f)
            for _, e in g.groupby(grp)
        ]

    order = {
        'high': 0,
        'medium': 1,
        'watch': 2
    }

    return sorted(
        events,
        key=lambda e: (
            order[e['severity']],
            -e['score']
        )
    )


# --------------------------------------------------
# MODEL EVALUATION USING SYNTHETIC FAULT INJECTION
# --------------------------------------------------

def evaluate(df, n=24, seed=0):
    """Evaluate detection using injected synthetic faults.

    The dataset has no labelled real faults.
    Therefore, these metrics measure detection of injected
    faults and must not be interpreted as real-world accuracy.
    """

    if n <= 0:
        raise ValueError("Number of injected faults must be positive.")

    base = detect(df)

    # Exclude anomalies already present in the original dataset.
    known = set(base.index[base.flagged])

    rng = np.random.default_rng(seed)

    kinds = [
        'bearing',
        'overheat',
        'underperf'
    ]

    d = df.copy()

    injected = {}

    for i in range(n):

        kind = kinds[i % len(kinds)]

        pool = [
            j for j in d.index
            if j not in known
            and j not in injected
            and (
                kind != 'underperf'
                or d.at[j, 'power_mw'] > 0.3
            )
        ]

        if not pool:
            raise ValueError(
                f"Not enough eligible rows to inject {kind} faults."
            )

        j = int(rng.choice(pool))

        # Inject synthetic fault patterns.
        if kind == 'bearing':

            d.at[j, 'vibration_mm_s'] *= 2.7

            d.at[j, 'temperature_c'] += 12

        elif kind == 'overheat':

            d.at[j, 'temperature_c'] += 18

        else:

            d.at[j, 'power_mw'] *= 0.4

        injected[j] = kind

    # Run detection again after fault injection.
    f = detect(d)

    hit = set(f.index[f.flagged]) - known

    true_positives = len(hit & set(injected))

    false_positives = len(hit - set(injected))

    # Calculate recall separately for each fault type.
    recall_by_fault = {}

    for kind in kinds:

        total = sum(
            1 for value in injected.values()
            if value == kind
        )

        detected = sum(
            1 for j, value in injected.items()
            if value == kind and j in hit
        )

        recall_by_fault[kind] = round(
            detected / total,
            2
        ) if total else None

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else None
    )

    return {

        'method': (
            f'synthetic fault injection ({n} faults: '
            'bearing, overheating, under-performance)'
        ),

        'injected': n,

        'detected': true_positives,

        'false_alarms': false_positives,

        'recall': round(true_positives / n, 2),

        'precision': (
            round(precision, 2)
            if precision is not None
            else None
        ),

        'recall_by_fault': recall_by_fault,

        'real_anomalies_in_data': len(known)
    }
