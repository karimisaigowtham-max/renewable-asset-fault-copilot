
"""Anomaly detection for solar and wind telemetry.

Methods:
1. Robust statistical z-scores using median and MAD.
2. Isolation Forest for unsupervised anomaly detection.

Severity:
- HIGH: statistical rule and Isolation Forest agree.
- MEDIUM: statistical rule only.
- WATCH: Isolation Forest flags a sufficiently unusual reading.
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
    "irradiance_wm2",
    "wind_speed_ms",
    "temperature_c",
    "vibration_mm_s",
    "power_mw",
]


# --------------------------------------------------
# ROBUST Z-SCORE
# --------------------------------------------------

def _rz(x, floor):
    """Calculate robust z-scores using median and MAD."""

    x = pd.Series(x)
    med = x.median()
    mad = (x - med).abs().median()

    denominator = max(1.4826 * mad, floor)

    return (x - med) / denominator


# --------------------------------------------------
# DATA LOADING AND PREPROCESSING
# --------------------------------------------------

def load(path):

    df = pd.read_csv(path)

    df[NUM] = df[NUM].apply(
        pd.to_numeric,
        errors="coerce"
    )

    df = (
        df.drop_duplicates()
        .sort_values(["asset_id", "timestamp_id"])
        .reset_index(drop=True)
    )

    df[NUM] = df.groupby("asset_id")[NUM].transform(
        lambda s: s.interpolate(limit_direction="both")
    )

    df["day"] = (
        df.groupby("asset_id").cumcount() // 24 + 1
    )

    df["asset_type"] = np.where(
        df.asset_id.str.upper().str.startswith("SOLAR"),
        "solar",
        "wind"
    )

    return df


# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------

def add_features(df):

    parts = []

    for _, g in df.groupby("asset_id"):

        g = g.copy()

        # Temperature deviation from the usual value
        # for the same hour.

        g["temp_resid"] = (
            g.temperature_c
            - g.groupby("hour").temperature_c.transform("median")
        )

        g["temp_z"] = _rz(g.temp_resid, 1.0).values

        # Vibration anomaly score.

        g["vib_z"] = _rz(
            g.vibration_mm_s,
            0.05
        ).values

        # Select the appropriate power driver.

        driver = (
            "irradiance_wm2"
            if g.asset_type.iloc[0] == "solar"
            else "wind_speed_ms"
        )

        # Estimate expected power using a cubic polynomial.

        coef = np.polyfit(
            g[driver],
            g.power_mw,
            3
        )

        residual = (
            g.power_mw
            - np.polyval(coef, g[driver])
        )

        # Refit after excluding large residuals.

        keep = residual.abs() <= (
            3 * max(
                1.4826 * residual.abs().median(),
                0.03
            )
        )

        if keep.sum() > 10:

            coef = np.polyfit(
                g.loc[keep, driver],
                g.loc[keep, "power_mw"],
                3
            )

        g["expected_power_mw"] = np.polyval(
            coef,
            g[driver]
        ).clip(min=0)

        g["power_resid"] = (
            g.power_mw - g.expected_power_mw
        )

        g["power_z"] = _rz(
            g.power_resid,
            0.03
        ).values

        parts.append(g)

    return pd.concat(parts).sort_index()


# --------------------------------------------------
# ANOMALY DETECTION
# --------------------------------------------------

def detect(df):

    f = add_features(df)

    # Statistical rules.

    f["rule_vib"] = f.vib_z > Z_THRESH

    f["rule_temp"] = f.temp_z > Z_THRESH

    f["rule_power"] = f.power_z < -Z_THRESH

    f["rule_flag"] = f[
        ["rule_vib", "rule_temp", "rule_power"]
    ].any(axis=1)

    # Isolation Forest.

    X = f[
        ["temp_z", "vib_z", "power_z"]
    ]

    iso = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=42
    )

    f["if_flag"] = iso.fit_predict(X) == -1

    f["if_score"] = -iso.score_samples(X)

    # Only sufficiently unusual Isolation Forest
    # results are included.

    strong = X.abs().max(axis=1) >= WATCH_Z

    f["flagged"] = (
        f.rule_flag
        | (f.if_flag & strong)
    )

    f["severity"] = np.select(
        [
            f.rule_flag & f.if_flag,
            f.rule_flag,
            f.flagged
        ],
        [
            "high",
            "medium",
            "watch"
        ],
        default="normal"
    )

    return f


# --------------------------------------------------
# OPERATOR RECOMMENDATIONS
# --------------------------------------------------

CHECKS = {

    "vib_temp": [
        "Inspect bearings and lubrication",
        "Check mounting bolts and looseness",
        "Check cooling fan and vents"
    ],

    "temp_high": [
        "Check cooling fan, vents and filters",
        "Look for blocked airflow or high load",
        "Verify the temperature sensor"
    ],

    "temp_low": [
        "Verify the temperature sensor reading",
        "Check operating conditions and ambient temperature",
        "Compare readings with similar assets"
    ],

    "vib": [
        "Inspect bearings and drive-train alignment",
        "Check mounting and balance",
        "Verify the vibration sensor"
    ],

    "power_wind": [
        "Check pitch/yaw alignment and curtailment orders",
        "Check for derating or a controller fault",
        "Compare with neighbouring turbines"
    ],

    "power_solar": [
        "Check inverter status and derating alarms",
        "Look for shading, soiling or string faults",
        "Compare with neighbouring inverters"
    ],

    "general": [
        "Verify the sensor readings",
        "Check operating conditions",
        "Compare with neighbouring assets"
    ]
}


# --------------------------------------------------
# EVENT EXPLANATION
# --------------------------------------------------

def _event(aid, e, f):

    base = f[f.asset_id == aid]

    asset_type = e.asset_type.iloc[0]

    vib = bool(e.rule_vib.any())
    temp_high = bool(e.rule_temp.any())
    power_low = bool(e.rule_power.any())

    # Determine the strongest signal for watch-only events.

    if not (vib or temp_high or power_low):

        col = (
            e[["temp_z", "vib_z", "power_z"]]
            .abs()
            .max()
            .idxmax()
        )

        vib = col == "vib_z"

        if col == "temp_z":
            temp_high = (
                e.temp_resid.abs().max() > 0
                and e.loc[
                    e.temp_resid.abs().idxmax(),
                    "temp_resid"
                ] > 0
            )

    # Identify the largest temperature deviation
    # while preserving its direction.

    temp_row = e.loc[
        e.temp_resid.abs().idxmax()
    ]

    temp_delta = float(temp_row.temp_resid)

    temp_low = (
        not temp_high
        and not vib
        and not power_low
        and temp_delta < 0
        and "temp_z" in e[
            ["temp_z", "vib_z", "power_z"]
        ].abs().max().idxmax()
    )

    # For events containing a high temperature rule,
    # describe the actual elevated temperature.

    if temp_high:
        temp_low = False

    # Power deviation and its direction.

    power_row = e.loc[
        e.power_resid.abs().idxmax()
    ]

    power_delta = float(power_row.power_resid)

    if not (vib or temp_high or temp_low or power_low):
        # Handle unusual power-only watch events.
        strongest = (
            e[["temp_z", "vib_z", "power_z"]]
            .abs()
            .max()
            .idxmax()
        )

        if strongest == "power_z":
            power_low = power_delta < 0

    parts = []

    if vib:

        parts.append(
            f"vibration peaked at "
            f"{e.vibration_mm_s.max():.2f} mm/s "
            f"(normal is about "
            f"{base.vibration_mm_s.median():.2f})"
        )

    if temp_high:

        parts.append(
            f"temperature was "
            f"{temp_delta:.1f} °C above the usual "
            f"for this hour "
            f"(reading {temp_row.temperature_c:.1f} °C)"
        )

    elif temp_low:

        parts.append(
            f"temperature was "
            f"{abs(temp_delta):.1f} °C below the usual "
            f"for this hour "
            f"(reading {temp_row.temperature_c:.1f} °C)"
        )

    if power_low:

        parts.append(
            f"power was "
            f"{abs(power_delta):.2f} MW below "
            f"the expected output based on "
            f"{'sunlight' if asset_type == 'solar' else 'wind speed'}"
        )

    # Select explanation and recommended checks.

    if vib and temp_high:

        key = "vib_temp"

        meaning = (
            "Vibration and heat rising together may indicate "
            "a mechanical problem such as a worn bearing "
            "or loose component."
        )

    elif temp_high:

        key = "temp_high"

        meaning = (
            "Elevated temperature without extra vibration "
            "may indicate a cooling or airflow problem."
        )

    elif temp_low:

        key = "temp_low"

        meaning = (
            "A lower-than-usual temperature is an unusual "
            "reading that should be checked against sensor "
            "accuracy and operating conditions."
        )

    elif vib:

        key = "vib"

        meaning = (
            "Extra vibration may indicate mechanical imbalance "
            "or a sensor issue."
        )

    elif power_low:

        key = f"power_{asset_type}"

        if asset_type == "wind":

            meaning = (
                "Low output at the observed wind speed may "
                "indicate derating, pitch/yaw issues or curtailment."
            )

        else:

            meaning = (
                "Low output under the observed sunlight may "
                "indicate inverter derating, shading or soiling."
            )

    else:

        key = "general"

        meaning = (
            "An unusual sensor pattern was detected. "
            "Further inspection is needed before identifying "
            "a possible cause."
        )

    # Event time.

    n = len(e)

    when = (
        f"day {int(e.day.iloc[0])}, "
        f"{int(e.hour.iloc[0]):02d}:00"
    )

    if n > 1:
        when += f" for {n} hours"

    # Event severity.

    severity = (
        "high"
        if (e.severity == "high").any()
        else "medium"
        if (e.severity == "medium").any()
        else "watch"
    )

    # Return event details.

    signals = []

    if vib:
        signals.append("vibration")

    if temp_high or temp_low:
        signals.append("temperature")

    if power_low:
        signals.append("power")

    return {

        "asset_id": aid,

        "when": when,

        "hours": n,

        "severity": severity,

        "start_index": int(e.timestamp_id.iloc[0]),

        "end_index": int(e.timestamp_id.iloc[-1]),

        "signals": signals,

        "explanation": (
            f"On {aid}, "
            f"{' and '.join(parts)}. "
            f"{meaning}"
        ),

        "recommended_checks": CHECKS[key],

        "score": float(e.if_score.max())
    }


# --------------------------------------------------
# GROUP ANOMALIES INTO EVENTS
# --------------------------------------------------

def build_events(f):

    events = []

    for aid, g in f[f.flagged].groupby("asset_id"):

        grp = (
            g.timestamp_id.diff()
            .fillna(1)
            .gt(1)
            .cumsum()
        )

        for _, e in g.groupby(grp):

            events.append(
                _event(aid, e, f)
            )

    order = {
        "high": 0,
        "medium": 1,
        "watch": 2
    }

    return sorted(
        events,
        key=lambda e: (
            order[e["severity"]],
            -e["score"]
        )
    )


# --------------------------------------------------
# MODEL EVALUATION USING SYNTHETIC FAULT INJECTION
# --------------------------------------------------

def evaluate(df, n=24, seed=0):

    """Evaluate detection by injecting synthetic faults.

    This is not a substitute for evaluation against
    real labelled operational failures.
    """

    if n <= 0 or n % 3 != 0:
        raise ValueError(
            "n must be a positive multiple of 3."
        )

    base = detect(df)

    known = set(
        base.index[base.flagged]
    )

    rng = np.random.default_rng(seed)

    kinds = [
        "bearing",
        "overheat",
        "underperf"
    ]

    d = df.copy()

    injected = {}

    for i in range(n):

        kind = kinds[i % 3]

        pool = [
            j for j in d.index
            if j not in known
            and j not in injected
            and (
                kind != "underperf"
                or d.at[j, "power_mw"] > 0.3
            )
        ]

        if not pool:
            raise ValueError(
                "Not enough suitable rows for fault injection."
            )

        j = int(rng.choice(pool))

        if kind == "bearing":

            d.at[j, "vibration_mm_s"] *= 2.7

            d.at[j, "temperature_c"] += 12

        elif kind == "overheat":

            d.at[j, "temperature_c"] += 18

        else:

            d.at[j, "power_mw"] *= 0.4

        injected[j] = kind

    f = detect(d)

    hit = (
        set(f.index[f.flagged])
        - known
    )

    true_positives = len(
        hit & set(injected)
    )

    false_positives = len(
        hit - set(injected)
    )

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
        )

    recall = true_positives / n

    precision = (
        true_positives
        / (true_positives + false_positives)
        if true_positives + false_positives
        else None
    )

    return {

        "method": (
            f"synthetic fault injection "
            f"({n} faults: bearing, overheating, "
            f"under-performance)"
        ),

        "injected": n,

        "detected": true_positives,

        "false_alarms": false_positives,

        "recall": round(recall, 2),

        "precision": (
            round(precision, 2)
            if precision is not None
            else None
        ),

        "recall_by_fault": recall_by_fault,

        "real_anomalies_in_data": len(known)
    }
