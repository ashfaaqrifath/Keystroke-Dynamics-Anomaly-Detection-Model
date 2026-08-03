import time
import threading
import os
import sys
import math
import pandas as pd
import numpy as np
from pynput import keyboard
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import joblib
from collections import deque, defaultdict, Counter

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

MODEL_FILE = os.path.join(BASE_DIR, "keystroke_dynamics_model.pkl")
SCALER_FILE = os.path.join(BASE_DIR, "ocsvm_scaler.pkl")
THRESHOLD_FILE = os.path.join(BASE_DIR, "ocsvm_threshold.pkl")
DIGRAPH_PROFILE_FILE = os.path.join(BASE_DIR, "digraph_profile.pkl")
TRAINING_DATA_FILE = os.path.join(BASE_DIR, "keystroke_dynamics_training.csv")
DETECTION_DATA_FILE = os.path.join(BASE_DIR, "anomaly_detection.csv")


IDLE_CHECK_INTERVAL = 4
MIN_FEATURE_EVENTS = 5
EVENT_WINDOW_SECONDS = 10
MIN_TRAINING_WINDOWS = 1000
MIN_DIGRAPH_SAMPLES = 3           # min occurrences of a digraph before trusting its profile
PAUSE_THRESHOLD = 0.2             # seconds; a press-to-press gap longer than this counts as a pause

# OCSVM hyperparameters (can be tuned)
NU_VALUES = [0.01, 0.02, 0.05, 0.1]
GAMMA_VALUES = ['scale', 'auto']
THRESHOLD_PERCENTILE = 5          # percentile of training scores to set threshold


FEATURE_COLUMNS = [
    "dwell_mean",
    "dwell_std",
    "dwell_median",
    "flight_mean",
    "flight_std",
    "flight_median",
    "pp_mean",
    "pp_std",
    "pp_median",
    "r_letter",
    "r_digit",
    "r_space",
    "r_backspace",
    "r_enter",
    "r_modifier",
    "r_other",
    "typing_rate",
    "key_entropy",
    "pause_ratio",
    "pause_mean",
    "digraph_dev_mean",
    "digraph_dev_max",
]


FEATURE_READABLE = {
    "dwell_mean": "Key hold time (avg)",
    "dwell_std": "Key hold variability",
    "dwell_median": "Key hold time (median)",
    "flight_mean": "Gap between keys (avg)",
    "flight_std": "Gap variability",
    "flight_median": "Gap between keys (median)",
    "pp_mean": "Key press interval (avg)",
    "pp_std": "Interval variability",
    "pp_median": "Key press interval (median)",
    "r_letter": "Letter key ratio",
    "r_digit": "Digit key ratio",
    "r_space": "Space key ratio",
    "r_backspace": "Backspace ratio",
    "r_enter": "Enter key ratio",
    "r_modifier": "Modifier key ratio",
    "r_other": "Other key ratio",
    "typing_rate": "Typing speed (keys/sec)",
    "key_entropy": "Key variety (entropy)",
    "pause_ratio": "Pause frequency",
    "pause_mean": "Pause length (avg)",
    "digraph_dev_mean": "Digraph timing deviation (avg)",
    "digraph_dev_max": "Digraph timing deviation (max)",
}

# ─── GLOBAL STATE ────────────────────────────────────────────────────────────
key_events = deque()          # (timestamp, type, key)
pressed_keys = {}             # key → press_time (for dwell)
data_updated = False

feature_vectors = []          # list of feature dicts for training
window_digraphs_list = []     # list of per-window digraph lists, parallel to feature_vectors
digraph_raw_accum = defaultdict(list)   # digraph -> list of raw press-to-press intervals (training only)
digraph_profile = {}          # digraph -> {'mean':.., 'std':..} personal timing profile

model = None
scaler = None
threshold = None
baseline_ready = False
feature_stats = {}            # mean/std per feature (for z‑score reporting)

detection_rows = 0

# ─── LOAD EXISTING DATA & MODEL ─────────────────────────────────────────────
if os.path.exists(TRAINING_DATA_FILE):
    df = pd.read_csv(TRAINING_DATA_FILE)
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0.0)
    feature_vectors = df.to_dict('records')
    print(f"Loaded {len(feature_vectors)} training feature windows.")

if os.path.exists(DIGRAPH_PROFILE_FILE):
    digraph_profile = joblib.load(DIGRAPH_PROFILE_FILE)
    print(f"Loaded digraph profile with {len(digraph_profile)} digraphs.")

if os.path.exists(DETECTION_DATA_FILE):
    detection_rows = len(pd.read_csv(DETECTION_DATA_FILE))
    print(f"Loaded {detection_rows} past detection rows.")

if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE) and os.path.exists(THRESHOLD_FILE):
    _loaded_model = joblib.load(MODEL_FILE)
    _loaded_scaler = joblib.load(SCALER_FILE)
    _loaded_threshold = joblib.load(THRESHOLD_FILE)

    expected_n = len(FEATURE_COLUMNS)
    loaded_n = getattr(_loaded_scaler, "n_features_in_", None)

    if loaded_n == expected_n:
        model, scaler, threshold = _loaded_model, _loaded_scaler, _loaded_threshold
        baseline_ready = True
        print("Loaded existing OCSVM model, scaler, and threshold.")
        # Build feature_stats for reporting
        if feature_vectors:
            df_temp = pd.DataFrame(feature_vectors).reindex(columns=FEATURE_COLUMNS, fill_value=0.0)
            for col in df_temp.columns:
                feature_stats[col] = {
                    'mean': df_temp[col].mean(),
                    'std': df_temp[col].std()
                }
    else:
        # Old model was trained on a different feature set. Refuse to load it
        # so we don't crash on a shape mismatch — retrain from scratch instead.
        print(f"Existing model expects {loaded_n} features but {expected_n} are now defined. "
              f"Ignoring the old model/scaler/threshold.")
        print("Delete keystroke_dynamics_training.csv too, then let it recollect "
              f"{MIN_TRAINING_WINDOWS} fresh windows so the profiles can be rebuilt.")
        model = OneClassSVM(kernel='rbf', nu=0.01, gamma='scale')
        scaler = StandardScaler()
        threshold = None
        baseline_ready = False
else:
    print(f"Will train OCSVM after {MIN_TRAINING_WINDOWS} active windows.")
    print(f"(each window = {EVENT_WINDOW_SECONDS}s of typing).")
    model = OneClassSVM(kernel='rbf', nu=0.01, gamma='scale')
    scaler = StandardScaler()
    threshold = None

# ─── SAVE / LOG FUNCTIONS ───────────────────────────────────────────────────
def save_training_data():
    pd.DataFrame(feature_vectors).reindex(columns=FEATURE_COLUMNS, fill_value=0).to_csv(
        TRAINING_DATA_FILE, index=False
    )

def save_detection_row(feature_dict, prediction, score, reasons):
    global detection_rows
    detection_columns = ["row", "timestamp", "prediction", "score", "reasons"] + FEATURE_COLUMNS
    detection_rows += 1
    row = {
        "row": detection_rows,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prediction": "anomaly" if prediction == -1 else "normal",
        "score": score,
        "reasons": ", ".join(reasons),
    }
    row.update(feature_dict)
    pd.DataFrame([row]).to_csv(
        DETECTION_DATA_FILE,
        columns=detection_columns,
        mode="a",
        header=not os.path.exists(DETECTION_DATA_FILE),
        index=False,
    )
    return detection_rows

# ─── KEYBOARD CALLBACKS ─────────────────────────────────────────────────────
def on_key_press(key):
    global data_updated
    t = time.time()
    key_events.append((t, 'press', key))
    pressed_keys[key] = t
    while key_events and key_events[0][0] < t - EVENT_WINDOW_SECONDS:
        key_events.popleft()
    data_updated = True

def on_key_release(key):
    global data_updated
    t = time.time()
    key_events.append((t, 'release', key))
    while key_events and key_events[0][0] < t - EVENT_WINDOW_SECONDS:
        key_events.popleft()
    if key in pressed_keys:
        del pressed_keys[key]
    data_updated = True

# ─── FEATURE EXTRACTION ─────────────────────────────────────────────────────
def get_key_category(key):
    """Buckets a key into one of the categories tracked in category_counts.
    Anything not explicitly matched (punctuation, delete, tab, arrows,
    function keys...) falls through to 'other'."""
    char = getattr(key, "char", None)
    if char is not None:
        if char.isalpha():
            return "letter"
        if char.isdigit():
            return "digit"
        return "other"

    if key == keyboard.Key.space:
        return "space"
    if key == keyboard.Key.enter:
        return "enter"
    if key == keyboard.Key.backspace:
        return "backspace"

    key_name = str(key).split(".")[-1].lower()
    if key_name.startswith(("shift", "ctrl", "alt")):
        return "modifier"
    return "other"

def compute_entropy(items):
    """Shannon entropy (bits) of a sequence — low = repetitive key choices,
    high = varied. A personal 'which keys do I favor' fingerprint."""
    if len(items) < 2:
        return 0.0
    counts = Counter(items)
    n = len(items)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())

def extract_digraphs(events):
    """
    Walk the raw event list and return [(digraph_str, press_to_press_interval), ...]
    for consecutive LETTER-key presses only. Any non-letter key (space, digit,
    backspace, punctuation, modifier...) breaks the chain, since digraph timing
    is only meaningful between two actual letters typed back to back.
    """
    digraphs = []
    last_letter = None
    last_press_time = None

    for t, etype, key in events:
        if etype != 'press':
            continue
        char = getattr(key, "char", None)
        if char is None or not char.isalpha():
            last_letter = None
            last_press_time = None
            continue
        char = char.lower()
        if last_letter is not None:
            digraphs.append((last_letter + char, t - last_press_time))
        last_letter = char
        last_press_time = t

    return digraphs

def compute_digraph_deviation(digraphs, profile):
    """
    Compare this window's digraph timings against the personal digraph profile.
    Returns (mean_abs_zscore, max_abs_zscore) across digraphs that exist in the
    profile. If nothing matches (or profile is empty), returns (0.0, 0.0).
    """
    if not digraphs or not profile:
        return 0.0, 0.0

    devs = []
    for dg, interval in digraphs:
        prof = profile.get(dg)
        if prof and prof['std'] > 0:
            devs.append(abs((interval - prof['mean']) / prof['std']))

    if not devs:
        return 0.0, 0.0

    return float(np.mean(devs)), float(max(devs))

def extract_features_from_raw():

    events = list(key_events)
    if len(events) < MIN_FEATURE_EVENTS:
        return None, None

    press_times = []
    press_chars = []          # key identity per press, used for entropy
    dwell_times = []
    flight_times = []
    last_release_time = None
    pending_press = {}        # key → press_time (for dwell)
    category_counts = {
        "letter": 0, "digit": 0, "space": 0,
        "backspace": 0, "enter": 0, "modifier": 0, "other": 0,
    }

    for t, etype, key in events:
        if etype == 'press':
            pending_press[key] = t
            press_times.append(t)
            char = getattr(key, "char", None)
            press_chars.append(char.lower() if char else str(key))

            cat = get_key_category(key)
            category_counts[cat if cat in category_counts else 'other'] += 1

            if last_release_time is not None:
                flight_times.append(t - last_release_time)
                last_release_time = None
        else:  # release
            if key in pending_press:
                dwell_times.append(t - pending_press.pop(key))
            last_release_time = t

    if len(press_times) < 2:
        return None, None

    # 1. Dwell statistics
    dwell_mean = np.mean(dwell_times) if dwell_times else 0.0
    dwell_std = np.std(dwell_times) if dwell_times else 0.0
    dwell_median = np.median(dwell_times) if dwell_times else 0.0

    # 2. Flight statistics
    flight_mean = np.mean(flight_times) if flight_times else 0.0
    flight_std = np.std(flight_times) if flight_times else 0.0
    flight_median = np.median(flight_times) if flight_times else 0.0

    # 3. Press-to-press interval statistics
    pp_intervals = [press_times[i] - press_times[i - 1] for i in range(1, len(press_times))]
    pp_mean = np.mean(pp_intervals) if pp_intervals else 0.0
    pp_std = np.std(pp_intervals) if pp_intervals else 0.0
    pp_median = np.median(pp_intervals) if pp_intervals else 0.0

    # 4. Key type ratios
    total_presses = len(press_times)
    r_letter = category_counts['letter'] / total_presses
    r_digit = category_counts['digit'] / total_presses
    r_space = category_counts['space'] / total_presses
    r_backspace = category_counts['backspace'] / total_presses
    r_enter = category_counts['enter'] / total_presses
    r_modifier = category_counts['modifier'] / total_presses
    r_other = category_counts['other'] / total_presses

    # 5. Typing rate (keys per second)
    duration = press_times[-1] - press_times[0]
    typing_rate = total_presses / duration if duration > 0 else 0.0

    # 6. Key entropy — how varied vs. repetitive the keys pressed are
    key_entropy = compute_entropy(press_chars)

    # 7. Pauses — reuses pp_intervals, no extra pass over events needed
    pause_gaps = [g for g in pp_intervals if g > PAUSE_THRESHOLD]
    pause_ratio = len(pause_gaps) / len(pp_intervals) if pp_intervals else 0.0
    pause_mean = float(np.mean(pause_gaps)) if pause_gaps else 0.0

    # Build feature dict (digraph_dev_mean / digraph_dev_max are added later by
    # the caller, once a digraph profile exists — see main_loop)
    features = {
        'dwell_mean': dwell_mean,
        'dwell_std': dwell_std,
        'dwell_median': dwell_median,
        'flight_mean': flight_mean,
        'flight_std': flight_std,
        'flight_median': flight_median,
        'pp_mean': pp_mean,
        'pp_std': pp_std,
        'pp_median': pp_median,
        'r_letter': r_letter,
        'r_digit': r_digit,
        'r_space': r_space,
        'r_backspace': r_backspace,
        'r_enter': r_enter,
        'r_modifier': r_modifier,
        'r_other': r_other,
        'typing_rate': typing_rate,
        'key_entropy': key_entropy,
        'pause_ratio': pause_ratio,
        'pause_mean': pause_mean,
    }

    digraphs = extract_digraphs(events)
    return features, digraphs

# ─── DIGRAPH PROFILE ─────────────────────────────────────────────────────────
def finalize_digraph_profile():
    """
    Called once, right before the first OCSVM training run. Builds the personal
    digraph timing profile from every digraph seen across the whole training
    period, then backfills digraph_dev_mean / digraph_dev_max into every stored
    training window (comparing each window's own digraphs against the finished
    profile) so the OCSVM gets to train on the full feature vector.
    """
    global digraph_profile

    profile = {}
    for dg, intervals in digraph_raw_accum.items():
        if len(intervals) >= MIN_DIGRAPH_SAMPLES:
            profile[dg] = {
                'mean': float(np.mean(intervals)),
                'std': float(np.std(intervals)),
            }
    digraph_profile = profile
    joblib.dump(digraph_profile, DIGRAPH_PROFILE_FILE)
    print(f"Built digraph profile: {len(digraph_profile)} digraphs "
          f"(min {MIN_DIGRAPH_SAMPLES} samples each).")

    for i, wd in enumerate(window_digraphs_list):
        dev_mean, dev_max = compute_digraph_deviation(wd, digraph_profile)
        feature_vectors[i]['digraph_dev_mean'] = dev_mean
        feature_vectors[i]['digraph_dev_max'] = dev_max

    save_training_data()

# ─── TRAINING ────────────────────────────────────────────────────────────────
def train_ocsvm():

    global model, scaler, threshold, baseline_ready, feature_stats

    if len(feature_vectors) < MIN_TRAINING_WINDOWS:
        print(f"Not enough windows ({len(feature_vectors)} < {MIN_TRAINING_WINDOWS}).")
        return

    print(f"\nTraining OCSVM on {len(feature_vectors)} windows...")
    X = pd.DataFrame(feature_vectors).reindex(columns=FEATURE_COLUMNS, fill_value=0).values

    scaler.fit(X)
    X_scaled = scaler.transform(X)

    best_fpr = float('inf')
    best_model = None
    best_nu = None
    best_gamma = None

    split_idx = int(0.8 * len(X_scaled))
    X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]

    for nu in NU_VALUES:
        for gamma in GAMMA_VALUES:
            clf = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)
            clf.fit(X_train)
            val_scores = clf.decision_function(X_val)
            th = np.percentile(val_scores, THRESHOLD_PERCENTILE)
            fpr = np.sum(val_scores < th) / len(val_scores)
            print(f"  nu={nu:.3f}, gamma={gamma}: val FPR={fpr:.4f}")
            if fpr < best_fpr:
                best_fpr = fpr
                best_model = clf
                best_nu = nu
                best_gamma = gamma

    if best_model is None:
        best_model = OneClassSVM(kernel='rbf', nu=0.01, gamma='scale')
        best_model.fit(X_scaled)
    else:
        best_model = OneClassSVM(kernel='rbf', nu=best_nu, gamma=best_gamma)
        best_model.fit(X_scaled)

    model = best_model

    scores = model.decision_function(X_scaled)
    threshold = float(np.percentile(scores, THRESHOLD_PERCENTILE))
    print(f"Threshold set at {THRESHOLD_PERCENTILE}th percentile: {threshold:.6f}")

    df_temp = pd.DataFrame(feature_vectors).reindex(columns=FEATURE_COLUMNS, fill_value=0.0)
    for col in df_temp.columns:
        feature_stats[col] = {'mean': df_temp[col].mean(), 'std': df_temp[col].std()}

    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(threshold, THRESHOLD_FILE)

    baseline_ready = True
    print(f"Model saved to {MODEL_FILE}, scaler to {SCALER_FILE}, threshold to {THRESHOLD_FILE}.")
    print("Anomaly detection active.\n")

# ─── DETECTION ──────────────────────────────────────────────────────────────
def detect_anomaly(feature_dict):

    global baseline_ready, model, scaler, threshold

    if not baseline_ready:
        return

    X = pd.DataFrame([feature_dict]).reindex(columns=FEATURE_COLUMNS, fill_value=0)
    X_scaled = scaler.transform(X)
    score = float(model.decision_function(X_scaled)[0])

    reasons = []
    for feat, value in feature_dict.items():
        stats = feature_stats.get(feat)
        if stats and stats['std'] > 0:
            z = (value - stats['mean']) / stats['std']
            if abs(z) > 2.0:
                reasons.append(FEATURE_READABLE.get(feat, feat))

    prediction = -1 if threshold is not None and score < threshold else 1
    save_detection_row(feature_dict, prediction, score, reasons)

    if prediction == -1:
        reason_text = ", ".join(reasons) if reasons else "unusual pattern"
        print(f"[{time.strftime('%H:%M:%S')}] ANOMALY (Score: {score:.1%}) - Reason: {reason_text}")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] NORMAL (Score: {score:.1%})")

# ─── MAIN LOOP ──────────────────────────────────────────────────────────────
def main_loop():
    global feature_vectors, window_digraphs_list, data_updated, baseline_ready

    while True:
        time.sleep(IDLE_CHECK_INTERVAL)

        if not data_updated:
            continue

        new_features, window_digraphs = extract_features_from_raw()
        if new_features is not None:
            was_training = not baseline_ready

            if was_training:
                feature_vectors.append(new_features)
                window_digraphs_list.append(window_digraphs)

                if len(feature_vectors) > 5000:
                    feature_vectors = feature_vectors[-5000:]
                    window_digraphs_list = window_digraphs_list[-5000:]
                save_training_data()

                for dg, interval in window_digraphs:
                    digraph_raw_accum[dg].append(interval)

                if len(feature_vectors) >= MIN_TRAINING_WINDOWS and not baseline_ready:
                    finalize_digraph_profile()
                    train_ocsvm()
            else:
                dev_mean, dev_max = compute_digraph_deviation(window_digraphs, digraph_profile)
                new_features["digraph_dev_mean"] = dev_mean
                new_features["digraph_dev_max"] = dev_max
                detect_anomaly(new_features)

        data_updated = False

# ─── STARTUP ────────────────────────────────────────────────────────────────
keyboard.Listener(on_press=on_key_press, on_release=on_key_release).start()
print("Keystroke dynamics OCSVM anomaly detector running...")
threading.Thread(target=main_loop, daemon=True).start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nExiting.")