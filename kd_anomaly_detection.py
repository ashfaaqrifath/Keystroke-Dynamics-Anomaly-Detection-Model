import time
import threading
import os
import sys
import pandas as pd
import numpy as np
from pynput import keyboard
from sklearn.ensemble import IsolationForest
import joblib
from collections import deque


WINDOW_SIZE = 300               # number of keystroke events kept in memory
MIN_TRAINING_WINDOWS = 1000       # How many feature windows before training
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(BASE_DIR, "kd_baseline_model.pkl")
TRAINING_DATA_FILE = os.path.join(BASE_DIR, "keystroke_dynamics_training.csv")
DETECTION_DATA_FILE = os.path.join(BASE_DIR, "anomaly_detection.csv")

IDLE_CHECK_INTERVAL = 4
MIN_FEATURE_EVENTS = 5
EVENT_WINDOW_SECONDS = 10
PAUSE_THRESHOLD_SECONDS = 1.0

FEATURE_COLUMNS = [
    "inter_press_mean",
    "inter_press_std",
    "inter_press_min",
    "inter_press_max",
    "inter_press_median",
    "dwell_mean",
    "dwell_std",
    "dwell_median",
    "flight_mean",
    "flight_std",
    "flight_median",
    "event_count",
    "press_count",
    "release_count",
    "key_press_rate",
    "pause_count",
    "pause_mean",
    "pause_max",
    "max_same_key_streak",
    "most_common_key_ratio",
    "unique_key_count",
    "char_key_count",
    "letter_key_count",
    "digit_key_count",
    "space_count",
    "enter_count",
    "backspace_count",
    "delete_count",
    "tab_count",
    "arrow_key_count",
    "modifier_key_count",
    "function_key_count",
    "special_key_count",
    "correction_key_ratio",
]

FEATURE_READABLE = {
    "inter_press_mean": "Typing speed average changed",
    "inter_press_std": "Typing speed consistency changed",
    "inter_press_min": "Fastest typing speed changed",
    "inter_press_max": "Slowest typing speed changed",
    "inter_press_median": "Typical typing speed changed",
    "dwell_mean": "Key hold duration average changed",
    "dwell_std": "Key hold consistency changed",
    "dwell_median": "Typical key hold duration changed",
    "flight_mean": "Time between keys average changed",
    "flight_std": "Time between keys consistency changed",
    "flight_median": "Typical time between keys changed",
    "event_count": "Number of key events changed",
    "press_count": "Number of key presses changed",
    "release_count": "Number of key releases changed",
    "key_press_rate": "Overall typing rate changed",
    "pause_count": "Number of typing pauses changed",
    "pause_mean": "Average pause duration changed",
    "pause_max": "Longest pause duration changed",
    "max_same_key_streak": "Repeated key usage changed",
    "most_common_key_ratio": "Key repetition ratio changed",
    "unique_key_count": "Vocabulary/Key variety changed",
    "char_key_count": "Character key usage changed",
    "letter_key_count": "Letter key usage changed",
    "digit_key_count": "Number key usage changed",
    "space_count": "Spacebar usage changed",
    "enter_count": "Enter key usage changed",
    "backspace_count": "Backspace usage changed",
    "delete_count": "Delete key usage changed",
    "tab_count": "Tab key usage changed",
    "arrow_key_count": "Arrow key usage changed",
    "modifier_key_count": "Modifier key (Ctrl/Shift/Alt) usage changed",
    "function_key_count": "Function key usage changed",
    "special_key_count": "Special key usage changed",
    "correction_key_ratio": "Typo correction frequency changed",
}


#timestamp, event_type, key
key_events = deque()
#for dwell calc
pressed_keys = {}

data_updated = False


feature_vectors = []
baseline_ready = False
feature_stats = {}   # Will store mean/std of each feature from training data
detection_rows = 0


if os.path.exists(TRAINING_DATA_FILE):
    dataframe = pd.read_csv(TRAINING_DATA_FILE)
    dataframe = dataframe.reindex(columns=FEATURE_COLUMNS, fill_value=0.0)
    feature_vectors = dataframe.to_dict('records')
    print(f"Loaded {len(feature_vectors)} training feature windows.")

if os.path.exists(DETECTION_DATA_FILE):
    detection_rows = len(pd.read_csv(DETECTION_DATA_FILE))
    print(f"Loaded {detection_rows} past keystroke detection data.")
    
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
    baseline_ready = True
    print("Loaded existing baseline model.")
    
    if baseline_ready and feature_vectors:
        df = pd.DataFrame(feature_vectors)
        for col in df.columns:
            feature_stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std()
            }
else:
    print(f"Will train baseline after {MIN_TRAINING_WINDOWS} active windows")
    print(f"(each window = {IDLE_CHECK_INTERVAL}s of typing).")
    model = IsolationForest(contamination=0.05, random_state=42)


def save_training_data():
    pd.DataFrame(feature_vectors).reindex(columns=FEATURE_COLUMNS, fill_value=0).to_csv(
        TRAINING_DATA_FILE,
        index=False,
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


def get_key_category(key):
    char = getattr(key, "char", None)
    if char is not None:
        if char.isalpha():
            return "letter"
        if char.isdigit():
            return "digit"
        return "char"

    key_name = str(key).split(".")[-1].lower()

    if key == keyboard.Key.space:
        return "space"
    if key == keyboard.Key.enter:
        return "enter"
    if key == keyboard.Key.backspace:
        return "backspace"
    if key == keyboard.Key.delete:
        return "delete"
    if key == keyboard.Key.tab:
        return "tab"
    if key in {keyboard.Key.up, keyboard.Key.down, keyboard.Key.left, keyboard.Key.right}:
        return "arrow"
    if key_name.startswith(("shift", "ctrl", "alt")):
        return "modifier"
    if key_name.startswith("f") and key_name[1:].isdigit():
        return "function"

    return "special"


def extract_features_from_raw():
    features = {column: 0 for column in FEATURE_COLUMNS}
    events = list(key_events)
    if len(events) < MIN_FEATURE_EVENTS:
        return None

    press_times = []
    release_times = []
    pressed_key_sequence = []
    dwell_times = []
    flight_times = []
    last_release_time = None
    pending_press = {}
    category_counts = {
        "char": 0,
        "letter": 0,
        "digit": 0,
        "space": 0,
        "enter": 0,
        "backspace": 0,
        "delete": 0,
        "tab": 0,
        "arrow": 0,
        "modifier": 0,
        "function": 0,
        "special": 0,
    }

    for evt in events:
        t, etype, key = evt
        if etype == 'press':
            pending_press[key] = t
            press_times.append(t)
            pressed_key_sequence.append(key)
            category_counts[get_key_category(key)] += 1
            if last_release_time is not None:
                flight_times.append(t - last_release_time)
                last_release_time = None
        else:
            release_times.append(t)
            if key in pending_press:
                dwell = t - pending_press.pop(key)
                dwell_times.append(dwell)
            last_release_time = t

    features["event_count"] = len(events)
    features["press_count"] = len(press_times)
    features["release_count"] = len(release_times)

    window_duration = max(events[-1][0] - events[0][0], 0.001)
    features["key_press_rate"] = len(press_times) / window_duration

    #Inter press intervals
    if len(press_times) >= 2:
        inter_press = np.diff(press_times[-100:])   #last 100 presses
        features['inter_press_mean'] = np.mean(inter_press)
        features['inter_press_std'] = np.std(inter_press)
        features['inter_press_min'] = np.min(inter_press)
        features['inter_press_max'] = np.max(inter_press)
        features['inter_press_median'] = np.median(inter_press)

        pauses = [interval for interval in inter_press if interval >= PAUSE_THRESHOLD_SECONDS]
        features['pause_count'] = len(pauses)
        if pauses:
            features['pause_mean'] = np.mean(pauses)
            features['pause_max'] = np.max(pauses)
    else:
        features['inter_press_mean'] = features['inter_press_std'] = 0

    #Dwell times
    if len(dwell_times) >= MIN_FEATURE_EVENTS:
        features['dwell_mean'] = np.mean(dwell_times)
        features['dwell_std'] = np.std(dwell_times)
        features['dwell_median'] = np.median(dwell_times)
    else:
        features['dwell_mean'] = features['dwell_std'] = 0

    #Flight times
    if len(flight_times) >= MIN_FEATURE_EVENTS:
        features['flight_mean'] = np.mean(flight_times)
        features['flight_std'] = np.std(flight_times)
        features['flight_median'] = np.median(flight_times)
    else:
        features['flight_mean'] = features['flight_std'] = 0

    if pressed_key_sequence:
        max_streak = 1
        current_streak = 1
        key_counts = {}

        for previous_key, current_key in zip(pressed_key_sequence, pressed_key_sequence[1:]):
            if current_key == previous_key:
                current_streak += 1
            else:
                max_streak = max(max_streak, current_streak)
                current_streak = 1

        max_streak = max(max_streak, current_streak)

        for key in pressed_key_sequence:
            key_counts[key] = key_counts.get(key, 0) + 1

        features["max_same_key_streak"] = max_streak
        features["most_common_key_ratio"] = max(key_counts.values()) / len(pressed_key_sequence)
        features["unique_key_count"] = len(key_counts)

    features["char_key_count"] = category_counts["char"]
    features["letter_key_count"] = category_counts["letter"]
    features["digit_key_count"] = category_counts["digit"]
    features["space_count"] = category_counts["space"]
    features["enter_count"] = category_counts["enter"]
    features["backspace_count"] = category_counts["backspace"]
    features["delete_count"] = category_counts["delete"]
    features["tab_count"] = category_counts["tab"]
    features["arrow_key_count"] = category_counts["arrow"]
    features["modifier_key_count"] = category_counts["modifier"]
    features["function_key_count"] = category_counts["function"]
    features["special_key_count"] = category_counts["special"]

    correction_count = category_counts["backspace"] + category_counts["delete"]
    if press_times:
        features["correction_key_ratio"] = correction_count / len(press_times)

    return features


def train_baseline():
    global baseline_ready, feature_stats
    if len(feature_vectors) >= MIN_TRAINING_WINDOWS and not baseline_ready:
        df = pd.DataFrame(feature_vectors).reindex(columns=FEATURE_COLUMNS, fill_value=0)
        model.fit(df)
        joblib.dump(model, MODEL_FILE)

        #feature means, std
        for col in df.columns:
            feature_stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std()
            }

        baseline_ready = True
        print(f"[{time.strftime('%H:%M:%S')}] BASELINE trained on {len(feature_vectors)} windows.")
        print("Anomaly detection active")



def detect_anomaly(feature_dict):
    if not baseline_ready:
        return

    X = pd.DataFrame([feature_dict]).reindex(columns=FEATURE_COLUMNS, fill_value=0)
    pred = model.predict(X)[0]          # -1 = anomaly, 1 = normal
    score = model.decision_function(X)[0]

    
    reasons = []
    for feat, value in feature_dict.items():
        if feat in feature_stats:
            mean = feature_stats[feat]['mean']
            std = feature_stats[feat]['std']
            if std > 0:
                z = (value - mean) / std
                if abs(z) > 2.0:
                    name = FEATURE_READABLE.get(feat, feat)
                    reasons.append(name)

    row_number = save_detection_row(feature_dict, pred, score, reasons)

    if pred == -1:
        reason_text = ", ".join(reasons) if reasons else "typing pattern drift"
        msg = f"[{time.strftime('%H:%M:%S')}] ANOMALY (Score: {score:.1%}) - Reason: {reason_text}"
        print(msg)

    if pred == 1:
        print(f"[{time.strftime('%H:%M:%S')}] NORMAL (Score: {score:.1%})")


def main_loop():
    global feature_vectors, data_updated

    while True:
        time.sleep(IDLE_CHECK_INTERVAL)

        if not data_updated:
            continue

        new_features = extract_features_from_raw()
        if new_features:
            was_training_window = not baseline_ready

            if was_training_window:
                feature_vectors.append(new_features)

                #recent 5000 windows
                if len(feature_vectors) > 5000:
                    feature_vectors = feature_vectors[-5000:]

                save_training_data()
                train_baseline()

            if baseline_ready and not was_training_window:
                detect_anomaly(new_features)

        data_updated = False


keyboard.Listener(on_press=on_key_press, on_release=on_key_release).start()

print("Keystroke dynamics anomaly detection model running...")
threading.Thread(target=main_loop, daemon=True).start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nExiting.")
