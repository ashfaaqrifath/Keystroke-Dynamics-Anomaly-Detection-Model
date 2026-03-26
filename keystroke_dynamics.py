import time
import threading
import os
import pandas as pd
import numpy as np
from pynput import keyboard
from sklearn.ensemble import IsolationForest
import joblib


WINDOW_SIZE = 300               # Max number of keystroke events kept in memory
MIN_WINDOWS_TO_TRAIN = 50       # How many feature windows are needed before training
MODEL_FILE = "keystroke_model.pkl"
DATA_FILE = "typing_features.csv"
IDLE_CHECK_INTERVAL = 5
MIN_EVENTS_FOR_FEATURE = 5


#timestamp, event_type, key
key_events = []
#for dwell calc
pressed_keys = {}

data_updated = False


feature_vectors = []
baseline_ready = False
feature_stats = {}   # Will store mean/std of each feature from training data


if os.path.exists(DATA_FILE):
    feature_vectors = pd.read_csv(DATA_FILE).to_dict('records')
    print(f"Loaded {len(feature_vectors)} past feature windows.")

if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
    baseline_ready = True
    print("Loaded existing baseline model.")
    
    if feature_vectors:
        df = pd.DataFrame(feature_vectors)
        for col in df.columns:
            feature_stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std()
            }
else:
    print(f"Will train baseline after {MIN_WINDOWS_TO_TRAIN} active windows")
    print(f"(each window = {IDLE_CHECK_INTERVAL}s of typing).")
    model = IsolationForest(contamination=0.05, random_state=42)


def on_key_press(key):
    global data_updated
    t = time.time()
    key_events.append((t, 'press', key))
    pressed_keys[key] = t
    if len(key_events) > WINDOW_SIZE:
        key_events.pop(0)
    data_updated = True

def on_key_release(key):
    global data_updated
    t = time.time()
    key_events.append((t, 'release', key))
    if len(key_events) > WINDOW_SIZE:
        key_events.pop(0)
    if key in pressed_keys:
        del pressed_keys[key]
    data_updated = True


def extract_features_from_raw():
    features = {}
    press_times = []
    dwell_times = []
    flight_times = []
    last_release_time = None
    pending_press = {}

    for evt in key_events:
        t, etype, key = evt
        if etype == 'press':
            pending_press[key] = t
            press_times.append(t)
            if last_release_time is not None:
                flight_times.append(t - last_release_time)
                last_release_time = None
        else:
            if key in pending_press:
                dwell = t - pending_press.pop(key)
                dwell_times.append(dwell)
            last_release_time = t

    #Inter press intervals
    if len(press_times) >= 2:
        inter_press = np.diff(press_times[-100:])   #last 100 presses
        features['inter_press_mean'] = np.mean(inter_press)
        features['inter_press_std'] = np.std(inter_press)
    else:
        features['inter_press_mean'] = features['inter_press_std'] = 0

    #Dwell times
    if len(dwell_times) >= MIN_EVENTS_FOR_FEATURE:
        features['dwell_mean'] = np.mean(dwell_times)
        features['dwell_std'] = np.std(dwell_times)
    else:
        features['dwell_mean'] = features['dwell_std'] = 0

    #Flight times
    if len(flight_times) >= MIN_EVENTS_FOR_FEATURE:
        features['flight_mean'] = np.mean(flight_times)
        features['flight_std'] = np.std(flight_times)
    else:
        features['flight_mean'] = features['flight_std'] = 0

    return features


def train_baseline():
    global baseline_ready, feature_stats
    if len(feature_vectors) >= MIN_WINDOWS_TO_TRAIN and not baseline_ready:
        df = pd.DataFrame(feature_vectors)
        model.fit(df)
        joblib.dump(model, MODEL_FILE)

        #feature means, standard deviation
        for col in df.columns:
            feature_stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std()
            }

        baseline_ready = True
        print(f"[{time.strftime('%H:%M:%S')}] ✅ Baseline trained on {len(feature_vectors)} windows.")
        print("Anomaly detection active")



def detect_anomaly(feature_dict):
    global feature_vectors
    if not baseline_ready:
        return

    X = pd.DataFrame([feature_dict])
    pred = model.predict(X)[0]          # -1 = anomaly, 1 = normal
    score = model.decision_function(X)[0]
    row_number = len(feature_vectors) + 1

    
    reasons = []
    for feat, value in feature_dict.items():
        if feat in feature_stats:
            mean = feature_stats[feat]['mean']
            std = feature_stats[feat]['std']
            if std > 0:
                z = (value - mean) / std
                if abs(z) > 2.0:
                    reasons.append(f"{feat}: {value:.4f}")

    if pred == -1:
        print(f"[{time.strftime('%H:%M:%S')}] ⚠️ ANOMALY (score: {score:.1%}) CSV row {row_number} ({', '.join(reasons)})")
        
    if pred == 1:
        print(f"[{time.strftime('%H:%M:%S')}] ✅ NORMAL (score: {score:.1%}) CSV row {row_number}")


def main_loop():
    global feature_vectors, data_updated, last_train_time

    last_train_time = time.time() - IDLE_CHECK_INTERVAL

    while True:

        if not data_updated:
            continue

        new_features = extract_features_from_raw()
        if new_features:
            feature_vectors.append(new_features)

            #recent 5000 windows
            if len(feature_vectors) > 5000:
                feature_vectors = feature_vectors[-5000:]

            pd.DataFrame(feature_vectors).to_csv(DATA_FILE, index=False)

            detect_anomaly(new_features)

            now = time.time()
            if now - last_train_time >= IDLE_CHECK_INTERVAL:
                train_baseline()
                last_train_time = now

        data_updated = False


keyboard.Listener(on_press=on_key_press, on_release=on_key_release).start()

print("Keystroke dynamics anomaly detection model running...")
threading.Thread(target=main_loop, daemon=True).start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nExiting.")