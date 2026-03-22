import time
import threading
import os
import pandas as pd
import numpy as np
from pynput import keyboard
from sklearn.ensemble import IsolationForest
import joblib

# ===========================
#  CONFIGURATION
# ===========================
WINDOW_SIZE = 300               # Max number of keystroke events kept in memory
MIN_WINDOWS_TO_TRAIN = 50       # How many feature windows are needed before training
MODEL_FILE = "keystroke_model.pkl"
DATA_FILE = "typing_features.csv"
IDLE_CHECK_INTERVAL = 5
MIN_EVENTS_FOR_FEATURE = 5

# ===========================
#  RAW DATA STORAGE
# ===========================
# Each event is a tuple: (timestamp, event_type, key)
# event_type is either 'press' or 'release'
key_events = []

# Dictionary to store press time of currently held keys (for dwell calculation)
pressed_keys = {}

# Flag: becomes True whenever a new keyboard event arrives
data_updated = False

# ===========================
#  AGGREGATED FEATURE STORAGE
# ===========================
feature_vectors = []        # List of dicts, each dict = one window's features
baseline_ready = False       # True after model has been trained

# ===========================
#  LOAD PREVIOUS DATA / MODEL
# ===========================
if os.path.exists(DATA_FILE) == True:
    feature_vectors = pd.read_csv(DATA_FILE).to_dict('records')
    print(f"Loaded {len(feature_vectors) + 1} past feature windows.")

if os.path.exists(MODEL_FILE) == True:
    model = joblib.load(MODEL_FILE)
    baseline_ready = True
    print("Loaded existing baseline model.")

if os.path.exists(DATA_FILE) == False or os.path.exists(MODEL_FILE) == False:
    print(f"Will train baseline after {MIN_WINDOWS_TO_TRAIN} active windows (each window = {IDLE_CHECK_INTERVAL}s of typing).")
    model = IsolationForest(contamination=0.05, random_state=42)

# else:
#     model = IsolationForest(contamination=0.05, random_state=42)
    


# ===========================
#  KEYBOARD EVENT HANDLERS
# ===========================
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
    # Remove the key from the pressed dict (it should be there)
    if key in pressed_keys:
        del pressed_keys[key]
    data_updated = True

# ===========================
#  FEATURE EXTRACTION
# ===========================
def extract_features_from_raw():
    
    features = {}

    press_times = []      # timestamps of all key presses
    dwell_times = []       # durations keys were held
    flight_times = []      # time between a release and the next press

    last_release_time = None
    pending_press = {}     # key -> press_time (temporary during replay)

    for evt in key_events:
        t, etype, key = evt
        if etype == 'press':
            pending_press[key] = t
            press_times.append(t)
            # If there was a previous release, compute flight time
            if last_release_time is not None:
                flight_times.append(t - last_release_time)
                last_release_time = None   # consume it
        else:  # release
            if key in pending_press:
                dwell = t - pending_press.pop(key)
                dwell_times.append(dwell)
            last_release_time = t

    # --- Inter-press intervals ---
    if len(press_times) >= 2:
        inter_press = np.diff(press_times[-100:])   # last 100 to keep recent
        features['inter_press_mean'] = np.mean(inter_press)
        features['inter_press_std'] = np.std(inter_press)
    else:
        features['inter_press_mean'] = features['inter_press_std'] = 0

    # --- Dwell times ---
    if len(dwell_times) >= MIN_EVENTS_FOR_FEATURE:
        features['dwell_mean'] = np.mean(dwell_times)
        features['dwell_std'] = np.std(dwell_times)
    else:
        features['dwell_mean'] = features['dwell_std'] = 0

    # --- Flight times ---
    if len(flight_times) >= MIN_EVENTS_FOR_FEATURE:
        features['flight_mean'] = np.mean(flight_times)
        features['flight_std'] = np.std(flight_times)
    else:
        features['flight_mean'] = features['flight_std'] = 0

    return features

# ===========================
#  TRAINING
# ===========================
def train_baseline():
    global baseline_ready
    if len(feature_vectors) >= MIN_WINDOWS_TO_TRAIN and not baseline_ready:
        df = pd.DataFrame(feature_vectors)
        model.fit(df)
        joblib.dump(model, MODEL_FILE)
        baseline_ready = True
        print(f"[{time.strftime('%H:%M:%S')}] ✅ Baseline trained on {len(feature_vectors)} windows.")
        print("Anomaly detection active")

# ===========================
#  ANOMALY DETECTION
# ===========================
def detect_anomaly(feature_dict):
    global feature_vectors
    if not baseline_ready:
        return
    X = pd.DataFrame([feature_dict])
    pred = model.predict(X)[0]          # -1 = anomaly, 1 = normal

    score = model.decision_function(X)[0]
        
    row_number = len(feature_vectors) + 1

    if pred == -1:
        print(f"[{time.strftime('%H:%M:%S')}] ⚠️ ANOMALY DETECTED (score: {score:.1%}) CSV row {row_number}")

    elif pred == 1:
        print(f"[{time.strftime('%H:%M:%S')}] ✅ Normal behavior (score: {score:.1%}) CSV row {row_number}")

# ===========================
#  MAIN LOOP
# ===========================
def main_loop():
    global feature_vectors, data_updated

    while True:
        time.sleep(IDLE_CHECK_INTERVAL)

        # If no new keystrokes since last check, skip feature extraction
        if not data_updated:
            continue

        # Compute features from current raw data
        new_features = extract_features_from_raw()
        if new_features:
            feature_vectors.append(new_features)

            # Keep only the most recent 5000 windows (optional, saves memory)
            if len(feature_vectors) > 5000:
                feature_vectors = feature_vectors[-5000:]

            # Save all features to CSV (overwrites each time, but keeps full history)
            pd.DataFrame(feature_vectors).to_csv(DATA_FILE, index=False)

            
            train_baseline()

            
            detect_anomaly(new_features)

        
        data_updated = False


# ===========================
#  START LISTENERS & MAIN LOOP
# ===========================
keyboard.Listener(on_press=on_key_press, on_release=on_key_release).start()

print("Keystroke dynamics anomaly detection model running...")

threading.Thread(target=main_loop, daemon=True).start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nExiting.")