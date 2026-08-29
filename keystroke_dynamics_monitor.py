import time
import threading
import os
import sys
import pandas as pd
import numpy as np
from pynput import keyboard
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import joblib
from collections import deque

                                                                                
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

MODEL_FILE = os.path.join(BASE_DIR, "keystroke_dynamics_model.pkl")
SCALER_FILE = os.path.join(BASE_DIR, "ocsvm_scaler.pkl")
THRESHOLD_FILE = os.path.join(BASE_DIR, "ocsvm_threshold.pkl")
TRAINING_DATA_FILE = os.path.join(BASE_DIR, "keystroke_dynamics_training.csv")
DETECTION_DATA_FILE = os.path.join(BASE_DIR, "keystroke_detections.csv")


IDLE_CHECK_INTERVAL = 4          
MIN_FEATURE_EVENTS = 5           
EVENT_WINDOW_SECONDS = 10        
MIN_TRAINING_WINDOWS = 1000
TRAINING_TARGET_ROWS = int(os.getenv("SENTRY_TRAINING_TARGET_ROWS", MIN_TRAINING_WINDOWS))

                                      
NU_VALUES = [0.01, 0.02, 0.05, 0.1]
GAMMA_VALUES = ['scale', 'auto']
THRESHOLD_PERCENTILE = 5                                                         


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
}

                                                                               
key_events = deque()                                  
pressed_keys = {}                                           
data_updated = False

feature_vectors = []                                              
model = None
scaler = None
threshold = None
baseline_ready = False
feature_stats = {}                                                          

detection_rows = 0

                                                                              
if os.path.exists(TRAINING_DATA_FILE):
    df = pd.read_csv(TRAINING_DATA_FILE)
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0.0)
    feature_vectors = df.to_dict('records')
    print(f"Loaded {len(feature_vectors)} training rows")

if os.path.exists(DETECTION_DATA_FILE):
    detection_rows = len(pd.read_csv(DETECTION_DATA_FILE))
    

if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE) and os.path.exists(THRESHOLD_FILE):
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    threshold = joblib.load(THRESHOLD_FILE)
    baseline_ready = True
    print("Loaded existing model")
                                    
    if feature_vectors:
        df_temp = pd.DataFrame(feature_vectors)
        for col in df_temp.columns:
            feature_stats[col] = {
                'mean': df_temp[col].mean(),
                'std': df_temp[col].std()
            }
else:
    print(f"Will train model after {TRAINING_TARGET_ROWS} rows")
    print(f"(Capturing = {EVENT_WINDOW_SECONDS}s of typing).")
    model = OneClassSVM(kernel='rbf', nu=0.01, gamma='scale')
    scaler = StandardScaler()
    threshold = None


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
    return "other"

def extract_features_from_raw():
    
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
        "letter": 0,
        "digit": 0,
        "space": 0,
        "backspace": 0,
        "enter": 0,
        "modifier": 0,
        "other": 0,
    }

    for t, etype, key in events:
        if etype == 'press':
            pending_press[key] = t
            press_times.append(t)
            pressed_key_sequence.append(key)
            cat = get_key_category(key)
                                                                        
            if cat in category_counts:
                category_counts[cat] += 1
            else:
                category_counts['other'] += 1

            if last_release_time is not None:
                flight_times.append(t - last_release_time)
                last_release_time = None
        else:           
            release_times.append(t)
            if key in pending_press:
                dwell = t - pending_press.pop(key)
                dwell_times.append(dwell)
            last_release_time = t

                                                         
    if len(press_times) < 2:
        return None

                         
    dwell_mean = np.mean(dwell_times) if dwell_times else 0.0
    dwell_std  = np.std(dwell_times)  if dwell_times else 0.0
    dwell_median = np.median(dwell_times) if dwell_times else 0.0

                          
    flight_mean = np.mean(flight_times) if flight_times else 0.0
    flight_std  = np.std(flight_times)  if flight_times else 0.0
    flight_median = np.median(flight_times) if flight_times else 0.0

                                           
    pp_intervals = []
    for i in range(1, len(press_times)):
        pp_intervals.append(press_times[i] - press_times[i-1])
    pp_mean = np.mean(pp_intervals) if pp_intervals else 0.0
    pp_std  = np.std(pp_intervals)  if pp_intervals else 0.0
    pp_median = np.median(pp_intervals) if pp_intervals else 0.0

                        
    total_presses = len(press_times)
    r_letter = category_counts.get('letter', 0) / total_presses
    r_digit  = category_counts.get('digit', 0) / total_presses
    r_space  = category_counts.get('space', 0) / total_presses
    r_backspace = category_counts.get('backspace', 0) / total_presses
    r_enter  = category_counts.get('enter', 0) / total_presses
    r_modifier = category_counts.get('modifier', 0) / total_presses
    r_other  = category_counts.get('other', 0) / total_presses

                                      
    if len(press_times) >= 2:
        duration = press_times[-1] - press_times[0]
        typing_rate = total_presses / duration if duration > 0 else 0.0
    else:
        typing_rate = 0.0

                        
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
    }
    return features

                                                                               
def train_ocsvm():

    global model, scaler, threshold, baseline_ready, feature_stats

    if len(feature_vectors) < TRAINING_TARGET_ROWS:
        
        return

    print(f"\nTraining model on {len(feature_vectors)} rows")
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
                                                   
            fp = np.sum(val_scores < th)
            fpr = fp / len(val_scores)
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
    print(f"Threshold: {THRESHOLD_PERCENTILE}th percentile: {threshold:.6f}")


    df_temp = pd.DataFrame(feature_vectors)
    for col in df_temp.columns:
        feature_stats[col] = {
            'mean': df_temp[col].mean(),
            'std': df_temp[col].std()
        }

                    
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(threshold, THRESHOLD_FILE)

    baseline_ready = True
    print("Anomaly detection active.\n")

                                                                            
def detect_anomaly(feature_dict):
    
    global baseline_ready, model, scaler, threshold

    if not baseline_ready:
        return

                                    
    X = pd.DataFrame([feature_dict]).reindex(columns=FEATURE_COLUMNS, fill_value=0)
    X_scaled = scaler.transform(X)
    score = float(model.decision_function(X_scaled)[0])

                                               
    reasons = []
    for feat, value in feature_dict.items():
        if feat in feature_stats:
            mean = feature_stats[feat]['mean']
            std = feature_stats[feat]['std']
            if std > 0:
                z = (value - mean) / std
                if abs(z) > 2.0:
                    reasons.append(FEATURE_READABLE.get(feat, feat))

    prediction = -1 if threshold is not None and score < threshold else 1

                                                      
    save_detection_row(feature_dict, prediction, score, reasons)

    if prediction == -1:
        reason_text = ", ".join(reasons) if reasons else "unusual pattern"
        print(f"[{time.strftime('%H:%M:%S')}] ANOMALY (Score: {score:.1%}) - Reason: {reason_text}")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] NORMAL (Score: {score:.1%})")

                                                                              
def main_loop():
    global feature_vectors, data_updated, baseline_ready

    while True:
        time.sleep(IDLE_CHECK_INTERVAL)

        if not data_updated:
            continue

        new_features = extract_features_from_raw()
        if new_features is not None:
            was_training = not baseline_ready

            if was_training:
                feature_vectors.append(new_features)
                                                                              
                if len(feature_vectors) > 5000:
                    feature_vectors = feature_vectors[-5000:]
                save_training_data()

                                                  
                if len(feature_vectors) >= TRAINING_TARGET_ROWS and not baseline_ready:
                    train_ocsvm()
            else:
                                 
                detect_anomaly(new_features)

        data_updated = False


keyboard.Listener(on_press=on_key_press, on_release=on_key_release).start()
print("Keystroke dynamics anomaly detection active")
threading.Thread(target=main_loop, daemon=True).start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nExit")
