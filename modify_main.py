"""
Brain-Computer Interface MLP Classifier (BrainLink Version)
For EEG signal relaxation/focus/blink state classification
USE RAW DATA AS INPUT
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import os
import glob
import warnings
warnings.filterwarnings('ignore')

from scipy import signal
from scipy.ndimage import median_filter


# Parameter Settings
class Config:
    DATASET_PATH = "bci_dataset_114-2"

    HIDDEN_LAYERS = (128, 64, 32)
    MAX_ITER = 300
    LEARNING_RATE = 0.001
    ALPHA = 0.001
    ACTIVATION = 'relu'
    SOLVER = 'adam'
    BATCH_SIZE = 64
    EARLY_STOPPING = True
    VALIDATION_FRACTION = 0.1
    N_ITER_NO_CHANGE = 10

    SAMPLING_RATE = 512
    SEGMENT_LENGTH = 4
    OVERLAP_RATIO = 0.5

    FEATURE_SELECTION = True
    N_FEATURES_SELECT = 20

    RANDOM_STATE = 42


# =========================================================
# PREPROCESSING
# =========================================================
def create_segments(data, segment_length_samples, overlap_samples):
    if len(data) < segment_length_samples:
        return []

    segments = []

    # ==========================
    # REMOVE TRANSITION NOISE
    # ==========================
    trim = int(2 * Config.SAMPLING_RATE)
    if len(data) > 2 * trim:
        data = data[trim:-trim]

    # bandpass filter
    b, a = signal.butter(
        4,
        [0.5/(Config.SAMPLING_RATE/2), 45/(Config.SAMPLING_RATE/2)],
        btype='band'
    )
    data = signal.filtfilt(b, a, data)

    # notch filter (60Hz noise)
    b, a = signal.iirnotch(60/(Config.SAMPLING_RATE/2), 30)
    data = signal.filtfilt(b, a, data)

    start = 0
    step = segment_length_samples - overlap_samples

    while start + segment_length_samples <= len(data):
        segment = data[start:start + segment_length_samples]

        segment = signal.detrend(segment)

        # normalize per segment (important for LOSO)
        segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-8)

        segments.append(segment)
        start += step

    return segments


# =========================================================
# FEATURE ENGINEERING (MAJOR IMPROVEMENT)
# =========================================================
def extract_features(segments):
    features = []

    for seg in segments:

        # ======================
        # TIME DOMAIN FEATURES
        # ======================
        ptp = np.max(seg) - np.min(seg)
        rms = np.sqrt(np.mean(seg**2))
        var = np.var(seg)
        abs_mean = np.mean(np.abs(seg))

        blink_score = np.percentile(np.abs(seg), 95)

        # ======================
        # FREQUENCY DOMAIN
        # ======================
        f, Pxx = signal.welch(seg, fs=Config.SAMPLING_RATE, nperseg=min(len(seg), 256))

        def band(low, high):
            idx = (f >= low) & (f <= high)
            return np.sum(Pxx[idx])

        delta = band(1, 4)
        theta = band(4, 8)
        alpha = band(8, 13)
        beta = band(13, 30)
        gamma = band(30, 45)

        total = delta + theta + alpha + beta + gamma + 1e-8

        feat = [
            delta / total,
            theta / total,
            alpha / total,
            beta / total,
            gamma / total,

            ptp,
            rms,
            var,
            abs_mean,
            blink_score,

            alpha / (beta + 1e-8),
            theta / (alpha + 1e-8)
        ]

        features.append(feat)

    return np.array(features)


# =========================================================
# DATA LOADING (UNCHANGED)
# =========================================================
def load_all_subjects():
    all_features = []
    all_labels = []
    all_subjects = []

    subject_folders = sorted([
        f.path for f in os.scandir(Config.DATASET_PATH) if f.is_dir()
    ])

    segment_length_samples = int(Config.SEGMENT_LENGTH * Config.SAMPLING_RATE)
    overlap_samples = int(segment_length_samples * Config.OVERLAP_RATIO)

    for subject_folder in subject_folders:
        subject_id = os.path.basename(subject_folder)

        relax_segments, focus_segments, blink_segments = [], [], []

        for file in glob.glob(os.path.join(subject_folder, "*_1_*.txt")):
            data = np.loadtxt(file)
            relax_segments += create_segments(data, segment_length_samples, overlap_samples)

        for file in glob.glob(os.path.join(subject_folder, "*_2_*.txt")):
            data = np.loadtxt(file)
            focus_segments += create_segments(data, segment_length_samples, overlap_samples)

        for file in glob.glob(os.path.join(subject_folder, "*_3_*.txt")):
            data = np.loadtxt(file)
            blink_segments += create_segments(data, segment_length_samples, overlap_samples)

        if len(relax_segments) == 0 or len(focus_segments) == 0 or len(blink_segments) == 0:
            continue

        Xr = extract_features(relax_segments)
        Xf = extract_features(focus_segments)
        Xb = extract_features(blink_segments)

        yr = np.zeros(len(Xr))
        yf = np.ones(len(Xf))
        yb = np.full(len(Xb), 2)

        X = np.vstack([Xr, Xf, Xb])
        y = np.hstack([yr, yf, yb])

        all_features.append(X)
        all_labels.append(y)
        all_subjects.extend([subject_id] * len(y))

        print(f"{subject_id}: {len(Xr)}, {len(Xf)}, {len(Xb)}")

    return np.vstack(all_features), np.hstack(all_labels), all_subjects


# =========================================================
# MODEL
# =========================================================
class EnhancedBCIClassifier:
    def __init__(self):
        self.model = MLPClassifier(
            hidden_layer_sizes=Config.HIDDEN_LAYERS,
            max_iter=Config.MAX_ITER,
            learning_rate_init=Config.LEARNING_RATE,
            alpha=Config.ALPHA,
            activation=Config.ACTIVATION,
            solver=Config.SOLVER,
            batch_size=Config.BATCH_SIZE,
            early_stopping=Config.EARLY_STOPPING,
            validation_fraction=Config.VALIDATION_FRACTION,
            n_iter_no_change=Config.N_ITER_NO_CHANGE,
            random_state=Config.RANDOM_STATE,
            verbose=False
        )

        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(f_classif, k=Config.N_FEATURES_SELECT) if Config.FEATURE_SELECTION else None

    def fit(self, X, y):
        X = self.scaler.fit_transform(X)

        if self.feature_selector is not None:
            self.feature_selector.k = min(Config.N_FEATURES_SELECT, X.shape[1])
            X = self.feature_selector.fit_transform(X, y)

        self.model.fit(X, y)
        return self

    def predict(self, X):
        X = self.scaler.transform(X)

        if self.feature_selector is not None:
            X = self.feature_selector.transform(X)

        # ==========================
        # POSTPROCESSING (IMPORTANT)
        # ==========================
        prob = self.model.predict_proba(X)
        pred = np.argmax(prob, axis=1)

        # smoothing (EEG critical)
        pred = median_filter(pred, size=5)

        # confidence fallback
        conf = np.max(prob, axis=1)
        pred[conf < 0.55] = 1  # focus as safe class

        return pred


# =========================================================
# LOSO (UNCHANGED)
# =========================================================
def leave_one_subject_out_validation():
    X, y, subjects = load_all_subjects()
    unique = sorted(list(set(subjects)))

    accs = []

    for test in unique:
        train_mask = np.array([s != test for s in subjects])
        test_mask = np.array([s == test for s in subjects])

        Xtr, Xte = X[train_mask], X[test_mask]
        ytr, yte = y[train_mask], y[test_mask]

        model = EnhancedBCIClassifier()
        model.fit(Xtr, ytr)

        pred = model.predict(Xte)
        acc = accuracy_score(yte, pred)

        print(test, acc)
        accs.append(acc)

    print("Mean:", np.mean(accs))


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    leave_one_subject_out_validation()