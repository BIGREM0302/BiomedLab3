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
# ==========================================
from scipy import signal
from scipy.ndimage import median_filter
import pywt
# ==========================================
warnings.filterwarnings('ignore')

# Parameter Settings
class Config:
    # Dataset path settings
    DATASET_PATH = "bci_dataset_114-2"
    
    # MLP model parameters
    HIDDEN_LAYERS = (64,32)
    MAX_ITER = 1000
    LEARNING_RATE = 0.001
    ALPHA = 0.0005
    ACTIVATION = 'relu'
    SOLVER = 'adam'
    BATCH_SIZE = 64
    EARLY_STOPPING = True
    VALIDATION_FRACTION = 0.1
    N_ITER_NO_CHANGE = 10
    
    # Signal processing parameters
    SAMPLING_RATE = 512    # BrainLink fixed sampling rate
    SEGMENT_LENGTH = 2     # Segment length in seconds
    OVERLAP_RATIO = 0.2    # Overlap ratio for segments
    
    # Feature selection parameters
    FEATURE_SELECTION = True
    N_FEATURES_SELECT = 30 # Modify preprocessing to extract truly effective features
    
    # Other settings
    RANDOM_STATE = 42

# ==========================================
# Preprocessing 
# ==========================================
def bandpass_filter(data, fs, low=1, high=45):  ### [MODIFIED]
    b, a = signal.butter(4, [low/(fs/2), high/(fs/2)], btype='band')
    return signal.filtfilt(b, a, data)


def notch_filter(data, fs, freq=50):  ### [MODIFIED]
    b, a = signal.iirnotch(freq/(fs/2), Q=30)
    return signal.filtfilt(b, a, data)
# =========================================

def create_segments(data, segment_length_samples, overlap_samples):
    """Split a single round of continuous EEG signal into multiple segments"""
    if len(data) < segment_length_samples:
        return []
    
    segments = []
    start = 0
    step = segment_length_samples - overlap_samples
    
    # === FILTER FIRST === ### [MODIFIED]
    data = bandpass_filter(data, Config.SAMPLING_RATE)
    data = notch_filter(data, Config.SAMPLING_RATE)

    while start + segment_length_samples <= len(data):
        segment = data[start:start + segment_length_samples]

        segment = signal.detrend(segment)
        segment = segment * np.hamming(len(segment))

        segments.append(segment)
        start += step

    return segments
    # =========================================

def bandpower(seg, fs, band):  ### [MODIFIED]
    f, Pxx = signal.welch(seg, fs=fs)
    idx = np.logical_and(f >= band[0], f <= band[1])
    return np.trapz(Pxx[idx], f[idx])


def extract_features(segments):
    """
    Perform feature engineering on segments
    """
    features = []
    bands = {
        'Delta': (1, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 50)
    }

    for seg in segments:
        feat = []

        # === 1. 保留原始時域特徵 (Blink 判斷關鍵) === ### [MODIFIED]
        # 必須在 Z-score 之前計算，保留原始振幅差距
        ptp_amplitude = np.max(seg) - np.min(seg)
        variance = np.var(seg)

        # === 2. PER-SEGMENT NORMALIZATION (Z-SCORE) === ### [MODIFIED]
        # 消除每個人的頭骨厚度、導電狀態造成的整體電壓差異
        seg_norm = (seg - np.mean(seg)) / (np.std(seg) + 1e-8)

        # === 3. RELATIVE BANDPOWER === ### [MODIFIED]
        abs_powers = []
        for b in bands.values():
            # 注意：這裡改用標準化後的 seg_norm 來算能量
            bp = bandpower(seg_norm, Config.SAMPLING_RATE, b)
            abs_powers.append(bp)
        
        # 計算總能量，並將各頻段能量轉為「比例 (0~1)」
        total_power = sum(abs_powers) + 1e-8
        for bp in abs_powers:
            feat.append(bp / total_power) 

        # === 4. 組合所有特徵 ===
        feat.append(variance)      # index 5
        feat.append(ptp_amplitude) # index 6 (稍後用來抓 Blink)
        feat.append(np.mean(np.abs(seg_norm)))
        feat.append(signal.skew(seg_norm)) if hasattr(signal, 'skew') else feat.append(0)
        feat.append(signal.kurtosis(seg_norm)) if hasattr(signal, 'kurtosis') else feat.append(0)

        features.append(feat)

    return np.array(features)


def load_all_subjects():
    """Load round-based data for all subjects in the group"""
    all_features = []
    all_labels = []
    all_subjects = []
    
    if not os.path.exists(Config.DATASET_PATH):
        print(f"Error: Directory '{Config.DATASET_PATH}' not found")
        return None, None, None
        
    subject_folders = sorted([f.path for f in os.scandir(Config.DATASET_PATH) if f.is_dir()])
    
    if len(subject_folders) < 2:
        print("Error: Not enough subjects. At least 2 subject folders are required for cross-validation.")
        return None, None, None
        
    print(f"Found {len(subject_folders)} subjects. Loading data...")
    
    segment_length_samples = int(Config.SEGMENT_LENGTH * Config.SAMPLING_RATE)
    overlap_samples = int(segment_length_samples * Config.OVERLAP_RATIO)

    for subject_folder in subject_folders:
        subject_id = os.path.basename(subject_folder)
        relax_segments = []
        focus_segments = []
        blink_segments = []
        
        # Load Task 1 (Relax) all rounds
        task1_files = glob.glob(os.path.join(subject_folder, "*_1_*.txt"))
        for file in task1_files:
            try:
                data = np.loadtxt(file)
                segs = create_segments(data, segment_length_samples, overlap_samples)
                relax_segments.extend(segs)
            except Exception as e:
                print(f"Error reading {file}: {e}")

        # Load Task 2 (Focus) all rounds
        task2_files = glob.glob(os.path.join(subject_folder, "*_2_*.txt"))
        for file in task2_files:
            try:
                data = np.loadtxt(file)
                segs = create_segments(data, segment_length_samples, overlap_samples)
                focus_segments.extend(segs)
            except Exception as e:
                print(f"Error reading {file}: {e}")
        
        # Load Task 3 (Blink) all rounds
        task3_files = glob.glob(os.path.join(subject_folder, "*_3_*.txt"))
        for file in task3_files:
            try:
                data = np.loadtxt(file)
                segs = create_segments(data, segment_length_samples, overlap_samples)
                blink_segments.extend(segs)
            except Exception as e:
                print(f"Error reading {file}: {e}")

        if len(relax_segments) == 0 or len(focus_segments) == 0 or len(blink_segments) == 0:
            print(f"Warning: Insufficient data for {subject_id}. Skipping.")
            continue

        # Extract features
        relax_features = extract_features(relax_segments)
        focus_features = extract_features(focus_segments)
        blink_features = extract_features(blink_segments)
        
        # Create labels (0=Relax, 1=Focus, 2=Blink)
        relax_labels = np.zeros(len(relax_features))
        focus_labels = np.ones(len(focus_features))
        blink_labels = np.full(len(blink_features), 2)
        
        # Combine subject data
        subject_features = np.vstack([relax_features, focus_features, blink_features])
        subject_labels = np.hstack([relax_labels, focus_labels, blink_labels])
        subject_ids = [subject_id] * len(subject_labels)
        
        all_features.append(subject_features)
        all_labels.append(subject_labels)
        all_subjects.extend(subject_ids)
        
        print(f" - {subject_id}: Successfully loaded {len(relax_segments)} Relax, {len(focus_segments)} Focus, {len(blink_segments)} Blink segments")
    
    if not all_features:
        return None, None, None
    
    return np.vstack(all_features), np.hstack(all_labels), all_subjects


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
        self.blink_threshold = 0  # === 新增：用來記憶 Blink 的專屬門檻 === ### [MODIFIED]
        
    def fit(self, X, y):
        # === 1. 計算 Blink 專屬物理門檻 === ### [MODIFIED]
        # 特徵 index 6 是我們剛剛算的 ptp_amplitude
        ptp_features = X[:, 6] 
        blink_ptp = ptp_features[y == 2]
        non_blink_ptp = ptp_features[y != 2]
        
        if len(blink_ptp) > 0 and len(non_blink_ptp) > 0:
            # 抓非眨眼的高標(95%)與眨眼的低標(5%)，取中間值當作安全門檻
            self.blink_threshold = (np.percentile(non_blink_ptp, 95) + np.percentile(blink_ptp, 5)) / 2
        
        # === 2. 濾除 Blink，只讓 MLP 學習純腦波 (Relax=0, Focus=1) === ### [MODIFIED]
        mask_eeg = (y == 0) | (y == 1)
        X_eeg = X[mask_eeg]
        y_eeg = y[mask_eeg]
        
        X_scaled = self.scaler.fit_transform(X_eeg)
        if self.feature_selector is not None:
            self.feature_selector.k = min(Config.N_FEATURES_SELECT, X_scaled.shape[1])
            X_selected = self.feature_selector.fit_transform(X_scaled, y_eeg)
        else:
            X_selected = X_scaled
        
        self.model.fit(X_selected, y_eeg)
        return self
        
    def predict(self, X, smoothing_window=5):
        predictions = np.zeros(len(X), dtype=int)
        
        # === 1. 第一層過濾：用物理規則抓出 Blink === ### [MODIFIED]
        ptp_features = X[:, 6]
        is_blink = ptp_features > self.blink_threshold
        predictions[is_blink] = 2  # 直接蓋章認定為 Blink (2)
        
        # === 2. 第二層過濾：剩下的交給 MLP 預測純腦波 === ### [MODIFIED]
        is_eeg = ~is_blink
        if np.any(is_eeg):
            X_eeg = X[is_eeg]
            X_scaled = self.scaler.transform(X_eeg)
            if self.feature_selector is not None:
                X_selected = self.feature_selector.transform(X_scaled)
            else:
                X_selected = X_scaled
                
            probs = self.model.predict_proba(X_selected)
            # 由於 MLP 只被餵過 0 和 1，所以預測結果只會是 0 或 1
            predictions[is_eeg] = np.argmax(probs, axis=1)
            
        # === 3. 整體時間平滑化 (Majority Vote) ===
        if smoothing_window > 1:
            predictions = median_filter(predictions, size=smoothing_window)

        return predictions
    
    def get_loss_curve(self):
        return self.model.loss_curve_ if hasattr(self.model, 'loss_curve_') else []


def leave_one_subject_out_validation():
    print("\nStarting Leave-One-Subject-Out (LOSO) Cross-Validation...")
    
    X, y, subjects = load_all_subjects()
    if X is None: return None
    
    unique_subjects = sorted(list(set(subjects)))
    results = {'accuracies': [], 'confusion_matrices': [], 'loss_curves': [], 'subject_names': []}
    
    print("\n" + "="*40)
    for test_subject in unique_subjects:
        train_mask = [s != test_subject for s in subjects]
        test_mask = [s == test_subject for s in subjects]
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        print(f"Training Model (Test Subject: {test_subject}) | Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        classifier = EnhancedBCIClassifier()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        
        results['accuracies'].append(accuracy)
        results['confusion_matrices'].append(cm)
        results['loss_curves'].append(classifier.get_loss_curve())
        results['subject_names'].append(test_subject)
        
        print(f" -> Accuracy: {accuracy:.3f}")
    
    return results


def plot_results(results):
    if results is None: return
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('BCI Classifier (Raw Data) - Group LOSO Results', fontsize=16)
    
    # 1. Accuracy distribution
    subject_names = results['subject_names']
    axes[0].bar(subject_names, results['accuracies'], 
                color=['green' if acc >= 0.7 else 'orange' if acc >= 0.65 else 'red' for acc in results['accuracies']])
    axes[0].set_title('Accuracy by Subject')
    axes[0].set_ylabel('Accuracy')
    axes[0].axhline(y=np.mean(results['accuracies']), color='r', linestyle='--', label=f'Mean: {np.mean(results["accuracies"]):.3f}')
    axes[0].axhline(y=0.65, color='blue', linestyle=':', label='Target: 0.65')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)
    
    # 2. Overall confusion matrix
    total_cm = np.sum(results['confusion_matrices'], axis=0)
    sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Relax', 'Focus', 'Blink'], yticklabels=['Relax', 'Focus', 'Blink'], ax=axes[1])
    axes[1].set_title('Overall Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    # 3. Training loss curves
    valid_loss_curves = [lc for lc in results['loss_curves'] if len(lc) > 0]
    if valid_loss_curves:
        for i, loss_curve in enumerate(valid_loss_curves):
            axes[2].plot(loss_curve, alpha=0.7, label=subject_names[i])
        axes[2].set_title('Training Loss Curves')
        axes[2].set_xlabel('Iteration')
        axes[2].set_ylabel('Loss')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bci_results_raw_data.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("BCI EEG Classification - Group Evaluation")
    print("=" * 60)
    
    results = leave_one_subject_out_validation()
    if results is None:
        print("Validation failed! Please check your directory structure.")
        return
    
    mean_accuracy = np.mean(results['accuracies'])
    std_accuracy = np.std(results['accuracies'])
    
    print("\n" + "="*40)
    print(f"Overall Mean Accuracy: {mean_accuracy:.3f} ± {std_accuracy:.3f}")
    
    total_cm = np.sum(results['confusion_matrices'], axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        relax_accuracy = total_cm[0, 0] / np.sum(total_cm[0, :]) if np.sum(total_cm[0, :]) > 0 else 0
        concentration_accuracy = total_cm[1, 1] / np.sum(total_cm[1, :]) if np.sum(total_cm[1, :]) > 0 else 0
        blink_accuracy = total_cm[2, 2] / np.sum(total_cm[2, :]) if np.sum(total_cm[2, :]) > 0 else 0
        relax_precision = total_cm[0, 0] / np.sum(total_cm[:, 0]) if np.sum(total_cm[:, 0]) > 0 else 0
        concentration_precision = total_cm[1, 1] / np.sum(total_cm[:, 1]) if np.sum(total_cm[:, 1]) > 0 else 0
        blink_precision = total_cm[2, 2] / np.sum(total_cm[:, 2]) if np.sum(total_cm[:, 2]) > 0 else 0

    print(f"\n[Relax Class]")
    print(f"  - Accuracy (Recall): {relax_accuracy:.3f} ({total_cm[0, 0]}/{np.sum(total_cm[0, :])})")
    print(f"  - Precision: {relax_precision:.3f} ({total_cm[0, 0]}/{np.sum(total_cm[:, 0])})")
    
    print(f"\n[Focus Class]")
    print(f"  - Accuracy (Recall): {concentration_accuracy:.3f} ({total_cm[1, 1]}/{np.sum(total_cm[1, :])})")
    print(f"  - Precision: {concentration_precision:.3f} ({total_cm[1, 1]}/{np.sum(total_cm[:, 1])})")
    
    print(f"\n[Blink Class]")
    print(f"  - Accuracy (Recall): {blink_accuracy:.3f} ({total_cm[2, 2]}/{np.sum(total_cm[2, :])})")
    print(f"  - Precision: {blink_precision:.3f} ({total_cm[2, 2]}/{np.sum(total_cm[:, 2])})")
    
    plot_results(results)
    print(f"\nResults saved to 'bci_results_raw_data.png'")

if __name__ == "__main__":
    main()