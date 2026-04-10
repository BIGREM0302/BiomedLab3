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
from scipy import stats
# ==========================================
warnings.filterwarnings('ignore')

# Parameter Settings
class Config:
    # Dataset path settings
    DATASET_PATH = "bci_dataset_114-2"
    
    # MLP model parameters
    HIDDEN_LAYERS = (32,)
    MAX_ITER = 1000
    LEARNING_RATE = 0.01
    ALPHA = 0.01
    ACTIVATION = 'relu'
    SOLVER = 'adam'
    BATCH_SIZE = 128
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
def bandpass_filter(data, fs, low=0.1, high=45):  ### [MODIFIED]
    b, a = signal.butter(4, [low/(fs/2), high/(fs/2)], btype='band')
    return signal.filtfilt(b, a, data)


def notch_filter(data, fs, freq=60):  ### [MODIFIED]
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
    features = []
    bands = {'Delta': (1, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30), 'Gamma': (30, 50)}

    for seg in segments:
        feat = []

        # === 💥 1. 暴力物理特徵 (不標準化，專門對付 Blink) === ### [MODIFIED]
        # 直接拿濾波後、最原始的電壓來算，眨眼的數字會是其他狀態的十幾倍
        raw_ptp = np.max(seg) - np.min(seg)
        raw_var = np.var(seg)
        
        # === 2. 逐段標準化 (為了算頻率能量，讓 Relax/Focus 站在同一起跑線) ===
        seg_norm = (seg - np.mean(seg)) / (np.std(seg) + 1e-8)

        # 3. 相對頻段能量 (使用標準化後的波形)
        abs_powers = []
        for b in bands.values():
            bp = bandpower(seg_norm, Config.SAMPLING_RATE, b)
            abs_powers.append(bp)
        
        total_power = sum(abs_powers) + 1e-8
        for bp in abs_powers:
            feat.append(bp / total_power)  

        # 4. 神經科學黃金指標
        feat.append(abs_powers[2] / (abs_powers[3] + 1e-8)) # Alpha/Beta
        feat.append(abs_powers[1] / (abs_powers[3] + 1e-8)) # Theta/Beta

        # 5. 組裝特徵 
        feat.append(raw_ptp)                  # ### [MODIFIED] 塞入原始峰對峰值
        feat.append(np.log(raw_var + 1e-8))   # ### [MODIFIED] 塞入原始變異數的 Log
        feat.append(stats.kurtosis(seg))      # 峰度 (尖銳度)

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
            hidden_layer_sizes=(32, 16),  # ### [MODIFIED] 輕量化網路，防死背
            max_iter=1000,
            learning_rate_init=0.005,
            alpha=0.1,                    # ### [MODIFIED] 增強 L2 正則化，強迫尋找泛化規律
            activation='relu',
            solver='adam',
            batch_size=64,
            early_stopping=True,
            random_state=42,
            verbose=False
        )
        self.scaler = StandardScaler()
        # 特徵已經精煉過，直接全拿
        self.feature_selector = SelectKBest(f_classif, k='all') 

    def fit(self, X, y):
        # === 💥 [MODIFIED] 取消所有手動門檻，讓 MLP 學習全部 3 個類別 ===
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self
        
    def predict(self, X, smoothing_window=5):
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)
        
        # 多分類標準作法：取機率最大的類別
        raw_predictions = np.argmax(probs, axis=1)
        
        # 加上時間多數決平滑化
        if smoothing_window > 1:
            raw_predictions = median_filter(raw_predictions, size=smoothing_window)

        return raw_predictions
    
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