"""
Brain-Computer Interface MLP Classifier (BrainLink Version)
For EEG signal relaxation/focus/blink state classification
WITH FEATURE ENGINEERING & PREPROCESSING (Compliant with README)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from scipy import signal
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# Parameter Settings
class Config:
    # Dataset path settings
    DATASET_PATH = "bci_dataset_114-2"
    
    # MLP model parameters
    HIDDEN_LAYERS = (128, 64, 32)
    MAX_ITER = 200             # [Allowed] 50 ~ 200
    LEARNING_RATE = 0.01       # [Allowed] 0.005 ~ 0.02
    ALPHA = 0.01               # [Allowed] 0.0001 ~ 0.05
    ACTIVATION = 'relu'        # [Fixed]
    SOLVER = 'adam'            # [Fixed]
    BATCH_SIZE = 64            # [Allowed] 32 ~ 128
    EARLY_STOPPING = True      # [Fixed]
    VALIDATION_FRACTION = 0.1
    N_ITER_NO_CHANGE = 10
    
    # Signal processing parameters
    SAMPLING_RATE = 512        # [Fixed] BrainLink fixed sampling rate
    SEGMENT_LENGTH = 4         # [Allowed] 2 ~ 6 秒
    OVERLAP_RATIO = 0.5        # [Allowed] 0.0 ~ 0.8
    
    # Feature selection parameters
    FEATURE_SELECTION = True
    N_FEATURES_SELECT = 10     # 配合我們提取的特徵數量
    
    # Other settings
    RANDOM_STATE = 42

def create_segments(data, segment_length_samples, overlap_samples):
    """Split a single round of continuous EEG signal into multiple segments"""
    if len(data) < segment_length_samples:
        return []
    
    segments = []
    start = 0
    step = segment_length_samples - overlap_samples
    
    # === student preprocessing ===
    # 建立 1-40 Hz Butterworth Bandpass Filter
    nyq = 0.5 * Config.SAMPLING_RATE
    low = 1.0 / nyq
    high = 40.0 / nyq
    b, a = signal.butter(4, [low, high], btype='band')
    
    while start + segment_length_samples <= len(data):
        segment = data[start:start + segment_length_samples]
        
        # 1. 濾除極低頻基線漂移與高頻雜訊
        segment_filtered = signal.filtfilt(b, a, segment)
        
        # 2. Artifact Rejection (剔除極端雜訊)
        if np.max(np.abs(segment_filtered)) > 800:
            start += step
            continue
            
        segments.append(segment_filtered)
        start += step
    
    return segments

def extract_features(segments):
    """
    Perform feature engineering on segments
    """
    features = []
    for seg in segments:
        # === student preprocessing ===
        
        # --- 時域特徵 (針對 Blink) ---
        var = np.var(seg)
        p2p = np.ptp(seg)
        rms = np.sqrt(np.mean(seg**2))
        zcr = ((seg[:-1] * seg[1:]) < 0).sum()
        
        # --- 頻域特徵 (針對 Relax/Focus) ---
        freqs, psd = signal.welch(seg, fs=Config.SAMPLING_RATE, nperseg=len(seg))
        
        delta_power = np.sum(psd[(freqs >= 1) & (freqs < 4)])
        theta_power = np.sum(psd[(freqs >= 4) & (freqs < 8)])
        alpha_power = np.sum(psd[(freqs >= 8) & (freqs < 13)])
        beta_power  = np.sum(psd[(freqs >= 13) & (freqs < 30)])
        gamma_power = np.sum(psd[(freqs >= 30) & (freqs <= 40)])
        
        beta_alpha_ratio = beta_power / alpha_power if alpha_power > 0 else 0
        
        current_feature = [
            var, p2p, rms, zcr, 
            delta_power, theta_power, alpha_power, beta_power, gamma_power, 
            beta_alpha_ratio
        ]
        
        features.append(current_feature)
        
    return np.array(features)

def load_all_subjects():
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
        
        # Load Task 1 (Relax)
        task1_files = glob.glob(os.path.join(subject_folder, "*_1_*.txt"))
        for file in task1_files:
            try:
                data = np.loadtxt(file)
                segs = create_segments(data, segment_length_samples, overlap_samples)
                relax_segments.extend(segs)
            except Exception as e:
                print(f"Error reading {file}: {e}")

        # Load Task 2 (Focus)
        task2_files = glob.glob(os.path.join(subject_folder, "*_2_*.txt"))
        for file in task2_files:
            try:
                data = np.loadtxt(file)
                segs = create_segments(data, segment_length_samples, overlap_samples)
                focus_segments.extend(segs)
            except Exception as e:
                print(f"Error reading {file}: {e}")
        
        # Load Task 3 (Blink)
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

        relax_features = extract_features(relax_segments)
        focus_features = extract_features(focus_segments)
        blink_features = extract_features(blink_segments)
        
        relax_labels = np.zeros(len(relax_features))
        focus_labels = np.ones(len(focus_features))
        blink_labels = np.full(len(blink_features), 2)
        
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
        
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        if self.feature_selector is not None:
            self.feature_selector.k = min(Config.N_FEATURES_SELECT, X_scaled.shape[1])
            X_selected = self.feature_selector.fit_transform(X_scaled, y)
        else:
            X_selected = X_scaled
        
        self.model.fit(X_selected, y)
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_scaled)
        else:
            X_selected = X_scaled
        
        # === student postprocessing ===
        probs = self.model.predict_proba(X_selected)
        predictions = []
        
        for p in probs:
            max_prob = np.max(p)
            pred_class = np.argmax(p)
            
            # 使用信心閾值做初步的過濾
            if max_prob < 0.40:
                predictions.append(pred_class) 
            else:
                predictions.append(pred_class)
                
        return np.array(predictions)
    
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
    fig.suptitle('BCI Classifier (Optimized) - Group LOSO Results', fontsize=16)
    
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
    
    total_cm = np.sum(results['confusion_matrices'], axis=0)
    sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Relax', 'Focus', 'Blink'], yticklabels=['Relax', 'Focus', 'Blink'], ax=axes[1])
    axes[1].set_title('Overall Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
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
    plt.savefig('bci_results_optimized.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("BCI EEG Classification - Group Evaluation (Optimized)")
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
    print(f"\nResults saved to 'bci_results_optimized.png'")

if __name__ == "__main__":
    main()