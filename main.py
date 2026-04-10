"""
Brain-Computer Interface MLP Classifier
FINAL VERSION: DYNAMIC SEGMENTATION (Class-Specific Windowing)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from scipy import signal
from scipy.stats import kurtosis, skew
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# Parameter Settings
class Config:
    DATASET_PATH = "bci_dataset_114-2"
    SKIP_SECONDS = 2.0                 # 捨棄每回合開頭前 2 秒
    
    # === 新增：動態切分策略 (你的完美提案) ===
    # 策略 A：針對連續狀態 (Relax / Focus)
    RF_SEG_LEN = 4.0                   # 窗口大一點，頻譜解析度才高
    RF_OVERLAP = 0.7                   # 重疊率高一點，資料量才多
    RF_MAX_THRES = 2500                # 剔除過大雜訊的門檻
    
    # 策略 B：針對瞬間狀態 (Blink)
    BLINK_SEG_LEN = 1.5                # 窗口縮小，聚焦眨眼瞬間，避免被背景稀釋
    BLINK_OVERLAP = 0.0                # 重疊率 0，不重複計算同一個眨眼
    BLINK_MIN_THRES = 500              # 【關鍵】至少要大於這個值，才承認這個片段「真的有眨眼」
    
    # MLP model parameters
    HIDDEN_LAYERS = (64, 32)           
    MAX_ITER = 200                     
    LEARNING_RATE = 0.005              
    ALPHA = 0.02                       
    ACTIVATION = 'relu'                
    SOLVER = 'adam'                    
    BATCH_SIZE = 128                   
    EARLY_STOPPING = True              
    VALIDATION_FRACTION = 0.1
    N_ITER_NO_CHANGE = 15
    SAMPLING_RATE = 512                
    FEATURE_SELECTION = True
    N_FEATURES_SELECT = 12             
    RANDOM_STATE = 42

def create_segments(data, segment_length_samples, overlap_samples, task_type):
    """根據不同任務類型，執行不同的切割與過濾策略"""
    skip_samples = int(Config.SKIP_SECONDS * Config.SAMPLING_RATE)
    if len(data) > skip_samples:
        data = data[skip_samples:]
    else:
        return []
        
    if len(data) < segment_length_samples:
        return []
    
    segments = []
    start = 0
    step = segment_length_samples - overlap_samples
    
    nyq = 0.5 * Config.SAMPLING_RATE
    low, high = 0.5 / nyq, 45.0 / nyq
    b, a = signal.butter(4, [low, high], btype='band')
    
    while start + segment_length_samples <= len(data):
        segment = data[start:start + segment_length_samples]
        segment_filtered = signal.filtfilt(b, a, segment)
        
        peak_amp = np.max(np.abs(segment_filtered))
        
        # === 核心邏輯：依照任務進行智能過濾 ===
        if task_type in [1, 2]: # Relax 或 Focus
            if peak_amp > Config.RF_MAX_THRES:
                start += step
                continue # 太大的是雜訊，丟棄
                
        elif task_type == 3:    # Blink
            # 如果這個小視窗內沒有出現足夠大的突波，代表它切到了「沒眨眼」的空白期
            if peak_amp < Config.BLINK_MIN_THRES:
                start += step
                continue # 沒有眨眼的片段直接丟棄，防止標籤污染！
            
        segments.append(segment_filtered)
        start += step
        
    return segments

def extract_features(segments):
    """提取不受視窗長度影響的強特徵"""
    features = []
    for seg in segments:
        # 1. 時域特徵 (變異數、峰度等本來就與長度無關)
        log_var = np.log10(np.var(seg) + 1e-7)
        kurt = kurtosis(seg)
        sk = skew(seg)
        zcr = ((seg[:-1] * seg[1:]) < 0).sum() / len(seg)
        
        # 2. 頻域特徵 (動態調整 nperseg 以適應不同的 seg_length)
        nperseg = min(len(seg), int(Config.SAMPLING_RATE * 1.0)) # 固定解析度為 1 秒
        freqs, psd = signal.welch(seg, fs=Config.SAMPLING_RATE, nperseg=nperseg)
        
        theta = np.log10(np.sum(psd[(freqs >= 4) & (freqs < 8)]) + 1e-9)
        alpha = np.log10(np.sum(psd[(freqs >= 8) & (freqs < 13)]) + 1e-9)
        beta  = np.log10(np.sum(psd[(freqs >= 13) & (freqs < 30)]) + 1e-9)
        total = np.log10(np.sum(psd[(freqs >= 1) & (freqs < 45)]) + 1e-9)
        
        rel_beta = beta - total
        mobility = np.sqrt(np.var(np.diff(seg)) / (np.var(seg) + 1e-7))
        
        current_feature = [
            log_var, kurt, sk, zcr, mobility,
            theta, alpha, beta, rel_beta, 
            beta - alpha,
            np.ptp(seg) / (np.std(seg) + 1e-7),
            np.max(np.abs(seg))
        ]
        features.append(current_feature)
    return np.array(features)

def load_all_subjects():
    all_features, all_labels, all_subjects = [], [], []
    if not os.path.exists(Config.DATASET_PATH): return None, None, None
    subject_folders = sorted([f.path for f in os.scandir(Config.DATASET_PATH) if f.is_dir()])
    
    # 計算兩套切割參數的樣本數
    rf_seg_len = int(Config.RF_SEG_LEN * Config.SAMPLING_RATE)
    rf_overlap = int(rf_seg_len * Config.RF_OVERLAP)
    blk_seg_len = int(Config.BLINK_SEG_LEN * Config.SAMPLING_RATE)
    blk_overlap = int(blk_seg_len * Config.BLINK_OVERLAP)

    for folder in subject_folders:
        sub_id = os.path.basename(folder)
        sub_segs = {1: [], 2: [], 3: []}
        
        for task in [1, 2, 3]:
            files = glob.glob(os.path.join(folder, f"*_{task}_*.txt"))
            for f in files:
                try:
                    with open(f, 'r', encoding='utf-8', errors='ignore') as file:
                        lines = file.readlines()
                        clean_data = []
                        for line in lines:
                            val = line.strip()
                            if val:
                                try: clean_data.append(float(val))
                                except ValueError: pass 
                    data = np.array(clean_data)
                    
                    # 依據任務類型套用不同的切割策略
                    if task in [1, 2]:
                        segs = create_segments(data, rf_seg_len, rf_overlap, task)
                    else:
                        segs = create_segments(data, blk_seg_len, blk_overlap, task)
                    sub_segs[task].extend(segs)
                        
                except Exception as e:
                    continue
        
        if not (sub_segs[1] and sub_segs[2] and sub_segs[3]): 
            print(f"Warning: Subject {sub_id} missing valid task data. Skipping.")
            continue

        f1 = extract_features(sub_segs[1])
        f2 = extract_features(sub_segs[2])
        f3 = extract_features(sub_segs[3])
        
        sub_feat = np.vstack([f1, f2, f3])
        sub_feat = StandardScaler().fit_transform(sub_feat) 
        
        sub_lab = np.hstack([np.zeros(len(f1)), np.ones(len(f2)), np.full(len(f3), 2)])
        all_features.append(sub_feat)
        all_labels.append(sub_lab)
        all_subjects.extend([sub_id] * len(sub_lab))
        
        print(f" - {sub_id}: Loaded Focus/Relax({len(f1)+len(f2)}) & Blink({len(f3)}) segments")
    
    if not all_features: return None, None, None
    return np.vstack(all_features), np.hstack(all_labels), all_subjects

class EnhancedBCIClassifier:
    def __init__(self):
        self.model = MLPClassifier(
            hidden_layer_sizes=Config.HIDDEN_LAYERS, max_iter=Config.MAX_ITER,
            learning_rate_init=Config.LEARNING_RATE, alpha=Config.ALPHA,
            activation=Config.ACTIVATION, solver=Config.SOLVER, batch_size=Config.BATCH_SIZE,
            early_stopping=Config.EARLY_STOPPING, random_state=Config.RANDOM_STATE
        )
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(f_classif, k=Config.N_FEATURES_SELECT) if Config.FEATURE_SELECTION else None
        
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        X_selected = self.feature_selector.fit_transform(X_scaled, y) if self.feature_selector else X_scaled
        self.model.fit(X_selected, y)
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_selected = self.feature_selector.transform(X_scaled) if self.feature_selector else X_scaled
        raw_preds = self.model.predict(X_selected)
        if len(raw_preds) > 5:
            return signal.medfilt(raw_preds, kernel_size=5) 
        return raw_preds

def leave_one_subject_out_validation():
    print("\nStarting Leave-One-Subject-Out (LOSO) Cross-Validation...")
    X, y, subjects = load_all_subjects()
    if X is None: return None
        
    unique_subjects = sorted(list(set(subjects)))
    results = {'accuracies': [], 'confusion_matrices': [], 'loss_curves': [], 'subject_names': []}
    
    for test_sub in unique_subjects:
        train_mask = [s != test_sub for s in subjects]
        test_mask = [s == test_sub for s in subjects]
        
        clf = EnhancedBCIClassifier().fit(X[train_mask], y[train_mask])
        y_pred = clf.predict(X[test_mask])
        
        results['accuracies'].append(accuracy_score(y[test_mask], y_pred))
        results['confusion_matrices'].append(confusion_matrix(y[test_mask], y_pred, labels=[0, 1, 2]))
        results['loss_curves'].append(clf.model.loss_curve_)
        results['subject_names'].append(test_sub)
        
        print(f" -> {test_sub} Accuracy: {results['accuracies'][-1]:.3f}")
    return results

def plot_results(results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('BCI Classifier - Dynamic Segmentation Strategy', fontsize=16)
    
    axes[0].bar(results['subject_names'], results['accuracies'], color='orange')
    axes[0].axhline(y=np.mean(results['accuracies']), color='r', linestyle='--', label=f'Mean: {np.mean(results["accuracies"]):.3f}')
    axes[0].axhline(y=0.65, color='b', linestyle=':', label='Target 0.65')
    axes[0].set_ylim(0, 1)
    axes[0].set_title('Accuracy by Subject')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    total_cm = np.sum(results['confusion_matrices'], axis=0)
    sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=['Relax', 'Focus', 'Blink'], yticklabels=['Relax', 'Focus', 'Blink'])
    axes[1].set_title('Overall Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    for i, lc in enumerate(results['loss_curves']): 
        axes[2].plot(lc, alpha=0.7, label=results['subject_names'][i])
    axes[2].set_title('Training Loss Curves')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Loss')
    if len(results['subject_names']) <= 10: 
        axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bci_results_optimized.png', dpi=300)
    plt.show()

def main():
    print("BCI EEG Classification - Processing...")
    print("=" * 60)
    res = leave_one_subject_out_validation()
    if res: 
        plot_results(res)
        print("\n" + "="*40)
        print(f"Overall Mean Accuracy: {np.mean(res['accuracies']):.3f}")
        print(f"Results saved to 'bci_results_optimized.png'")

if __name__ == "__main__":
    main()