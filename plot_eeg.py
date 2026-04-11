import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import signal
import os

SAMPLING_RATE = 512  # Hz

DATASET_PATH = "bci_dataset_114-2"
VIS_PATH = "vis"

class Config:
    DATASET_PATH = "bci_dataset_114-2"
    SKIP_SECONDS = 2.0                 # 捨棄每回合開頭前 2 秒
    
    # === 策略 A：針對連續狀態 (Relax / Focus) ===
    RF_SEG_LEN = 4.0                   # 窗口大一點，頻譜解析度才高
    RF_OVERLAP = 0.7                   # 重疊率高一點，資料量才多
    # 【關鍵修改 1】嚴格過濾！真正的腦波不會超過 800，超過的都是肌肉或眼動雜訊，直接丟棄！
    RF_MAX_THRES = 800                 
    
    # === 策略 B：針對瞬間狀態 (Blink) ===
    BLINK_SEG_LEN = 1.5                # 窗口縮小，聚焦眨眼瞬間，避免被背景稀釋
    BLINK_OVERLAP = 0.0                # 重疊率 0，不重複計算同一個眨眼
    BLINK_MIN_THRES = 500              # 必須有大於 500 的突波才承認是眨眼
    
    # MLP model parameters
    HIDDEN_LAYERS = (64, 32)           
    MAX_ITER = 200                     
    LEARNING_RATE = 0.005              
    ALPHA = 0.05                       # 提高正規化強度，防止模型死背特徵
    ACTIVATION = 'relu'                
    SOLVER = 'adam'                    
    BATCH_SIZE = 128                   
    EARLY_STOPPING = True              
    VALIDATION_FRACTION = 0.1
    N_ITER_NO_CHANGE = 15
    SAMPLING_RATE = 512                
    FEATURE_SELECTION = False
    N_FEATURES_SELECT = 10             # 精簡為 10 個最強比例與複雜度特徵
    RANDOM_STATE = 42

def build_path(student_id, task, idx):
    filename = f"{student_id}_{task}_{idx}.txt"
    full_path = os.path.join(DATASET_PATH, student_id, filename)
    return full_path

def load_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            val = line.strip()
            if val:
                try:
                    data.append(float(val))
                except ValueError:
                    pass
    return np.array(data)

def plot_signal(data, file_name = "vis.png"):
    n = len(data)
    time = np.arange(n) / SAMPLING_RATE  # 秒

    plt.figure(figsize=(12, 4))
    plt.plot(time, data)
    plt.title("EEG Signal")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(VIS_PATH, file_name))

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
    low, high = 0.5 / nyq, 45.0 / nyq # normalize frequency, since max frequency = 1/2 * sample_rate
    b, a = signal.butter(4, [low, high], btype='band') # order = 4
    data_filetered = signal.filtfilt(b, a, data)
    plot_signal(data_filetered, "raw_data_filtered")
    while start + segment_length_samples <= len(data):
        segment = data[start:start + segment_length_samples]
        segment_filtered = signal.filtfilt(b, a, segment) # use filter
        plot_signal(segment_filtered, f"seg_{start}_filt.png")
        peak_amp = np.max(np.abs(segment_filtered)) 
        print(peak_amp)
        # === 核心邏輯：依照任務進行智能過濾 ===
        if task_type == 1: # Relax
            # 放寬 Relax 的標準，多收一點資料進來訓練
            if peak_amp > 1500: 
                print(f"seg_{start} peak amplitude violation for Task 1(Relax)")
                start += step # just don't put this segment into []
                continue 
        elif task_type == 2: # Focus
            if peak_amp > Config.RF_MAX_THRES:
                print(f"seg_{start} peak amplitude violation for Task 2(Focus)")
                start += step
                continue
                
        elif task_type == 3:    # Blink
            # 如果這個小視窗內沒有出現足夠大的突波，代表它切到了「沒眨眼」的空白期
            if peak_amp < Config.BLINK_MIN_THRES:
                print(f"seg_{start} peak amplitude too low for Task 3(Blink)")
                start += step
                continue # 沒有眨眼的片段直接丟棄，防止標籤污染
        print(f"seg_{start} is included")
        segments.append(segment_filtered)
        start += step
        
    return segments


def main():
    rf_seg_len = int(Config.RF_SEG_LEN * Config.SAMPLING_RATE)
    rf_overlap = int(rf_seg_len * Config.RF_OVERLAP)
    blk_seg_len = int(Config.BLINK_SEG_LEN * Config.SAMPLING_RATE)
    blk_overlap = int(blk_seg_len * Config.BLINK_OVERLAP)
    # 情況 1：直接給完整路徑
    if len(sys.argv) == 2:
        filename = sys.argv[1]
    
    # 情況 2：給三個參數（學號 task idx）
    elif len(sys.argv) == 4:
        student_id = sys.argv[1]
        task = sys.argv[2]
        idx = sys.argv[3]
        filename = build_path(student_id, task, idx)
        print(f"自動組合路徑: {filename}")
    
    # 情況 3：互動輸入
    else:
        student_id = input("學號: ")
        task = input("task (1=Relax, 2=Focus, 3=Blink): ")
        idx = input("第幾個: ")
        filename = build_path(student_id, task, idx)
        print(f"自動組合路徑: {filename}")

    if not os.path.exists(filename):
        print("❌ 檔案不存在！")
        return
    
    try:
        data = load_data(filename)
        print(f"Loaded {len(data)} samples ({len(data)/SAMPLING_RATE:.2f} seconds)")
        plot_signal(data, "raw_data.png")
        print("Task:", task)
        task = int(task)
        if task in [1, 2]:
            print("Seg_len:", rf_seg_len)
            print("Overlap_len", rf_overlap)
            segs = create_segments(data, rf_seg_len, rf_overlap, task)
        else:
            segs = create_segments(data, blk_seg_len, blk_overlap, task)
        
    except FileNotFoundError:
        print("檔案找不到！請確認路徑是否正確")

if __name__ == "__main__":
    main()