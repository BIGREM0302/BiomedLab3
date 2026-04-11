import numpy as np
import os

def generate_synthetic_eeg(subject_id="b12909999"):
    fs = 512  # 取樣率 [cite: 24]
    duration = 20  # 總時長 (20s 任務 + 20s 休息) [cite: 214]
    n_samples = fs * duration
    t = np.linspace(0, duration, n_samples)
    
    output_dir = f"bci_dataset_114-2/{subject_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    for task in [1, 2, 3]:  # 1: Relax, 2: Focus, 3: Blink [cite: 29]
        for round_idx in range(1, 31):  # 每類別 30 回合 [cite: 218]
            
            # 1. 基礎噪音與基線漂移 (Baseline Drift)
            eeg = np.random.normal(0, 50, n_samples)  # 白噪音
            drift = 100 * np.sin(2 * np.pi * 0.05 * t)  # 低頻漂移
            eeg += drift
            
            # 2. 注入狀態特徵 (僅針對前 20 秒任務段)
            task_mask = t <= 20
            
            if task == 1:  # Relax: 強 Alpha 波 (10Hz)
                alpha = 150 * np.sin(2 * np.pi * 10 * t[task_mask])
                eeg[task_mask] += alpha
                
            elif task == 2:  # Focus: 高頻 Beta 波 (20Hz)
                beta = 60 * np.sin(2 * np.pi * 20 * t[task_mask])
                eeg[task_mask] += beta
                
            elif task == 3:  # Blink: 隨機插入 5 個巨大突波
                blink_times = np.random.choice(np.where(task_mask)[0], 5, replace=False)
                for idx in blink_times:
                    # 模擬 0.2 秒的眨眼脈衝
                    width = int(0.2 * fs)
                    if idx + width < n_samples:
                        pulse = 1200 * np.hanning(width) 
                        eeg[idx:idx+width] += pulse
            
            # 3. 儲存檔案
            filename = f"{subject_id}_{task}_{round_idx}.txt"
            filepath = os.path.join(output_dir, filename)
            np.savetxt(filepath, eeg, fmt='%.0f')
            
    print(f"成功生成受試者 {subject_id} 的 90 個虛擬數據檔於 {output_dir}")

if __name__ == "__main__":
    generate_synthetic_eeg()