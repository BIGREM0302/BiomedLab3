import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def plot_task_fixed_y(dataset_path, student_id, rounds, start_sec=8, end_sec=13, fs=512):
    """
    固定 Y 軸範圍並在所有子圖間共用 Y 軸
    """
    tasks = [1, 2, 3]
    rounds = [17, 18, 19]
    task_names = {1: "Relax", 2: "Focus", 3: "Blink"}

    start_idx = int(start_sec * fs)
    end_idx = int(end_sec * fs)

    # === 修改 1：加入 sharey=True 讓所有圖表共用 Y 軸尺度 ===
    fig, axes = plt.subplots(3, 3, figsize=(10, 6), sharex=True, sharey=True)
    fig.suptitle(f"EEG Segment (Fixed Y): {student_id}", fontsize=14)

    for i, task_id in enumerate(tasks):
        for j, round_id in enumerate(rounds):
            ax = axes[i, j]
            pattern = os.path.join(dataset_path, student_id, f"*_{task_id}_{round_id}.txt")
            files = glob.glob(pattern)
            
            if files:
                try:
                    raw_data = np.loadtxt(files[0])
                    actual_end = min(end_idx, len(raw_data))
                    data_slice = raw_data[start_idx:actual_end]
                    time = np.arange(start_idx, actual_end) / fs
                    
                    ax.plot(time, data_slice, lw=0.6, color='C'+str(i))
                    ax.set_title(f"{task_names[task_id]} R{round_id}", fontsize=9)
                    ax.grid(True, alpha=0.3)
                    
                    # === 修改 2：手動固定 Y 軸顯示範圍 ===
                    # 建議設在 -800 到 800 之間，或是根據你觀察到的最大值調整
                    ax.set_ylim(-1000, 1000) 
                    
                except Exception as e:
                    ax.text(0.5, 0.5, "Load Error", ha='center', fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# 執行指令

if __name__ == "__main__":
    student_id = 'b12901035'
    rounds = [17, 18, 19]
    start_sec = 8
    end_sec = 16

    plot_task_fixed_y('bci_dataset_114-2', student_id, rounds, start_sec, end_sec)