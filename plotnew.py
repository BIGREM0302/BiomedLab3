import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def plot_task_fixed_y(dataset_path, student_id, rounds, start_sec=8, end_sec=13, fs=512):
    """
    固定 Y 軸範圍並在所有子圖間共用 Y 軸，支援動態數量的回合 (Rounds)
    """
    tasks = [1, 2, 3]
    task_names = {1: "Relax", 2: "Focus", 3: "Blink"}

    start_idx = int(start_sec * fs)
    end_idx = int(end_sec * fs)

    # === 修改 1：動態根據傳入的 rounds 數量決定欄數 ===
    n_tasks = len(tasks)
    n_rounds = len(rounds)
    
    # 為了容納 10 個 column，我們把圖片寬度動態拉長 (每個 round 寬度約 2.5，最多不超過 25)
    fig_width = min(2.5 * n_rounds, 25)
    fig, axes = plt.subplots(n_tasks, n_rounds, figsize=(fig_width, 6), sharex=True, sharey=True)
    fig.suptitle(f"EEG Segment (Fixed Y): {student_id} (Rounds: {rounds[0]}~{rounds[-1]})", fontsize=16)

    for i, task_id in enumerate(tasks):
        for j, round_id in enumerate(rounds):
            # 確保 axes 的索引正確 (即使只有 1 個 round 也能處理)
            if n_tasks == 1 and n_rounds == 1: ax = axes
            elif n_tasks == 1: ax = axes[j]
            elif n_rounds == 1: ax = axes[i]
            else: ax = axes[i, j]
                
            pattern = os.path.join(dataset_path, student_id, f"*_{task_id}_{round_id}.txt")
            files = glob.glob(pattern)
            
            if files:
                try:
                    # 加入強健的讀取方式，避免被 \r\n 或壞字元報錯
                    with open(files[0], 'r', encoding='utf-8', errors='ignore') as file:
                        clean_data = [float(line.strip()) for line in file if line.strip()]
                    raw_data = np.array(clean_data)
                    
                    actual_end = min(end_idx, len(raw_data))
                    data_slice = raw_data[start_idx:actual_end]
                    time = np.arange(start_idx, actual_end) / fs
                    
                    ax.plot(time, data_slice, lw=0.8, color='C'+str(i))
                    ax.set_title(f"{task_names[task_id]} R{round_id}", fontsize=10)
                    ax.grid(True, alpha=0.3)
                    
                    # === 修改 2：手動固定 Y 軸顯示範圍 ===
                    # 配合真實的眨眼數據，將範圍拉寬至正負 1500
                    ax.set_ylim(-1500, 1500) 
                    
                except Exception as e:
                    ax.text(0.5, 0.5, "Load Error", ha='center', va='center', transform=ax.transAxes, color='red')
            else:
                ax.text(0.5, 0.5, "No File", ha='center', va='center', transform=ax.transAxes, color='gray')
                
            # 隱藏內側的座標軸數字，讓版面更乾淨
            if i < n_tasks - 1:
                ax.tick_params(labelbottom=False)
            if j > 0:
                ax.tick_params(labelleft=False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88) # 留出標題空間
    return fig

if __name__ == "__main__":
    # 設定資料夾路徑與學號
    path = "bci_dataset_114-2"
    sid = "b12901028" # 可替換為其他組員學號
    
    # === 關鍵修改：指定要畫的 Round 列表 ===
    # 這裡用 range 產生 1 到 10 的列表: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    target_rounds = list(range(1, 11)) 
    
    # 呼叫繪圖函數
    plot_task_fixed_y(dataset_path=path, student_id=sid, rounds=target_rounds)
    
    # 儲存圖片 (會自動命名包含學號)
    plt.savefig(f'eeg_10_rounds_plot_{sid}.png', dpi=300, bbox_inches='tight')
    print(f"圖片已儲存為 eeg_10_rounds_plot_{sid}.png")
    
    # 顯示圖表
    plt.show()