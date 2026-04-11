import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
import glob

def bandpass_filter(data, fs, low=0.1, high=45):
    b, a = signal.butter(4, [low/(fs/2), high/(fs/2)], btype='band')
    return signal.filtfilt(b, a, data)

def notch_filter(data, fs, freq=4):
    b, a = signal.iirnotch(freq/(fs/2), Q=10)
    return signal.filtfilt(b, a, data)

def bandstop_filter(data, fs, low=3, high=5, order=4):
    b, a = signal.butter(
        order,
        [low/(fs/2), high/(fs/2)],
        btype='bandstop'
    )
    return signal.filtfilt(b, a, data)



def plot_task_fixed_y(dataset_path, student_id, rounds, start_sec, end_sec, fs=512):

    tasks = [1, 2, 3]
    # rounds = [17, 18, 19]
    task_names = {1: "Relax", 2: "Focus", 3: "Blink"}

    start_idx = int(start_sec * fs)
    end_idx = int(end_sec * fs)

    
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
                    # =============================================
                    # ==== adding filters =========================
                    # raw_data = bandpass_filter(raw_data, fs)
                    # raw_data = notch_filter(raw_data, fs)
                    # raw_data = bandstop_filter(raw_data, fs)
                    # =============================================
                    actual_end = min(end_idx, len(raw_data))
                    data_slice = raw_data[start_idx:actual_end]
                    time = np.arange(start_idx, actual_end) / fs
                    
                    ax.plot(time, data_slice, lw=0.6, color='C'+str(i))
                    ax.set_title(f"{task_names[task_id]} R{round_id}", fontsize=9)
                    ax.grid(True, alpha=0.3)
                    
                    
                    ax.set_ylim(-1000, 1000) 
                    
                except Exception as e:
                    ax.text(0.5, 0.5, "Load Error", ha='center', fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



if __name__ == "__main__":
    student_id = 'b12901016'
    rounds = [18, 19, 20]
    start_sec = 10
    end_sec = 15

    plot_task_fixed_y('bci_dataset_114-2', student_id, rounds, start_sec, end_sec)