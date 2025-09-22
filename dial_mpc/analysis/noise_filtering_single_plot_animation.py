import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import signal

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 14,
    'axes.linewidth': 1.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.alpha': 0.4,
    'figure.facecolor': 'white'
})

colors = {
    'white_noise': '#d62728',    # Red (chaotic, unstructured)
    'filtered': '#2ca02c',       # Green (smooth, structured)
}

def apply_butterworth_lowpass(data, fs=50.0, cutoff=2.0):
    """Apply second-order Butterworth low-pass filter with forward-backward filtering (zero phase)."""
    nyquist = fs / 2.0
    normalized_cutoff = cutoff / nyquist
    b, a = signal.butter(2, normalized_cutoff, btype='low')
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def create_single_plot_noise_animation():
    """Create a visually clear, single-plot animation of noise filtering."""
    
    # Signal parameters
    fs = 50.0
    dt = 1.0 / fs
    cutoff_freq = 2.0
    window_duration = 5.0
    total_duration = 15.0
    
    t_total = np.arange(0, total_duration, dt)
    window_samples = int(window_duration * fs)
    
    # Generate signals
    np.random.seed(42)
    white_noise = np.random.normal(0, 1, len(t_total))
    filtered_noise = apply_butterworth_lowpass(white_noise, fs, cutoff_freq)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.set_ylim(-3.5, 3.5)
    ax.set_xlabel('Time (s)', fontweight='bold', fontsize=14)
    ax.set_ylabel('Noise Amplitude', fontweight='bold', fontsize=14)
    ax.grid(True)
    
    # Initialize plot elements
    # White noise will be in the background
    line_white, = ax.plot([], [], color=colors['white_noise'], linewidth=1.5, 
                         alpha=0.5, label='Original White Noise', zorder=1)
    
    # Shaded region for white noise
    fill_collection = ax.fill_between([], [], [], color=colors['white_noise'], 
                                      alpha=0.1, zorder=0)
    
    # Add a "glow" effect for the filtered line
    line_filtered_glow, = ax.plot([], [], color=colors['filtered'], linewidth=6.5, 
                                 alpha=0.3, zorder=9)
    # Filtered noise will be prominent in the foreground
    line_filtered, = ax.plot([], [], color=colors['filtered'], linewidth=3.5, 
                            label='Low-Pass Filtered Noise (Smooth)', zorder=10)
    
    ax.legend(loc='upper right')
    
    def animate(frame):
        nonlocal fill_collection
        
        # Calculate current time window
        progress = frame / 200.0
        current_time = progress * (total_duration - window_duration)
        
        start_idx = int(current_time * fs)
        end_idx = start_idx + window_samples
        
        if end_idx >= len(t_total):
            end_idx = len(t_total) - 1
            start_idx = end_idx - window_samples

        t_window = t_total[start_idx:end_idx] - t_total[start_idx]
        white_window = white_noise[start_idx:end_idx]
        filtered_window = filtered_noise[start_idx:end_idx]
        
        # Update plot data
        line_white.set_data(t_window, white_window)
        line_filtered.set_data(t_window, filtered_window)
        line_filtered_glow.set_data(t_window, filtered_window)
        
        # Update shaded region
        # This is done by removing the old one and adding a new one
        fill_collection.remove()
        fill_collection = ax.fill_between(t_window, white_window, 0, 
                                          color=colors['white_noise'], alpha=0.1, zorder=0)

        ax.set_xlim(0, window_duration)
        
        return [line_white, line_filtered, line_filtered_glow, fill_collection]

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    # Create and save animation
    anim = animation.FuncAnimation(fig, animate, frames=201, interval=120, blit=True, repeat=True)
    
    try:
        print("Saving single-plot noise filtering animation...")
        anim.save('noise_filtering_single_plot.gif', writer='pillow', fps=12, dpi=120)
        print("Files saved: noise_filtering_single_plot.gif")
    except Exception as e:
        print(f"Error saving animation: {e}")

    plt.show()

if __name__ == "__main__":
    create_single_plot_noise_animation() 