#!/usr/bin/env python3
"""
Animated plot illustrating the effect of band-limited noise filter in DIAL-MPC.
Shows the filtering process in real-time for immediate understanding.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scienceplots

# Set style for clean, presentation-ready plots
plt.style.use(['science', 'no-latex'])

# Set up the plot style for a clean, professional look
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 14,
    'axes.linewidth': 1.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.alpha': 0.4,
    'figure.facecolor': 'white'
})

# Colorblind-friendly Wong palette
COLORS = {
    'orange': '#E69F00',
    'sky_blue': '#56B4E9',
    'green': '#009E73',
    'yellow': '#F0E442',
    'blue': '#0173B2',
    'red': '#CC79A7',
    'black': '#000000',
    'gray': '#808080'
}

def generate_bandlimited_noise(fmax, dt, T, num_harmonics=8, seed=42):
    """Generate band-limited noise by summing random Fourier modes up to fmax."""
    np.random.seed(seed)
    t = np.arange(T) * dt
    
    # Sample frequencies uniformly in (0, fmax]
    freqs = np.random.uniform(1e-3, fmax, num_harmonics)
    phases = 2 * np.pi * np.random.uniform(0, 1, num_harmonics)
    amps = np.random.normal(0, 1, num_harmonics) / np.sqrt(num_harmonics)
    
    # Sum sinusoids
    signal_sum = np.zeros_like(t)
    for i in range(num_harmonics):
        signal_sum += amps[i] * np.cos(2 * np.pi * freqs[i] * t + phases[i])
    
    # Zero-mean and unit-variance normalization
    signal_sum = signal_sum - np.mean(signal_sum)
    signal_sum = signal_sum / (np.std(signal_sum) + 1e-8)
    
    return signal_sum

def create_bandlimited_animation():
    """Create a fast, animated plot showing the band-limited filtering effect."""
    
    # Parameters
    fmax = 2.0  # Hz
    num_harmonics = 8
    fs = 100.0  # Hz
    dt = 1.0 / fs
    duration = 4.0  # seconds
    T = int(duration / dt)
    
    t = np.arange(T) * dt
    
    # Generate signals
    np.random.seed(42)
    white_noise = np.random.normal(0, 1, T)
    white_noise = white_noise - np.mean(white_noise)
    white_noise = white_noise / np.std(white_noise)
    
    bandlimited_noise = generate_bandlimited_noise(fmax, dt, T, num_harmonics)
    
    # Create figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
    
    # Animation phases for 8-second total duration
    total_frames = 80 # 8 seconds at 10 fps
    phase1_frames = 25   # Show white noise
    phase2_frames = 30   # Show filtering process
    phase3_frames = 25   # Show final band-limited result
    
    def animate(frame):
        ax.clear()
        
        if frame < phase1_frames:
            # Phase 1: Show white noise
            progress = frame / phase1_frames
            show_length = int(progress * len(t))
            if show_length > 0:
                ax.plot(t[:show_length], white_noise[:show_length],
                           color=COLORS['gray'], linewidth=2, label='White Noise')
            ax.set_title('Step 1: Standard White Noise (Rapid, Uncorrelated)', color=COLORS['gray'], weight='bold', fontsize=18, pad=15)

        elif frame < phase1_frames + phase2_frames:
            # Phase 2: Show filtering process
            progress = (frame - phase1_frames) / phase2_frames
            blended_signal = (1 - progress) * white_noise + progress * bandlimited_noise
            ax.plot(t, blended_signal, color=COLORS['orange'], linewidth=2.5,
                        alpha=0.7 + 0.3 * progress, label='Filtering...')
            ax.set_title(f'Step 2: Applying Band-Limited Filter (f_max = {fmax} Hz)', color=COLORS['orange'], weight='bold', fontsize=18, pad=15)

        else:
            # Phase 3: Show final band-limited result
            ax.plot(t, bandlimited_noise, color=COLORS['green'], linewidth=2.5,
                        label=f'Band-Limited Noise')
            # Highlight smoothness with annotation
            ax.annotate('Result: Smooth, Correlated Trajectories',
                           xy=(2.0, bandlimited_noise[int(2.0/dt)] + 0.2),
                           xytext=(1.5, 2.5),
                           arrowprops=dict(arrowstyle='->', color=COLORS['green'], lw=2),
                           fontsize=12, ha='center', color=COLORS['green'], weight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            ax.set_title('Step 3: Band-Limited Result', color=COLORS['green'], weight='bold', fontsize=18, pad=15)
        
        # Common formatting
        ax.set_xlabel('Time (s)', fontsize=12, weight='bold')
        ax.set_ylabel('Noise Amplitude', fontsize=12, weight='bold')
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, duration)
        ax.set_ylim(-3.5, 3.5)
        
        # Completely remove top and right spines and their ticks
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(top=False, right=False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=total_frames, 
                                  interval=100, repeat=True, blit=False)
    
    # Adjust layout to ensure title fits within image bounds
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(top=0.85)  # Leave more room for the title
    
    # Save as GIF
    print("Generating fast band-limited noise filter animation...")
    anim.save('bandlimited_noise_fast_animation.gif', writer='pillow', fps=10, dpi=120)
    print("Saved bandlimited_noise_fast_animation.gif")
    
    plt.show()
    
    return anim

if __name__ == "__main__":
    create_bandlimited_animation() 