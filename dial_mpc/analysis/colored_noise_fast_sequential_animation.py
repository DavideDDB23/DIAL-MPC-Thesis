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
    'white': '#606060',    # Visible Gray
    'pink': '#FF69B4',     # Hot Pink
    'brown': '#A52A2A',    # Brown
}

def generate_colored_noise_from_source(white_noise_fft, beta, length, fs=100.0):
    """Generate colored noise from a pre-computed FFT of a white noise source."""
    freqs = np.fft.rfftfreq(length, 1/fs)
    amp = np.where(freqs == 0.0, 0.0, freqs ** (-beta / 2.0))
    
    colored_fft = white_noise_fft * amp
    colored = np.fft.irfft(colored_fft, n=length)
    
    colored = colored - np.mean(colored)
    colored = colored / (np.std(colored) + 1e-8)
    return colored

def create_fast_sequential_noise_animation():    
    # Signal parameters
    fs = 100.0
    dt = 1.0 / fs
    total_duration = 8.0
    
    t_total = np.arange(0, total_duration, dt)
    
    # Generate a single white noise source signal for consistency
    np.random.seed(42)
    white_noise_source = np.random.normal(0, 1, len(t_total))
    white_noise_source_fft = np.fft.rfft(white_noise_source)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    ax.set_ylim(-3.5, 3.5)
    ax.set_xlabel('Time (s)', fontweight='bold', fontsize=14)
    ax.set_ylabel('Noise Amplitude', fontweight='bold', fontsize=14)
    ax.grid(True)
    
    line, = ax.plot([], [], linewidth=3)
    glow_line, = ax.plot([], [], linewidth=7, alpha=0.3)
    title = ax.set_title('', fontsize=20, fontweight='bold', pad=20)
    
    # Animation parameters
    fps = 30
    stay_duration = 2  # seconds
    morph_duration = 1 # seconds
    pause_duration = 1 # seconds

    stay_frames = stay_duration * fps
    morph_frames = morph_duration * fps
    pause_frames = pause_duration * fps

    # Frame breakdown
    f_white_end = stay_frames
    f_morph1_end = f_white_end + morph_frames
    f_pink_end = f_morph1_end + stay_frames
    f_morph2_end = f_pink_end + morph_frames
    f_brown_end = f_morph2_end + stay_frames
    total_frames = f_brown_end + pause_frames
    
    def animate(frame):
        # Determine current phase
        if frame < f_white_end: # Stay on White
            beta = 0.0
            current_color = colors['white']
            title_text = f'White Noise (β = {beta:.1f})'
        elif frame < f_morph1_end: # Morph to Pink
            progress = (frame - f_white_end) / morph_frames
            beta = progress
            start_color = np.array(plt.cm.colors.to_rgb(colors['white']))
            end_color = np.array(plt.cm.colors.to_rgb(colors['pink']))
            current_color = start_color * (1 - progress) + end_color * progress
            title_text = f'Morphing to Pink... (β = {beta:.2f})'
        elif frame < f_pink_end: # Stay on Pink
            beta = 1.0
            current_color = colors['pink']
            title_text = f'Pink Noise (β = {beta:.1f})'
        elif frame < f_morph2_end: # Morph to Brown
            progress = (frame - f_pink_end) / morph_frames
            beta = 1.0 + progress
            start_color = np.array(plt.cm.colors.to_rgb(colors['pink']))
            end_color = np.array(plt.cm.colors.to_rgb(colors['brown']))
            current_color = start_color * (1 - progress) + end_color * progress
            title_text = f'Morphing to Brown... (β = {beta:.2f})'
        else: # Stay on Brown
            beta = 2.0
            current_color = colors['brown']
            title_text = f'Brown Noise (β = {beta:.1f})'

        current_noise = generate_colored_noise_from_source(white_noise_source_fft, beta, len(t_total), fs)

        # Update scrolling window
        window_duration = 5.0
        window_samples = int(window_duration * fs)
        
        total_scroll_duration = total_duration - window_duration
        scroll_progress = frame / (f_brown_end -1) # Scroll continuously through active phases
        scroll_progress = min(scroll_progress, 1.0) # Clamp progress at the end
        start_time = scroll_progress * total_scroll_duration
        
        start_idx = int(start_time * fs)
        end_idx = start_idx + window_samples
        
        t_window = t_total[start_idx:end_idx] - t_total[start_idx]
        noise_window = current_noise[start_idx:end_idx]

        line.set_data(t_window, noise_window)
        line.set_color(current_color)
        glow_line.set_data(t_window, noise_window)
        glow_line.set_color(current_color)
        
        title.set_text(title_text)
        title.set_color(current_color)
        
        ax.set_xlim(0, window_duration)
        
        return [line, glow_line, title]

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    anim = animation.FuncAnimation(fig, animate, frames=total_frames, interval=1000/fps, blit=True)
    
    try:
        print("Saving fast sequential colored noise animation...")
        anim.save('colored_noise_fast_sequential.gif', writer='pillow', fps=fps, dpi=120)
        print("Files saved: colored_noise_fast_sequential.gif")
    except Exception as e:
        print(f"Error saving animation: {e}")

    plt.show()

if __name__ == "__main__":
    create_fast_sequential_noise_animation() 