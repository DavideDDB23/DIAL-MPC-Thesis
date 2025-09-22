import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
import scienceplots
import argparse

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 14,
    'axes.linewidth': 1.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.alpha': 0.4,
    'figure.facecolor': 'white'
})

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

def create_reward_landscape(x, y):
    """Create a synthetic reward landscape with correlation structure."""
    r1 = -0.5 * ((x - 1)**2 + (y - 0.5)**2)  # Peak 1
    r2 = -0.3 * ((x + 1)**2 + (y + 0.5)**2)  # Peak 2
    r3 = -0.8 * ((x - 0.5)**2 + 2*(y + 1)**2)  # Valley
    ridge = -0.1 * (x - 2*y)**2  # Ridge showing correlation
    return r1 + r2 + r3 + ridge

def confidence_ellipse(mean, cov, ax, n_std=2.0, **kwargs):
    """Draw confidence ellipse for 2D Gaussian distribution."""
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    order = eigenvals.argsort()[::-1]
    eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]
    
    angle = np.degrees(np.arctan2(*eigenvecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigenvals)
    
    ellip = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellip)
    return ellip

def sample_from_covariance(mean, cov, n_samples=50):
    """Sample points from 2D Gaussian distribution."""
    return np.random.multivariate_normal(mean, cov, n_samples)

def create_covariance_evolution_animation(single_panel=False):
    """Create animation showing evolution from fixed to adaptive covariance."""
    
    # Setup
    figsize = (8, 7) if single_panel else (15, 6)
    fig, axes = plt.subplots(1, 2 if not single_panel else 1, figsize=figsize)
    if single_panel:
        ax1 = axes
        ax2 = None
    else:
        ax1, ax2 = axes
    
    # Create reward landscape
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = create_reward_landscape(X, Y)
    
    # Animation parameters for an 8-second total duration
    fps = 10
    stay_duration_s = 2   # Time to show fixed covariance
    morph_duration_s = 4  # Time for transition
    pause_duration_s = 2  # Pause at the end before looping
    
    stay_frames = int(stay_duration_s * fps)
    morph_frames = int(morph_duration_s * fps)
    pause_frames = int(pause_duration_s * fps)
    total_frames = stay_frames + morph_frames + pause_frames
    
    # Initial parameters
    mean_init = np.array([0.0, 0.0])
    cov_fixed = np.array([[1.0, 0.0], [0.0, 1.0]])  # Isotropic
    
    # Target adaptive covariance (learned structure)
    cov_adaptive = np.array([[2.0, 1.2], [1.2, 0.8]])  # Correlated
    mean_adaptive = np.array([0.5, 0.2])  # Shifted toward reward
    
    def animate(frame):
        ax1.clear()
        if not single_panel:
            ax2.clear()
        
        # Determine animation progress
        if frame < stay_frames:  # Phase 1: Fixed
            progress = 0.0
        elif frame < stay_frames + morph_frames:  # Phase 2: Morphing
            progress = (frame - stay_frames) / morph_frames
        else:  # Phase 3: Paused at adaptive
            progress = 1.0

        if progress == 0.0:
            title1 = "Fixed Covariance (MPPI/DIAL-MPC)"
            title2 = "Isotropic Distribution" if not single_panel else ""
            color = COLORS['gray']
        elif progress < 1.0:
            title1 = "Adaptive Covariance (VIGAS)"
            title2 = f"Learning... {int(progress*100)}%" if not single_panel else ""
            color = COLORS['orange']
        else: # progress == 1.0
            title1 = "Adaptive Covariance (VIGAS)"
            title2 = "Learning... 100%" if not single_panel else ""
            color = COLORS['blue']

        # Interpolate between fixed and adaptive
        current_cov = (1 - progress) * cov_fixed + progress * cov_adaptive
        current_mean = (1 - progress) * mean_init + progress * mean_adaptive
        
        # Left plot: Covariance visualization with reward landscape
        contour = ax1.contour(X, Y, Z, levels=10, alpha=0.3, colors='lightblue')
        ax1.contourf(X, Y, Z, levels=10, alpha=0.2, cmap='Blues')
        
        # Draw confidence ellipse
        confidence_ellipse(current_mean, current_cov, ax1, n_std=2.0, 
                          facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
        
        # Sample and plot points
        samples = sample_from_covariance(current_mean, current_cov, n_samples=100)
        ax1.scatter(samples[:, 0], samples[:, 1], c=color, alpha=0.6, s=20)
        ax1.scatter(*current_mean, c='red', s=100, marker='*', label='Mean', zorder=5)
        
        ax1.set_xlim(-3, 3)
        ax1.set_ylim(-3, 3)
        ax1.set_xlabel('Control Dimension 1', fontweight='bold')
        ax1.set_ylabel('Control Dimension 2', fontweight='bold')
        ax1.set_title(title1, color=color, fontweight='bold', fontsize=16, pad=15)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')
        
        # Remove top and right spines
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Right plot: Covariance matrix visualization
        if not single_panel:
            ax2.imshow(current_cov, cmap='RdYlBu_r', aspect='equal', vmin=-0.5, vmax=2.5)
            
            # Add text annotations for matrix values
            for i in range(2):
                for j in range(2):
                    text = ax2.text(j, i, f'{current_cov[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontweight='bold', fontsize=14)
            
            ax2.set_xticks([0, 1])
            ax2.set_yticks([0, 1])
            ax2.set_xticklabels(['Dim 1', 'Dim 2'], fontweight='bold')
            ax2.set_yticklabels(['Dim 1', 'Dim 2'], fontweight='bold')
            ax2.set_title(title2, color=color, fontweight='bold', fontsize=16, pad=15)
            
            # Add correlation information
            correlation = current_cov[0, 1] / np.sqrt(current_cov[0, 0] * current_cov[1, 1])
            ax2.text(0.5, -0.35, f'Correlation: {correlation:.3f}', 
                    transform=ax2.transAxes, ha='center', fontweight='bold', fontsize=12)
            
            # Remove spines for right plot too
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax2.spines['bottom'].set_visible(False)
            ax2.tick_params(left=False, bottom=False)

        # Add phase indicator
        if progress == 0.0:
            phase_text = "Phase 1: Isotropic Exploration"
            phase_color = COLORS['gray']
        else: # progress > 0.0, combines morphing and final pause states
            phase_text = f"Phase 2: Learning Structure (Iter {int(progress * 60)})"
            if progress < 1.0:
                phase_color = COLORS['orange']
            else: # progress == 1.0
                phase_color = COLORS['blue']
        
        fig.suptitle(phase_text, fontsize=20, fontweight='bold', color=phase_color)
        
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=total_frames, 
                                  interval=1000/fps, repeat=True, blit=False)
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, bottom=0.15, top=0.80, wspace=0.3) # Add space for suptitle and axis labels
    
    # Save as GIF
    filename_suffix = "_single_panel" if single_panel else ""
    filename = f'fixed_to_adaptive_covariance{filename_suffix}.gif'
    print(f"Generating {filename}...")
    anim.save(filename, writer='pillow', fps=fps, dpi=120)
    print(f"Saved {filename}")
    
    plt.show()
    
    return anim

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate covariance evolution animation.')
    parser.add_argument('--single-panel', action='store_true', 
                        help='Generate a single-panel version for presentations.')
    args = parser.parse_args()

    np.random.seed(42)  # For reproducible results
    create_covariance_evolution_animation(single_panel=args.single_panel) 