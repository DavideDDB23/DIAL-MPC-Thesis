import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scienceplots

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

def sample_lowrank_diagonal(mean, L, d, n_samples=100):
    """
    Sample from low-rank plus diagonal covariance: Σ = L·L^T + diag(d)
    Uses the VIGAS sampling formula: Y = μ + eps_k @ L.T + eps_d * sqrt(d)
    """
    D, k = L.shape
    eps_k = np.random.normal(0, 1, (n_samples, k))
    lowrank_samples = eps_k @ L.T
    eps_d = np.random.normal(0, 1, (n_samples, D))
    diagonal_samples = eps_d * np.sqrt(d)[None, :]
    return mean[None, :] + lowrank_samples + diagonal_samples

def create_lowrank_diagonal_animation():
    """Create single-plot animation showing low-rank plus diagonal sampling construction."""
    
    # Setup single plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # VIGAS parameters for demonstration
    mean = np.array([0.0, 0.0])
    
    # Low-rank component: creates correlation structure
    L = np.array([[1.2, 0.8], [0.6, 1.0]])  # 2x2 matrix for correlation
    
    # Diagonal component: individual variances
    d = np.array([0.4, 0.3])
    
    # Animation parameters for 10-second duration
    fps = 10
    total_duration = 10
    total_frames = total_duration * fps
    
    # Pre-generate random components for 6 sample points
    n_points = 6
    np.random.seed(42)
    eps_k_samples = np.random.normal(0, 1, (n_points, 2))
    eps_d_samples = np.random.normal(0, 1, (n_points, 2))
    
    def animate(frame):
        ax.clear()
        
        # Determine animation progress
        progress = frame / (total_frames - 1)
        
        # Animation phases (2.5 seconds each)
        if progress < 0.25:
            phase = "Step 1: Start at the Mean (μ)"
            phase_color = COLORS['black']
            show = ['mean']
        elif progress < 0.5:
            phase = "Step 2: Add Low-Rank Correlation (L·ε_k)"
            phase_color = COLORS['orange']
            show = ['mean', 'lowrank']
        elif progress < 0.75:
            phase = "Step 3: Add Residual Variance (√d·ε_d)"
            phase_color = COLORS['gray']
            show = ['mean', 'lowrank', 'diagonal']
        else:
            phase = "Result: Multiple VIGAS Samples"
            phase_color = COLORS['blue']
            show = ['mean', 'lowrank', 'diagonal', 'final']
        
        # Show construction for all 6 sample points
        for i in range(n_points):
            # Calculate components for each sample point
            mean_point = mean
            lowrank_component = eps_k_samples[i] @ L.T
            diagonal_component = np.sqrt(d) * eps_d_samples[i]

            # Show construction step by step
            if 'mean' in show:
                ax.scatter(*mean_point, c=COLORS['black'], s=80, marker='s', 
                          label='μ (Mean)' if i == 0 else "", alpha=0.8)

            if 'lowrank' in show:
                point_after_lr = mean_point + lowrank_component
                ax.scatter(*point_after_lr, c=COLORS['orange'], s=70, marker='^', 
                          label='μ + L·ε_k' if i == 0 else "", alpha=0.8)
                ax.annotate('', xy=point_after_lr, xytext=mean_point,
                           arrowprops=dict(arrowstyle='->', color=COLORS['orange'], 
                                         lw=2, alpha=0.7))

            if 'diagonal' in show:
                point_after_lr = mean_point + lowrank_component
                final_point = point_after_lr + diagonal_component
                ax.scatter(*final_point, c=COLORS['blue'], s=90, marker='o', 
                          label='Final Samples' if i == 0 else "", zorder=5, alpha=0.9)
                ax.annotate('', xy=final_point, xytext=point_after_lr,
                           arrowprops=dict(arrowstyle='->', color=COLORS['gray'], 
                                         lw=2, linestyle='--', alpha=0.7))
        
        if 'final' in show:
            # Add faint cloud of other samples for context
            np.random.seed(100)  # Different seed to avoid overlap with main points
            all_samples = sample_lowrank_diagonal(mean, L, d, 80)
            ax.scatter(all_samples[:, 0], all_samples[:, 1], c=COLORS['blue'], 
                      alpha=0.12, s=15, zorder=0)

        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_xlabel('Control Dimension 1', fontweight='bold')
        ax.set_ylabel('Control Dimension 2', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', framealpha=0.9, fontsize=12)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        fig.suptitle(phase, fontsize=20, fontweight='bold', color=phase_color)
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=total_frames, 
                                  interval=1000/fps, repeat=True, blit=False)
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, bottom=0.12, top=0.85)
    
    # Save as GIF
    print("Generating clearer low-rank plus diagonal sampling construction animation...")
    anim.save('lowrank_diagonal_construction_clear.gif', writer='pillow', fps=fps, dpi=120)
    print("Saved lowrank_diagonal_construction_clear.gif")
    
    plt.show()
    
    return anim

if __name__ == "__main__":
    np.random.seed(42)  # For reproducible results
    create_lowrank_diagonal_animation() 