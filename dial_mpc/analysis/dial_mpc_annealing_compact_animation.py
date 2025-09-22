import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set up the plot style for clean, professional look matching thesis plots
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.alpha': 0.3,
    'figure.facecolor': 'white'
})

# DIAL-MPC parameters (from the thesis and code)
N_diffuse = 8      # Number of diffusion iterations
H_nodes = 10       # Prediction horizon (control nodes)
alpha_traj = 0.7   # Trajectory annealing factor
alpha_horizon = 0.8 # Action-level annealing factor

# Consistent colorblind-friendly colors matching thesis plots
colors = {
    'dial': '#ff7f0e',          # Orange (DIAL-MPC)
    'vigas': '#1f77b4',         # Blue (VIGAS/nominal)
    'mppi': '#2ca02c',          # Green (MPPI)
    'high_noise': '#d62728',    # Red (high exploration)
    'med_noise': '#ff7f0e',     # Orange (medium)
    'low_noise': '#2ca02c',     # Green (low/refined)
    'envelope': '#1f77b4',      # Blue (matching nominal)
}

def compute_annealing_schedule():
    """Compute the DIAL-MPC annealing schedule."""
    traj_factors = alpha_traj ** np.arange(N_diffuse)
    action_factors = alpha_horizon ** np.arange(H_nodes)[::-1]
    combined_schedule = traj_factors[:, None] * action_factors[None, :]
    return combined_schedule

def generate_sample_trajectories(iteration, combined_schedule, n_samples=15):
    """Generate sample trajectories for the given diffusion iteration."""
    t = np.linspace(0, 2 * np.pi, H_nodes)
    nominal = 0.4 * np.sin(t) + 0.1 * np.cos(2 * t)  # Smoother nominal
    current_noise_profile = combined_schedule[iteration, :]
    
    noise = np.random.randn(n_samples, H_nodes) * current_noise_profile
    trajectories = nominal + noise
    return trajectories, nominal

def create_compact_animation():
    """Create a compact, intuitive DIAL-MPC annealing animation."""
    fig, (ax_heatmap, ax_samples) = plt.subplots(
        2, 1, figsize=(10, 8.5), 
        gridspec_kw={'height_ratios': [1.2, 1.8]},
    )
    
    fig_title = fig.suptitle('', fontsize=16, fontweight='bold')
    
    combined_schedule = compute_annealing_schedule()
    
    # --- Top Plot: Annealing Schedule with Clear Labels ---
    im = ax_heatmap.imshow(combined_schedule, aspect='auto', origin='lower',
                           cmap='Oranges', vmin=0, vmax=combined_schedule.max())
    ax_heatmap.set_ylabel('Diffusion Iteration', fontsize=12, fontweight='bold')
    ax_heatmap.set_title('1. Dual Annealing Schedule', fontsize=14, fontweight='bold', pad=10)
    ax_heatmap.set_yticks(np.arange(N_diffuse))
    ax_heatmap.set_yticklabels([f'{i+1}' for i in range(N_diffuse)])
    ax_heatmap.set_xticks(np.arange(H_nodes))
    ax_heatmap.set_xticklabels([]) # Remove x-ticks, explained by text below
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_heatmap, fraction=0.046, pad=0.04)
    cbar.set_label('Noise Scale', rotation=270, labelpad=15, fontweight='bold')
    
    # Correctly position vertical annotations to avoid overlap and align with variance levels
    ax_heatmap.text(-1.8, N_diffuse - 2.5, 'Low Var.\n(Refine)', 
                    rotation=90, ha='center', va='center', fontweight='bold',
                    color=colors['low_noise'], fontsize=10)
    ax_heatmap.text(-1.8, 1.5, 'High Var.\n(Explore)', 
                    rotation=90, ha='center', va='center', fontweight='bold', 
                    color=colors['high_noise'], fontsize=10)
    
    ax_heatmap.text(2, -2.0, 'Immediate Actions\n(Low Noise)', 
                    ha='center', va='top', fontweight='bold',
                    color=colors['low_noise'], fontsize=10)
    ax_heatmap.text(7.5, -2.0, 'Future Actions\n(High Noise)', 
                    ha='center', va='top', fontweight='bold',
                    color=colors['high_noise'], fontsize=10)
    
    # --- Bottom Plot: Effect on Trajectories ---
    ax_samples.set_ylim(-1.5, 1.5)
    ax_samples.set_xlabel('Control Horizon (Time Steps)', fontsize=14, fontweight='bold')
    ax_samples.set_ylabel('Control Value', fontsize=14, fontweight='bold')
    ax_samples.set_title('2. Effect on Sampled Trajectories', fontsize=14, fontweight='bold', pad=15)
    ax_samples.grid(True, alpha=0.3)
    ax_samples.set_xticks(np.arange(0, H_nodes, 2))
    
    # Animation elements
    highlight_rect = Rectangle((-0.5, -0.5), H_nodes, 1,
                              linewidth=4, edgecolor='black', facecolor='none')
    ax_heatmap.add_patch(highlight_rect)

    nominal_line, = ax_samples.plot([], [], color=colors['vigas'], linewidth=4, 
                                   label='Nominal Trajectory', zorder=10)
    sample_lines = []
    
    iteration_text = ax_heatmap.text(0.98, 0.95, '', transform=ax_heatmap.transAxes,
                                    ha='right', va='top', fontsize=14, fontweight='bold',
                                    bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.3'))

    def animate(frame):
        iteration = frame
        
        highlight_rect.set_y(iteration - 0.5)
        
        trajectories, nominal = generate_sample_trajectories(iteration, combined_schedule)
        
        for line in sample_lines:
            line.remove()
        sample_lines.clear()
        
        if iteration < N_diffuse // 3:
            phase = "EXPLORATION PHASE"
            phase_color = colors['high_noise']
            sample_color = colors['high_noise']
            alpha_val = 0.6
            description = "High variance → Global search"
        elif iteration < 2 * N_diffuse // 3:
            phase = "TRANSITION PHASE"
            phase_color = colors['med_noise']
            sample_color = colors['med_noise']
            alpha_val = 0.7
            description = "Medium variance → Balanced search"
        else:
            phase = "REFINEMENT PHASE"
            phase_color = colors['low_noise']
            sample_color = colors['low_noise']
            alpha_val = 0.8
            description = "Low variance → Local optimization"
        
        fig_title.set_text(f'DIAL-MPC: {phase}\n({description})')
        fig_title.set_color(phase_color)
        iteration_text.set_text(f'Iter. {iteration + 1}/{N_diffuse}')
        
        for _, traj in enumerate(trajectories):
            line, = ax_samples.plot(np.arange(H_nodes), traj, 
                                   color=sample_color, linewidth=1.5, alpha=alpha_val)
            sample_lines.append(line)
        
        nominal_line.set_data(np.arange(H_nodes), nominal)
        
        current_std = np.sqrt(combined_schedule[iteration, :])
        for patch in ax_samples.collections:
            if hasattr(patch, '_alpha') and patch._alpha == 0.15:
                patch.remove()
        
        ax_samples.fill_between(np.arange(H_nodes), 
                               nominal - current_std, nominal + current_std,
                               color=colors['envelope'], alpha=0.15, 
                               label='±1σ Noise Variance' if iteration == 0 else "")
        
        for child in ax_samples.get_children():
            if isinstance(child, plt.Annotation):
                child.remove()
        
        if iteration == 0:
            ax_samples.annotate('Wide exploration', 
                              xy=(7, 1.0), xytext=(4.5, 1.2),
                              arrowprops=dict(arrowstyle='->', color=colors['high_noise'], lw=2),
                              ha='center', fontsize=11, fontweight='bold', color=colors['high_noise'])
        elif iteration == N_diffuse - 1:
            ax_samples.annotate('Focused refinement', 
                              xy=(7, 0.3), xytext=(4, 0.9),
                              arrowprops=dict(arrowstyle='->', color=colors['low_noise'], lw=2),
                              ha='center', fontsize=11, fontweight='bold', color=colors['low_noise'])

    legend_elements = [
        plt.Line2D([0], [0], color=colors['vigas'], linewidth=4, label='Nominal Trajectory'),
        plt.Line2D([0], [0], color=colors['high_noise'], linewidth=2, alpha=0.7, label='Sample Trajectories'),
        mpatches.Patch(color=colors['envelope'], alpha=0.15, label='±1σ Noise Variance')
    ]
    ax_samples.legend(handles=legend_elements, loc='upper left', framealpha=0.9)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.08, left=0.15, right=0.95, hspace=0.5)

    anim = animation.FuncAnimation(fig, animate, frames=N_diffuse,
                                   interval=1400, repeat=True, blit=False)
    
    try:
        print("Saving improved DIAL-MPC animation...")
        anim.save('dial_mpc_annealing_clear.gif', writer='pillow', fps=0.7, dpi=120)
        anim.save('dial_mpc_annealing_clear.mp4', writer='ffmpeg', fps=0.7, dpi=120,
                  extra_args=['-vcodec', 'libx264'])
        print("Files saved: dial_mpc_annealing_clear.gif, dial_mpc_annealing_clear.mp4")
    except Exception as e:
        print(f"Error saving animation: {e}")

    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    create_compact_animation() 