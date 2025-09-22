import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.alpha': 0.3,
    'figure.facecolor': 'white'
})

colors = {
    'dial': '#ff7f0e',          # Orange (DIAL-MPC)
    'vigas': '#1f77b4',         # Blue (VIGAS/nominal)
    'mppi': '#2ca02c',          # Green (MPPI)
    'high_noise': '#d62728',    # Red (high exploration)
    'med_noise': '#ff7f0e',     # Orange (medium)
    'low_noise': '#2ca02c',     # Green (low/refined)
    'envelope': '#1f77b4',      # Blue (matching nominal)
}

def create_exploration_comparison():
    """Create a side-by-side comparison of Fixed vs Adaptive Exploration."""
    
    fig, (ax_fixed, ax_adaptive) = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle('Exploration Strategies in Model Predictive Control', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    # Time steps for the trajectory
    time_steps = np.linspace(0, 10, 50)
    
    # Create a nominal trajectory (sinusoidal with some complexity)
    nominal_traj = 2.0 * np.sin(0.8 * time_steps) + 0.5 * np.cos(1.5 * time_steps) + 0.1 * time_steps
    
    # --- LEFT PANEL: Fixed Exploration ---
    ax_fixed.set_title('Fixed Exploration\n(e.g., MPPI, DIAL-MPC)', 
                       fontsize=16, fontweight='bold', pad=20)
    
    # Fixed exploration: constant variance throughout
    fixed_variance = np.ones_like(time_steps) * 1.2
    
    # Plot nominal trajectory
    ax_fixed.plot(time_steps, nominal_traj, color=colors['vigas'], 
                  linewidth=4, label='Nominal Trajectory', zorder=10)
    
    # Plot constant exploration envelope
    ax_fixed.fill_between(time_steps, 
                          nominal_traj - fixed_variance, 
                          nominal_traj + fixed_variance,
                          color=colors['high_noise'], alpha=0.3, 
                          label='Exploration Envelope')
    
    # Add sample trajectories with fixed noise
    np.random.seed(42)
    for i in range(8):
        noise = np.random.normal(0, fixed_variance * 0.6, len(time_steps))
        sample_traj = nominal_traj + noise
        ax_fixed.plot(time_steps, sample_traj, color=colors['high_noise'], 
                      alpha=0.6, linewidth=1.5, zorder=5)
    
    # Add constant-sized circles at key points to emphasize fixed nature
    key_points = [10, 25, 40]
    for i, point in enumerate(key_points):
        circle = plt.Circle((time_steps[point], nominal_traj[point]), 
                           radius=1.2, fill=False, color=colors['high_noise'], 
                           linewidth=3, alpha=0.8)
        ax_fixed.add_patch(circle)
        
        if i == 1:  # Middle point
            ax_fixed.annotate('Constant\nExploration', 
                             xy=(time_steps[point], nominal_traj[point] + 1.8),
                             ha='center', va='bottom', fontweight='bold',
                             color=colors['high_noise'], fontsize=11,
                             bbox=dict(boxstyle="round,pad=0.3", 
                                     facecolor='white', alpha=0.8))
    
    ax_fixed.set_ylim(-4, 6)
    ax_fixed.set_xlabel('Time Steps', fontweight='bold', fontsize=14)
    ax_fixed.set_ylabel('Control Value', fontweight='bold', fontsize=14)
    ax_fixed.grid(True, alpha=0.3)
    ax_fixed.legend(loc='upper left', framealpha=0.9)
    
    # --- RIGHT PANEL: Adaptive Exploration ---
    ax_adaptive.set_title('Adaptive Exploration\n(e.g., VIGAS)', 
                          fontsize=16, fontweight='bold', pad=20)
    
    # Adaptive exploration: decreasing variance over time
    adaptive_variance = 2.0 * np.exp(-0.3 * time_steps / time_steps[-1] * 8) + 0.2
    
    # Plot nominal trajectory
    ax_adaptive.plot(time_steps, nominal_traj, color=colors['vigas'], 
                     linewidth=4, label='Nominal Trajectory', zorder=10)
    
    # Plot adaptive exploration envelope
    ax_adaptive.fill_between(time_steps, 
                            nominal_traj - adaptive_variance, 
                            nominal_traj + adaptive_variance,
                            color=colors['vigas'], alpha=0.2, 
                            label='Adaptive Exploration Envelope')
    
    # Add sample trajectories with adaptive noise
    np.random.seed(42)
    colors_adaptive = [colors['high_noise'], colors['med_noise'], colors['low_noise']]
    n_segments = 3
    segment_length = len(time_steps) // n_segments
    
    for i in range(8):
        sample_traj = np.zeros_like(time_steps)
        for seg in range(n_segments):
            start_idx = seg * segment_length
            end_idx = (seg + 1) * segment_length if seg < n_segments - 1 else len(time_steps)
            
            # Decreasing noise over segments
            noise_scale = adaptive_variance[start_idx:end_idx] * 0.6
            noise = np.random.normal(0, noise_scale, end_idx - start_idx)
            sample_traj[start_idx:end_idx] = nominal_traj[start_idx:end_idx] + noise
            
            # Plot each segment with appropriate color
            color = colors_adaptive[seg]
            alpha = 0.8 if seg == 0 else 0.7 if seg == 1 else 0.6
            ax_adaptive.plot(time_steps[start_idx:end_idx], sample_traj[start_idx:end_idx], 
                            color=color, alpha=alpha, linewidth=1.5, zorder=5)
    
    # Add shrinking circles at key points to emphasize adaptive nature
    for i, point in enumerate(key_points):
        radius = adaptive_variance[point] * 0.8
        color = colors_adaptive[i] if i < len(colors_adaptive) else colors['low_noise']
        
        circle = plt.Circle((time_steps[point], nominal_traj[point]), 
                           radius=radius, fill=False, color=color, 
                           linewidth=3, alpha=0.8)
        ax_adaptive.add_patch(circle)
        
        # Add phase labels
        if i == 0:
            ax_adaptive.annotate('Exploration', 
                               xy=(time_steps[point], nominal_traj[point] + radius + 0.3),
                               ha='center', va='bottom', fontweight='bold',
                               color=colors['high_noise'], fontsize=10)
        elif i == 1:
            ax_adaptive.annotate('Transition', 
                               xy=(time_steps[point], nominal_traj[point] + radius + 0.3),
                               ha='center', va='bottom', fontweight='bold',
                               color=colors['med_noise'], fontsize=10)
        elif i == 2:
            ax_adaptive.annotate('Refinement', 
                               xy=(time_steps[point], nominal_traj[point] + radius + 0.3),
                               ha='center', va='bottom', fontweight='bold',
                               color=colors['low_noise'], fontsize=10)
    
    # Add arrows showing the adaptation direction
    ax_adaptive.annotate('', xy=(time_steps[35], 4.5), xytext=(time_steps[15], 4.5),
                        arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    ax_adaptive.text(time_steps[25], 5.0, 'Adaptive Learning', 
                    ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax_adaptive.set_ylim(-4, 6)
    ax_adaptive.set_xlabel('Time Steps', fontweight='bold', fontsize=14)
    ax_adaptive.set_ylabel('Control Value', fontweight='bold', fontsize=14)
    ax_adaptive.grid(True, alpha=0.3)
    ax_adaptive.legend(loc='upper right', framealpha=0.9)
    
    # Add comparison annotations
    fig.text(0.25, 0.08, '• Constant exploration\n• Inefficient sampling\n• No learning', 
             ha='center', va='top', fontsize=11, 
             bbox=dict(boxstyle="round,pad=0.4", facecolor=colors['high_noise'], alpha=0.1))
    
    fig.text(0.75, 0.08, '• Decreasing exploration\n• Efficient sampling\n• Learns optimal regions', 
             ha='center', va='top', fontsize=11,
             bbox=dict(boxstyle="round,pad=0.4", facecolor=colors['vigas'], alpha=0.1))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.2)
    
    plt.savefig('fixed_vs_adaptive_exploration.png', dpi=300, bbox_inches='tight')
    print("Saved: fixed_vs_adaptive_exploration.png")
    
    plt.show()

if __name__ == "__main__":
    create_exploration_comparison() 