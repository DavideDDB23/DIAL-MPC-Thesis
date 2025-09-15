import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set up the plot style for clean, professional look
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
N_diffuse = 8  # Number of diffusion iterations
H_nodes = 10   # Prediction horizon (control nodes)
alpha_traj = 0.7  # Trajectory annealing factor
alpha_horizon = 0.8  # Action-level annealing factor

# Colorblind-friendly colors
colors = {
    'high_variance': '#d62728',      # Red
    'medium_variance': '#ff7f0e',    # Orange  
    'low_variance': '#2ca02c',       # Green
    'trajectory': '#1f77b4',        # Blue
    'action': '#9467bd',            # Purple
    'background': '#f0f0f0'         # Light gray
}

def compute_annealing_schedule():
    """Compute the DIAL-MPC annealing schedule."""
    # Trajectory-level annealing (exponential decay)
    traj_factors = alpha_traj ** np.arange(N_diffuse)
    
    # Action-level annealing (horizon shaping)
    action_factors = alpha_horizon ** np.arange(H_nodes)[::-1]
    
    # Combined schedule: outer product
    combined_schedule = traj_factors[:, None] * action_factors[None, :]
    
    return traj_factors, action_factors, combined_schedule

def generate_sample_trajectories(iteration, n_samples=15):
    """Generate sample trajectories for the given diffusion iteration."""
    traj_factors, action_factors, combined_schedule = compute_annealing_schedule()
    
    # Base nominal trajectory (smooth sinusoidal)
    t = np.linspace(0, 2*np.pi, H_nodes)
    nominal = 0.5 * np.sin(t) + 0.2 * np.cos(2*t)
    
    # Current noise levels
    current_noise = combined_schedule[iteration, :]
    
    # Generate sample trajectories
    trajectories = []
    for _ in range(n_samples):
        noise = np.random.randn(H_nodes) * current_noise
        trajectory = nominal + noise
        trajectories.append(trajectory)
    
    return np.array(trajectories), nominal

def create_animation():
    """Create the DIAL-MPC annealing animation."""
    fig = plt.figure(figsize=(16, 10))
    
    # Create subplots
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2], width_ratios=[1, 1], 
                         hspace=0.3, wspace=0.3)
    
    # Trajectory-level annealing plot
    ax_traj = fig.add_subplot(gs[0, 0])
    # Action-level annealing plot  
    ax_action = fig.add_subplot(gs[0, 1])
    # Combined heatmap
    ax_heatmap = fig.add_subplot(gs[1, :])
    # Sample trajectories
    ax_samples = fig.add_subplot(gs[2, :])
    
    # Compute schedules
    traj_factors, action_factors, combined_schedule = compute_annealing_schedule()
    
    # Setup trajectory-level plot
    ax_traj.set_xlim(-0.5, N_diffuse-0.5)
    ax_traj.set_ylim(0, 1.1)
    ax_traj.set_xlabel('Diffusion Iteration')
    ax_traj.set_ylabel('Trajectory Noise Scale')
    ax_traj.set_title('Trajectory-Level Annealing\n(Outer Loop)', fontweight='bold')
    ax_traj.grid(True, alpha=0.3)
    
    # Setup action-level plot
    ax_action.set_xlim(-0.5, H_nodes-0.5)
    ax_action.set_ylim(0, 1.1)
    ax_action.set_xlabel('Control Node')
    ax_action.set_ylabel('Action Noise Scale')
    ax_action.set_title('Action-Level Annealing\n(Horizon Shaping)', fontweight='bold')
    ax_action.grid(True, alpha=0.3)
    
    # Setup heatmap
    im = ax_heatmap.imshow(combined_schedule, aspect='auto', origin='lower',
                          cmap='Reds', vmin=0, vmax=combined_schedule.max())
    ax_heatmap.set_xlabel('Control Node')
    ax_heatmap.set_ylabel('Diffusion Iteration')
    ax_heatmap.set_title('Combined Annealing Schedule\n(Trajectory × Action)', fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_heatmap, fraction=0.046, pad=0.04)
    cbar.set_label('Noise Scale', rotation=270, labelpad=15)
    
    # Setup sample trajectories plot
    ax_samples.set_xlim(-0.5, H_nodes-0.5)
    ax_samples.set_ylim(-2, 2)
    ax_samples.set_xlabel('Control Node')
    ax_samples.set_ylabel('Control Value')
    ax_samples.set_title('Sample Trajectories Evolution', fontweight='bold')
    ax_samples.grid(True, alpha=0.3)
    
    # Animation elements
    traj_bars = ax_traj.bar(range(N_diffuse), traj_factors, 
                           color=colors['trajectory'], alpha=0.7)
    action_bars = ax_action.bar(range(H_nodes), action_factors,
                               color=colors['action'], alpha=0.7)
    
    # Current iteration markers
    traj_highlight = Rectangle((0, 0), 1, 1, facecolor=colors['high_variance'], 
                              alpha=0.8, transform=ax_traj.transData)
    action_highlight_patches = []
    
    ax_traj.add_patch(traj_highlight)
    
    # Text annotations
    iteration_text = fig.text(0.02, 0.98, '', fontsize=16, fontweight='bold', 
                             transform=fig.transFigure, va='top')
    phase_text = fig.text(0.02, 0.94, '', fontsize=12, 
                         transform=fig.transFigure, va='top')
    
    def animate(frame):
        iteration = frame
        
        # Update iteration counter
        iteration_text.set_text(f'Diffusion Iteration: {iteration + 1}/{N_diffuse}')
        
        # Update phase description
        if iteration < N_diffuse // 3:
            phase = "Exploration Phase\n(High Variance → Global Search)"
            phase_color = colors['high_variance']
        elif iteration < 2 * N_diffuse // 3:
            phase = "Transition Phase\n(Medium Variance → Balanced)"
            phase_color = colors['medium_variance']  
        else:
            phase = "Refinement Phase\n(Low Variance → Local Optimization)"
            phase_color = colors['low_variance']
            
        phase_text.set_text(phase)
        phase_text.set_color(phase_color)
        
        # Update trajectory-level highlighting
        traj_highlight.set_xy((iteration - 0.4, 0))
        traj_highlight.set_height(traj_factors[iteration])
        
        # Color trajectory bars based on current iteration
        for i, bar in enumerate(traj_bars):
            if i <= iteration:
                # Completed iterations - color by variance level
                val = traj_factors[i]
                if val > 0.7:
                    bar.set_color(colors['high_variance'])
                elif val > 0.3:
                    bar.set_color(colors['medium_variance'])
                else:
                    bar.set_color(colors['low_variance'])
                bar.set_alpha(0.8)
            else:
                # Future iterations - grayed out
                bar.set_color('lightgray')
                bar.set_alpha(0.4)
        
        # Clear previous action highlighting
        for patch in action_highlight_patches:
            patch.remove()
        action_highlight_patches.clear()
        
        # Update action-level highlighting with current combined values
        current_combined = combined_schedule[iteration, :]
        for j in range(H_nodes):
            val = current_combined[j]
            if val > 0.4:
                color = colors['high_variance']
            elif val > 0.15:
                color = colors['medium_variance'] 
            else:
                color = colors['low_variance']
                
            highlight = Rectangle((j - 0.4, 0), 0.8, action_factors[j],
                                facecolor=color, alpha=0.6, 
                                transform=ax_action.transData)
            ax_action.add_patch(highlight)
            action_highlight_patches.append(highlight)
        
        # Update heatmap highlighting
        # Clear previous highlighting
        for patch in ax_heatmap.patches:
            patch.remove()
            
        # Add current iteration highlight
        highlight_rect = Rectangle((-0.5, iteration-0.5), H_nodes, 1,
                                 linewidth=3, edgecolor='black', 
                                 facecolor='none', transform=ax_heatmap.transData)
        ax_heatmap.add_patch(highlight_rect)
        
        # Update sample trajectories
        ax_samples.clear()
        ax_samples.set_xlim(-0.5, H_nodes-0.5)
        ax_samples.set_ylim(-2, 2)
        ax_samples.set_xlabel('Control Node')
        ax_samples.set_ylabel('Control Value')
        ax_samples.set_title('Sample Trajectories Evolution', fontweight='bold')
        ax_samples.grid(True, alpha=0.3)
        
        # Generate and plot sample trajectories
        trajectories, nominal = generate_sample_trajectories(iteration)
        
        # Plot sample trajectories with varying alpha based on variance
        alpha_samples = max(0.1, min(0.7, combined_schedule[iteration, 0]))
        for traj in trajectories:
            ax_samples.plot(range(H_nodes), traj, color='red', 
                          alpha=alpha_samples, linewidth=1)
        
        # Plot nominal trajectory
        ax_samples.plot(range(H_nodes), nominal, color=colors['trajectory'], 
                       linewidth=3, label='Nominal Trajectory')
        
        # Add variance envelope
        current_std = np.sqrt(current_combined)
        ax_samples.fill_between(range(H_nodes), nominal - current_std, 
                              nominal + current_std, color=colors['trajectory'], 
                              alpha=0.2, label='Noise Envelope')
        
        ax_samples.legend(loc='upper right')
        
        # Add annotations for key concepts
        if iteration == 0:
            ax_samples.annotate('High exploration\n(diverse samples)', 
                              xy=(H_nodes//2, 1.5), xytext=(H_nodes//2, 1.8),
                              arrowprops=dict(arrowstyle='->', color='red', lw=2),
                              ha='center', fontweight='bold', color='red')
        elif iteration == N_diffuse - 1:
            ax_samples.annotate('Refined convergence\n(focused samples)', 
                              xy=(H_nodes//2, 0.5), xytext=(H_nodes//2, 1.2),
                              arrowprops=dict(arrowstyle='->', color='green', lw=2),
                              ha='center', fontweight='bold', color='green')
        
        return (list(traj_bars) + list(action_bars) + 
                [traj_highlight, iteration_text, phase_text] + 
                action_highlight_patches + ax_heatmap.get_children() + 
                ax_samples.get_children())
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=N_diffuse, 
                                  interval=1500, repeat=True, blit=False)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color=colors['high_variance'], label='High Variance (Exploration)'),
        mpatches.Patch(color=colors['medium_variance'], label='Medium Variance (Transition)'),
        mpatches.Patch(color=colors['low_variance'], label='Low Variance (Refinement)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
              bbox_to_anchor=(0.5, 0.02))
    
    # Save animation
    print("Saving DIAL-MPC annealing animation...")
    # Use a try-except block to catch potential issues with saving
    try:
        anim.save('dial_mpc_annealing.gif', writer='pillow', fps=0.67, dpi=100)
        anim.save('dial_mpc_annealing.mp4', writer='ffmpeg', fps=0.67, dpi=100, 
                  extra_args=['-vcodec', 'libx264'])
        print("DIAL-MPC annealing animation created successfully!")
        print("Files saved: dial_mpc_annealing.gif, dial_mpc_annealing.mp4")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Please ensure you have ffmpeg installed and accessible in your system's PATH.")
        print("You can install it via: conda install ffmpeg -c conda-forge")

    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout to make space for legend
    plt.show()
    
    return anim

if __name__ == "__main__":
    # Set random seed for reproducible trajectories
    np.random.seed(42)
    anim = create_animation()
    