import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as patches

plt.style.use('default')
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
fig.patch.set_facecolor('white')

# Parameters
total_time_steps = 25
prediction_horizon = 6
current_step = 0
max_animation_steps = 15
num_samples = 20  # Number of sampled trajectories

# Time axis
time_points = np.arange(total_time_steps)

# Generate a reference trajectory
reference_trajectory = 2 + 0.8 * np.sin(0.3 * time_points) + 0.4 * np.cos(0.5 * time_points + 0.5)

executed_trajectory = []
executed_time_points = []
nominal_trajectory = reference_trajectory.copy()

horizon_color = '#E8F4FD'  # Light blue
horizon_edge_color = '#2E86AB'  # Darker blue
nominal_color = '#2E86AB'  # Blue for nominal
sampled_color = '#F28482'  # Coral for samples
executed_color = '#32CD32'  # Lime Green
executed_point_color = '#FF6B35'  # Orange for current execution

def animate(frame):
    global current_step, executed_trajectory, executed_time_points, nominal_trajectory
    
    ax.clear()
    
    # Reset executed trajectory at the start of each animation cycle
    if frame % max_animation_steps == 0:
        executed_trajectory = []
        executed_time_points = []
        nominal_trajectory = reference_trajectory.copy()
    
    ax.set_xlim(-1, 20)
    ax.set_ylim(-0.6, 4.5)
    ax.set_xlabel('Time Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Control/State Value', fontsize=12, fontweight='bold')
    
    current_step = frame % max_animation_steps
    
    # Define prediction horizon window
    horizon_start = current_step
    horizon_end = min(current_step + prediction_horizon, total_time_steps)
    
    # Draw prediction horizon shaded region
    horizon_rect = Rectangle((horizon_start, -0.6), prediction_horizon, 5.1, 
                           facecolor=horizon_color, edgecolor=horizon_edge_color,
                           alpha=0.3, linewidth=2)
    ax.add_patch(horizon_rect)
    
    # Add prediction horizon label
    ax.text(horizon_start + prediction_horizon/2, 4.2, 'Prediction Horizon', 
           ha='center', va='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=horizon_edge_color))
    
    # Generate and draw sampled trajectories within horizon
    if horizon_end > horizon_start:
        horizon_time = np.arange(horizon_start, horizon_end)
        
        # Base nominal trajectory for this horizon
        nominal_segment = nominal_trajectory[horizon_start:horizon_end]
        
        # Generate multiple sampled trajectories
        sample_rewards = []
        sample_trajectories = []
        
        for i in range(num_samples):
            # Add noise to create sampled trajectory
            noise_std = 0.15 + 0.1 * np.random.rand()  # Varying noise levels
            sample_noise = noise_std * np.random.randn(len(nominal_segment))
            sampled_traj = nominal_segment + sample_noise
            sample_trajectories.append(sampled_traj)
            
            # Simulate reward (higher for trajectories closer to reference in this horizon)
            reward = -np.sum((sampled_traj - reference_trajectory[horizon_start:horizon_end])**2)
            sample_rewards.append(reward)
            
            # Draw sampled trajectory
            ax.plot(horizon_time, sampled_traj, '-', color=sampled_color, 
                   linewidth=1, alpha=0.4)
        
        # Draw nominal trajectory
        ax.plot(horizon_time, nominal_segment, '-', color=nominal_color, 
               linewidth=4, alpha=0.9, label='Nominal Trajectory')
        
        # Highlight the first point (to be executed)
        if len(nominal_segment) > 0:
            ax.plot(horizon_start, nominal_segment[0], 'o', 
                   color=executed_point_color, markersize=12, 
                   markeredgecolor='black', markeredgewidth=2,
                   label='Next Executed Point')
        
        # Update nominal trajectory using reward-weighted average
        if len(sample_trajectories) > 0:
            sample_rewards = np.array(sample_rewards)
            # Softmax weighting
            weights = np.exp(sample_rewards / 0.1)
            weights = weights / np.sum(weights)
            
            # Update nominal for next iteration
            for i, (traj, weight) in enumerate(zip(sample_trajectories, weights)):
                if i == 0:
                    updated_segment = weight * traj
                else:
                    updated_segment += weight * traj
            
            # Smooth update to avoid jumps
            if current_step < max_animation_steps - 1:
                nominal_trajectory[horizon_start:horizon_end] = (
                    0.7 * nominal_segment + 0.3 * updated_segment
                )
    
    # Add sample trajectories label
    ax.plot([], [], '-', color=sampled_color, linewidth=1, alpha=0.4, 
           label=f'Sampled Rollouts (N={num_samples})')
    
    # Build up executed trajectory
    if current_step > 0 and len(executed_trajectory) < current_step:
        # Add the executed point
        executed_time_points.append(current_step - 1)
        executed_trajectory.append(nominal_trajectory[current_step - 1])
    
    # Draw executed trajectory
    if executed_trajectory:
        ax.plot(executed_time_points, executed_trajectory, 'o-', 
               color=executed_color, linewidth=3, markersize=6,
               label='Executed Trajectory')
    
    # Add arrow showing the receding horizon movement
    if current_step > 0:
        arrow = patches.FancyArrowPatch((current_step - 0.5, 0.8), (current_step + 0.5, 0.8),
                                      arrowstyle='->', mutation_scale=15, 
                                      color='black', linewidth=2)
        ax.add_patch(arrow)
        ax.text(current_step, 0.95, 'Horizon Shift', ha='center', va='center', 
               fontsize=9, fontweight='bold')
    
    ax.axvline(x=current_step, color='red', linestyle='-', alpha=0.7, linewidth=2)
    ax.text(current_step, -0.35, f't = {current_step}', ha='center', va='top',
           fontsize=10, fontweight='bold', 
           bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.7))
    
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize=10)
    
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

# Create animation
anim = animation.FuncAnimation(fig, animate, frames=max_animation_steps, 
                             interval=1200, repeat=True, blit=False)

plt.tight_layout(pad=2.0)
plt.subplots_adjust(bottom=0.15, top=0.90, left=0.10, right=0.95)

# Save as GIF
print("Creating MPPI animation...")
anim.save('mppi_control_animation.gif', writer='pillow', fps=0.8, dpi=150)
print("Animation saved as 'mppi_control_animation.gif'")

plt.show() 