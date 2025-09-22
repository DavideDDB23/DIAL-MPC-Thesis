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

# Time axis
time_points = np.arange(total_time_steps)

# Generate a reference trajectory (sinusoidal for visual appeal)
reference_trajectory = 2 + 0.8 * np.sin(0.3 * time_points) + 0.3 * np.sin(0.7 * time_points)

executed_trajectory = []
executed_time_points = []

horizon_color = '#E8F4FD'  # Light blue
horizon_edge_color = '#2E86AB'  # Darker blue
predicted_color = '#F28482'  # Coral red
executed_color = '#2E86AB'  # Blue
executed_point_color = '#F28482'  # Coral for current execution
reference_color = '#32CD32'  # Lime Green (colorblind-friendly)

def animate(frame):
    global current_step, executed_trajectory, executed_time_points
    
    ax.clear()
    
    # Reset executed trajectory at the start of each animation cycle
    if frame % max_animation_steps == 0:
        executed_trajectory = []
        executed_time_points = []
    
    ax.set_xlim(-1, 20)
    ax.set_ylim(-0.6, 4.2)
    ax.set_xlabel('Time Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Control/State Value', fontsize=12, fontweight='bold')

    
    ax.plot(time_points[:20], reference_trajectory[:20], '--', 
            color=reference_color, alpha=0.8, linewidth=2, label='Reference')
    
    current_step = frame % max_animation_steps
    
    # Define prediction horizon window
    horizon_start = current_step
    horizon_end = min(current_step + prediction_horizon, total_time_steps)
    
    # Draw prediction horizon shaded region
    horizon_rect = Rectangle((horizon_start, -0.6), prediction_horizon, 4.8, 
                           facecolor=horizon_color, edgecolor=horizon_edge_color,
                           alpha=0.3, linewidth=2)
    ax.add_patch(horizon_rect)
    
    # Add prediction horizon label
    ax.text(horizon_start + prediction_horizon/2, 3.0, 'Prediction Horizon', 
           ha='center', va='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=horizon_edge_color))
    
    # Generate predicted trajectory within horizon
    if horizon_end > horizon_start:
        horizon_time = np.arange(horizon_start, horizon_end)
        predicted_traj = reference_trajectory[horizon_start:horizon_end]
        
        # Add some variation to show it's a prediction
        prediction_noise = 0.1 * np.random.randn(len(predicted_traj))
        predicted_traj += prediction_noise
        
        # Draw predicted trajectory
        ax.plot(horizon_time, predicted_traj, 'o-', color=predicted_color, 
               linewidth=3, markersize=8, alpha=0.8, label='Predicted Trajectory')
        
        # Highlight the first point (to be executed)
        if len(predicted_traj) > 0:
            ax.plot(horizon_start, predicted_traj[0], 'o', 
                   color=executed_point_color, markersize=12, 
                   markeredgecolor='black', markeredgewidth=2,
                   label='Next Executed Point')
    
    # Build up executed trajectory
    if current_step > 0 and len(executed_trajectory) < current_step:
        # Add the executed point
        executed_time_points.append(current_step - 1)
        executed_trajectory.append(reference_trajectory[current_step - 1] + 
                                 0.05 * np.random.randn())  # Small execution noise
    
    # Draw executed trajectory
    if executed_trajectory:
        ax.plot(executed_time_points, executed_trajectory, 'o-', 
               color=executed_color, linewidth=3, markersize=6,
               label='Executed Trajectory')
    
    # Add arrow showing the receding horizon movement
    if current_step > 0:
        arrow = patches.FancyArrowPatch((current_step - 0.5, 0.5), (current_step + 0.5, 0.5),
                                      arrowstyle='->', mutation_scale=15, 
                                      color='black', linewidth=2)
        ax.add_patch(arrow)
        ax.text(current_step, 0.65, 'Horizon Shift', ha='center', va='center', 
               fontsize=9, fontweight='bold')
    
    # Current time indicator
    ax.axvline(x=current_step, color='red', linestyle='-', alpha=0.7, linewidth=2)
    ax.text(current_step, -0.25, f't = {current_step}', ha='center', va='top',
           fontsize=10, fontweight='bold', 
           bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.7))
    
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=10)
    
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

anim = animation.FuncAnimation(fig, animate, frames=max_animation_steps, 
                             interval=800, repeat=True, blit=False)

plt.tight_layout(pad=2.0)
plt.subplots_adjust(bottom=0.15, top=0.90, left=0.10, right=0.95)

# Save as GIF
print("Creating MPC animation...")
anim.save('mpc_receding_horizon_animation.gif', writer='pillow', fps=1.25, dpi=150)
print("Animation saved as 'mpc_receding_horizon_animation.gif'")

plt.show() 