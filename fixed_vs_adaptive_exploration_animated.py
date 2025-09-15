import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import PolyCollection
import matplotlib.patches as mpatches

# Set up the plot style matching thesis plots
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 14,
    'axes.linewidth': 1.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.alpha': 0.3,
    'figure.facecolor': 'white'
})

# Consistent colorblind-friendly colors
colors = {
    'nominal': '#1f77b4',       # Blue (VIGAS/Nominal)
    'fixed': '#d62728',         # Red (Fixed/Wasted Exploration)
    'explore': '#d62728',       # Red (High variance)
    'transition': '#ff7f0e',    # Orange (Medium variance)
    'refine': '#2ca02c',        # Green (Low variance)
}

def create_animated_exploration_comparison():
    """Create a clearer, animated side-by-side comparison diagram."""
    
    fig, (ax_fixed, ax_adaptive) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    
    time_steps = np.linspace(0, 10, 100)
    nominal_traj = 1.5 * np.sin(0.8 * time_steps) + 0.3 * np.cos(2.5 * time_steps)
    
    # --- LEFT PANEL: Fixed Exploration (Static) ---
    ax_fixed.set_title('Fixed Exploration (MPPI-DIAL-MPC)', 
                       fontsize=18, fontweight='bold', pad=20)
    
    fixed_variance = np.ones_like(time_steps) * 1.0
    
    ax_fixed.plot(time_steps, nominal_traj, color=colors['nominal'], 
                  linewidth=4, label='Nominal Trajectory', zorder=10)
    
    ax_fixed.fill_between(time_steps, 
                          nominal_traj - fixed_variance, 
                          nominal_traj + fixed_variance,
                          color=colors['fixed'], alpha=0.2, 
                          label='Fixed Variance (±1σ)')
    
    ax_fixed.annotate('Constant, inefficient exploration', 
                      xy=(3, nominal_traj[30] - 1.0), 
                      xytext=(0.5, -3.0),
                      arrowprops=dict(arrowstyle='->', lw=2.5, color=colors['fixed']),
                      ha='left', va='bottom', fontweight='bold',
                      fontsize=12, color=colors['fixed'],
                      bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.8))
    
    ax_fixed.set_xlabel('Time Steps', fontweight='bold', fontsize=14)
    ax_fixed.set_ylabel('Control Value', fontweight='bold', fontsize=14)
    ax_fixed.grid(True)
    ax_fixed.legend(loc='upper right', bbox_to_anchor=(1, 0.9375), framealpha=0.9)
    ax_fixed.set_ylim(-3.5, 4.5)

    # --- RIGHT PANEL: Adaptive Exploration (Animated) ---
    ax_adaptive.set_title('Adaptive Exploration (VIGAS)', 
                          fontsize=18, fontweight='bold', pad=20)
    
    ax_adaptive.plot(time_steps, nominal_traj, color=colors['nominal'], 
                     linewidth=4, label='Nominal Trajectory', zorder=10)
    
    # Placeholders for animated elements
    poly_collection = PolyCollection([], facecolors='red', alpha=0.4)
    ax_adaptive.add_collection(poly_collection)
    
    # Annotations - initially invisible
    explore_text = ax_adaptive.annotate('Global Search\n(High Variance)', 
                         xy=(1.5, nominal_traj[15] + 1.2), ha='center',
                         fontweight='bold', color=colors['explore'], fontsize=11, visible=False)
    
    refine_text = ax_adaptive.annotate('Local Refinement\n(Low Variance)', 
                         xy=(8.5, nominal_traj[85] + 1.3), ha='center',
                         fontweight='bold', color=colors['refine'], fontsize=11, visible=False)
    
    # Move Adaptive Learning arrow above the plot
    ax_adaptive.annotate('', xy=(7, 4.2), xytext=(3, 4.2),
                         arrowprops=dict(arrowstyle='simple,head_width=0.7,head_length=0.8', 
                                         lw=3, color='black'))
    ax_adaptive.text(5, 4.4, 'Adaptive Learning', 
                     ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    ax_adaptive.set_xlabel('Time Steps', fontweight='bold', fontsize=14)
    ax_adaptive.grid(True)
    
    legend_patch = mpatches.Patch(color=colors['nominal'], alpha=0.2, label='Adaptive Variance (±1σ)')
    handles, labels = ax_adaptive.get_legend_handles_labels()
    handles.append(legend_patch)
    # Move legend to be vertically centered on the right
    ax_adaptive.legend(handles=handles, loc='upper right', bbox_to_anchor=(1, 0.9375), framealpha=0.9)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.4) # Added hspace and re-confirmed top
    
    def animate(frame):
        # Total frames: 120 for animation + 30 for pause = 150 total
        if frame < 120:
            # Animation phase: frame goes from 0 to 119
            progress = frame / 119.0
        else:
            # Pause phase: stay at final state
            progress = 1.0
        
        # Animate the variance shrinking
        start_variance = 1.5 * np.exp(-0.0 * time_steps) + 0.15
        end_variance = 1.5 * np.exp(-0.4 * time_steps) + 0.15
        current_variance = start_variance * (1 - progress) + end_variance * progress
        
        y1 = nominal_traj - current_variance
        y2 = nominal_traj + current_variance
        
        # Create polygons
        verts = []
        for i in range(len(time_steps) - 1):
            quad = [(time_steps[i], y1[i]), (time_steps[i+1], y1[i+1]), 
                    (time_steps[i+1], y2[i+1]), (time_steps[i], y2[i])]
            verts.append(quad)
        
        # Update polygon collection
        poly_collection.set_verts(verts)
        
        # Update colors based on progress
        num_polygons = len(verts)
        gradient_colors = np.zeros((num_polygons, 4))
        
        # Interpolate between colors
        explore_color = np.array(plt.cm.Reds(0.5))
        transition_color = np.array(plt.cm.Oranges(0.5))
        refine_color = np.array(plt.cm.Greens(0.5))
        
        for i in range(num_polygons):
            time_progress = i / num_polygons
            if time_progress < 0.4:
                color = explore_color
            elif time_progress < 0.7:
                # Interpolate from red to orange
                interp = (time_progress - 0.4) / 0.3
                color = explore_color * (1 - interp) + transition_color * interp
            else:
                # Interpolate from orange to green
                interp = (time_progress - 0.7) / 0.3
                color = transition_color * (1 - interp) + refine_color * interp
            gradient_colors[i, :] = color
            
        poly_collection.set_facecolors(gradient_colors)
        
        # Make annotations appear at the right time
        explore_text.set_visible(progress > 0.1)
        refine_text.set_visible(progress > 0.8)
        
        return [poly_collection, explore_text, refine_text]

    # Create and save animation - slower with pause at end
    anim = animation.FuncAnimation(fig, animate, frames=150, interval=80, blit=True, repeat=True)
    
    try:
        print("Saving SLOWER ANIMATED diagram with end pause...")
        anim.save('fixed_vs_adaptive_exploration.gif', writer='pillow', fps=12, dpi=120)
        print("Saved: fixed_vs_adaptive_exploration.gif")
    except Exception as e:
        print(f"Error saving animation: {e}")

    plt.show()

if __name__ == "__main__":
    create_animated_exploration_comparison() 