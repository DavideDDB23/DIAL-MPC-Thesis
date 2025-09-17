#!/usr/bin/env python3
"""
Animated visualization of VIGAS Sample Budget Scalability Analysis.
Shows non-monotonic sample-performance relationship across tasks.
Demonstrates task-specific optimal budgets and challenges "more is better" assumption.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scienceplots

# Set up the plot style for consistency
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
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

def create_sample_scaling_animation():
    """Create animation showing VIGAS sample budget scalability."""
    
    # Data from the table
    tasks = ['Go2 Trot', 'Go2 SeqJump', 'H1 Jog', 'H1 Loco']
    sample_counts = [512, 896, 1024]
    
    # Performance data (higher is better for all metrics after normalization)
    data = {
        'Go2 Trot': {
            'rewards': [-0.0213, -0.0196, -0.0199],  # Higher magnitude is better (less negative)
            'runtimes': [81, 128, 147],  # seconds
            'tracking_errors': [0.058, 0.060, 0.057],  # Lower is better
            'optimal_idx': 1,  # 896 samples for best reward
            'color': COLORS['blue']
        },
        'Go2 SeqJump': {
            'rewards': [10.379, 10.382, 10.379],  # Higher is better
            'runtimes': [105, 158, 173],  # seconds
            'contact_rewards': [0.675, 0.700, 0.700],  # Higher is better
            'optimal_idx': 1,  # 896 samples for best contact reward
            'color': COLORS['green']
        },
        'H1 Jog': {
            'rewards': [-0.0940, -0.0924, -0.0928],  # Higher magnitude is better
            'runtimes': [346, 619, 659],  # seconds
            'optimal_idx': 1,  # 896 samples for best reward
            'color': COLORS['orange']
        },
        'H1 Loco': {
            'rewards': [-0.0598, -0.0575, -0.0576],  # Higher magnitude is better
            'runtimes': [47, 72, 78],  # seconds
            'optimal_idx': 1,  # 896 samples for best reward
            'color': COLORS['red']
        }
    }
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    axes = [ax1, ax2, ax3, ax4]
    
    # Animation parameters for an 8-second total duration (2s active build-up + 6s pause)
    fps = 12
    active_duration = 2
    pause_duration = 6
    active_frames = active_duration * fps
    pause_frames = pause_duration * fps
    total_frames = active_frames + pause_frames

    def animate(frame):
        # Clear all axes
        for ax in axes:
            ax.clear()
        
        # Determine effective frame, capping at last active frame for the pause
        effective_frame = min(frame, active_frames - 1)
        progress = effective_frame / (active_frames - 1) if active_frames > 1 else 1.0
        
        # Plot each task simultaneously
        for i, (task_name, task_data) in enumerate(data.items()):
            ax = axes[i]
            
            # Show points based on progress
            points_to_show = min(3, int(progress * 3) + 1)
            
            # Plot performance vs sample count
            x_data = sample_counts[:points_to_show]
            y_data = task_data['rewards'][:points_to_show]
            
            # Plot line and points
            if len(x_data) > 1:
                ax.plot(x_data, y_data, 'o-', color=task_data['color'], 
                       linewidth=3, markersize=8, alpha=0.8)
            else:
                ax.plot(x_data, y_data, 'o', color=task_data['color'], 
                       markersize=8, alpha=0.8)
            
            # Highlight optimal point if visible
            if points_to_show > task_data['optimal_idx']:
                optimal_x = sample_counts[task_data['optimal_idx']]
                optimal_y = task_data['rewards'][task_data['optimal_idx']]
                ax.plot(optimal_x, optimal_y, 'o', color='gold', 
                       markersize=12, markeredgecolor='black', markeredgewidth=2,
                       zorder=10, label='Optimal')
            
            # Add runtime annotations only at the end
            if progress >= 1.0:
                for j in range(points_to_show):
                    runtime_text = f"{task_data['runtimes'][j]//60:02d}:{task_data['runtimes'][j]%60:02d}"
                    ax.annotate(runtime_text, 
                               (sample_counts[j], task_data['rewards'][j]),
                               xytext=(5, 10), textcoords='offset points',
                               fontsize=9, alpha=0.7,
                               bbox=dict(boxstyle='round,pad=0.2', 
                                       facecolor='white', alpha=0.7))
            
            # Formatting
            ax.set_xlim(400, 1100)
            if 'Go2' in task_name:
                if 'Trot' in task_name:
                    ax.set_ylim(-0.025, -0.018)
                else:  # SeqJump
                    ax.set_ylim(10.375, 10.385)
            else:  # H1 tasks
                if 'Jog' in task_name:
                    ax.set_ylim(-0.095, -0.092)
                else:  # Loco
                    ax.set_ylim(-0.062, -0.056)
            
            ax.set_xlabel('Number of Samples', fontweight='bold')
            ax.set_ylabel('Mean Reward', fontweight='bold')
            ax.set_title(task_name, fontweight='bold', fontsize=14, 
                        color=task_data['color'])
            ax.grid(True, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Add legend if optimal point is shown
            if points_to_show > task_data['optimal_idx']:
                ax.legend(loc='upper right', fontsize=10)
        
        # Overall title based on phase
        if progress < 1.0:
            title = "Task-by-Task Performance vs Sample Count"
            title_color = COLORS['blue']
        else: # analysis or pause
            title = "Non-Monotonic Sample-Performance Relationship"
            title_color = COLORS['orange']
        
        fig.suptitle(title, fontsize=18, fontweight='bold', color=title_color)
        
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=total_frames,
                                  interval=1000/fps, repeat=True, blit=False)
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.08, bottom=0.20, top=0.915, wspace=0.25, hspace=0.35)
    
    # Save as GIF
    print("Generating VIGAS sample scaling analysis animation...")
    anim.save('vigas_sample_scaling.gif', writer='pillow', fps=fps, dpi=120)
    print("Saved vigas_sample_scaling.gif")
    
    plt.show()
    
    return anim

if __name__ == "__main__":
    create_sample_scaling_animation() 