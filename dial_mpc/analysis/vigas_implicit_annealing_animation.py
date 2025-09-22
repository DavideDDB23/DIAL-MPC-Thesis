#!/usr/bin/env python3
"""
Animated visualization of VIGAS Implicit Annealing Behavior.
Single plot showing sampling distribution evolution with variance inset.
Designed for immediate understanding in presentations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
import scienceplots

# Set up the plot style for consistency
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 14,
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

def create_reward_landscape(x, y):
    """Create a simple reward landscape with clear optimal region."""
    # Single main peak for clarity, made slightly broader for smoother convergence
    main_peak = 8 * np.exp(-((x - 0.8)**2 + (y - 0.3)**2) / 1.5)
    # Background penalty
    background = -0.05 * (x**2 + y**2)
    return main_peak + background

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

def simulate_vigas_iteration(samples, rewards, temperature):
    """Simulate one VIGAS iteration."""
    weights = np.exp(rewards / temperature)
    weights = weights / np.sum(weights)
    
    new_mean = np.average(samples, weights=weights, axis=0)
    centered_samples = samples - new_mean[None, :]
    weighted_cov = np.cov(centered_samples.T, aweights=weights)
    
    return new_mean, weighted_cov, weights

def create_vigas_implicit_annealing_animation():
    """Create single-plot animation showing VIGAS implicit annealing."""
    
    # Setup single plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Create reward landscape
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = create_reward_landscape(X, Y)
    
    # Animation parameters for 8-second duration + 2s pause, focused on 8 iterations
    fps = 12
    active_duration = 8
    pause_duration = 2
    active_frames = active_duration * fps
    pause_frames = pause_duration * fps
    total_frames = active_frames + pause_frames
    n_iterations = 8
    
    # Initial parameters - Higher temperature for more gradual annealing
    initial_mean = np.array([-0.5, -0.5])
    initial_cov = np.array([[2.5, 0.0], [0.0, 2.5]])
    temperature = 1.5
    
    # Pre-compute all iterations for smooth animation
    all_means = [initial_mean]
    all_covs = [initial_cov]
    variance_history = [np.trace(initial_cov)]
    
    np.random.seed(42)
    mean, cov = initial_mean.copy(), initial_cov.copy()
    
    for i in range(n_iterations):
        samples = np.random.multivariate_normal(mean, cov, 60)
        rewards = np.array([create_reward_landscape(s[0], s[1]) for s in samples])
        mean, cov, _ = simulate_vigas_iteration(samples, rewards, temperature)
        
        all_means.append(mean)
        all_covs.append(cov)
        variance_history.append(np.trace(cov))
    
    def animate(frame):
        ax.clear()
        
        # Determine current iteration, pausing at the end
        if frame < active_frames:
            progress = frame / (active_frames - 1)
        else:
            progress = 1.0  # Stay at the final state during the pause
            
        current_iter = min(int(progress * n_iterations), n_iterations)
        
        # Show reward landscape (subtle)
        ax.contourf(X, Y, Z, levels=20, alpha=0.15, cmap='viridis')
        
        # Show evolution of sampling distribution
        max_ellipses = 5  # Show last 5 iterations for trajectory
        start_idx = max(0, current_iter - max_ellipses + 1)
        
        for i in range(start_idx, current_iter + 1):
            alpha = 0.3 + 0.7 * (i - start_idx) / max(1, max_ellipses - 1)
            linewidth = 1 + 2 * (i - start_idx) / max(1, max_ellipses - 1)
            
            if i == current_iter:
                # Current distribution - most prominent
                confidence_ellipse(all_means[i], all_covs[i], ax, n_std=2.0,
                                 facecolor=COLORS['blue'], alpha=0.3,
                                 edgecolor=COLORS['blue'], linewidth=3)
                
                # Show sample points for current iteration
                np.random.seed(42 + i)
                samples = np.random.multivariate_normal(all_means[i], all_covs[i], 40)
                ax.scatter(samples[:, 0], samples[:, 1], c=COLORS['blue'], 
                          alpha=0.6, s=25, zorder=5)
            else:
                # Previous distributions - fading trail
                confidence_ellipse(all_means[i], all_covs[i], ax, n_std=2.0,
                                 facecolor='none', edgecolor=COLORS['gray'],
                                 linewidth=linewidth, alpha=alpha, linestyle='--')
        
        # Mark optimal point
        ax.scatter(0.8, 0.3, c='gold', s=120, marker='*', edgecolor='black',
                  linewidth=2, label='Optimum', zorder=10)
        
        # Add arrow showing convergence direction
        if current_iter > 3:
            start_mean = all_means[max(0, current_iter-3)]
            end_mean = all_means[current_iter]
            ax.annotate('', xy=end_mean, xytext=start_mean,
                       arrowprops=dict(arrowstyle='->', color=COLORS['red'],
                                     lw=3, alpha=0.8))
            ax.text((start_mean[0] + end_mean[0])/2, (start_mean[1] + end_mean[1])/2 + 0.3,
                   'Converging', ha='center', fontweight='bold', 
                   color=COLORS['red'], fontsize=12)
        
        # Main plot formatting
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_xlabel('Control Dimension 1', fontweight='bold')
        ax.set_ylabel('Control Dimension 2', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Create inset for variance plot
        inset = fig.add_axes([0.15, 0.18, 0.32, 0.25])  # [left, bottom, width, height] in bottom-left corner
        
        iters = np.arange(current_iter + 1)
        current_variances = variance_history[:current_iter + 1]
        
        inset.plot(iters, current_variances, 'o-', color=COLORS['blue'], 
                  linewidth=2, markersize=4)
        inset.fill_between(iters, current_variances, alpha=0.3, color=COLORS['blue'])
        
        inset.set_xlim(0, n_iterations)
        inset.set_ylim(0, variance_history[0] * 1.1)
        inset.set_xlabel('Iteration', fontweight='bold', fontsize=10, labelpad=0)
        inset.set_ylabel('Variance', fontweight='bold', fontsize=10)
        inset.set_title('Implicit Annealing', fontweight='bold', fontsize=11)
        inset.grid(True, alpha=0.3)
        inset.spines['top'].set_visible(False)
        inset.spines['right'].set_visible(False)
        
        # Add annotation on inset
        if current_iter > n_iterations // 2:
            inset.annotate('Automatic\nDecrease', 
                          xy=(current_iter * 0.7, current_variances[-1] + 0.2),
                          xytext=(current_iter * 0.3, variance_history[0] * 0.6),
                          arrowprops=dict(arrowstyle='->', color=COLORS['orange'], lw=1.5),
                          fontsize=9, ha='center', color=COLORS['orange'],
                          weight='bold')
        
        # Dynamic title
        if current_iter == 0:
            title = "VIGAS Implicit Annealing: Initial Exploration"
            color = COLORS['gray']
        elif progress < 1.0: # Use progress to detect if we are still animating
            title = f"VIGAS Implicit Annealing: Learning (Iter {current_iter})"
            color = COLORS['orange']
        else: # Paused at the end
            title = f"VIGAS Implicit Annealing: Converged (Iter {n_iterations})"
            color = COLORS['blue']
        
        fig.suptitle(title, fontsize=18, fontweight='bold', color=color)
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=total_frames,
                                  interval=1000/fps, repeat=True, blit=False)
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, bottom=0.12, top=0.93) # Increased top to reduce space
    
    # Save as GIF
    print("Generating clear VIGAS implicit annealing animation...")
    anim.save('vigas_implicit_annealing_clear.gif', writer='pillow', fps=fps, dpi=120)
    print("Saved vigas_implicit_annealing_clear.gif")
    
    plt.show()
    
    return anim

if __name__ == "__main__":
    np.random.seed(42)
    create_vigas_implicit_annealing_animation() 