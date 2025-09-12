import matplotlib.pyplot as plt
import numpy as np
import scienceplots

# Use science plots style for publication quality
plt.style.use(['science', 'no-latex'])

# Configure font to avoid missing glyphs with scienceplots and no-latex
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

# Data from the results tables
data = {
    'Go2 Trot': [
        ('MPPI', -0.201, 2048),
        ('DIAL-MPC', -0.0322, 2048),
        ('DIAL-MPC + Low-Pass', -0.0413, 2048),
        ('DIAL-MPC + Colored', -0.0329, 2048),
        ('DIAL-MPC + Band-Limited', -0.0253, 2048),
        ('VIGAS', -0.0213, 512)
    ],
    'Go2 Sequential Jump': [
        ('MPPI', 10.1, 2048),
        ('DIAL-MPC', 10.378, 2048),
        ('VIGAS', 10.382, 896)
    ],
    'Go2 Crate Climb': [
        ('MPPI', -0.506, 2048),
        ('DIAL-MPC', -0.0231, 2048),
        ('VIGAS', -0.0090, 896)
    ],
    'H1 Locomotion': [
        ('MPPI', -0.367, 2048),
        ('DIAL-MPC', -0.0693, 2048),
        ('VIGAS', -0.0598, 512)
    ],
    'H1 Jog': [
        ('MPPI', -0.305, 2048),
        ('DIAL-MPC', -0.0968, 2048),
        ('VIGAS', -0.0924, 896)
    ],
    'H1 Push Crate': [
        ('MPPI', -0.581, 2048),
        ('DIAL-MPC', 0.0153, 2048),
        ('VIGAS', 0.0253, 896)
    ]
}

# Colorblind-friendly colors (Wong palette)
colors = {
    'VIGAS': '#0173B2',          # Blue
    'DIAL-MPC': '#E69F00',       # Orange  
    'DIAL-MPC + Low-Pass': '#CC79A7',     # Reddish purple
    'DIAL-MPC + Colored': '#009E73',      # Bluish green
    'DIAL-MPC + Band-Limited': '#D55E00', # Vermillion
    'MPPI': '#56B4E9'            # Sky blue
}

# Create figure with subplots for better organization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Separate Go2 Trot (with filters) and other tasks for better visualization
focus_task = 'Go2 Trot'
other_tasks = [t for t in data.keys() if t != focus_task]

# Plot 1: Go2 Trot with all algorithms including filters
task_data = data[focus_task]

# Control variation data for Go2 Trot (lower is better)
control_variation = {
    'MPPI': 236,
    'DIAL-MPC': 168,
    'DIAL-MPC + Low-Pass': 151,
    'DIAL-MPC + Colored': 178,
    'DIAL-MPC + Band-Limited': 164,
    'VIGAS': 161,
}

# Sample counts for legend
sample_counts = {
    'MPPI': 2048,
    'DIAL-MPC': 2048,
    'DIAL-MPC + Low-Pass': 2048,
    'DIAL-MPC + Colored': 2048,
    'DIAL-MPC + Band-Limited': 2048,
    'VIGAS': 512,
}

cv_to_size = lambda cv: 150.0 + 850.0 * (cv_max - cv) / (cv_max - cv_min + 1e-8)

for algorithm, reward, samples in task_data:
    cv = control_variation[algorithm]
    n_samples = sample_counts[algorithm]
    label = f"{algorithm} (N={n_samples})"
    
    ax1.scatter(cv, reward, 
              c=colors[algorithm], 
              s=200,  # Fixed size
              alpha=0.85,
              edgecolors='white',
              linewidth=2,
              label=label,
              zorder=5)

# Add clear annotations
vigas_cv = control_variation['VIGAS']
vigas_reward = next((reward for alg, reward, samples in task_data if alg == 'VIGAS'))

# Highlight the smooth control region
ax1.axvspan(145, 170, alpha=0.2, color='lightgreen', zorder=1)
ax1.text(157, -0.18, 'Smooth Control\nRegion', 
         fontsize=11, ha='center', va='center', rotation=0,
         bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))

ax1.set_xlim(140, 250)
ax1.set_ylim(-0.22, -0.01)
ax1.set_xlabel('Control Variation (Lower is Better)', fontsize=12)
ax1.set_ylabel('Mean Reward (Higher is Better)', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9, bbox_to_anchor=(-0.1, 1.02), loc='upper right', frameon=True, labelspacing=1.0)

# Plot 2: Other tasks (no filters shown)

# Include all tasks in the right plot for a complete relative comparison
other_tasks = list(data.keys())

# Collect all data points for other tasks (excluding filters) and calculate improvement relative to MPPI
normalized_points = []
for task_name in other_tasks:
    task_data = [(algorithm, reward, samples) for algorithm, reward, samples in data[task_name] 
                 if not algorithm.startswith('DIAL-MPC +')]
    
    # Find MPPI baseline for this task
    mppi_reward = next(reward for alg, reward, samples in task_data if alg == 'MPPI')
    
    for algorithm, reward, samples in task_data:
        if algorithm == 'MPPI':
            improvement = 0.0  # MPPI is the baseline
        else:
            # Calculate percentage improvement relative to MPPI
            if mppi_reward != 0:
                improvement = ((reward - mppi_reward) / abs(mppi_reward)) * 100
            else:
                improvement = (reward - mppi_reward) * 100  # For cases where MPPI reward is 0
        
        normalized_points.append((algorithm, improvement, samples, task_name))

# Plot points for DIAL-MPC and VIGAS directly without jitter
shown_labels_ax2 = set()
for alg, improvement, samples, task_name in normalized_points:
    if alg == 'MPPI': # Skip plotting MPPI points, keep only the baseline line
        continue 
    
    label = alg if alg not in shown_labels_ax2 else ""
    ax2.scatter(samples, improvement,
                c=colors[alg], s=180, alpha=0.9,
                edgecolors='white', linewidth=2, zorder=5,
                label=label)
    shown_labels_ax2.add(alg)

# Add efficiency region highlight
ax2.axvspan(400, 1000, alpha=0.15, color='green', zorder=1)
ax2.text(700, 120, 'Efficient Region\n(< 1000 samples)', 
         fontsize=11, ha='center', va='center', rotation=0,
         bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))

# Add baseline line at 0% improvement (MPPI level)
ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5, zorder=2)
ax2.text(1800, -5, 'MPPI Baseline\n(0% improvement)', 
         fontsize=10, ha='center', va='top')

ax2.set_xlim(400, 2200)
ax2.set_ylim(-50, 150)
ax2.set_xlabel('Samples per MPC Step (Lower is Better)', fontsize=12)
ax2.set_ylabel('Improvement over MPPI (%)', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11, bbox_to_anchor=(-0.1, 1.02), loc='upper right', frameon=True)

plt.tight_layout()
plt.show()

plt.savefig('pareto_efficiency_plot.pdf', dpi=300, bbox_inches='tight')
plt.savefig('pareto_efficiency_plot.png', dpi=300, bbox_inches='tight')