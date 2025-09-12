import matplotlib.pyplot as plt
import numpy as np
import scienceplots

# Use science plots style for publication quality
plt.style.use(['science', 'no-latex'])

# Configure font to avoid missing glyphs with scienceplots and no-latex
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

# Data from the results tables (full data)
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

# Create the second plot (All Tasks Comparison)
fig2, ax2 = plt.subplots(figsize=(8, 7))

# Plot 2: All tasks (excluding filters) normalized by MPPI
all_tasks = list(data.keys())

# Collect all data points for all tasks (excluding filters) and calculate improvement relative to MPPI
normalized_points = []
for task_name in all_tasks:
    task_data_alg_only = [(algorithm, reward, samples) for algorithm, reward, samples in data[task_name] 
                 if not algorithm.startswith('DIAL-MPC +')]
    
    # Find MPPI baseline for this task
    mppi_reward = next(reward for alg, reward, samples in task_data_alg_only if alg == 'MPPI')
    
    for algorithm, reward, samples in task_data_alg_only:
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

plt.savefig('all_tasks_comparison_plot.pdf', dpi=300, bbox_inches='tight')
plt.savefig('all_tasks_comparison_plot.png', dpi=300, bbox_inches='tight')
plt.show() 