import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from matplotlib.lines import Line2D

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
    'H1 Locomotion': [
        ('MPPI', -0.367, 2048),
        ('DIAL-MPC', -0.0693, 2048),
        ('DIAL-MPC + Low-Pass', -0.0953, 2048),
        ('DIAL-MPC + Colored', -0.0843, 2048),
        ('DIAL-MPC + Band-Limited', -1.857, 2048),
        ('VIGAS', -0.0598, 512)
    ],
    'H1 Jog': [
        ('MPPI', -0.305, 2048),
        ('DIAL-MPC', -0.0968, 2048),
        ('DIAL-MPC + Low-Pass', -0.128, 2048),
        ('DIAL-MPC + Colored', -0.0944, 2048),
        ('DIAL-MPC + Band-Limited', -0.138, 2048),
        ('VIGAS', -0.0924, 896)
    ],
}

# Control variation data for each task (lower is better)
control_variation = {
    'Go2 Trot': {
        'MPPI': 236, 'DIAL-MPC': 168, 'DIAL-MPC + Low-Pass': 151,
        'DIAL-MPC + Colored': 178, 'DIAL-MPC + Band-Limited': 164, 'VIGAS': 161,
    },
    'H1 Locomotion': {
        'MPPI': 221, 'DIAL-MPC': 124, 'DIAL-MPC + Low-Pass': 100,
        'DIAL-MPC + Colored': 127, 'DIAL-MPC + Band-Limited': 166, 'VIGAS': 163,
    },
    'H1 Jog': {
        'MPPI': 821, 'DIAL-MPC': 152, 'DIAL-MPC + Low-Pass': 129,
        'DIAL-MPC + Colored': 148, 'DIAL-MPC + Band-Limited': 153, 'VIGAS': 151,
    }
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

# Task-specific axis ranges and smooth-region highlights
axis_ranges = {
    'Go2 Trot': dict(xlim=(140, 250), ylim=(-0.22, -0.01), smooth=(145, 170), smooth_text=(157, -0.18)),
    'H1 Locomotion': dict(xlim=(90, 230), ylim=(-2.0, -0.04), smooth=(95, 130), smooth_text=(112, -1.7)),
    'H1 Jog': dict(xlim=(120, 850), ylim=(-0.35, -0.08), smooth=(125, 160), smooth_text=(142, -0.33)),
}

# Unified legend handles
base_samples = 2048
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=f'MPPI (N={base_samples})', markerfacecolor=colors['MPPI'], markersize=12),
    Line2D([0], [0], marker='o', color='w', label=f'DIAL-MPC (N={base_samples})', markerfacecolor=colors['DIAL-MPC'], markersize=12),
    Line2D([0], [0], marker='o', color='w', label=f'DIAL-MPC + Low-Pass (N={base_samples})', markerfacecolor=colors['DIAL-MPC + Low-Pass'], markersize=12),
    Line2D([0], [0], marker='o', color='w', label=f'DIAL-MPC + Colored (N={base_samples})', markerfacecolor=colors['DIAL-MPC + Colored'], markersize=12),
    Line2D([0], [0], marker='o', color='w', label=f'DIAL-MPC + Band-Limited (N={base_samples})', markerfacecolor=colors['DIAL-MPC + Band-Limited'], markersize=12),
    Line2D([0], [0], marker='o', color='w', label='VIGAS (N=512 for Trot/Loco, N=896 for Jog)', markerfacecolor=colors['VIGAS'], markersize=12)
]

# Generate a combined figure: 2 plots on top, legend + H1 Jog on bottom
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.22)
ax_trot = fig.add_subplot(gs[0, 0])
ax_loco = fig.add_subplot(gs[0, 1])
legend_ax = fig.add_subplot(gs[1, 0])
ax_jog = fig.add_subplot(gs[1, 1])


def plot_task(ax, task_name):
    task_data = data[task_name]
    task_cv = control_variation[task_name]

    for algorithm, reward, _ in task_data:
        cv = task_cv[algorithm]
        ax.scatter(cv, reward, c=colors[algorithm], s=110, alpha=0.9,
                   edgecolors='white', linewidth=1.2, label=algorithm, zorder=5)

    # Apply ranges and highlight smooth region
    rng = axis_ranges[task_name]
    ax.set_xlim(*rng['xlim']); ax.set_ylim(*rng['ylim'])
    ax.axvspan(rng['smooth'][0], rng['smooth'][1], alpha=0.18, color='lightgreen', zorder=1)
    ax.text(rng['smooth_text'][0], rng['smooth_text'][1], 'Smooth Control\nRegion',
            fontsize=8, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', alpha=0.8))

    # Labels (no global title; keep subplot titles)
    ax.set_xlabel('Control Variation (Lower is Better)', fontsize=9)
    ax.set_ylabel('Mean Reward (Higher is Better)', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25)


# Plot the three tasks
plot_task(ax_trot, 'Go2 Trot')
ax_trot.set_title('Go2 Trot', fontsize=10, fontweight='bold')
plot_task(ax_loco, 'H1 Locomotion')
ax_loco.set_title('H1 Locomotion', fontsize=10, fontweight='bold')
plot_task(ax_jog, 'H1 Jog')
ax_jog.set_title('H1 Jog', fontsize=10, fontweight='bold')

# Single consolidated legend (bottom-left)
legend_ax.legend(handles=legend_elements, loc='center', frameon=True, fontsize=9, labelspacing=0.9, handlelength=1.1, borderpad=0.8)
legend_ax.axis('off')

plt.subplots_adjust(left=0.08, right=0.985, top=0.94, bottom=0.09)
plt.savefig('control_variation_overview.pdf', dpi=300, bbox_inches='tight')
plt.savefig('control_variation_overview.png', dpi=300, bbox_inches='tight')
plt.show() 