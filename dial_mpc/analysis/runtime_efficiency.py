import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import scienceplots

plt.style.use(['science', 'no-latex'])
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

def parse_runtime(time_str):
    """Convert runtime string (mm:ss or h:mm:ss) to seconds"""
    parts = time_str.split(':')
    if len(parts) == 2:  # mm:ss
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:  # h:mm:ss
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0

runtime_data = {
    'Go2 Trot': {
        'MPPI': '07:12',
        'DIAL-MPC': '07:01', 
        'DIAL-MPC + Low-Pass': '06:47',
        'DIAL-MPC + Colored': '06:53',
        'DIAL-MPC + Band-Limited': '06:48',
        'VIGAS': '01:21'
    },
    'Go2 Sequential Jump': {
        'MPPI': '05:35',
        'DIAL-MPC': '05:02',
        'VIGAS': '02:36'
    },
    'Go2 Crate Climb': {
        'MPPI': '29:02',
        'DIAL-MPC': '28:49',
        'VIGAS': '13:35'
    },
    'H1 Locomotion': {
        'MPPI': '02:17',
        'DIAL-MPC': '03:11',
        'DIAL-MPC + Low-Pass': '02:32',
        'DIAL-MPC + Colored': '02:32',
        'DIAL-MPC + Band-Limited': '02:15',
        'VIGAS': '00:47'
    },
    'H1 Jog': {
        'MPPI': '31:42',
        'DIAL-MPC': '22:01',
        'DIAL-MPC + Low-Pass': '24:52',
        'DIAL-MPC + Colored': '26:03',
        'DIAL-MPC + Band-Limited': '25:41',
        'VIGAS': '10:19'
    },
    'H1 Push Crate': {
        'MPPI': '47:21',
        'DIAL-MPC': '52:50',
        'VIGAS': '18:27'
    }
}

colors = {
    'MPPI': '#56B4E9',          # Sky blue  
    'DIAL-MPC': '#E69F00',      # Orange
    'DIAL-MPC + Low-Pass': '#CC79A7',     # Reddish purple
    'DIAL-MPC + Colored': '#009E73',      # Bluish green
    'DIAL-MPC + Band-Limited': '#D55E00', # Vermillion
    'VIGAS': '#0173B2'          # Blue
}

runtime_seconds = {}
for task, algorithms in runtime_data.items():
    runtime_seconds[task] = {}
    for alg, time_str in algorithms.items():
        runtime_seconds[task][alg] = parse_runtime(time_str)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

tasks = list(runtime_data.keys())

for i, task in enumerate(tasks):
    ax = axes[i]
    
    # Get algorithms and their runtimes for this task
    task_data = runtime_seconds[task]
    algorithms = list(task_data.keys())
    runtimes = [task_data[alg] for alg in algorithms]
    
    # Create bars
    bars = ax.bar(range(len(algorithms)), runtimes, 
                  color=[colors[alg] for alg in algorithms],
                  alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Highlight VIGAS bar
    vigas_idx = algorithms.index('VIGAS')
    bars[vigas_idx].set_alpha(1.0)
    bars[vigas_idx].set_linewidth(2.0)
    
    # Add speedup annotations for VIGAS
    vigas_time = runtimes[vigas_idx]
    if 'DIAL-MPC' in algorithms:
        dial_idx = algorithms.index('DIAL-MPC')
        dial_time = runtimes[dial_idx]
        speedup = dial_time / vigas_time
        
        ax.annotate(f'{speedup:.1f}× faster', 
                   xy=(vigas_idx, vigas_time), 
                   xytext=(vigas_idx, vigas_time + max(runtimes) * 0.15),
                   ha='center', va='bottom',
                   fontsize=10, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    
    def format_time(x, pos):
        minutes = int(x // 60)
        seconds = int(x % 60)
        return f'{minutes}:{seconds:02d}'
    
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(format_time))
    
    ax.set_title(task, fontsize=12, fontweight='bold')
    ax.set_ylabel('Wall-Clock Time (mm:ss)', fontsize=10)
    
    ax.set_xticks(range(len(algorithms)))
    ax.set_xticklabels([alg.replace('DIAL-MPC + ', '') for alg in algorithms], 
                       rotation=45, ha='right', fontsize=9)
    
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

legend_elements = [Line2D([0], [0], color=colors[alg], lw=4, label=alg) 
                   for alg in colors.keys() if any(alg in task_data for task_data in runtime_data.values())]

fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), 
           ncol=6, fontsize=10, frameon=False)

plt.tight_layout()
plt.subplots_adjust(bottom=0.12, top=0.95, hspace=0.4, wspace=0.3)

plt.savefig('dial_mpc/analysis/runtime_efficiency.png', dpi=300, bbox_inches='tight')
plt.show() 