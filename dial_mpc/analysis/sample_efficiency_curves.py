import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import scienceplots

plt.style.use(['science', 'no-latex'])
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

COLORS = {
    'DIAL-MPC': '#E69F00',      # Orange
    'VIGAS': '#0173B2'          # Blue
}

TASK_NAME_MAPPING = {
    'unitree_go2_walk': 'Go2 Trot',
    'unitree_h1_walk': 'H1 Locomotion',
    'unitree_h1_loco': 'H1 Locomotion', 
    'unitree_h1_jog': 'H1 Jog'
}

def parse_args():
    parser = argparse.ArgumentParser(description='Sample efficiency curves from real learning data')
    parser.add_argument('--npz', nargs='+', required=True,
                        help='Paths to learning curve npz files saved by dial_core.py')
    parser.add_argument('--out_png', default='dial_mpc/analysis/sample_efficiency_curves.png',
                        help='Output PNG path')
    parser.add_argument('--out_pdf', default='dial_mpc/analysis/sample_efficiency_curves.pdf',
                        help='Output PDF path')
    parser.add_argument('--tasks', nargs='+', 
                        default=['Go2 Trot', 'H1 Locomotion', 'H1 Jog'],
                        help='Tasks to include in plots')
    return parser.parse_args()

def _label_for_algo(meta_algo: str) -> str:
    if meta_algo is None:
        return 'Unknown'
    if meta_algo.strip().upper() == 'VIGAS':
        return 'VIGAS'
    if 'DIAL-MPC' in meta_algo:
        return 'DIAL-MPC'
    if 'MPPI' in meta_algo:
        return 'MPPI'
    return meta_algo

def load_learning_curves(npz_paths: list[str]):
    """Load learning curve data from NPZ files"""
    datasets = {}
    
    for path in npz_paths:
        try:
            data = np.load(path, allow_pickle=True)
        except Exception as e:
            print(f"Skipping {path}: could not load ({e})")
            continue
            
        env_name = str(data.get('env_name', ''))
        
        # Use file path to disambiguate when env_name is generic
        if 'unitree_go2_trot' in path:
            task_name = 'Go2 Trot'
        elif 'unitree_h1_jog' in path and env_name == 'unitree_h1_walk':
            task_name = 'H1 Jog'
        elif 'unitree_h1_loco' in path and env_name in ['unitree_h1_walk', 'unitree_h1_loco']:
            task_name = 'H1 Locomotion'  
        else:
            task_name = TASK_NAME_MAPPING.get(env_name, env_name)
            
        algo_meta = str(data.get('algo', ''))
        algo_name = _label_for_algo(algo_meta)
        
        print(f"File: {path} -> env_name: '{env_name}' -> task_name: '{task_name}' -> algo: '{algo_name}'")
        
        if task_name not in datasets:
            datasets[task_name] = {}
            
        # Prefer running-avg series for smoother curves
        x_steps = data.get('running_avg_steps', data.get('step_numbers', []))
        y_rewards = data.get('running_avg_rewards', data.get('step_rewards', []))
        
        datasets[task_name][algo_name] = {
            'x_steps': x_steps,
            'rewards': y_rewards,
            'final_reward': float(data.get('final_reward', 0)),
            'total_steps': int(len(data.get('step_numbers', []))),
            'samples_per_step': int(data.get('samples_per_step', 512)),
            'path': path
        }
        
        print(f"Loaded {algo_name} data for {task_name}: {len(datasets[task_name][algo_name]['rewards'])} points")
    
    return datasets

def plot_sample_efficiency(datasets, tasks, out_png: str, out_pdf: str):
    """Plot sample efficiency curves using real data with a 2x2 layout (top: 2 plots, bottom-right: 1 plot; bottom-left: legend)."""
    preferred_order = ['Go2 Trot', 'H1 Locomotion', 'H1 Jog']
    # Auto-select up to three tasks in preferred order from what's available
    available_tasks = [t for t in preferred_order if t in datasets]
    # If none matched, fall back to the provided tasks list intersection
    if not available_tasks:
        available_tasks = [task for task in tasks if task in datasets]
    if not available_tasks:
        raise ValueError(f"No data available for requested tasks: {tasks}")

    # Build figure with 2x2 gridspec (equal sizes)
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.0], wspace=0.25, hspace=0.35)
    axes_by_task = {}

    # Place first two tasks on the top row
    if len(available_tasks) >= 1:
        axes_by_task[available_tasks[0]] = fig.add_subplot(gs[0, 0])
    if len(available_tasks) >= 2:
        axes_by_task[available_tasks[1]] = fig.add_subplot(gs[0, 1])
    # Third task (if present) goes to bottom-right cell; bottom-left is for legend box
    legend_ax = fig.add_subplot(gs[1, 0])
    legend_ax.axis('off')
    if len(available_tasks) >= 3:
        axes_by_task[available_tasks[2]] = fig.add_subplot(gs[1, 1])

    # Plot each task
    for idx, task in enumerate(available_tasks[:3]):
        ax = axes_by_task[task]
        task_data = datasets[task]

        max_steps = 0
        vigas_data = None
        dial_data = None
        y_min_plot = np.inf
        y_max_plot = -np.inf

        for algo_name, data in task_data.items():
            if algo_name not in ['VIGAS', 'DIAL-MPC']:
                continue
            x = np.asarray(data['x_steps'])
            y = np.asarray(data['rewards'])
            if len(x) == 0 or len(y) == 0:
                continue
            color = COLORS.get(algo_name, '#999999')
            linewidth = 2.5 if algo_name == 'VIGAS' else 2.0
            ax.plot(x, y, color=color, linewidth=linewidth, alpha=0.9,
                    label=f"{algo_name} ({data['samples_per_step']} samp/step)")
            ax.scatter(x[-1], y[-1], color=color, s=80 if algo_name == 'VIGAS' else 60,
                       zorder=5, edgecolors='white', linewidth=1)
            max_steps = max(max_steps, int(x[-1]) if len(x) else 0)
            # Track y-range to place annotation low in the plot
            try:
                y_min_plot = min(y_min_plot, float(np.nanmin(y)))
                y_max_plot = max(y_max_plot, float(np.nanmax(y)))
            except Exception:
                pass
            if algo_name == 'VIGAS':
                vigas_data = data
            elif algo_name == 'DIAL-MPC':
                dial_data = data

        # Add sample efficiency annotation if VIGAS and DIAL-MPC are present
        if vigas_data and dial_data and vigas_data['samples_per_step'] < dial_data['samples_per_step']:
            efficiency_gain = (dial_data['samples_per_step'] - vigas_data['samples_per_step']) / dial_data['samples_per_step'] * 100

            # Arrow to VIGAS curve: choose a target near 80% of steps
            xv = np.asarray(vigas_data['x_steps'])
            yv = np.asarray(vigas_data['rewards'])
            if len(xv) > 0:
                target_step = 0.8 * (xv[-1] if len(xv) else max_steps)
                j = int(np.argmin(np.abs(xv - target_step)))
                arrow_xy = (float(xv[j]), float(yv[j]))
                # Place text much lower in y near bottom of plot and to the left
                y_range = (y_max_plot - y_min_plot) if np.isfinite(y_min_plot) and np.isfinite(y_max_plot) else 0.1
                y_low = y_min_plot + 0.12 * y_range if np.isfinite(y_min_plot) else arrow_xy[1] - 0.03
                x_left = max(0.07 * max_steps, arrow_xy[0] - 0.35 * max_steps)
                text_xy = (float(x_left), float(y_low))
                ax.annotate(f'{efficiency_gain:.0f}% fewer\nsamples needed',
                            xy=arrow_xy, xytext=text_xy,
                            ha='center', va='top',
                            bbox=dict(boxstyle='round,pad=0.35', facecolor='gold', alpha=0.85),
                            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

        ax.set_xlabel('MPC Steps', fontsize=11)
        ax.set_ylabel('Mean Reward', fontsize=11)
        ax.set_title(task, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_xlim(0, max_steps * 1.02 if max_steps > 0 else None)

    # Consolidated legend in bottom-left legend box
    counts_by_algo = {'DIAL-MPC': None, 'VIGAS': None}
    for task in available_tasks:
        for algo_name, data in datasets[task].items():
            if algo_name in counts_by_algo and counts_by_algo[algo_name] is None:
                counts_by_algo[algo_name] = data['samples_per_step']
    counts_by_algo['DIAL-MPC'] = counts_by_algo['DIAL-MPC'] or 2048
    counts_by_algo['VIGAS'] = counts_by_algo['VIGAS'] or 512

    legend_elements = [
        Line2D([0], [0], color=COLORS['DIAL-MPC'], lw=2, label=f"DIAL-MPC (N={counts_by_algo['DIAL-MPC']})"),
        Line2D([0], [0], color=COLORS['VIGAS'], lw=2.5, label="VIGAS (N=512 for Trot/Loco, N=896 for Jog)")
    ]
    legend_ax.legend(handles=legend_elements, loc='center', fontsize=10, framealpha=0.95, frameon=True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, left=0.08, right=0.98, bottom=0.08, hspace=0.35, wspace=0.25)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    args = parse_args()
    datasets = load_learning_curves(args.npz)
    plot_sample_efficiency(datasets, args.tasks, args.out_png, args.out_pdf) 