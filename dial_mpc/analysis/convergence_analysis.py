import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import scienceplots
from matplotlib.patches import Rectangle

plt.style.use(['science', 'no-latex'])
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

colors = {
    'DIAL-MPC': '#E69F00',      # Orange
    'VIGAS': '#0173B2'          # Blue
}

# Sample counts used by each algorithm
SAMPLE_COUNTS = {
    'Go2 Trot': {'DIAL-MPC': 2048, 'VIGAS': 512},
    'H1 Locomotion': {'DIAL-MPC': 2048, 'VIGAS': 512},
    'H1 Jog': {'DIAL-MPC': 2048, 'VIGAS': 896},
}

TASK_NAME_MAPPING = {
    'unitree_go2_walk': 'Go2 Trot',
    'unitree_h1_walk': 'H1 Locomotion',
    'unitree_h1_loco': 'H1 Locomotion',
    'unitree_h1_jog': 'H1 Jog',
}

def parse_args():
    parser = argparse.ArgumentParser(description='Convergence analysis from real inner-iteration traces')
    parser.add_argument('--npz', nargs='+', required=True,
                        help='Paths to convergence_traces npz files saved by dial_core.py')
    parser.add_argument('--out_png', default='dial_mpc/analysis/convergence_analysis.png',
                        help='Output PNG path')
    parser.add_argument('--out_pdf', default='dial_mpc/analysis/convergence_analysis.pdf',
                        help='Output PDF path')
    return parser.parse_args()


def _label_for_algo(meta_algo: str) -> str:
    if meta_algo is None:
        return 'Unknown'
    if meta_algo.strip().upper() == 'VIGAS':
        return 'VIGAS'
    if meta_algo.startswith('DIAL-MPC'):
        return 'DIAL-MPC'
    if 'DIAL' in meta_algo.upper():
        return 'DIAL-MPC'
    return meta_algo


def load_convergence_runs(npz_paths: list[str]):
    datasets: dict[str, dict[str, list[dict]] ] = {}
    for path in npz_paths:
        try:
            data = np.load(path, allow_pickle=True)
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue
        env_name = str(data.get('env_name', ''))
        algo_meta = str(data.get('algo', ''))
        algo = _label_for_algo(algo_meta)
        # Disambiguate task from path when env_name is generic
        if 'unitree_go2_trot' in path:
            task_name = 'Go2 Trot'
        elif 'unitree_h1_jog' in path:
            task_name = 'H1 Jog'
        elif 'unitree_h1_loco' in path:
            task_name = 'H1 Locomotion'
        else:
            task_name = TASK_NAME_MAPPING.get(env_name, env_name)
        print(f"Loaded: task={task_name}, algo={algo}, file={os.path.basename(path)}")
        run = {
            'mean_trace': np.asarray(data.get('mean_trace', []), dtype=float),
            'iters': np.asarray(data.get('iters', []), dtype=int),
            'traces': np.asarray(data.get('traces', []), dtype=float),
        }
        datasets.setdefault(task_name, {}).setdefault(algo, []).append(run)
    return datasets


def average_traces(runs: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(runs) == 0:
        return np.array([]), np.array([]), np.array([])
    max_len = max(len(r['mean_trace']) for r in runs)
    padded = np.full((len(runs), max_len), np.nan, dtype=float)
    for i, r in enumerate(runs):
        mt = r['mean_trace']
        L = len(mt)
        padded[i, :L] = mt
    avg = np.nanmean(padded, axis=0)
    std = np.nanstd(padded, axis=0)
    iters = np.arange(1, max_len + 1)
    return iters, avg, std


def calculate_convergence_metrics(iters, trace):
    if len(trace) < 2:
        return None
    
    initial_reward = trace[0]
    final_reward = trace[-1]
    total_improvement = final_reward - initial_reward
    
    if total_improvement <= 0:
        return None
    
    # Find iterations to reach certain percentages of improvement
    percentages = [50, 90, 95]
    iter_to_percent = {}
    
    for pct in percentages:
        target_reward = initial_reward + (pct/100) * total_improvement
        # Find first iteration where reward >= target_reward
        achieved_idx = np.where(trace >= target_reward)[0]
        if len(achieved_idx) > 0:
            iter_to_percent[pct] = iters[achieved_idx[0]]
        else:
            iter_to_percent[pct] = iters[-1]  # Never achieved, use final iteration
    
    return {
        'total_improvement': total_improvement,
        'iter_to_50': iter_to_percent.get(50),
        'iter_to_90': iter_to_percent.get(90),
        'iter_to_95': iter_to_percent.get(95),
        'convergence_rate': total_improvement / len(iters),  # improvement per iteration
    }


def plot_convergence(datasets, out_png: str, out_pdf: str):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    tasks_order = ['Go2 Trot', 'H1 Locomotion', 'H1 Jog']

    # Panel mapping
    panel_axes = {
        'Go2 Trot': axes[0, 0],
        'H1 Locomotion': axes[0, 1],
        'H1 Jog': axes[1, 0],
        'Average': axes[1, 1],
    }

    # Plot task-specific panels
    for task in tasks_order:
        ax = panel_axes[task]
        
        title_text = task
        if task in SAMPLE_COUNTS:
            vigas_samples = SAMPLE_COUNTS[task].get('VIGAS', 'N/A')
            dial_samples = SAMPLE_COUNTS[task].get('DIAL-MPC', 'N/A')
            title_text += ''
        
        ax.set_title(title_text, fontsize=10, fontweight='bold')
        
        if task not in datasets:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            continue
            
        ymins, ymaxs = [], []
        
        for algo in ['DIAL-MPC', 'VIGAS']:
            runs = datasets[task].get(algo, [])
            iters, avg, _ = average_traces(runs)
            if len(iters) == 0:
                continue
                
            ax.plot(iters, avg, color=colors[algo], linewidth=2.5 if algo == 'VIGAS' else 2,
                    label=algo, zorder=3)
            ymins.append(np.nanmin(avg))
            ymaxs.append(np.nanmax(avg))
            
            # Calculate and display convergence metrics
            metrics = calculate_convergence_metrics(iters, avg)
            if metrics:
                 text = f"{algo}:\n90%: {metrics['iter_to_90']:.0f} iters"
                 if algo == 'VIGAS':
                     text += f"\nΔR: {metrics['total_improvement']:.3f}"
                 
                 if algo == 'VIGAS':
                     x_pos, y_pos = 0.98, 0.85  # Top-right
                     ha, va = 'right', 'top'
                 else:
                     x_pos, y_pos = 0.02, 0.15   # Bottom-left
                     ha, va = 'left', 'bottom'
                 
                 ax.text(x_pos, y_pos, text, transform=ax.transAxes,
                        fontsize=7, verticalalignment=va, horizontalalignment=ha,
                        bbox=dict(boxstyle="round,pad=0.25", facecolor=colors[algo], alpha=0.2, edgecolor=colors[algo], linewidth=0.5),
                        zorder=5)
            
            # Enhanced adaptation phase visualization for VIGAS
            if algo == 'VIGAS' and len(iters) > 0:
                adaptation_start = int(0.4 * len(iters))  # Start of adaptation phase
                adaptation_end = int(0.8 * len(iters))    # End of adaptation phase
                
                # Add shaded region for adaptation phase
                if ymins and ymaxs:
                    ymin, ymax = min(ymins), max(ymaxs)
                    ax.axvspan(adaptation_start, adaptation_end, alpha=0.1, color=colors[algo], zorder=1)
                    
                    # Add annotation
                    mid_adapt = (adaptation_start + adaptation_end) / 2
                    y_annot = np.interp(mid_adapt, iters, avg)
                    ax.annotate('Adaptation\nPhase', xy=(mid_adapt, y_annot),
                               xytext=(mid_adapt, y_annot + 0.1 * (ymax - ymin)),
                               ha='center', fontsize=8, color=colors[algo],
                               arrowprops=dict(arrowstyle='->', color=colors[algo], alpha=0.7),
                               zorder=4)
        
        ax.set_xlabel('Inner Iteration', fontsize=9)
        ax.set_ylabel('Best Reward Found', fontsize=9)
        ax.grid(True, alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        
        if ymins and ymaxs:
            ymin, ymax = min(ymins), max(ymaxs)
            pad = 0.1 * (ymax - ymin + 1e-8)
            ax.set_ylim(ymin - pad, ymax + pad)

    # Enhanced average panel
    ax_avg = panel_axes['Average']
    ax_avg.set_title('Convergence Patterns Across Tasks', fontsize=11, fontweight='bold')
    
    for algo in ['DIAL-MPC', 'VIGAS']:
        # Collect all runs across tasks for this algo
        runs_all = []
        for task in tasks_order:
            runs_all.extend(datasets.get(task, {}).get(algo, []))
        iters, avg, std = average_traces(runs_all)
        if len(iters) == 0:
            continue
            
        ax_avg.plot(iters, avg, color=colors[algo], linewidth=2.5 if algo == 'VIGAS' else 2,
                   label=algo, alpha=0.9, zorder=3)
        ax_avg.fill_between(iters, avg - std, avg + std, color=colors[algo], alpha=0.2, zorder=2)
        
        # Add overall convergence metrics
        metrics = calculate_convergence_metrics(iters, avg)
        if metrics:
             text = f"{algo} (Avg):\n90%: {metrics['iter_to_90']:.1f} iters\nRate: {metrics['convergence_rate']:.4f}/iter"
             
             if algo == 'VIGAS':
                 x_pos, y_pos = 0.98, 0.95  # Top-right
                 ha, va = 'right', 'top'
             else:
                 x_pos, y_pos = 0.02, 0.05   # Bottom-left  
                 ha, va = 'left', 'bottom'
             
             ax_avg.text(x_pos, y_pos, text, transform=ax_avg.transAxes,
                        fontsize=7, verticalalignment=va, horizontalalignment=ha,
                        bbox=dict(boxstyle="round,pad=0.25", facecolor=colors[algo], alpha=0.2, edgecolor=colors[algo], linewidth=0.5),
                        zorder=5)
    
    ax_avg.set_xlabel('Inner Iteration', fontsize=9)
    ax_avg.set_ylabel('Best Reward Found', fontsize=9)
    ax_avg.grid(True, alpha=0.3, zorder=0)
    ax_avg.set_axisbelow(True)

    legend_elements = [
        Line2D([0], [0], color=colors['DIAL-MPC'], lw=2, label='DIAL-MPC (Fixed Annealing, N = 2048)'),
        Line2D([0], [0], color=colors['VIGAS'], lw=2.5, label='VIGAS (Adaptive Exploration, N=512 for Trot/Loco, N=896 for Jog)')
    ]
    fig.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.005, 0.98),
               fontsize=10, framealpha=0.95, frameon=True, borderaxespad=0.0,
               labelspacing=0.6, handlelength=2.2, handletextpad=0.7)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, left=0.26, hspace=0.35, wspace=0.3)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    datasets = load_convergence_runs(args.npz)
    plot_convergence(datasets, args.out_png, args.out_pdf) 