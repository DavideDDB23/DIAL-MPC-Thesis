import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scienceplots

# Configure matplotlib to avoid font issues
plt.style.use(['science', 'no-latex'])
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

# Colorblind-friendly colors (Wong palette)
COLORS = {
    'Target': '#000000',       # Black for target
    'MPPI': '#56B4E9',         # Sky blue
    'DIAL-MPC': '#E69F00',     # Orange
    'VIGAS': '#0173B2'         # Blue
}


def parse_args():
    parser = argparse.ArgumentParser(description='Behavioral trajectories from real rollout metrics for Go2 trot')
    parser.add_argument('--npz', nargs='+', required=True,
                        help='Paths to rollout metrics npz files saved by dial_core.py (provide multiple for different algorithms)')
    parser.add_argument('--out_png', default='dial_mpc/analysis/behavioral_trajectories.png',
                        help='Output PNG path')
    parser.add_argument('--out_pdf', default='dial_mpc/analysis/behavioral_trajectories.pdf',
                        help='Output PDF path')
    parser.add_argument('--feet', nargs=2, type=int, default=[0, 3],
                        help='Foot indices to plot (default: 0=FL, 3=RR)')
    parser.add_argument('--joint_indices', nargs=2, type=int, default=[0, 1],
                        help='Joint indices to plot (default: first two actuated joints)')
    parser.add_argument('--env', default='unitree_go2_walk', help='Expected env_name in files')
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


def load_datasets(npz_paths: list[str], expected_env: str):
    datasets = []
    for path in npz_paths:
        try:
            data = np.load(path, allow_pickle=True)
        except Exception as e:
            print(f"Skipping {path}: could not load ({e})")
            continue

        env_name = str(data.get('env_name')) if 'env_name' in data.files else None
        if env_name != expected_env:
            print(f"Skipping {path}: env_name={env_name} != {expected_env}")
            continue

        algo_meta = str(data.get('algo')) if 'algo' in data.files else None
        label = _label_for_algo(algo_meta)

        ds = {
            'label': label,
            'time': data['time'] if 'time' in data.files else None,
            'feet_z': data['feet_z'] if 'feet_z' in data.files else None,
            'feet_z_tar': data['feet_z_tar'] if 'feet_z_tar' in data.files else None,
            'base_vel_body': data['base_vel_body'] if 'base_vel_body' in data.files else None,
            'vel_tar': data['vel_tar'] if 'vel_tar' in data.files else None,
            'joint_angles': data['joint_angles'] if 'joint_angles' in data.files else None,
            'controls': data['controls'] if 'controls' in data.files else None,
            'path': path,
        }
        # Convert object-typed None back to None if present
        if isinstance(ds['feet_z'], np.ndarray) and ds['feet_z'].dtype == object:
            ds['feet_z'] = None
        if isinstance(ds['feet_z_tar'], np.ndarray) and ds['feet_z_tar'].dtype == object:
            ds['feet_z_tar'] = None

        datasets.append(ds)

    # Keep a stable order: MPPI, DIAL-MPC, VIGAS if available
    order = {'MPPI': 0, 'DIAL-MPC': 1, 'VIGAS': 2}
    datasets.sort(key=lambda d: order.get(d['label'], 99))
    return datasets


def plot_behavior(datasets, out_png: str, out_pdf: str, feet_indices: list[int]):
    if len(datasets) == 0:
        raise ValueError('No valid datasets to plot. Check --npz paths and env filter.')

    # Validate that time arrays exist; if not, derive from length and assume dt=0.02
    for ds in datasets:
        if ds['time'] is None:
            for key in ['feet_z', 'base_vel_body', 'joint_angles', 'controls']:
                if isinstance(ds.get(key), np.ndarray):
                    T = ds[key].shape[0]
                    ds['time'] = np.arange(T) * 0.02
                    break

    # Larger figure with two rows: feet (2 cols), velocity (full width)
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.5], hspace=0.35, wspace=0.25)
    # Leave room on the right for outside legends
    plt.subplots_adjust(right=0.80)

    # 1. Foot Height Trajectories for two selected feet
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    for i, (idx, ax) in enumerate(zip(feet_indices, [ax1, ax2])):
        # Plot target if available (from any dataset that has it)
        for ds in datasets:
            if ds['feet_z_tar'] is not None:
                ax.plot(ds['time'], ds['feet_z_tar'][:, idx], color=COLORS['Target'], linewidth=2.4,
                        linestyle='--', label='Target', alpha=0.9)
                break

        for ds in datasets:
            if ds['feet_z'] is None:
                continue
            color = COLORS.get(ds['label'], '#999999')
            ax.plot(ds['time'], ds['feet_z'][:, idx], color=color, linewidth=2.4,
                    label=ds['label'], alpha=0.9)

        foot_name = f'Foot {idx}'
        ax.set_title(f'{foot_name} Height Tracking', fontsize=13, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Height (m)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=11)
        ax.set_ylim(bottom=-0.02)
        if i == 0:
            # Place left subplot legend outside on the left to avoid overlap with right subplot
            ax.legend(loc='upper right', bbox_to_anchor=(-0.18, 1.0), fontsize=11,
                      labelspacing=0.3, framealpha=0.95, frameon=True, ncol=1)
        else:
            # Place right subplot legend outside on the right
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=11,
                      labelspacing=0.3, framealpha=0.95, frameon=True, ncol=1)

    # 2. Body Velocity Tracking (forward vx in body frame)
    ax3 = fig.add_subplot(gs[1, :])
    # Plot target from any dataset that has it
    for ds in datasets:
        if ds['vel_tar'] is not None:
            ax3.plot(ds['time'], ds['vel_tar'][:, 0], color=COLORS['Target'], linewidth=2.6,
                     linestyle='--', label='Target Velocity', alpha=0.9)
            break

    for ds in datasets:
        if ds['base_vel_body'] is None:
            continue
        color = COLORS.get(ds['label'], '#999999')
        ax3.plot(ds['time'], ds['base_vel_body'][:, 0], color=color, linewidth=2.6,
                 label=ds['label'], alpha=0.9)

    ax3.set_title('Body Velocity Tracking Performance (vx)', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('Forward Velocity (m/s)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=11)
    ax3.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=11,
               labelspacing=0.3, framealpha=0.95, frameon=True, ncol=1)

    # Save first, then show
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    datasets = load_datasets(args.npz, args.env)
    plot_behavior(datasets, args.out_png, args.out_pdf, args.feet) 