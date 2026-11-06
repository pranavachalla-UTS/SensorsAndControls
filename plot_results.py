import argparse
import csv
import json
import os
from pathlib import Path
# Keep the plotting cache with generated files, not in the user's home directory.
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / 'results' / '.mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from robot_common import RESULTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', default='nominal')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    path = RESULTS / args.condition / f'trial_{args.seed}.csv'
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    t = np.array([float(r['time_s']) for r in rows])
    def values(name):
        return np.array([float(r[name]) for r in rows])
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    joint = 1
    axes[0].plot(t, values(f'true_q_{joint}'), label='True angle (evaluation)')
    axes[0].plot(t, values(f'reference_q_{joint}'), '--', label='Reference')
    axes[0].plot(t, values(f'estimated_q_{joint}'), ':', label='KF estimate')
    axes[0].set_ylabel('Shoulder angle [rad]')
    axes[0].legend()
    error = values(f'true_v_{joint}') - values(f'estimated_v_{joint}')
    sigma = np.sqrt(values(f'variance_v_{joint}'))
    axes[1].plot(t, error, label='Velocity estimation error')
    axes[1].fill_between(t, -2*sigma, 2*sigma, alpha=0.25, label='KF +/-2 sigma')
    axes[1].set_ylabel('Velocity error [rad/s]')
    axes[1].legend()
    axes[2].plot(t, values(f'torque_nm_{joint}'))
    axes[2].set_ylabel('Shoulder torque [Nm]')
    axes[2].set_xlabel('Simulation time [s]')
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle(f'{args.condition}: seed {args.seed}; CPU ray camera; ideal suction')
    fig.tight_layout()
    fig.savefig(RESULTS / f'{args.condition}_tracking.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
