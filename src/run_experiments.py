import argparse
import json
import numpy as np
from robot_common import RESULTS
from run_pick import run_trial


def wilson(successes, n):
    # Wilson 95% interval communicates uncertainty from a small trial sample.
    p = successes / n
    denominator = 1 + 1.96**2/n
    centre = (p + 1.96**2/(2*n)) / denominator
    half = 1.96*np.sqrt(p*(1-p)/n + 1.96**2/(4*n*n)) / denominator
    return [float(centre-half), float(centre+half)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=10)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error('--trials must be positive')
    conditions = {'nominal': {}, 'sensor_noise': {'encoder_std': 0.004, 'depth_std': 0.005},
                  'dropout': {'dropout': 0.25}, 'torque_pulse': {'disturbance': 2.0}}
    summaries = {}
    all_trials = {}
    for name, settings in conditions.items():
        # Reuse seeds across conditions to compare the same object placements.
        results = []
        for seed in range(args.trials):
            row = run_trial(seed=seed, output_dir=RESULTS / 'raw' / name, **settings)
            results.append(row)
            print(name, seed, row['success'], row['reason'], flush=True)
        successes = sum(row['success'] for row in results)
        summary = {'trials': len(results), 'successes': successes,
                   'success_rate': successes / len(results),
                   'success_95_percent_interval': wilson(successes, len(results)),
                   'settings': settings, 'camera_backend': 'ray',
                   'failures': [{'seed': r['seed'], 'reason': r['reason']} for r in results if not r['success']]}
        for metric in ['localisation_error_m', 'angle_rmse_rad', 'velocity_rmse_rad_s',
                       'tracking_rmse_rad', 'torque_rms_nm', 'two_sigma_coverage']:
            values = [r[metric] for r in results if metric in r]
            summary[metric] = {'mean': float(np.mean(values)) if values else None,
                               'count': len(values)}
        summaries[name] = summary
        all_trials[name] = results
        (RESULTS / 'trials.json').write_text(json.dumps(all_trials, indent=2))
        (RESULTS / 'experiment_summary.json').write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
