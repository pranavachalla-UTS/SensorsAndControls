import argparse
from pathlib import Path
import runpy
import sys
import unittest

# One entry point keeps the normal commands in one place.
PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / 'src'))
COMMANDS = {
    'pick': 'run_pick',
    'trials': 'run_experiments',
    'calibrate': 'calibrate_camera',
    'model': 'analyse_model',
    'preview': 'view_project',
    'detect': 'detect_target',
    'approach': 'approach_target',
    'plot': 'plot_results',
}


def main():
    parser = argparse.ArgumentParser(description='UR3 vision-guided pick project')
    parser.add_argument('command', choices=[*COMMANDS, 'test'])
    args, remaining = parser.parse_known_args()
    if args.command == 'test':
        suite = unittest.defaultTestLoader.discover(str(PROJECT / 'tests'))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    # Pass options such as --seed or --viewer to the chosen small script.
    sys.argv = [COMMANDS[args.command] + '.py', *remaining]
    runpy.run_module(COMMANDS[args.command], run_name='__main__')


if __name__ == '__main__':
    main()
