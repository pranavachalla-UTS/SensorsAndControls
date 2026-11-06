import numpy as np
from robot_common import RESULTS


def main():
    surface = np.load(RESULTS / 'target_surface_base.npy')
    # Base Z points upward. This waypoint is 8 cm above the detected surface.
    approach = surface + [0, 0, 0.08]
    np.save(RESULTS / 'target_approach_base.npy', approach)
    print('Approach in base frame [m]:', np.round(approach, 4))
    print('Use python main.py pick for the complete pipeline with a fresh camera capture')


if __name__ == '__main__':
    main()
