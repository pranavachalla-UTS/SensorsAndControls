import numpy as np
from robot_common import RESULTS
from vision import intrinsic_matrix, localise


def main():
    # RGB, depth, and transform come from the same capture and robot pose.
    frame = np.load(RESULTS / 'capture.npz')
    height, width = frame['depth'].shape
    K = intrinsic_matrix(height, width, float(frame['fovy']))
    surface, mask = localise(frame['rgb'], frame['depth'], K, frame['camera_to_base'])
    np.save(RESULTS / 'target_surface_base.npy', surface)
    print('Detected surface in base frame [m]:', np.round(surface, 4))


if __name__ == '__main__':
    main()
