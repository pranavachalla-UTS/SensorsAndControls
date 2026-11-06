import argparse
import mujoco
import numpy as np
from robot_common import RESULTS, HOME, load_scene, estimated_data, camera_to_base
from vision import capture, capture_rays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--camera', choices=['ray', 'raster'], default='ray')
    args = parser.parse_args()
    model, data = load_scene()
    # This is a static capture/preview, not a dynamic control demonstration.
    estimate = estimated_data(model, HOME)
    if args.camera == 'ray':
        rgb, depth = capture_rays(model, data)
    else:
        with mujoco.Renderer(model, height=480, width=640) as renderer:
            rgb, depth = capture(renderer, data)
    RESULTS.mkdir(exist_ok=True)
    np.savez(RESULTS / 'capture.npz', rgb=rgb, depth=depth,
             camera_to_base=camera_to_base(model, estimate),
             fovy=model.cam_fovy[model.camera('wrist_camera').id])
    print('Saved results/capture.npz; no scene files were rebuilt')
    if args.viewer:
        from mujoco import viewer
        viewer.launch(model, data)


if __name__ == '__main__':
    main()
