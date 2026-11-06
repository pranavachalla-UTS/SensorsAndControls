import json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from robot_common import PROJECT, HOME, indices
from vision import capture_rays, intrinsic_matrix


def calibrate():
    # A known green sphere is moved through known optical-frame positions.
    # This is an offline simulated calibration fixture, never a pick target input.
    root = ET.parse(PROJECT / 'dynamic_scene.xml').getroot()
    root.find('compiler').set('meshdir', str(PROJECT))
    marker = ET.SubElement(root.find('worldbody'), 'body', name='calibration_marker', mocap='true')
    ET.SubElement(marker, 'geom', type='sphere', size='0.004', rgba='0 1 0 1')
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    data = mujoco.MjData(model)
    qi, vi = indices(model)
    data.qpos[qi] = HOME
    mujoco.mj_forward(model, data)
    cam = model.camera('wrist_camera').id
    R = data.cam_xmat[cam].reshape(3, 3) @ np.diag([1, -1, -1])
    origin = data.cam_xpos[cam].copy()
    observations = []
    for z in [0.10, 0.15, 0.20]:
        for x in [-0.3, -0.1, 0.1, 0.3]:
            for y in [-0.25, 0, 0.25]:
                point = np.array([x*z, y*z, z])
                data.mocap_pos[0] = origin + R @ point
                mujoco.mj_forward(model, data)
                rgb, depth = capture_rays(model, data)
                mask = (rgb[:, :, 1] > 200) & (rgb[:, :, 0] < 20)
                rows, cols = np.nonzero(mask)
                if len(rows) >= 8:
                    observations.append([x, y, cols.mean(), rows.mean()])
    values = np.array(observations)
    if len(values) < 24:
        raise RuntimeError('Calibration fixture has too few visible reference points')
    validation = np.arange(len(values)) % 3 == 0
    train = values[~validation]
    fx, cx = np.linalg.lstsq(np.c_[train[:, 0], np.ones(len(train))], train[:, 2], rcond=None)[0]
    fy, cy = np.linalg.lstsq(np.c_[train[:, 1], np.ones(len(train))], train[:, 3], rcond=None)[0]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    held = values[validation]
    residual = np.c_[fx*held[:, 0] + cx, fy*held[:, 1] + cy] - held[:, 2:]
    result = {'K': K.tolist(), 'width': 640, 'height': 480, 'backend': 'ray',
              'method': 'known simulated sphere centres; independent held-out positions',
              'training_points': len(train), 'validation_points': len(held),
              'validation_rmse_pixels': float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
              'nominal_K': intrinsic_matrix(480, 640, model.cam_fovy[cam]).tolist(),
              'extrinsics': 'Known simulated mount and encoder FK; no real hand-eye calibration claimed'}
    folder = PROJECT / 'calibration'
    folder.mkdir(exist_ok=True)
    (folder / 'intrinsics.json').write_text(json.dumps(result, indent=2))
    np.savetxt(folder / 'observations.csv', values, delimiter=',',
               header='X_over_Z,Y_over_Z,u_pixels,v_pixels', comments='')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    calibrate()
