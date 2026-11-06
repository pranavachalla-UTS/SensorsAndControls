import numpy as np
import cv2
from robot_common import camera_to_base


def intrinsic_matrix(height, width, fovy):
    focal = height / (2 * np.tan(np.deg2rad(fovy) / 2))
    return np.array([[focal, 0, (width - 1) / 2],
                     [0, focal, (height - 1) / 2], [0, 0, 1]])


def capture(renderer, data):
    renderer.disable_depth_rendering()
    renderer.update_scene(data, camera='wrist_camera')
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera='wrist_camera')
    depth = renderer.render().copy()
    return rgb, depth


def localise(rgb, depth, K, T_BC):
    colour = rgb.astype(float)
    red, green, blue = colour[:, :, 0], colour[:, :, 1], colour[:, :, 2]
    mask = ((red > 60) & (red > 1.8 * green) & (red > 1.8 * blue)
            & np.isfinite(depth) & (depth > 0) & (depth < 2))
    count, labels, stats, centres = cv2.connectedComponentsWithStats(mask.astype('uint8'))
    if count < 2:
        raise RuntimeError('No red target visible: stop rather than use object ground truth')
    label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    if stats[label, cv2.CC_STAT_AREA] < 20:
        raise RuntimeError('Target has fewer than 20 valid pixels')
    # Remove boundary pixels, where depth mixes object sides and background.
    inside = cv2.erode((labels == label).astype('uint8'), np.ones((5, 5), np.uint8))
    if inside.sum() < 20:
        raise RuntimeError('Target interior is too small for reliable depth')
    rows, columns = np.nonzero(inside)
    # This near-downward view is dominated by the top face of the cube.
    z = depth[rows, columns]
    points = np.column_stack(((columns - K[0, 2]) * z / K[0, 0],
                              (rows - K[1, 2]) * z / K[1, 1], z))
    points_base = points @ T_BC[:3, :3].T + T_BC[:3, 3]
    # Top-face pixels give a better lateral centre than visible side faces.
    # Use all interior pixels for height so positive noise is not selected.
    top = points_base[:, 2] > np.percentile(points_base[:, 2], 80) - 0.002
    surface = np.median(points_base[top], axis=0)
    surface[2] = np.median(points_base[:, 2])
    return surface, mask


def capture_rays(model, data, height=480, width=640):
    # CPU pinhole sensor for systems without a graphics context.
    # Ray hits create sensor pixels only; geom IDs are never sent to localisation.
    import mujoco
    camera = model.camera('wrist_camera').id
    K = intrinsic_matrix(height, width, model.cam_fovy[camera])
    rows, columns = np.indices((height, width))
    optical = np.column_stack(((columns.ravel() - K[0, 2]) / K[0, 0],
                               (rows.ravel() - K[1, 2]) / K[1, 1], np.ones(height*width)))
    lengths = np.linalg.norm(optical, axis=1)
    rotation = data.cam_xmat[camera].reshape(3, 3) @ np.diag([1, -1, -1])
    rays = np.ascontiguousarray((optical / lengths[:, None]) @ rotation.T)
    ids = np.full(height*width, -1, dtype=np.int32)
    distances = np.full(height*width, -1.0)
    mujoco.mj_multiRay(model, data, data.cam_xpos[camera], rays.ravel(), None,
                       True, -1, ids, distances, None, height*width, 3.0)
    rgb = np.zeros((height*width, 3), dtype=np.uint8)
    valid = ids >= 0
    rgb[valid] = (255 * model.geom_rgba[ids[valid], :3]).astype(np.uint8)
    # Depth cameras report optical Z, not radial range along the ray.
    depth = np.where(valid, distances / lengths, np.nan)
    return rgb.reshape(height, width, 3), depth.reshape(height, width)
