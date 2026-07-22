import argparse
import re
import time

import mujoco as mj
import mujoco.viewer as mjv
import numpy as np

from general_motion_retargeting.utils.lafan_vendor import utils
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh


SCENE_XML = """
<mujoco model="lafan1_bvh_viewer">
  <option gravity="0 0 0" />
  <visual>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.75 0.75 0.75" />
    <rgba haze="0.15 0.18 0.22 1" />
    <global azimuth="140" elevation="-15" />
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.08 0.1 0.14" rgb2="0.02 0.03 0.05" width="512" height="3072" />
    <texture name="grid" type="2d" builtin="checker" rgb1="0.28 0.3 0.32" rgb2="0.18 0.2 0.22" width="512" height="512" />
    <material name="floor" texture="grid" texrepeat="8 8" reflectance="0.08" />
  </asset>
  <worldbody>
    <light pos="0 -2 4" dir="0 0 -1" diffuse="0.8 0.8 0.8" />
    <geom name="floor" type="plane" size="0 0 0.05" material="floor" />
  </worldbody>
</mujoco>
"""


def read_frame_time(path):
    with open(path, encoding="utf-8") as bvh_file:
        for line in bvh_file:
            match = re.match(r"\s*Frame Time:\s+([\d.]+)", line)
            if match:
                return float(match.group(1))
    raise ValueError(f"Frame Time not found in {path}")


def add_sphere(scene, position, radius, rgba):
    geom = scene.geoms[scene.ngeom]
    mj.mjv_initGeom(
        geom,
        type=mj.mjtGeom.mjGEOM_SPHERE,
        size=np.array([radius, 0.0, 0.0]),
        pos=np.asarray(position, dtype=np.float64),
        mat=np.eye(3).ravel(),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def add_bone(scene, parent_position, child_position, width, rgba):
    geom = scene.geoms[scene.ngeom]
    mj.mjv_initGeom(
        geom,
        type=mj.mjtGeom.mjGEOM_CAPSULE,
        size=np.array([width, 0.0, 0.0]),
        pos=np.zeros(3),
        mat=np.eye(3).ravel(),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    mj.mjv_connector(
        geom,
        type=mj.mjtGeom.mjGEOM_CAPSULE,
        width=width,
        from_=np.asarray(parent_position, dtype=np.float64),
        to=np.asarray(child_position, dtype=np.float64),
    )
    scene.ngeom += 1


def main():
    parser = argparse.ArgumentParser(description="Play an original LAFAN1 BVH skeleton without retargeting.")
    parser.add_argument("--bvh_file", required=True)
    parser.add_argument("--start_frame", default=0, type=int)
    args = parser.parse_args()

    animation = read_bvh(args.bvh_file)
    _, global_positions = utils.quat_fk(animation.quats, animation.pos, animation.parents)
    coordinate_rotation = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    positions = global_positions @ coordinate_rotation.T / 100.0
    positions[:, :, 2] -= positions[:, :, 2].min()
    frame_time = read_frame_time(args.bvh_file)

    model = mj.MjModel.from_xml_string(SCENE_XML)
    data = mj.MjData(model)
    with mjv.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 145
        viewer.cam.elevation = -12
        frame_idx = args.start_frame % len(positions)
        next_frame_time = time.perf_counter()

        while viewer.is_running():
            viewer.user_scn.ngeom = 0
            frame_positions = positions[frame_idx]
            for joint_idx, parent_idx in enumerate(animation.parents):
                if parent_idx >= 0:
                    add_bone(
                        viewer.user_scn,
                        frame_positions[parent_idx],
                        frame_positions[joint_idx],
                        width=0.018,
                        rgba=(0.25, 0.72, 0.92, 1.0),
                    )
                add_sphere(
                    viewer.user_scn,
                    frame_positions[joint_idx],
                    radius=0.028 if joint_idx else 0.04,
                    rgba=(0.96, 0.72, 0.24, 1.0),
                )

            viewer.cam.lookat[:] = frame_positions[0]
            viewer.cam.lookat[2] = 0.9
            viewer.sync()

            frame_idx = (frame_idx + 1) % len(positions)
            next_frame_time += frame_time
            delay = next_frame_time - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_frame_time = time.perf_counter()


if __name__ == "__main__":
    main()
