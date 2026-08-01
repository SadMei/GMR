import argparse
import pathlib
import pickle

import imageio.v2 as imageio
import mujoco
import numpy as np

from general_motion_retargeting.utils.mujoco_model import joint_names_in_qpos_order


def load_motion(path: pathlib.Path):
    with path.open("rb") as file:
        motion = pickle.load(file)
    required = ("fps", "root_pos", "root_rot", "dof_pos", "dof_names")
    missing = [key for key in required if key not in motion]
    if missing:
        raise KeyError(f"{path}: missing motion fields {missing}")
    return motion


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a MuJoCo robot motion pickle.")
    parser.add_argument("--motion", type=pathlib.Path, required=True)
    parser.add_argument("--xml", type=pathlib.Path, required=True)
    parser.add_argument("--video", type=pathlib.Path, required=True)
    parser.add_argument("--max_frames", type=int, default=600)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    motion = load_motion(args.motion)
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    model_joint_names = joint_names_in_qpos_order(model)
    if list(motion["dof_names"]) != model_joint_names:
        raise ValueError(
            f"Motion/XML joint order mismatch:\n  motion={motion['dof_names']}\n  xml={model_joint_names}"
        )

    root_pos = np.asarray(motion["root_pos"], dtype=np.float32)
    root_rot_xyzw = np.asarray(motion["root_rot"], dtype=np.float32)
    root_rot_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float32)
    frame_count = min(len(root_pos), len(root_rot_wxyz), len(dof_pos), max(args.max_frames, 1))
    if model.nq != 7 + dof_pos.shape[1]:
        raise ValueError(f"Expected model.nq={7 + dof_pos.shape[1]}, got {model.nq}")

    args.video.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 2.4
    camera.azimuth = 180.0
    camera.elevation = -12.0

    with imageio.get_writer(args.video, fps=float(motion["fps"]), codec="libx264") as writer:
        for frame_id in range(frame_count):
            data.qpos[:3] = root_pos[frame_id]
            data.qpos[3:7] = root_rot_wxyz[frame_id]
            data.qpos[7:] = dof_pos[frame_id]
            mujoco.mj_forward(model, data)

            base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
            camera.lookat[:] = data.xpos[base_id]
            renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())

    renderer.close()
    print(f"[INFO] Recorded {frame_count} frames at {float(motion['fps']):g} Hz to {args.video}")


if __name__ == "__main__":
    main()
