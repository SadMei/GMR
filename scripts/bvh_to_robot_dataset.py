import argparse
import fnmatch
import os
import pickle

import mujoco as mj
import numpy as np
from rich import print
from tqdm import tqdm

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.grounding import RobotGrounder
from general_motion_retargeting.utils.lafan1 import load_bvh_file


def joint_names_in_qpos_order(model):
    joints = []
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mj.mjtJoint.mjJNT_FREE:
            continue
        joints.append((int(model.jnt_qposadr[joint_id]), mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id)))
    return [name for _, name in sorted(joints)]


def body_positions(model, qpos_sequence):
    data = mj.MjData(model)
    body_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id) for body_id in range(1, model.nbody)]
    positions = np.empty((len(qpos_sequence), len(body_names), 3), dtype=np.float32)
    for frame_idx, qpos in enumerate(qpos_sequence):
        data.qpos[:] = qpos
        mj.mj_forward(model, data)
        positions[frame_idx] = data.xpos[1:]
    return positions, body_names


def retarget_file(source_path, target_path, robot, motion_fps, ground_offset, verbose):
    human_frames, human_height = load_bvh_file(source_path, format="lafan1")
    retargeter = GMR(
        src_human="bvh_lafan1",
        tgt_robot=robot,
        actual_human_height=human_height,
        verbose=verbose,
    )

    qpos = np.empty((len(human_frames), retargeter.model.nq), dtype=np.float64)
    for frame_idx, human_frame in enumerate(tqdm(human_frames, desc=os.path.basename(source_path), leave=False)):
        qpos[frame_idx] = retargeter.retarget(human_frame)

    if robot == "bsrl":
        qpos = RobotGrounder(robot).ground_qpos_sequence(qpos, ground_offset=ground_offset)

    qpos[:, :2] -= qpos[0, :2]

    world_body_pos, body_names = body_positions(retargeter.model, qpos)
    root_pos = qpos[:, :3].astype(np.float32)
    root_rot_xyzw = qpos[:, 3:7][:, [1, 2, 3, 0]].astype(np.float32)
    dof_pos = qpos[:, 7:].astype(np.float32)

    root_model = mj.MjModel.from_xml_path(retargeter.xml_file)
    root_data = mj.MjData(root_model)
    local_body_pos = np.empty_like(world_body_pos)
    for frame_idx, frame_qpos in enumerate(qpos):
        local_qpos = frame_qpos.copy()
        local_qpos[:3] = 0.0
        local_qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        root_data.qpos[:] = local_qpos
        mj.mj_forward(root_model, root_data)
        local_body_pos[frame_idx] = root_data.xpos[1:]

    key_body_names = ["link_right_ankle_roll", "link_left_ankle_roll"] if robot == "bsrl" else []
    key_body_indices = [body_names.index(name) for name in key_body_names]
    motion_data = {
        "fps": float(motion_fps),
        "root_pos": root_pos,
        "root_rot": root_rot_xyzw,
        "dof_pos": dof_pos,
        "dof_names": joint_names_in_qpos_order(retargeter.model),
        "local_body_pos": local_body_pos,
        "link_body_list": body_names,
        "key_body_pos": world_body_pos[:, key_body_indices],
        "key_body_names": key_body_names,
        "source_motion": os.path.basename(source_path),
    }

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "wb") as output_file:
        pickle.dump(motion_data, output_file)

    print(f"[green]Saved {target_path}[/green] ({len(qpos)} frames, root z {root_pos[:, 2].min():.3f}..{root_pos[:, 2].max():.3f})")


def main():
    parser = argparse.ArgumentParser(description="Batch-retarget LAFAN1 BVH files to a robot.")
    parser.add_argument("--src_folder", required=True, help="Folder containing BVH files.")
    parser.add_argument("--tgt_folder", required=True, help="Folder for GMR pickle files.")
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--override", action="store_true")
    parser.add_argument("--motion_fps", default=30, type=int)
    parser.add_argument("--ground_offset", default=0.0, type=float)
    parser.add_argument(
        "--include",
        nargs="*",
        default=["*.bvh"],
        help="Optional filename patterns, for example: walk*.bvh run*.bvh.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    source_files = []
    for dirpath, _, filenames in os.walk(args.src_folder):
        source_files.extend(
            os.path.join(dirpath, name)
            for name in sorted(filenames)
            if name.lower().endswith(".bvh") and any(fnmatch.fnmatch(name, pattern) for pattern in args.include)
        )

    if not source_files:
        raise FileNotFoundError(f"No BVH files found in {args.src_folder}")

    for source_path in source_files:
        relative_path = os.path.relpath(source_path, args.src_folder)
        target_path = os.path.join(args.tgt_folder, os.path.splitext(relative_path)[0] + ".pkl")
        if os.path.exists(target_path) and not args.override:
            print(f"Skipping existing file {target_path}")
            continue
        retarget_file(
            source_path,
            target_path,
            args.robot,
            args.motion_fps,
            args.ground_offset,
            args.verbose,
        )

    print(f"Done. Saved motions to {args.tgt_folder}")


if __name__ == "__main__":
    main()
