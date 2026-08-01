import mujoco as mj
import numpy as np


def joint_names_in_qpos_order(model: mj.MjModel) -> list[str]:
    """Return scalar joint names in their qpos storage order."""
    joints = []
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mj.mjtJoint.mjJNT_FREE:
            continue
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id)
        joints.append((int(model.jnt_qposadr[joint_id]), name))
    return [name for _, name in sorted(joints)]


def body_positions(model: mj.MjModel, qpos_sequence: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Evaluate non-world body positions for a sequence of MuJoCo poses."""
    data = mj.MjData(model)
    body_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id) for body_id in range(1, model.nbody)]
    positions = np.empty((len(qpos_sequence), len(body_names), 3), dtype=np.float32)
    for frame_idx, qpos in enumerate(qpos_sequence):
        data.qpos[:] = qpos
        mj.mj_forward(model, data)
        positions[frame_idx] = data.xpos[1:]
    return positions, body_names


def local_body_positions(model: mj.MjModel, qpos_sequence: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Evaluate body positions with the floating base fixed at the identity pose."""
    local_qpos = np.asarray(qpos_sequence).copy()
    local_qpos[:, :3] = 0.0
    local_qpos[:, 3:7] = (1.0, 0.0, 0.0, 0.0)
    return body_positions(model, local_qpos)
