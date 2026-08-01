import argparse
from pathlib import Path

import mujoco as mj
import mujoco.viewer as mjv
import numpy as np


def resolve_urdf(path_str):
    candidate = Path(path_str)
    candidates = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.append((Path.cwd() / candidate).resolve())
        candidates.append((Path(__file__).resolve().parent.parent / candidate).resolve())

    for path in candidates:
        if path.exists():
            return path

    urdf_dir = Path(__file__).resolve().parent.parent / "assets" / "bsrl" / "urdf"
    available = ", ".join(sorted(p.name for p in urdf_dir.glob("*.urdf")))
    raise FileNotFoundError(f"URDF not found: {path_str}. Available URDFs: {available}")


def add_visual_floor(viewer):
    viewer.user_scn.ngeom = 0
    floor = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mj.mjv_initGeom(
        floor,
        type=mj.mjtGeom.mjGEOM_BOX,
        size=np.array([3.0, 3.0, 0.02]),
        pos=np.array([0.0, 0.0, -0.02]),
        mat=np.eye(3).reshape(-1),
        rgba=np.array([0.92, 0.93, 0.95, 1.0]),
    )
    viewer.user_scn.ngeom += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--urdf",
        type=str,
        default="assets/BSRL_urdf/urdf/export_floating.urdf",
        help="URDF file to inspect.",
    )
    parser.add_argument(
        "--root-height",
        type=float,
        default=0.881216,
        help="Initial base height for floating-base models.",
    )
    args = parser.parse_args()

    urdf_path = resolve_urdf(args.urdf)

    model = mj.MjModel.from_xml_path(str(urdf_path))
    data = mj.MjData(model)

    # Brighter scene than the repo defaults so we can inspect link orientations.
    model.vis.headlight.diffuse[:] = [0.95, 0.95, 0.95]
    model.vis.headlight.ambient[:] = [0.55, 0.55, 0.55]
    model.vis.headlight.specular[:] = [0.15, 0.15, 0.15]
    model.vis.rgba.haze[:] = [1.0, 1.0, 1.0, 1.0]
    model.vis.global_.azimuth = -140
    model.vis.global_.elevation = -25

    # For floating-base URDFs, hold the robot in a readable static pose.
    if model.nq >= 7:
        data.qpos[0:3] = [0.0, 0.0, args.root_height]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

    mj.mj_forward(model, data)

    with mjv.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        viewer.cam.distance = 2.4
        viewer.cam.elevation = -18
        viewer.cam.azimuth = 145

        while viewer.is_running():
            mj.mj_forward(model, data)
            add_visual_floor(viewer)
            viewer.sync()


if __name__ == "__main__":
    main()
