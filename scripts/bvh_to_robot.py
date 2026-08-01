import argparse
import pathlib
import time
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.params import (
    BSRL_GROUNDING_CONVENTION,
    BSRL_JOINT_COORDINATE_CONVENTION,
    BSRL_KEY_BODY_NAMES,
)
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from general_motion_retargeting.utils.mujoco_model import (
    body_positions,
    joint_names_in_qpos_order,
    local_body_positions,
)
from general_motion_retargeting.grounding import RobotGrounder
from rich import print
from tqdm import tqdm
import os
import numpy as np

if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bvh_file",
        help="BVH motion file to load.",
        required=True,
        type=str,
    )
    
    parser.add_argument(
        "--format",
        choices=["lafan1", "nokov"],
        default="lafan1",
    )
    
    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion.",
    )
    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "booster_t1", "stanford_toddy", "fourier_n1", "engineai_pm01", "pal_talos", "bsrl"],
        default="unitree_g1",
    )
    
    
    parser.add_argument(
        "--record_video",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--video_path",
        type=str,
        default=str(pathlib.Path(__file__).resolve().parents[1] / "videos" / "example.mp4"),
    )

    parser.add_argument(
        "--rate_limit",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
    )
    
    parser.add_argument(
        "--motion_fps",
        default=30,
        type=int,
    )

    parser.add_argument(
        "--max_frames",
        default=None,
        type=int,
        help="Stop after this many frames. By default, process the full motion.",
    )
    
    args = parser.parse_args()

    if args.loop and args.save_path is not None:
        parser.error("--loop cannot be combined with --save_path")
    
    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []

    
    # Load SMPLX trajectory
    lafan1_data_frames, actual_human_height = load_bvh_file(args.bvh_file, format=args.format)
    
    
    # Initialize the retargeting system
    retargeter = GMR(
        src_human=f"bvh_{args.format}",
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
    )

    motion_fps = args.motion_fps
    grounder = RobotGrounder(args.robot) if args.robot == "bsrl" else None
    
    robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                            motion_fps=motion_fps,
                                            transparent_robot=0,
                                            record_video=args.record_video,
                                            video_path=args.video_path,
                                            # video_width=2080,
                                            # video_height=1170
                                            )
    
    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds
    
    print(f"mocap_frame_rate: {motion_fps}")
    
    # Create tqdm progress bar for the total number of frames
    frame_count = len(lafan1_data_frames)
    if args.max_frames is not None:
        frame_count = min(frame_count, max(args.max_frames, 0))
    pbar = tqdm(total=frame_count, desc="Retargeting")
    
    # Start the viewer
    i = 0
    


    while i < frame_count:
        
        # FPS measurement
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time
            
        # Update progress bar
        pbar.update(1)

        # Update task targets.
        smplx_data = lafan1_data_frames[i]

        # retarget
        qpos = retargeter.retarget(smplx_data)
        if grounder is not None:
            qpos = grounder.ground_qpos(qpos)
        

        # visualize
        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retargeter.scaled_human_data,
            rate_limit=args.rate_limit,
            follow_camera=True,
            # human_pos_offset=np.array([0.0, 0.0, 0.0])
        )

        if args.save_path is not None:
            # Save the exact grounded pose shown by the online viewer.
            qpos_list.append(qpos.copy())

        i += 1
        if args.loop and i >= frame_count:
            i = 0
   
        
    if args.save_path is not None:
        import pickle
        qpos_array = np.asarray(qpos_list)
        root_pos = qpos_array[:, :3]
        # save from wxyz to xyzw
        root_rot = qpos_array[:, 3:7][:, [1, 2, 3, 0]]
        dof_pos = qpos_array[:, 7:]
        world_body_pos, body_names = body_positions(retargeter.model, qpos_array)
        local_body_pos, local_body_names = local_body_positions(retargeter.model, qpos_array)
        if local_body_names != body_names:
            raise RuntimeError("World and local body orders differ")

        key_body_names = list(BSRL_KEY_BODY_NAMES) if args.robot == "bsrl" else []
        key_body_indices = [body_names.index(name) for name in key_body_names]
        
        motion_data = {
            "fps": motion_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": local_body_pos,
            "link_body_list": body_names,
            "dof_names": joint_names_in_qpos_order(retargeter.model),
            "key_body_pos": world_body_pos[:, key_body_indices],
            "key_body_names": key_body_names,
            "source_motion": os.path.basename(args.bvh_file),
        }
        if args.robot == "bsrl":
            motion_data["joint_coordinate_convention"] = BSRL_JOINT_COORDINATE_CONVENTION
            motion_data["grounding_convention"] = BSRL_GROUNDING_CONVENTION
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")

    # Close progress bar
    pbar.close()
    
    robot_motion_viewer.close()
       
