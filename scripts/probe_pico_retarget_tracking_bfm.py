"""Probe Pico/XRobot -> GMR -> tracking_bfm sparse command without Redis.

Run from the RoboJuDo repo in an environment that can import RoboJuDo, GMR,
XRobotToolkit, and MuJoCo:

    conda run -n robojudo python scripts/probe_pico_retarget_tracking_bfm.py

This script does not run a policy. It only verifies that Pico streaming,
retargeting, MuJoCo FK, and training-compatible sparse command extraction work
in the same Python environment.
"""

from __future__ import annotations

import argparse
import time

import mujoco as mj
import numpy as np

try:
    import mujoco.viewer as mjv
except Exception:  # pragma: no cover - viewer import can fail on headless systems.
    mjv = None

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import ROBOT_BASE_DICT, ROBOT_XML_DICT, XRobotStreamer

from robojudo.tools.tracking_bfm_sparse_command import (
    DEFAULT_ANCHOR_BODY_NAME,
    DEFAULT_EE_BODY_NAMES,
    MujocoRetargetSnapshotBuilder,
    extract_tracking_bfm_sparse_command,
)


def _fmt_vec(vec: np.ndarray, precision: int = 3) -> str:
    return "[" + ", ".join(f"{float(x):+.{precision}f}" for x in np.asarray(vec).reshape(-1)) + "]"


def _left_exit_pressed(controller_data) -> bool:
    if not isinstance(controller_data, dict):
        return False
    left = controller_data.get("LeftController", {})
    return bool(left.get("key_one", False))


def _print_command(command: dict, qpos: np.ndarray, dt: float) -> None:
    print(
        "tracking_bfm "
        f"dt={dt * 1000.0:6.2f}ms "
        f"qpos={qpos.shape[0]:02d} "
        f"h={float(command['anchor_height_w'][0]):+.3f} "
        f"v_b={_fmt_vec(command['base_lin_vel_b'])} "
        f"w_b={_fmt_vec(command['base_ang_vel_b'])} "
        f"ee_L={_fmt_vec(command['ee_pose'][:3])} "
        f"ee_R={_fmt_vec(command['ee_pose'][9:12])}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Pico retargeting for RoboJuDo tracking_bfm.")
    parser.add_argument("--robot", choices=["unitree_g1", "unitree_g1_with_hands"], default="unitree_g1")
    parser.add_argument("--actual-human-height", type=float, default=1.6)
    parser.add_argument("--target-fps", type=float, default=60.0)
    parser.add_argument("--print-hz", type=float, default=10.0)
    parser.add_argument("--max-seconds", type=float, default=0.0, help="0 means run until Ctrl-C or left X/key_one.")
    parser.add_argument("--no-viewer", action="store_true", help="Disable MuJoCo viewer and run terminal-only.")
    parser.add_argument("--anchor-body-name", default=DEFAULT_ANCHOR_BODY_NAME)
    parser.add_argument("--left-ee-body-name", default=DEFAULT_EE_BODY_NAMES[0])
    parser.add_argument("--right-ee-body-name", default=DEFAULT_EE_BODY_NAMES[1])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    xml_file = ROBOT_XML_DICT[args.robot]
    robot_base = ROBOT_BASE_DICT[args.robot]
    body_names = (args.anchor_body_name, args.left_ee_body_name, args.right_ee_body_name)

    print("Initializing XRobot streamer...")
    streamer = XRobotStreamer()
    print("Initializing GMR retargeter...")
    retarget = GMR(
        src_human="xrobot",
        tgt_robot="unitree_g1",
        actual_human_height=args.actual_human_height,
    )
    print(f"Loading MuJoCo model: {xml_file}")
    model = mj.MjModel.from_xml_path(str(xml_file))
    data = mj.MjData(model)
    snapshot_builder = MujocoRetargetSnapshotBuilder(model, data, body_names)

    viewer_ctx = None
    viewer = None
    if not args.no_viewer:
        if mjv is None:
            raise RuntimeError("mujoco.viewer is unavailable. Re-run with --no-viewer for terminal-only probing.")
        viewer_ctx = mjv.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False)
        viewer = viewer_ctx.__enter__()

    target_period = 1.0 / max(args.target_fps, 1.0)
    print_period = 1.0 / max(args.print_hz, 1.0)
    start_time = time.time()
    last_print_time = 0.0
    last_retarget_time = time.time()

    print("Streaming. Press left X/key_one, close viewer, or Ctrl-C to exit.")
    try:
        while True:
            loop_start = time.time()
            if args.max_seconds > 0.0 and loop_start - start_time >= args.max_seconds:
                break
            if viewer is not None and not viewer.is_running():
                break

            smplx_data, _left_hand_data, _right_hand_data, controller_data, _headset_data = streamer.get_current_frame()
            if _left_exit_pressed(controller_data):
                print("Exit requested from left controller key_one.")
                break
            if smplx_data is None:
                time.sleep(target_period)
                continue

            qpos = np.asarray(retarget.retarget(smplx_data, offset_to_ground=True), dtype=np.float32)
            timestamp_ns = int(time.time() * 1e9)
            snapshot = snapshot_builder.build(qpos, timestamp_ns=timestamp_ns)
            command = extract_tracking_bfm_sparse_command(
                snapshot,
                anchor_body_name=args.anchor_body_name,
                ee_body_names=(args.left_ee_body_name, args.right_ee_body_name),
            )

            if viewer is not None:
                viewer.cam.lookat = data.xpos[model.body(robot_base).id]
                viewer.cam.distance = 3.0
                viewer.sync()

            now = time.time()
            if now - last_print_time >= print_period:
                dt = now - last_retarget_time
                last_print_time = now
                _print_command(command, qpos=qpos, dt=dt)
            last_retarget_time = now

            elapsed = time.time() - loop_start
            if elapsed < target_period:
                time.sleep(target_period - elapsed)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if viewer_ctx is not None:
            viewer_ctx.__exit__(None, None, None)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
