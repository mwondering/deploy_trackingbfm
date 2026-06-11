# Fix OMP perfmance issue on ARM platform (Jetson)
import os
import platform

if platform.machine().startswith("aarch64"):
    os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import logging
import time

import robojudo.pipeline
from robojudo.config.config_manager import ConfigManager
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg
from robojudo.pipeline.rl_pipeline import RlPipeline

logger = logging.getLogger("robojudo")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="g1",
        help="Name of the config class to use",
    )
    parser.add_argument(
        "--motion-file",
        type=str,
        default=None,
        help="Path to a tracking_bfm npz motion file for playback controllers",
    )
    args = parser.parse_args()
    return args


def _apply_motion_file_override(cfg: RlPipelineCfg, motion_file: str | None) -> None:
    if motion_file is None:
        return
    for cfg_ctrl in getattr(cfg, "ctrl", []) or []:
        if hasattr(cfg_ctrl, "motion_file"):
            cfg_ctrl.motion_file = motion_file
            return
    raise ValueError(f"Config {cfg.__class__.__name__} does not support --motion-file")


def _sim_default_pose_message(cfg: RlPipelineCfg) -> str:
    for cfg_ctrl in getattr(cfg, "ctrl", []) or []:
        if getattr(cfg_ctrl, "ctrl_type", None) == "WbTeleopNpzPlaybackCtrl":
            if bool(getattr(cfg_ctrl, "auto_start", False)):
                return "Sim mode — holding default pose; motion playback will auto-start"
            return "Sim mode — holding default pose, press | to start replay"
        if bool(getattr(cfg_ctrl, "auto_start", False)):
            return "Sim mode — holding default pose; motion playback will auto-start"
    return "Sim mode — holding default pose, press R to start motion"


def main():
    args = parse_args()
    logger.info(f"Using config: {args.config}")
    config_manager = ConfigManager(config_name=args.config)

    cfg: RlPipelineCfg = config_manager.get_cfg()
    _apply_motion_file_override(cfg, args.motion_file)

    pipeline_type = cfg.pipeline_type

    pipeline_class: type[RlPipeline] = getattr(robojudo.pipeline, pipeline_type)
    logger.info(f"Using pipeline: {pipeline_type} -> {pipeline_class}")

    pipeline = pipeline_class(cfg=cfg)

    if not cfg.env.is_sim:
        pipeline.prepare()
    elif getattr(pipeline, "_has_default_pose_mode", False):
        pipeline._set_default_pose_mode(True)
        logger.warning(_sim_default_pose_message(cfg))

    while True:
        time_start = time.time()
        pipeline.step()
        time_end = time.time()
        time_diff = time_end - time_start

        # keep the pipeline running at the desired frequency
        if not cfg.run_fullspeed:
            time_diff = pipeline.dt - time_diff
            if time_diff > 0:
                time.sleep(time_diff)
            else:
                if not cfg.env.is_sim:
                    logger.error(f"Warning: frame drop -> {time_diff}")
                    if time_diff < -0.2:
                        logger.critical("Exiting due to excessive frame drop")
                        pipeline.env.shutdown()
                        time.sleep(10)
                        break


if __name__ == "__main__":
    main()
