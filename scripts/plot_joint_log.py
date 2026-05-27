from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import msgpack
import msgpack_numpy
import numpy as np

msgpack_numpy.patch()

FIELD_ALIASES = {
    "angle": "pos",
    "pos": "pos",
    "position": "pos",
    "velocity": "vel",
    "vel": "vel",
    "target": "target",
    "pd_target": "target",
}

FIELD_LABELS = {
    "pos": "joint position [rad]",
    "vel": "joint velocity [rad/s]",
    "target": "PD target [rad]",
}


def get_latest_folder(path: str | Path = "logs") -> Path:
    root = Path(path)
    folders = [folder for folder in root.iterdir() if folder.is_dir()]
    if not folders:
        raise FileNotFoundError(f"No log folders found under {root}")
    return max(folders, key=lambda folder: folder.stat().st_mtime)


def read_msgpack(log_dir: str | Path) -> list[dict]:
    log_dir = Path(log_dir)
    log_files = sorted(log_dir.glob("*.msgpack"))
    if not log_files:
        raise FileNotFoundError(f"No .msgpack file found in {log_dir}")

    records = []
    with open(log_files[0], "rb") as f:
        unpacker = msgpack.Unpacker(f, raw=False)
        records.extend(unpacker)
    return records


def load_joint_names(log_dir: str | Path) -> list[str] | None:
    config_path = Path(log_dir) / "config.json"
    if not config_path.is_file():
        return None

    with open(config_path) as f:
        config = json.load(f)

    try:
        joint_names = config["env"]["dof"]["joint_names"]
    except (KeyError, TypeError):
        return None
    return list(joint_names)


def resolve_joint_index(joint: str, joint_names: list[str] | None) -> int:
    try:
        index = int(joint)
    except ValueError:
        if joint_names is None:
            raise ValueError("Joint name lookup requires config.json with env.dof.joint_names") from None
        if joint not in joint_names:
            candidates = ", ".join(joint_names)
            raise ValueError(f"Unknown joint name '{joint}'. Available joints: {candidates}") from None
        return joint_names.index(joint)

    if index < 0:
        raise ValueError("Joint index must be non-negative")
    return index


def normalize_fields(fields: list[str]) -> list[str]:
    normalized = []
    for field in fields:
        key = FIELD_ALIASES.get(field)
        if key is None:
            choices = ", ".join(sorted(FIELD_ALIASES))
            raise ValueError(f"Unknown field '{field}'. Choices: {choices}")
        if key not in normalized:
            normalized.append(key)
    return normalized


def extract_joint_series(
    log_dir: str | Path,
    joint_index: int,
    start: int | None = None,
    end: int | None = None,
) -> dict[str, np.ndarray]:
    records = read_msgpack(log_dir)
    records = records[slice(start, end)]
    if not records:
        raise ValueError("No frames selected from log")

    time_values = np.asarray([record.get("time", i) for i, record in enumerate(records)], dtype=np.float64)
    time_values = time_values - time_values[0]

    pos = []
    vel = []
    target = []
    for record in records:
        env_data = record["env_data"]
        pos.append(env_data["dof_pos"][joint_index])
        vel.append(env_data["dof_vel"][joint_index])
        target.append(record["pd_target"][joint_index])

    return {
        "time_s": time_values,
        "pos": np.asarray(pos, dtype=np.float64),
        "vel": np.asarray(vel, dtype=np.float64),
        "target": np.asarray(target, dtype=np.float64),
    }


def plot_joint_series(
    series: dict[str, np.ndarray],
    fields: list[str],
    *,
    joint_label: str,
    log_dir: str | Path,
    output: str | Path | None,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(fields), 1, figsize=(12, 3.5 * len(fields)), sharex=True)
    if len(fields) == 1:
        axes = [axes]

    for ax, field in zip(axes, fields, strict=True):
        ax.plot(series["time_s"], series[field], label=field)
        ax.set_ylabel(FIELD_LABELS[field])
        ax.grid(True)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("time [s]")
    fig.suptitle(f"{joint_label} from {Path(log_dir).name}")
    fig.tight_layout()

    if output is not None:
        plt.savefig(output, dpi=150)
        print(f"saved {output}")
    if show or output is None:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot joint position, velocity, and PD target from RoboJuDo log.msgpack."
    )
    log_group = parser.add_mutually_exclusive_group(required=True)
    log_group.add_argument("--log", type=Path, help="Path to a log folder containing log.msgpack")
    log_group.add_argument("--latest", action="store_true", help="Use the latest folder under --logs-root")
    parser.add_argument("--logs-root", type=Path, default=Path("logs"), help="Root folder used with --latest")
    parser.add_argument("--joint", required=True, help="Joint index or joint name from config.json")
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["pos", "vel", "target"],
        help="Fields to plot: pos/angle, vel/velocity, target/pd_target",
    )
    parser.add_argument("--start", type=int, default=None, help="Start frame index")
    parser.add_argument("--end", type=int, default=None, help="End frame index, exclusive")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG path to save. If omitted, an interactive window is shown.",
    )
    parser.add_argument("--show", action="store_true", help="Show the plot even when --output is set")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_dir = get_latest_folder(args.logs_root) if args.latest else args.log
    assert log_dir is not None

    joint_names = load_joint_names(log_dir)
    joint_index = resolve_joint_index(args.joint, joint_names)
    fields = normalize_fields(args.fields)
    if joint_names is not None and joint_index < len(joint_names):
        joint_label = joint_names[joint_index]
    else:
        joint_label = f"joint_{joint_index}"

    series = extract_joint_series(log_dir, joint_index=joint_index, start=args.start, end=args.end)
    plot_joint_series(
        series,
        fields,
        joint_label=joint_label,
        log_dir=log_dir,
        output=args.output,
        show=args.show,
    )


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "TkAgg")
    main()
