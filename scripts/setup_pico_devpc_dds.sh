#!/usr/bin/env bash
# Setup helper for Pico -> RoboJuDo -> Unitree G1 deployment from a development PC.
#
# Data path:
#   Pico headset/controllers
#     -> this development PC
#     -> RoboJuDo policy inference
#     -> Unitree DDS over Ethernet
#     -> G1 low-level controller
#
# This script does NOT install proprietary/private packages such as
# xrobotoolkit_sdk or general_motion_retargeting. It prints the checks you need
# to run after installing those packages from your internal source.
#
# Usage examples:
#   bash scripts/setup_pico_devpc_dds.sh
#   ROBOJUDO_UNITREE_NET_IF=enp3s0 bash scripts/setup_pico_devpc_dds.sh
#   SKIP_INSTALL=1 bash scripts/setup_pico_devpc_dds.sh
#
# Important environment variables:
#   ROBOJUDO_UNITREE_NET_IF       Network interface connected to the robot.
#   ROBOJUDO_TRACKING_BFM_ONNX    Absolute path to tracking_bfm ONNX export.
#   ROBOJUDO_TRACKING_BFM_ENV_YAML Absolute path to tracking_bfm params/env.yaml.
#   SKIP_INSTALL                  Set to 1 to skip pip/submodule installation.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
ROBOJUDO_UNITREE_NET_IF="${ROBOJUDO_UNITREE_NET_IF:-eth0}"
ROBOJUDO_TRACKING_BFM_ONNX="${ROBOJUDO_TRACKING_BFM_ONNX:-}"
ROBOJUDO_TRACKING_BFM_ENV_YAML="${ROBOJUDO_TRACKING_BFM_ENV_YAML:-}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"

log() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

warn() {
  printf '\n[WARN] %s\n' "$*" >&2
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    warn "Command '$1' was not found."
    return 1
  fi
}

log "Project root: ${PROJECT_ROOT}"
log "Python command: ${PYTHON_BIN}"

if ! require_command "${PYTHON_BIN}"; then
  cat >&2 <<'EOF'

Create and activate a Python 3.11 environment first. Example:

  conda create -n robojudo python=3.11 -y
  conda activate robojudo

Then rerun this script.
EOF
  exit 1
fi

log "Python version"
"${PYTHON_BIN}" --version

if [[ "${SKIP_INSTALL}" != "1" ]]; then
  log "Installing RoboJuDo in editable mode with dev and pico extras"
  # -e means editable install: source changes in this repo are used immediately.
  # [dev,pico] installs optional dependency groups declared in pyproject.toml:
  #   dev  -> pytest, ruff, pre-commit
  #   pico -> onnxruntime, mujoco
  "${PYTHON_BIN}" -m pip install -e ".[dev,pico]"

  log "Installing bundled Unitree C++ Python binding"
  # Unitree SDK2 must already be installed on the development PC before this.
  "${PYTHON_BIN}" submodule_install.py unitree_cpp
else
  log "SKIP_INSTALL=1, skipping pip install and unitree_cpp build"
fi

log "Checking robot-facing network interface"
if command -v ip >/dev/null 2>&1; then
  ip -br addr || true
  ip -br link || true
else
  warn "'ip' command not found. On Linux, install iproute2 or inspect network interfaces manually."
fi

cat <<EOF

Configured robot network interface:

  ROBOJUDO_UNITREE_NET_IF=${ROBOJUDO_UNITREE_NET_IF}

If this is wrong, rerun with the interface connected to the robot, for example:

  ROBOJUDO_UNITREE_NET_IF=enp3s0 bash scripts/setup_pico_devpc_dds.sh

To make it persistent in your current shell:

  export ROBOJUDO_UNITREE_NET_IF=${ROBOJUDO_UNITREE_NET_IF}
EOF

if [[ -n "${ROBOJUDO_TRACKING_BFM_ONNX}" ]]; then
  export ROBOJUDO_TRACKING_BFM_ONNX
else
  warn "ROBOJUDO_TRACKING_BFM_ONNX is not set. Set it to your deploy_model.onnx path before running real deployment."
fi

if [[ -n "${ROBOJUDO_TRACKING_BFM_ENV_YAML}" ]]; then
  export ROBOJUDO_TRACKING_BFM_ENV_YAML
else
  warn "ROBOJUDO_TRACKING_BFM_ENV_YAML is not set. Set it to your tracking_bfm params/env.yaml path before running real deployment."
fi

export ROBOJUDO_UNITREE_NET_IF

log "Verifying RoboJuDo pico real config wiring"
"${PYTHON_BIN}" - <<'PY'
from robojudo.config.config_manager import ConfigManager

for name in ("g1_tracking_bfm_pico_light_real", "g1_tracking_bfm_pico_retarget_real"):
    cfg = ConfigManager(name).get_cfg()
    print(
        name,
        "env=", cfg.env.env_type,
        "net_if=", cfg.env.unitree.net_if,
        "ctrl=", cfg.ctrl[0].ctrl_type,
        "policy_ctrl=", cfg.policy.ctrl_type,
        "safety=", cfg.do_safety_check,
    )
PY

log "Optional Pico package checks"
cat <<'EOF'
Run these after installing Pico/GMR packages from your internal source:

  python -c "import xrobotoolkit_sdk; print('xrobotoolkit_sdk OK')"
  python -c "import general_motion_retargeting; print('general_motion_retargeting OK')"

For retarget mode, run the non-robot Pico probe first:

  python scripts/probe_pico_retarget_tracking_bfm.py --robot unitree_g1
EOF

log "Ready-to-run commands"
cat <<'EOF'
After confirming the robot-facing interface, model paths, Pico SDK, Unitree SDK2,
physical emergency stop, and robot support setup:

  python scripts/run_pipeline.py -c g1_tracking_bfm_pico_light_real

Only after light mode and the retarget probe are stable:

  python scripts/run_pipeline.py -c g1_tracking_bfm_pico_retarget_real
EOF
