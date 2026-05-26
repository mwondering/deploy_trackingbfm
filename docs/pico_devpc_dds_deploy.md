# Pico Dev PC DDS Deployment

This guide describes the lowest-cost deployment path for Pico-controlled G1:
run Pico input, policy inference, and Unitree command publication on a development PC,
then send commands to the robot through a wired Unitree DDS network.

No custom receiver script is required on the robot onboard computer for this path.
RoboJuDo publishes Unitree DDS topics directly through `UnitreeCppEnv`.

## Data Flow

```text
Pico headset/controllers
  -> development PC
  -> Pico controller (`PicoLightSparseCtrl` or `PicoRetargetTrackingBfmCtrl`)
  -> ONNX policy (`TrackingBfmSparseOnnxPolicy`)
  -> Unitree DDS (`UnitreeCppEnv`, `rt/lowcmd`)
  -> G1 low-level controller
```

Robot state comes back over DDS from `rt/lowstate` and `rt/odommodestate`.

## Fresh Development PC Setup

```bash
git clone <your-robojudo-repo-url>
cd RoboJuDo

conda create -n robojudo python=3.11 -y
conda activate robojudo
```

Install Unitree SDK2 first, then run the setup helper:

```bash
bash scripts/setup_pico_devpc_dds.sh
```

The script runs `pip install -e ".[dev,pico]"`, builds the bundled `unitree_cpp`
binding, prints network interfaces, and verifies the two Pico real configs.

Install Pico runtime packages in the same environment:

- `xrobotoolkit_sdk` is required for `PicoLightSparseCtrl`.
- `general_motion_retargeting` is required for `PicoRetargetTrackingBfmCtrl`.
- Retarget mode also uses MuJoCo for FK; the `pico` extra installs `mujoco`.

The Pico SDK and GMR packages may come from private/internal install instructions,
so they are not hard-pinned in `requirements.txt`.

## Model Paths

Point the config at the tracking_bfm export on the development PC:

```bash
export ROBOJUDO_TRACKING_BFM_ONNX=/absolute/path/to/deploy_model.onnx
export ROBOJUDO_TRACKING_BFM_ENV_YAML=/absolute/path/to/params/env.yaml
```

If these variables are not set, RoboJuDo keeps the existing developer-machine defaults.

## Network Setup

Connect the development PC to the robot control network with Ethernet.
Find the robot-facing network interface:

```bash
ip addr
```

Set the interface used by `UnitreeCppEnv`:

```bash
export ROBOJUDO_UNITREE_NET_IF=enp3s0
```

If this variable is not set, the pico real configs default to `eth0`.

## Preflight Checks

For light Pico input:

```bash
python -c "from robojudo.controller.pico_light_sparse_ctrl import PicoLightSparseCtrl"
```

For full-body retarget input:

```bash
python scripts/probe_pico_retarget_tracking_bfm.py --robot unitree_g1
```

For robot DDS communication:

```bash
python -c "from robojudo.config.config_manager import ConfigManager; cfg=ConfigManager('g1_tracking_bfm_pico_light_real').get_cfg(); print(cfg.env.unitree.net_if, cfg.env.env_type)"
```

## Run

Start with the lighter and lower-latency controller:

```bash
python scripts/run_pipeline.py -c g1_tracking_bfm_pico_light_real
```

Use full-body retarget mode only after light mode and the Pico/GMR probe are stable:

```bash
python scripts/run_pipeline.py -c g1_tracking_bfm_pico_retarget_real
```

## Existing Communication Scripts

The main communication entrypoint is `scripts/run_pipeline.py`.
It builds the configured environment, controller, and policy, then calls `env.step()`
each policy tick.

For G1 real deployment, `UnitreeCppEnv` is the communication layer. It:

- initializes Unitree DDS on `cfg.env.unitree.net_if`;
- subscribes to robot state;
- publishes PD targets to `rt/lowcmd`;
- optionally reads Unitree odometry from `rt/odommodestate`.

There is no repository script that must run on the robot onboard computer as a custom
receiver for the direct-DDS path.

## Safety Notes

- Verify the physical emergency stop before enabling real motor commands.
- Start with conservative Pico scales in `PicoLightSparseCtrlCfg`.
- Keep `do_safety_check=True` for real configs.
- Do not run retarget mode on the real robot until the probe script is stable at the expected rate.
