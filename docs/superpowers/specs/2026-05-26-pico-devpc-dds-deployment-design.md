# Pico Dev PC DDS Deployment Design

## Goal

Support Pico-controlled G1 deployment from a fresh development PC using a wired Ethernet connection to the robot's Unitree DDS network.

## Architecture

The development PC runs the full RoboJuDo pipeline: Pico controller input, optional retargeting, ONNX policy inference, and Unitree command publication. The robot onboard computer does not run a custom receiver service. Unitree's existing low-level DDS services receive `rt/lowcmd` and publish `rt/lowstate` / `rt/odommodestate`.

## Components

- `PicoLightSparseCtrl`: reads `xrobotoolkit_sdk` controller/headset data and emits sparse tracking commands.
- `PicoRetargetTrackingBfmCtrl`: reads `XRobotStreamer`, runs GMR retargeting and MuJoCo FK, then emits sparse tracking commands.
- `TrackingBfmSparseOnnxPolicy`: consumes sparse command terms and robot state, runs ONNX inference on the development PC.
- `UnitreeCppEnv`: joins the robot DDS network through `net_if`, subscribes to robot state, and publishes PD targets to `rt/lowcmd`.

## Configuration

Add real-robot configs for:

- `g1_tracking_bfm_pico_light_real`
- `g1_tracking_bfm_pico_retarget_real`

Both use `G1RealEnvCfg(env_type="UnitreeCppEnv")`, keep `born_place_align=False`, enable safety checks, and default to `net_if="eth0"` as a placeholder that users change to their development PC robot-facing NIC.

## Setup Documentation

Document the fresh development PC path:

1. Clone and install RoboJuDo.
2. Install base Python dependencies.
3. Install Unitree SDK2 and `unitree_cpp`.
4. Install Pico runtime dependencies (`xrobotoolkit_sdk`; plus GMR for retarget mode).
5. Connect Ethernet to the robot, identify the NIC with `ip addr`, and set `net_if`.
6. Run the Pico probe first, then run the real config.

## Safety

Initial deployment should start with the light Pico path, conservative scales, robot lifted or supported where appropriate, and a verified physical emergency stop. The real configs keep `do_safety_check=True`; future work should add explicit Pico frame timeout handling and PD target rate limiting.

## Non-Goals

- No custom board-computer relay/agent in this iteration.
- No Wi-Fi or Redis/ZMQ transport.
- No automatic installation of proprietary Pico SDK packages.
