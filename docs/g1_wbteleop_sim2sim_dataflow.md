# `g1_wbteleop_sim2sim` 完整数据流

启动命令：

```bash
python scripts/run_pipeline.py -c g1_wbteleop_sim2sim
```

配置：[`g1_wbteleop_sim2sim`](../robojudo/config/g1/g1_custom_cfg.py#L172)
- controller = `PicoRetargetTrackingBfmCtrl`
- policy = `WbTeleopOnnxPolicy`（**不是** `TrackingBfmSparseOnnxPolicy`）
- hold_policy = `G1UnitreeWoGaitPolicyCfg`，0.75s 过渡混合

## 流程图

```text
启动: python scripts/run_pipeline.py -c g1_wbteleop_sim2sim
  └─ ConfigManager → cfg_registry.get("g1_wbteleop_sim2sim")   [config_manager.py:16]
  └─ pipeline = RlPipeline(cfg)  → while True: pipeline.step()  [run_pipeline.py:81-83]

╔══════════════════════════════════════════════════════════════════╗
║  子进程 (后台不停地算, async worker 内部)                          ║
╚══════════════════════════════════════════════════════════════════╝
producer_factory = _make_..._producer(cfg)                  [pico ctrl:211]
  └─ cfg.async_read 改成 False  (切断套娃递归)
  └─ ctrl = PicoRetargetTrackingBfmCtrl(async_read=False)   ← 真正干活的实体
        └─ 连 Pico VR streamer / 加载 GMR 重定向 / snapshot builder
  └─ return ctrl.get_data   ──►  producer

while not stop_event:                            [worker:119]
  output = producer() = _get_data_sync()         [pico ctrl:365]
    ├─ streamer.get_current_frame()              读一帧人体 (smplx_data)
    ├─ _step_state_machine()                     手柄按键 → idle/active/pause/exit
    └─ if state==active: _active_output()         [pico ctrl:320]
         ├─ retarget.retarget(smplx) → qpos       人体动作 → 机器人关节
         ├─ snapshot_builder.build(qpos)
         ├─ extract_tracking_bfm_sparse_command() → ee_pose 等(本任务忽略)
         └─ wbteleop_extractor.extract()  → command / ref_limb_ee_pose_b /
                                             motion_ref_ang_vel   ★本任务真正要用的
  └─ _put_latest(output_queue, output)           放"取餐台", 只留最新一份 [worker:140]

══════════════════════ 进程边界 (Queue 传送) ══════════════════════

╔══════════════════════════════════════════════════════════════════╗
║  主进程  pipeline.step()  每帧循环                  [rl_pipeline.py:574] ║
╚══════════════════════════════════════════════════════════════════╝
env.update() / env.get_data()  → env_data        机器人当前状态 [581]
        │
ctrl_data = ctrl_manager.get_ctrl_data(env_data) [586]
  └─ controller.inst.get_data()                  [ctrl_manager.py:66]
       └─ PicoRetargetTrackingBfmCtrl.get_data() (异步空壳)  [pico ctrl:406]
            └─ worker.get_data() ← 从取餐台瞬间取最新一份, 零等待
  └─ process_triggers() 把 "_commands" 拆出去     [pico ctrl:415]
  └─ ctrl_data["PicoRetargetTrackingBfmCtrl"] = 这份指令   [ctrl_manager.py:69]
  └─ ctrl_data["COMMANDS"] = 合并的离散命令
        │
_maybe_interpolate_command(ctrl_data)            [587]
  └─ 对 ctrl["ref_limb_ee_pose_b"] 做插值平滑(异步频率低 → 防抖动)
        │
active_policy = WbTeleopOnnxPolicy               (cfg.policy, 不是 sparse 策略)
        │
obs = active_policy.get_observation(env_data, ctrl_data)   [598 → wbteleop:427]
  └─ ctrl = ctrl_data["PicoRetargetTrackingBfmCtrl"]
  └─ 参考指令:  command(58) / ref_limb_ee_pose_b(36) / motion_ref_ang_vel(3)
  └─ 机器人当前: robot_limb_ee_pose_b(36)  ← FK 自 env_data
  └─ 按训练约定顺序拼成 observation 向量
        │
pd_target = active_policy.get_pd_target(obs)     [602]
  └─ action = get_action(obs):
       └─ session.run(...) 喂 ONNX 网络出 action  [wbteleop:468]
  └─ pd_target = action + default_pos
        │
pd_target = _apply_hold_to_policy_blend(pd_target)  [624]
  └─ 启动 0.75s 内与 hold_policy(站立) 过渡混合
        │
env.step(pd_target)                              [628]  → 仿真机器人电机
        │
post_step_callback() / 左臂关节实时对比图 / 可视化   [632,637]
        │
        └────────── 回到 while, 下一帧 ──────────┘
```

## 一句话总览

| 阶段 | 在哪 | 干什么 |
|------|------|--------|
| 生产（慢） | 子进程 | 读 VR → 重定向 → 出 `ref_limb_ee_pose_b` 等参考指令，放取餐台 |
| 取数据（快） | 主进程 `ctrl_manager` | 零等待取最新一份，汇总成 `ctrl_data` |
| 平滑 | `_maybe_interpolate_command` | 插值 `ref_limb_ee_pose_b` 防抖 |
| 决策 | `WbTeleopOnnxPolicy` | 参考指令 + 机器人当前态 → obs → ONNX → action |
| 执行 | `env.step` | action→pd_target（混合 hold）→ 电机 |

## 关键文件

| 角色 | 文件 |
|------|------|
| 启动入口 | [scripts/run_pipeline.py](../scripts/run_pipeline.py) |
| 配置 | [robojudo/config/g1/g1_custom_cfg.py](../robojudo/config/g1/g1_custom_cfg.py#L172) |
| 主循环 | [robojudo/pipeline/rl_pipeline.py](../robojudo/pipeline/rl_pipeline.py#L574) |
| 控制器汇总 | [robojudo/controller/ctrl_manager.py](../robojudo/controller/ctrl_manager.py#L59) |
| 控制器 | [robojudo/controller/pico_retarget_tracking_bfm_ctrl.py](../robojudo/controller/pico_retarget_tracking_bfm_ctrl.py) |
| 异步 worker | [robojudo/controller/utils/process_latest_output_worker.py](../robojudo/controller/utils/process_latest_output_worker.py) |
| 指令插值 | [robojudo/tools/streamed_command_interpolator.py](../robojudo/tools/streamed_command_interpolator.py) |
| 策略 | [robojudo/policy/wbteleop_onnx_policy.py](../robojudo/policy/wbteleop_onnx_policy.py) |
