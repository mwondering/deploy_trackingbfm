import importlib
import sys
import types

import numpy as np

unitree_cpp_stub = types.ModuleType("unitree_cpp")
unitree_cpp_stub.RobotState = object
unitree_cpp_stub.SportState = object
unitree_cpp_stub.UnitreeController = object
sys.modules.setdefault("unitree_cpp", unitree_cpp_stub)


def _env_for_filter():
    UnitreeCppEnv = importlib.import_module("robojudo.environment.unitree_cpp_env").UnitreeCppEnv
    env = UnitreeCppEnv.__new__(UnitreeCppEnv)
    env.num_dofs = 3
    env.position_limits = np.array(
        [
            [-1.0, 1.0],
            [-0.5, 0.5],
            [-2.0, 2.0],
        ],
        dtype=np.float32,
    )
    env._dof_pos = np.array([0.9, 0.0, -1.5], dtype=np.float32)
    env._dof_vel = np.array([0.0, 1.0, -2.0], dtype=np.float32)
    env.stiffness = np.array([10.0, 10.0, 20.0], dtype=np.float32)
    env.damping = np.array([1.0, 2.0, 1.0], dtype=np.float32)
    env.torque_limits = np.array([5.0, 5.0, 10.0], dtype=np.float32)
    env._last_pd_target_cmd = None
    env.limit_pd_target_effort = True
    env.clip_pd_target = False
    env.pd_target_max_delta = None
    return env


def test_filter_pd_target_allows_targets_outside_joint_limits_when_effort_allows():
    env = _env_for_filter()

    filtered = env._filter_pd_target(np.array([1.2, 0.2, -1.8], dtype=np.float32))

    np.testing.assert_allclose(filtered, np.array([1.2, 0.2, -1.8], dtype=np.float32), atol=1e-6)


def test_filter_pd_target_limits_targets_by_training_effort_range():
    env = _env_for_filter()

    filtered = env._filter_pd_target(np.array([2.0, 2.0, -3.0], dtype=np.float32))

    np.testing.assert_allclose(filtered, np.array([1.4, 0.7, -2.1], dtype=np.float32), atol=1e-6)
