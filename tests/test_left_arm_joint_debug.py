from __future__ import annotations

import sys
import types

import numpy as np

from robojudo.tools.left_arm_joint_debug import LeftArmJointDebugPlot


class _FakeLine:
    def __init__(self):
        self.data = None

    def set_data(self, x, y):
        self.data = (np.asarray(x), np.asarray(y))


class _FakeSpine:
    def __init__(self):
        self.color = None
        self.linewidth = None

    def set_color(self, color):
        self.color = color

    def set_linewidth(self, linewidth):
        self.linewidth = linewidth


class _FakeAxis:
    def __init__(self):
        self.plot_kwargs = []
        self.axis_off_called = False
        self.facecolor = None
        self.spines = {name: _FakeSpine() for name in ("top", "bottom", "left", "right")}

    def plot(self, *args, **kwargs):
        del args
        self.plot_kwargs.append(kwargs)
        return [_FakeLine()]

    def set_title(self, title, fontsize=None):
        del title, fontsize

    def set_ylabel(self, ylabel, fontsize=None):
        del ylabel, fontsize

    def set_xlabel(self, xlabel):
        del xlabel

    def grid(self, *args, **kwargs):
        del args, kwargs

    def set_xlim(self, *args, **kwargs):
        del args, kwargs

    def set_ylim(self, *args, **kwargs):
        del args, kwargs

    def set_facecolor(self, color):
        self.facecolor = color

    def set_xticks(self, ticks):
        del ticks

    def set_yticks(self, ticks):
        del ticks

    def tick_params(self, *args, **kwargs):
        del args, kwargs

    def set_axis_off(self):
        self.axis_off_called = True


class _FakeWindow:
    def __init__(self):
        self.geometry_value = None
        self.minsize_value = None

    def geometry(self, value):
        self.geometry_value = value

    def minsize(self, width, height):
        self.minsize_value = (width, height)


class _FakeManager:
    def __init__(self):
        self.window = _FakeWindow()
        self.title = None
        self.resize_value = None

    def set_window_title(self, title):
        self.title = title

    def resize(self, width, height):
        self.resize_value = (width, height)


class _FakeCanvas:
    def __init__(self):
        self.manager = _FakeManager()

    def draw_idle(self):
        pass

    def flush_events(self):
        pass


class _FakeFigure:
    def __init__(self):
        self.canvas = _FakeCanvas()
        self.subplots_adjust_args = None

    def suptitle(self, title):
        del title

    def tight_layout(self):
        raise ValueError("height and width must be > 0")

    def show(self):
        pass

    def set_size_inches(self, *args, **kwargs):
        del args, kwargs

    def subplots_adjust(self, *args, **kwargs):
        self.subplots_adjust_args = (args, kwargs)


def _fake_matplotlib_modules(monkeypatch, axes_holder):
    fake_matplotlib = types.ModuleType("matplotlib")
    fake_matplotlib.__path__ = []
    fake_matplotlib.rcParams = {}

    fake_plt = types.ModuleType("matplotlib.pyplot")
    fake_plt.ion = lambda: None
    fake_plt.get_backend = lambda: "TkAgg"
    fake_plt.pause = lambda seconds: None
    fake_plt.show = lambda block=False: None

    def _subplots(rows, cols, sharex, figsize):
        del sharex, figsize
        axes = np.asarray([[_FakeAxis() for _ in range(cols)] for _ in range(rows)])
        if rows == 1 and cols == 1:
            axes_out = axes[0, 0]
        elif rows == 1:
            axes_out = axes[0]
        else:
            axes_out = axes
        axes_holder.append(axes)
        return _FakeFigure(), axes_out

    fake_plt.subplots = _subplots
    fake_matplotlib.pyplot = fake_plt
    monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_plt)


def test_left_arm_plot_stays_enabled_when_tight_layout_backend_reports_zero_size(monkeypatch):
    fake_plt = types.SimpleNamespace(
        ion=lambda: None,
        subplots=lambda rows, cols, sharex, figsize: (
            _FakeFigure(),
            np.asarray([[_FakeAxis() for _ in range(cols)] for _ in range(rows)]),
        ),
        get_backend=lambda: "FakeAgg",
        pause=lambda seconds: None,
    )
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_plt)

    plot = LeftArmJointDebugPlot()

    assert plot._enabled is True


def test_left_arm_plot_disables_matplotlib_toolbar_before_creating_figure(monkeypatch):
    fake_matplotlib = types.ModuleType("matplotlib")
    fake_matplotlib.__path__ = []
    fake_matplotlib.rcParams = {}

    fake_plt = types.ModuleType("matplotlib.pyplot")
    fake_plt.ion = lambda: None
    fake_plt.get_backend = lambda: "TkAgg"
    fake_plt.pause = lambda seconds: None

    def _subplots(rows, cols, sharex, figsize):
        del sharex, figsize
        assert fake_matplotlib.rcParams["toolbar"] == "None"
        return _FakeFigure(), np.asarray([[_FakeAxis() for _ in range(cols)] for _ in range(rows)])

    fake_plt.subplots = _subplots
    fake_matplotlib.pyplot = fake_plt
    monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_plt)

    plot = LeftArmJointDebugPlot()

    assert plot._enabled is True


def test_left_arm_plot_keeps_matplotlib_panels_visible(monkeypatch):
    axes_holder = []
    _fake_matplotlib_modules(monkeypatch, axes_holder)

    plot = LeftArmJointDebugPlot()

    axes = axes_holder[0]
    flat_axes = list(axes.reshape(-1))
    assert plot._enabled is True
    assert axes.shape == (1, 1)
    assert all(axis.facecolor is not None for axis in flat_axes)
    assert not any(axis.axis_off_called for axis in flat_axes)
    assert len(flat_axes[0].plot_kwargs) == 2
    assert all(kwargs.get("marker") == "." for kwargs in flat_axes[0].plot_kwargs)
    manager = plot._fig.canvas.manager
    assert manager.window.geometry_value == "1000x520+60+60"
    assert manager.window.minsize_value == (700, 360)


def test_left_arm_plot_draws_only_shoulder_roll_retarget_and_actual(monkeypatch):
    axes_holder = []
    _fake_matplotlib_modules(monkeypatch, axes_holder)
    plot = LeftArmJointDebugPlot(update_hz=1_000_000.0)

    retarget = np.asarray([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], dtype=np.float32)
    actual = np.asarray([20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0], dtype=np.float32)

    plot.push(1.0, retarget_joints=retarget, actual_joints=actual)
    plot.push(2.0, retarget_joints=retarget + 1.0, actual_joints=actual + 1.0)
    plot.maybe_update()

    retarget_line, actual_line = plot._lines
    np.testing.assert_allclose(retarget_line.data[1], [11.0, 12.0])
    np.testing.assert_allclose(actual_line.data[1], [21.0, 22.0])
    assert len(plot._lines) == 2
    assert not hasattr(plot, "_raw")

