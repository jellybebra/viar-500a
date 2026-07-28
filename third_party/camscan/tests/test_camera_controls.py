"""Tests for the V4L2 controls used by the camera settings window."""

from types import SimpleNamespace

from camscan import camera


V4L2_OUTPUT = """
User Controls

    brightness 0x00980900 (int) : min=-10 max=10 step=1 default=-1 value=2
    white_balance_automatic 0x0098090c (bool) : default=1 value=1
    power_line_frequency 0x00980918 (menu) : min=0 max=2 default=1 value=1
        0: Disabled
        1: 50 Hz
        2: 60 Hz
"""


def camera_without_device():
    instance = camera.Camera.__new__(camera.Camera)
    instance.index = 0
    return instance


def test_get_controls_parses_ranges_boolean_and_menu(monkeypatch):
    monkeypatch.setattr(camera.shutil, "which", lambda command: "/usr/bin/v4l2-ctl")
    monkeypatch.setattr(
        camera.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=V4L2_OUTPUT,
            stderr="",
        ),
    )

    controls = camera_without_device().get_controls()

    assert controls["brightness"] == {
        "type": "int",
        "menu": {},
        "min": -10,
        "max": 10,
        "step": 1,
        "default": -1,
        "value": 2,
    }
    assert controls["white_balance_automatic"]["value"] == 1
    assert controls["power_line_frequency"]["menu"] == {
        0: "Disabled",
        1: "50 Hz",
        2: "60 Hz",
    }


def test_set_control_uses_argument_list_without_shell(monkeypatch):
    calls = []
    monkeypatch.setattr(camera.shutil, "which", lambda command: "/usr/bin/v4l2-ctl")

    def fake_run(arguments, **kwargs):
        calls.append(arguments)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(camera.subprocess, "run", fake_run)

    assert camera_without_device().set_control("brightness", -3)
    assert calls == [
        [
            "v4l2-ctl",
            "-d",
            "/dev/video0",
            "--set-ctrl=brightness=-3",
        ]
    ]


def test_set_control_rejects_untrusted_name(monkeypatch):
    monkeypatch.setattr(camera.shutil, "which", lambda command: "/usr/bin/v4l2-ctl")
    assert not camera_without_device().set_control("brightness;reboot", 1)
