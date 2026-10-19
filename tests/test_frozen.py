"""Opt-in checks against the actual one-file executable, not the source."""

import os
from pathlib import Path
import subprocess

import pytest


@pytest.fixture
def executable():
    configured = os.environ.get("VIDEO_PEDAL_EXECUTABLE")
    if not configured:
        pytest.skip("set VIDEO_PEDAL_EXECUTABLE to test a frozen build")
    path = Path(configured).resolve()
    assert path.is_file(), f"Missing executable: {path}"
    return path


def run_frozen(executable, argument, tmp_path):
    env = os.environ.copy()
    env["APPDATA"] = str(tmp_path / "appdata")
    env["XDG_CONFIG_HOME"] = str(tmp_path / "config")
    result = subprocess.run(
        [str(executable), argument], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (tmp_path / "appdata").exists()
    assert not (tmp_path / "config").exists()
    return result.stdout


def test_frozen_help(executable, tmp_path):
    output = run_frozen(executable, "--help", tmp_path)
    assert "usage:" in output
    assert "--no-vcam" in output
    assert "--no-pedal" in output


def test_frozen_dependencies(executable, tmp_path):
    output = run_frozen(executable, "--check-packaged-dependencies", tmp_path)
    assert "Packaged dependencies OK" in output
