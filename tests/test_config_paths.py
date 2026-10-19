from pathlib import Path

import pytest

import loop_pedal


@pytest.fixture
def frozen_home(monkeypatch, tmp_path):
    monkeypatch.setattr(loop_pedal.sys, "frozen", True, raising=False)
    monkeypatch.setattr(loop_pedal.sys, "_MEIPASS", str(tmp_path / "extraction"), raising=False)
    monkeypatch.setattr(loop_pedal, "__file__", str(tmp_path / "extraction" / "loop_pedal.py"))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    monkeypatch.delenv("APPDATA", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    return tmp_path / "home"


def test_source_config_stays_beside_script(monkeypatch, tmp_path):
    monkeypatch.setattr(loop_pedal.sys, "frozen", False, raising=False)
    monkeypatch.setattr(loop_pedal, "__file__", str(tmp_path / "source" / "loop_pedal.py"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "other"))
    assert Path(loop_pedal.get_config_path()) == tmp_path / "source" / "config.json"


@pytest.mark.parametrize(
    "platform,variable,default",
    [
        ("linux", "XDG_CONFIG_HOME", Path(".config")),
        ("win32", "APPDATA", Path("AppData") / "Roaming"),
    ],
)
@pytest.mark.parametrize("override", [None, "", "relative", "absolute"])
def test_frozen_config_location(monkeypatch, tmp_path, frozen_home, platform, variable,
                                default, override):
    monkeypatch.setattr(loop_pedal.sys, "platform", platform)
    if override is not None:
        monkeypatch.setenv(variable, str(tmp_path / "custom") if override == "absolute" else override)
    base = tmp_path / "custom" if override == "absolute" else frozen_home / default
    assert Path(loop_pedal.get_config_path()) == base / "video-pedal" / "config.json"
    assert not base.exists()


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_frozen_config_survives_extraction_and_working_directory_changes(
    monkeypatch, tmp_path, frozen_home, platform
):
    monkeypatch.setattr(loop_pedal.sys, "platform", platform)
    first_path = loop_pedal.get_config_path()
    monkeypatch.setattr(loop_pedal, "CONFIG_PATH", first_path)
    assert loop_pedal.load_config() == {}
    assert not Path(first_path).parent.exists()
    loop_pedal.save_config({"key": "f13", "stealth_threshold": 0.75})
    monkeypatch.setattr(loop_pedal.sys, "_MEIPASS", str(tmp_path / "next-extraction"))
    monkeypatch.setattr(loop_pedal, "__file__", str(tmp_path / "next-extraction" / "loop_pedal.py"))
    monkeypatch.chdir(tmp_path)
    assert loop_pedal.get_config_path() == first_path
    monkeypatch.setattr(loop_pedal, "CONFIG_PATH", loop_pedal.get_config_path())
    loop_pedal.save_config({"live_key": "f14"})
    assert loop_pedal.load_config() == {
        "key": "f13", "live_key": "f14", "stealth_threshold": 0.75,
    }
    args = loop_pedal.parse_args([])
    assert (args.key, args.live_key, args.stealth_threshold) == ("f13", "f14", 0.75)
    assert args.save_dir == "./saved_loops"
