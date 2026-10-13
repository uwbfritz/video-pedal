import json
import os
import queue
import tempfile
import numpy as np
import pytest

import sys
from loop_pedal import (
    set_window_borderless,
    FREEZE, LIVE, LOOPING, RECORDING,
    LoopPedal, Pedal, RawCodec, ToastNotifier, draw_hud,
    draw_control_dock, draw_help_modal, draw_text_shadow, draw_vector_icon,
    draw_key_config_modal, draw_state_border,
    draw_rounded_rect, draw_glass_pill,
    key_to_str, load_config, parse_args, save_config
)



def make_canvas(w=640, h=480):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestStateBorders:
    def test_draw_state_border_live(self):
        img = make_canvas()
        draw_state_border(img, LIVE)
        # Top-left border pixel should be green (50, 205, 50)
        assert img[0, 0, 1] == 205

    def test_draw_state_border_looping(self):
        img = make_canvas()
        draw_state_border(img, LOOPING)
        # Should have strong red and green components (amber/orange)
        assert img[0, 0, 2] > 150

    def test_draw_state_border_recording(self):
        img = make_canvas()
        draw_state_border(img, RECORDING)
        # Flashing red border
        assert img[0, 0, 2] > 100

    def test_draw_state_border_freeze(self):
        img = make_canvas()
        draw_state_border(img, FREEZE)
        # Ice-blue border (high blue/green)
        assert img[0, 0, 0] > 150


class TestToastNotifier:
    def test_toast_shows_and_draws(self):
        toast = ToastNotifier(duration=1.0)
        toast.show("TEST TOAST", (0, 255, 0))
        img = make_canvas()
        toast.draw(img)
        # Non-black pixels drawn
        assert np.any(img > 0)

    def test_toast_expires(self):
        toast = ToastNotifier(duration=0.01)
        toast.show("EXPIRED", (0, 255, 0))
        toast.start_time -= 0.1  # Force expiration
        img = make_canvas()
        toast.draw(img)
        # Nothing drawn because it expired
        assert np.all(img == 0)


class TestKeyConfigModal:
    def test_draw_key_config_modal(self):
        img = make_canvas()
        draw_key_config_modal(img, step=1, current_pedal="alt_r", current_live="ctrl_r")
        assert np.any(img > 0)

        img2 = make_canvas()
        draw_key_config_modal(img2, step=2, current_pedal="alt_r", current_live="ctrl_r")
        assert np.any(img2 > 0)


class TestMiniHud:
    def test_draw_hud_mini_mode(self):
        img = make_canvas(w=360, h=202)
        info = {"state": LOOPING, "elapsed": 3.0, "total": 10.0, "progress": 0.3, "slot": 2}
        draw_hud(img, info, key="alt_r", live_key="ctrl_r", mini=True)
        assert np.any(img > 0)
        # Verify the top area is completely unobstructed (no top app bar in mini mode)
        assert np.all(img[:40, :] == 0)
        # Verify the bottom area has the floating pill
        assert np.any(img[160:, :] > 0)

    def test_draw_rounded_rect_and_glass_pill(self):
        img = make_canvas(w=200, h=100)
        draw_rounded_rect(img, (10, 10), (190, 90), (0, 255, 0), thickness=-1, radius=8)
        assert np.any(img > 0)
        draw_glass_pill(img, (15, 15), (185, 85), bg_color=(20, 20, 20), border_color=(255, 255, 255))
        assert np.any(img > 0)


class TestConfigPersistence:
    def test_save_and_load_config(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_file = os.path.join(tmpdir, "config.json")
            monkeypatch.setattr("loop_pedal.CONFIG_PATH", cfg_file)

            save_config({"key": "f13", "live_key": "f14"})
            assert os.path.exists(cfg_file)

            loaded = load_config()
            assert loaded["key"] == "f13"
            assert loaded["live_key"] == "f14"


class TestPedalRebindAndCapture:
    @pytest.fixture
    def Key(self):
        return pytest.importorskip("pynput.keyboard").Key

    def test_key_to_str(self, Key):
        assert key_to_str(Key.ctrl_r) == "ctrl_r"
        assert key_to_str(Key.alt_r) == "alt_r"

    def test_rebind(self):
        q = queue.Queue()
        pedal = Pedal("alt_r", q, "ctrl_r")
        assert pedal._key is not None
        # Rebind to different keys without crashing
        assert pedal.rebind("shift_r", "cmd_r") is not None

    def test_capture_callback_intercepts_press(self, Key):
        q = queue.Queue()
        pedal = Pedal("alt_r", q, "ctrl_r")
        captured = []
        pedal.set_capture_callback(captured.append)

        pedal._on_press(Key.space)
        assert captured == ["space"]
        assert q.empty()  # Should not emit pedal 'down' while capturing!


class TestCliArgs:
    def test_topmost_and_mini_flags(self):
        args = parse_args(["--topmost", "--mini", "--no-border"])
        assert args.topmost is True
        assert args.mini is True
        assert args.no_border is True
        assert args.frameless is True

    def test_no_frameless_flag(self):
        args = parse_args(["--no-frameless"])
        assert args.frameless is False


class TestBorderlessWindow:
    def test_set_window_borderless_nonexistent(self):
        assert set_window_borderless("non_existent_window_12345", True) is False

    @pytest.mark.skipif(sys.platform != "win32", reason="Win32 specific window test")
    def test_set_window_borderless_win32(self):
        import cv2
        win_name = "test_borderless_unit"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        try:
            assert set_window_borderless(win_name, True) is True
            assert set_window_borderless(win_name, False) is True
        finally:
            cv2.destroyAllWindows()


class TestModernUiComponents:
    def test_draw_text_shadow(self):
        import cv2
        img = make_canvas(w=300, h=100)
        draw_text_shadow(img, "TEST SHADOW", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        assert np.any(img > 0)

    def test_draw_vector_icons(self):
        img = make_canvas(w=400, h=100)
        for idx, icon in enumerate(["rec", "stop", "play", "pause", "boom", "gear", "help", "save", "expand"]):
            draw_vector_icon(img, icon, (30 + idx * 40, 50), 20, (0, 200, 255))
        assert np.any(img > 0)

    def test_draw_help_modal(self):
        img = make_canvas(w=800, h=600)
        draw_help_modal(img, current_pedal="alt_r", current_live="ctrl_r", mouse_pos=(100, 100))
        assert np.any(img > 0)

    def test_draw_control_dock_registers_buttons(self):
        img = make_canvas(w=800, h=600)
        buttons = []
        info = {
            "state": LIVE,
            "elapsed": None,
            "total": None,
            "progress": None,
            "slot": 1,
            "ping_pong": False,
            "slots": {1: {"filled": True, "duration": 4.5}, 2: {"filled": False, "duration": 0.0}}
        }
        draw_control_dock(img, info, key="alt_r", live_key="ctrl_r",
                          align_mode="ghost", mouse_pos=(400, 570), clickable_buttons=buttons)
        assert np.any(img > 0)
        assert len(buttons) >= 8
        button_ids = [b["id"] for b in buttons]
        assert "toggle_record" in button_ids
        assert "toggle_freeze" in button_ids
        assert "slot_1" in button_ids
        assert "slot_2" in button_ids
        assert "toggle_ping_pong" in button_ids
        assert "cycle_align" in button_ids
        assert "toggle_help" in button_ids
        assert "toggle_mini" in button_ids

    def test_hud_info_includes_slots_metadata(self):
        pedal = LoopPedal(RawCodec(), fps=10, max_seconds=10, min_seconds=0.2, crossfade_seconds=0.2)
        info = pedal.hud_info()
        assert "slots" in info
        assert 1 in info["slots"]
        assert info["slots"][1]["filled"] is False
        assert info["slots"][1]["duration"] == 0.0

