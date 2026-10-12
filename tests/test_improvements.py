import os
import tempfile
import numpy as np
import pytest

from loop_pedal import (
    FREEZE, LIVE, LOOPING, RECORDING, LoopPedal, RawCodec,
    blend_overlay, build_loop, build_ping_pong_loop,
    compute_alignment_score, find_best_seam,
    load_loop_from_file, parse_args, save_loop_to_file
)


def gray(value, size=32):
    return np.full((size, size, 3), value, dtype=np.uint8)


def frames(values, size=32):
    return [gray(v, size=size) for v in values]


def level(frame):
    return int(frame[0, 0, 0])


class TestSmartSeam:
    def test_find_best_seam_picks_closest_posture(self):
        # 10 frames where frame 0 is level 10, tail has 100, 200, 12, 90
        # The best match to frame 0 is frame 8 (level 12)
        vals = [10, 50, 70, 80, 90, 100, 100, 200, 12, 90]
        src = frames(vals)
        h, t = find_best_seam(src, crossfade=1, codec=RawCodec(), max_search=4)
        assert h == 0
        assert t == 8  # 12 is closest to 10

    def test_find_best_seam_short_input_fallback(self):
        src = frames([10, 20])
        h, t = find_best_seam(src, crossfade=1, codec=RawCodec())
        assert (h, t) == (0, 1)

    def test_build_loop_with_smart_seam(self):
        vals = [10, 50, 70, 80, 90, 100, 100, 200, 12, 90]
        src = frames(vals)
        out = build_loop(src, crossfade=1, codec=RawCodec(), smart_seam=True)
        # Should trim to 0..8 then crossfade 1 frame
        assert len(out) == 9 - 1


class TestPingPong:
    def test_build_ping_pong_loop(self):
        src = frames([10, 20, 30, 40])
        out = build_ping_pong_loop(src, crossfade=0, codec=RawCodec())
        # [10, 20, 30, 40] + reverse [30, 20] = 6 frames
        levels = [level(f) for f in out]
        assert levels == [10, 20, 30, 40, 30, 20]


class TestAlignmentModes:
    def test_ghost_mode(self):
        live, loop = gray(0), gray(100)
        out = blend_overlay(live, loop, 0.5, mode="ghost")
        assert level(out) == 50

    def test_edge_mode(self):
        live = np.zeros((32, 32, 3), dtype=np.uint8)
        loop = np.zeros((32, 32, 3), dtype=np.uint8)
        # Draw a bright box in the loop
        loop[10:20, 10:20] = 255
        out = blend_overlay(live, loop, mode="edge")
        assert out.shape == live.shape
        # Edge pixels should be cyan outline
        assert np.any(out > 0)

    def test_diff_mode(self):
        live, loop = gray(0), gray(100)
        out = blend_overlay(live, loop, mode="diff")
        assert out.shape == live.shape

    def test_split_mode(self):
        live, loop = gray(0), gray(100)
        out = blend_overlay(live, loop, mode="split")
        assert out.shape == live.shape
        # Left half should be loop (100), right half should be live (0)
        assert int(out[0, 0, 0]) == 100
        assert int(out[0, -1, 0]) == 0

    def test_compute_alignment_score_identical(self):
        f = gray(50)
        score = compute_alignment_score(f, f)
        assert score == 1.0

    def test_compute_alignment_score_different(self):
        f1, f2 = gray(0), gray(255)
        score = compute_alignment_score(f1, f2)
        assert score < 0.2


class TestFreezeMode:
    def test_freeze_holds_frame_and_dissolves_to_live(self):
        pedal = LoopPedal(RawCodec(), fps=10, max_seconds=5, min_seconds=0.2, crossfade_seconds=0.2)
        pedal.process(gray(10))
        pedal.freeze(gray(10))
        assert pedal.state == FREEZE

        # Sending different live frames during freeze should still output frozen 10
        assert level(pedal.process(gray(80))) == 10
        assert level(pedal.process(gray(90))) == 10

        # go_live dissolves from frozen (10) to live (70)
        pedal.go_live()
        f1 = pedal.process(gray(70))
        f2 = pedal.process(gray(70))
        f3 = pedal.process(gray(70))
        assert level(f3) == 70
        assert pedal.state == LIVE


class TestSlotsAndPresets:
    def test_multi_slot_recording_and_switching(self):
        pedal = LoopPedal(RawCodec(), fps=10, max_seconds=5, min_seconds=0.1, crossfade_seconds=0.0)
        
        # Record into slot 1
        pedal.pedal_down()
        for v in [1, 2, 3]:
            pedal.process(gray(v))
        pedal.pedal_up()
        assert pedal.state == LOOPING
        assert pedal.active_slot == 1
        assert level(pedal.process(gray(0))) == 1

        # Switch to slot 2 and record
        pedal.select_slot(2)
        assert pedal.active_slot == 2
        pedal.pedal_down()
        for v in [10, 20, 30]:
            pedal.process(gray(v))
        pedal.pedal_up()
        assert level(pedal.process(gray(0))) == 10

        # Switch back to slot 1
        pedal.select_slot(1)
        assert level(pedal.process(gray(0))) == 1


class TestPersistence:
    def test_save_and_load_loop(self):
        codec = RawCodec()
        src = frames([10, 50, 100], size=64)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_loop.mp4")
            assert save_loop_to_file(src, codec, path, fps=10)
            assert os.path.exists(path)

            loaded = load_loop_from_file(path, codec)
            assert len(loaded) == 3
            assert loaded[0].shape == (64, 64, 3)

    def test_slot_save_and_load(self):
        pedal = LoopPedal(RawCodec(), fps=10, max_seconds=5, min_seconds=0.1, crossfade_seconds=0.0)
        pedal.pedal_down()
        for v in [50, 100, 150]:
            pedal.process(gray(v, size=64))
        pedal.pedal_up()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "slot1.mp4")
            assert pedal.save_slot(path, 1)

            pedal2 = LoopPedal(RawCodec(), fps=10, max_seconds=5, min_seconds=0.1, crossfade_seconds=0.0)
            assert pedal2.load_slot(path, 1)
            assert pedal2.state == LOOPING
            assert len(pedal2._slots[1]) == 3


class TestWindowsKeyErgonomics:
    def test_windows_live_key_default(self, monkeypatch):
        monkeypatch.setattr("sys.platform", "win32")
        monkeypatch.setattr("loop_pedal.load_config", lambda: {})
        args = parse_args([])
        assert args.live_key == "ctrl_r"

    def test_mac_live_key_default(self, monkeypatch):
        monkeypatch.setattr("sys.platform", "darwin")
        monkeypatch.setattr("loop_pedal.load_config", lambda: {})
        args = parse_args([])
        assert args.live_key == "cmd_r"
