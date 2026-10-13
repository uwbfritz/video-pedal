import time
import numpy as np
import pytest

from loop_pedal import (
    FREEZE, LIVE, LOOPING, RECORDING,
    LoopPedal, RawCodec, StudioEffects,
    STUDIO_BLURS, STUDIO_LUTS,
    compute_presence_score,
    draw_control_dock, parse_args
)


def gray(value: int, size: int = 64) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def sample_face_frame(size: int = 120) -> np.ndarray:
    """A synthetic frame with center brightness resembling a person."""
    img = np.full((size, size, 3), 40, dtype=np.uint8)
    # Draw center 'torso/face'
    y, x = np.ogrid[:size, :size]
    mask = ((x - size // 2) ** 2 + (y - size // 2) ** 2) < (size // 3) ** 2
    img[mask] = [180, 150, 130]
    return img


class TestStudioEffects:
    def test_init_and_constants(self):
        assert "warm" in STUDIO_LUTS
        assert "noir" in STUDIO_LUTS
        assert "studio" in STUDIO_BLURS
        studio = StudioEffects()
        assert studio.lut_name == "normal"
        assert studio.blur_mode == "off"

    def test_cycle_lut(self):
        studio = StudioEffects()
        initial = studio.lut_name
        seen = [initial]
        for _ in range(len(STUDIO_LUTS)):
            nxt = studio.cycle_lut()
            seen.append(nxt)
        assert seen[-1] == initial
        assert len(set(seen)) == len(STUDIO_LUTS)

    def test_cycle_blur(self):
        studio = StudioEffects()
        initial = studio.blur_mode
        seen = [initial]
        for _ in range(len(STUDIO_BLURS)):
            nxt = studio.cycle_blur()
            seen.append(nxt)
        assert seen[-1] == initial
        assert len(set(seen)) == len(STUDIO_BLURS)

    def test_set_lut_and_blur(self):
        studio = StudioEffects()
        studio.set_lut("warm")
        assert studio.lut_name == "warm"
        studio.set_lut("invalid_lut")
        assert studio.lut_name == "warm"

        studio.set_blur("studio")
        assert studio.blur_mode == "studio"
        studio.set_blur("invalid_blur")
        assert studio.blur_mode == "studio"

    def test_apply_lut_transformations(self):
        img = sample_face_frame(80)
        studio = StudioEffects()

        # normal does nothing
        assert np.array_equal(studio.apply_lut(img), img)

        for lut in ["warm", "cool", "vibrant", "noir", "soft"]:
            studio.set_lut(lut)
            res = studio.apply_lut(img)
            assert res.shape == img.shape
            assert res.dtype == np.uint8
            assert not np.array_equal(res, img)

    def test_apply_blur_modes(self):
        img = sample_face_frame(80)
        studio = StudioEffects()

        assert np.array_equal(studio.apply_blur(img), img)

        for mode in ["light", "studio", "strong"]:
            studio.set_blur(mode)
            res = studio.apply_blur(img)
            assert res.shape == img.shape
            assert res.dtype == np.uint8
            assert not np.array_equal(res, img)

    def test_compound_apply(self):
        img = sample_face_frame(80)
        studio = StudioEffects(lut_name="warm", blur_mode="studio")
        res = studio.apply(img)
        assert res.shape == img.shape
        assert res.dtype == np.uint8
        assert not np.array_equal(res, img)


class TestAutoCut:
    def test_auto_cut_toggle(self):
        pedal = LoopPedal(RawCodec(), fps=30.0, max_seconds=10.0, min_seconds=1.0, crossfade_seconds=0.0)
        assert pedal.auto_cut is True
        pedal.auto_cut_armed = True
        pedal.toggle_auto_cut()
        assert pedal.auto_cut is False
        assert pedal.auto_cut_armed is False

    def test_request_go_live_while_live(self):
        pedal = LoopPedal(RawCodec(), fps=30.0, max_seconds=10.0, min_seconds=1.0, crossfade_seconds=0.0)
        assert pedal.state == LIVE
        res = pedal.request_go_live()
        assert res is False
        assert pedal.auto_cut_armed is False

    def test_request_go_live_arms_and_overrides(self):
        pedal = LoopPedal(RawCodec(), fps=30.0, max_seconds=10.0, min_seconds=1.0, crossfade_seconds=0.0)
        pedal.state = LOOPING
        pedal._player = "dummy"

        # 1st request -> arms
        res1 = pedal.request_go_live()
        assert res1 is True
        assert pedal.auto_cut_armed is True
        assert pedal.state == LOOPING

        # 2nd request -> manual force override
        res2 = pedal.request_go_live()
        assert res2 is False
        assert pedal.auto_cut_armed is False
        assert pedal.state == LIVE

    def test_request_go_live_when_auto_cut_disabled(self):
        pedal = LoopPedal(RawCodec(), fps=30.0, max_seconds=10.0, min_seconds=1.0, crossfade_seconds=0.0, auto_cut=False)
        pedal.state = LOOPING
        pedal._player = "dummy"
        res = pedal.request_go_live()
        assert res is False
        assert pedal.state == LIVE

    def test_check_auto_cut_match(self):
        pedal = LoopPedal(RawCodec(), fps=30.0, max_seconds=10.0, min_seconds=1.0, crossfade_seconds=0.0)
        pedal.state = LOOPING
        pedal._player = "dummy"
        pedal.request_go_live()
        assert pedal.auto_cut_armed is True

        # Low score: stays armed
        assert pedal.check_auto_cut(0.65) is None
        assert pedal.auto_cut_armed is True

        # High score: matches and cuts to live
        assert pedal.check_auto_cut(0.85) == "matched"
        assert pedal.auto_cut_armed is False
        assert pedal.state == LIVE

    def test_check_auto_cut_timeout(self):
        pedal = LoopPedal(RawCodec(), fps=30.0, max_seconds=10.0, min_seconds=1.0, crossfade_seconds=0.0)
        pedal.state = LOOPING
        pedal._player = "dummy"
        pedal.request_go_live()
        pedal.auto_cut_arm_time = time.time() - 10.0  # Expired timeout

        assert pedal.check_auto_cut(0.50) == "timeout"
        assert pedal.auto_cut_armed is False
        assert pedal.state == LIVE


class TestAutoStealth:
    def test_toggle_auto_stealth(self):
        pedal = LoopPedal(RawCodec(), fps=30.0, max_seconds=10.0, min_seconds=1.0, crossfade_seconds=0.0)
        assert pedal.auto_stealth is False
        pedal.toggle_auto_stealth()
        assert pedal.auto_stealth is True

    def test_stealth_requires_loop(self):
        pedal = LoopPedal(RawCodec(), fps=30.0, max_seconds=10.0, min_seconds=1.0, crossfade_seconds=0.0, auto_stealth=True)
        seated_frame = sample_face_frame(80)
        # No loop in slots -> returns "no_loop"
        assert pedal.check_stealth_presence(seated_frame) == "no_loop"
        info = pedal.hud_info()
        assert info["has_loop"] is False

    def test_compute_presence_score(self):
        seated = sample_face_frame(80)
        empty = gray(20, size=80)
        # Identical frame gives high score
        assert compute_presence_score(seated, seated) > 0.95
        # Empty desk vs seated person drops significantly
        score = compute_presence_score(empty, seated)
        assert score < 0.60

    def test_stealth_absence_and_return(self):
        pedal = LoopPedal(RawCodec(), fps=30.0, max_seconds=10.0, min_seconds=1.0, crossfade_seconds=0.0, auto_stealth=True, stealth_threshold=0.65)
        seated_frame = sample_face_frame(80)
        empty_frame = gray(20, size=80)

        # Preload slot 1 with a seated frame loop
        pedal._slots[1] = [seated_frame]
        pedal.stealth_absence_threshold = 3  # short threshold for test
        pedal.stealth_return_required = 2

        # In LIVE state with empty frame
        assert pedal.check_stealth_presence(empty_frame) is None
        assert pedal.check_stealth_presence(empty_frame) is None
        # Third absence frame triggers stealth
        res = pedal.check_stealth_presence(empty_frame)
        assert res == "stealth_on"
        assert pedal.state == LOOPING
        assert pedal.stealth_engaged_by_sensor is True

        # Now looping under stealth: user returns
        assert pedal.check_stealth_presence(seated_frame) is None
        res_ret = pedal.check_stealth_presence(seated_frame)
        assert res_ret == "stealth_off"
        assert pedal.state == LIVE
        assert pedal.stealth_engaged_by_sensor is False

    def test_stealth_threshold_adjustment(self):
        pedal = LoopPedal(RawCodec(), fps=30.0, max_seconds=10.0, min_seconds=1.0, crossfade_seconds=0.0, stealth_threshold=0.65)
        assert pedal.stealth_threshold == 0.65

        # Test set_stealth_threshold
        th = pedal.set_stealth_threshold(0.70)
        assert th == 0.70
        assert pedal.stealth_threshold == 0.70
        assert pedal.stealth_return_threshold >= 0.70

        # Clamping
        assert pedal.set_stealth_threshold(0.10) == 0.30
        assert pedal.set_stealth_threshold(0.99) == 0.90

        # Cycle presets
        pedal.set_stealth_threshold(0.55)
        assert pedal.cycle_stealth_threshold() == 0.60
        assert pedal.cycle_stealth_threshold() == 0.65
        assert pedal.cycle_stealth_threshold() == 0.70
        assert pedal.cycle_stealth_threshold() == 0.75
        assert pedal.cycle_stealth_threshold() == 0.55


class TestCLIAndDockIntegration:
    def test_parse_args_new_flags(self):
        args = parse_args(["--auto-stealth", "--stealth-threshold", "0.72", "--no-auto-cut", "--lut", "vibrant", "--blur", "studio"])
        assert args.auto_stealth is True
        assert args.stealth_threshold == 0.72
        assert args.auto_cut is False
        assert args.lut == "vibrant"
        assert args.blur == "studio"

    def test_dock_has_feature_buttons(self):
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        info = {
            "state": LOOPING,
            "slot": 1,
            "slots": {1: {"filled": True, "duration": 4.0}},
            "auto_cut": True,
            "auto_cut_armed": False,
            "auto_stealth": True,
            "stealth_threshold": 0.65,
            "has_loop": True,
            "studio_lut": "warm",
            "studio_blur": "studio",
            "ping_pong": False,
        }
        buttons = []
        draw_control_dock(img, info, key="alt_r", live_key="ctrl_r",
                          clickable_buttons=buttons)
        btn_ids = {b["id"] for b in buttons}

        assert "toggle_auto_cut" in btn_ids
        assert "toggle_auto_stealth" in btn_ids
        assert "cycle_stealth_th" in btn_ids
        assert "cycle_lut" in btn_ids
        assert "cycle_blur" in btn_ids

