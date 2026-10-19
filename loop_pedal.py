#!/usr/bin/env python3
"""
loop_pedal.py -- a loop pedal for your webcam.

Hold the pedal key: the live camera keeps going out to the call while the
frames are recorded. Release it: the recording plays on a loop to the call
instead of the live feed. Press the live key once: the loop dissolves into
the live feed. While a loop plays, the preview ghosts it over the live camera
so you can line yourself up before going live.

The output is published as a virtual camera ("OBS Virtual Camera" on macOS and
Windows, a v4l2loopback device on Linux), which Zoom, Meet, Teams, Discord and
friends see as an ordinary webcam.

    python loop_pedal.py                 # preview window + virtual camera
    python loop_pedal.py --no-vcam       # preview only, no OBS needed
    python loop_pedal.py --list-cameras  # find the index of your real webcam
    python loop_pedal.py --key f13       # use a different pedal key
    python loop_pedal.py --live-key f14  # use a different go-live key
    python loop_pedal.py --overlay 0     # preview shows exactly what the call sees
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import time

import cv2
import numpy as np

LIVE, RECORDING, LOOPING, FREEZE = "LIVE", "REC", "LOOP", "FREEZE"


def get_config_path() -> str:
    if not getattr(sys, "frozen", False):
        return str(Path(__file__).absolute().parent / "config.json")
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
        if not base.is_absolute():
            base = Path.home() / "AppData" / "Roaming"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
        if not base.is_absolute():
            base = Path.home() / ".config"
    return str(base / "video-pedal" / "config.json")


CONFIG_PATH = get_config_path()


def load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_config(data: dict) -> None:
    try:
        current = load_config()
        current.update(data)
        Path(CONFIG_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)
    except Exception as exc:
        print(f"Warning: could not save config: {exc}")


# --------------------------------------------------------------------------- #
# Frame storage
# --------------------------------------------------------------------------- #

class JpegCodec:
    """Frames are kept in RAM as JPEG so a 30 s clip is ~100 MB, not ~2.5 GB."""

    def __init__(self, quality: int = 90):
        self._params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]

    def encode(self, frame: np.ndarray) -> bytes:
        ok, buf = cv2.imencode(".jpg", frame, self._params)
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return buf.tobytes()

    def decode(self, data: bytes) -> np.ndarray:
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


class RawCodec:
    """Lossless passthrough. Used by the tests; fine for short clips too."""

    def encode(self, frame: np.ndarray) -> np.ndarray:
        return frame.copy()

    def decode(self, data: np.ndarray) -> np.ndarray:
        return data


class Recorder:
    """Collects encoded frames while the pedal is held; keeps only the newest max_frames."""

    def __init__(self, codec, max_frames: int):
        self._codec = codec
        self._frames: collections.deque = collections.deque(maxlen=max(1, max_frames))

    def push(self, frame: np.ndarray) -> None:
        self._frames.append(self._codec.encode(frame))

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def capacity(self) -> int:
        return self._frames.maxlen

    def take(self) -> list:
        frames = list(self._frames)
        self._frames.clear()
        return frames


def find_best_seam(frames: list, crossfade: int, codec, max_search: int | None = None) -> tuple[int, int]:
    """Find head_idx and tail_idx (0 <= head_idx < tail_idx < len(frames))
    such that decoded frames[tail_idx] and frames[head_idx] have minimum visual difference,
    preserving at least 2 * crossfade frames between them.
    """
    n = len(frames)
    k = max(0, int(crossfade))
    if n <= 2 * k:
        return 0, n - 1

    search = max_search if max_search is not None else min(n // 4, 45)
    if search < 1:
        return 0, n - 1

    def get_thumb(idx: int) -> np.ndarray:
        raw = codec.decode(frames[idx])
        if raw is None:
            return np.zeros((36, 64), dtype=np.uint8)
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) if (len(raw.shape) == 3 and raw.shape[2] >= 3) else raw
        if len(gray.shape) == 3 and gray.shape[2] == 1:
            gray = gray[:, :, 0]
        return cv2.resize(gray, (64, 36), interpolation=cv2.INTER_NEAREST)

    head_thumbs = {i: get_thumb(i) for i in range(search)}
    tail_thumbs = {j: get_thumb(j) for j in range(n - search, n)}

    best_diff = float("inf")
    best_pair = (0, n - 1)

    for h, h_thumb in head_thumbs.items():
        for t, t_thumb in tail_thumbs.items():
            if (t - h) < 2 * k:
                continue
            diff = float(np.mean(np.abs(t_thumb.astype(np.int16) - h_thumb.astype(np.int16))))
            if diff < best_diff:
                best_diff = diff
                best_pair = (h, t)

    return best_pair


def build_loop(frames: list, crossfade: int, codec, smart_seam: bool = False) -> list:
    """Turn a recording into a list of encoded frames that plays end-to-start without a jump cut.

    The last `crossfade` frames are dissolved into the first `crossfade` frames, so
    the seam is a short fade rather than a cut. The result has len(frames) - crossfade
    frames. `crossfade` is clamped to half the recording.

    If `smart_seam` is True, trims candidate head and tail to match posture before crossfading.
    """
    if smart_seam and len(frames) > 2 * max(1, int(crossfade)):
        h_idx, t_idx = find_best_seam(frames, crossfade, codec)
        frames = frames[h_idx:t_idx + 1]

    n = len(frames)
    k = max(0, min(int(crossfade), n // 2))
    if k == 0:
        return list(frames)
    body = list(frames[k:n - k])
    seam = []
    for i in range(k):
        alpha = (i + 1) / (k + 1)
        tail = codec.decode(frames[n - k + i])
        head = codec.decode(frames[i])
        seam.append(codec.encode(cv2.addWeighted(tail, 1.0 - alpha, head, alpha, 0.0)))
    return body + seam


def build_ping_pong_loop(frames: list, crossfade: int, codec) -> list:
    """Turn a recording into a forward-reverse boomerang loop with smooth turnarounds."""
    if len(frames) <= 2:
        return list(frames)
    reverse_frames = frames[-2:0:-1]
    combined = list(frames) + list(reverse_frames)
    return build_loop(combined, crossfade, codec)


def compute_alignment_score(live: np.ndarray, loop: np.ndarray) -> float:
    """Calculates a normalized 0.0 .. 1.0 posture alignment score.
    Higher score means live posture matches loop posture closer.
    """
    if live.shape != loop.shape:
        loop = cv2.resize(loop, (live.shape[1], live.shape[0]))
    g_live = cv2.cvtColor(live, cv2.COLOR_BGR2GRAY) if (len(live.shape) == 3 and live.shape[2] >= 3) else live
    g_loop = cv2.cvtColor(loop, cv2.COLOR_BGR2GRAY) if (len(loop.shape) == 3 and loop.shape[2] >= 3) else loop
    t_live = cv2.resize(g_live, (64, 36), interpolation=cv2.INTER_NEAREST)
    t_loop = cv2.resize(g_loop, (64, 36), interpolation=cv2.INTER_NEAREST)
    mae = float(np.mean(np.abs(t_live.astype(np.float32) - t_loop.astype(np.float32))))
    return max(0.0, min(1.0, 1.0 - (mae / 60.0)))


def compute_presence_score(live: np.ndarray, ref: np.ndarray) -> float:
    """Calculates center-weighted presence/posture score against reference loop.
    Focuses on central torso and head region where seated person appears.
    Returns 0.0 .. 1.0 (higher = person present and posture matches).
    """
    if live.shape != ref.shape:
        ref = cv2.resize(ref, (live.shape[1], live.shape[0]))
    g_live = cv2.cvtColor(live, cv2.COLOR_BGR2GRAY) if (len(live.shape) == 3 and live.shape[2] >= 3) else live
    g_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY) if (len(ref.shape) == 3 and ref.shape[2] >= 3) else ref
    t_live = cv2.resize(g_live, (64, 36), interpolation=cv2.INTER_NEAREST)
    t_ref = cv2.resize(g_ref, (64, 36), interpolation=cv2.INTER_NEAREST)

    full_mae = float(np.mean(np.abs(t_live.astype(np.float32) - t_ref.astype(np.float32))))
    full_score = max(0.0, min(1.0, 1.0 - (full_mae / 60.0)))

    # Central torso/head ROI: vertical 10%..88% (rows 4..32), horizontal 15%..85% (cols 10..54)
    c_live = t_live[4:32, 10:54]
    c_ref = t_ref[4:32, 10:54]
    c_mae = float(np.mean(np.abs(c_live.astype(np.float32) - c_ref.astype(np.float32))))
    c_score = max(0.0, min(1.0, 1.0 - (c_mae / 60.0)))

    return float(0.70 * c_score + 0.30 * full_score)


def blend_overlay(live: np.ndarray, loop: np.ndarray, alpha: float = 0.5,
                  mode: str = "ghost") -> np.ndarray:
    """Preview only: displays loop alignment over live camera.

    Modes:
      - 'ghost': classic alpha blend
      - 'edge': luminous Canny edge contour of loop over live view
      - 'diff': visual color-mapped difference (darker = better match)
      - 'split': side-by-side split view
    """
    a = min(1.0, max(0.0, float(alpha)))
    if loop.shape != live.shape:
        loop = cv2.resize(loop, (live.shape[1], live.shape[0]))

    if mode == "edge":
        loop_gray = cv2.cvtColor(loop, cv2.COLOR_BGR2GRAY) if (len(loop.shape) == 3 and loop.shape[2] >= 3) else loop
        edges = cv2.Canny(loop_gray, 40, 120)
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        out = live.copy()
        if len(out.shape) == 3 and out.shape[2] >= 3:
            out[edges > 0] = [255, 230, 0]  # vibrant cyan outline
        else:
            out[edges > 0] = 255
        return out

    if mode == "diff":
        diff = cv2.absdiff(live, loop)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) if (len(diff.shape) == 3 and diff.shape[2] >= 3) else diff
        diff_norm = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX)
        if len(live.shape) == 3 and live.shape[2] >= 3:
            colored_diff = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
            mask = (diff_norm > 40).astype(np.float32)[:, :, None]
            return (live.astype(np.float32) * (1.0 - 0.6 * mask) + colored_diff.astype(np.float32) * (0.6 * mask)).astype(np.uint8)
        return diff

    if mode == "split":
        h, w = live.shape[:2]
        mid = w // 2
        out = live.copy()
        out[:, :mid] = loop[:, :mid]
        if w > 20:
            cv2.line(out, (mid, 0), (mid, h), (0, 255, 255), 2)
            cv2.putText(out, "LOOP", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(out, "LIVE", (mid + 10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        return out

    return cv2.addWeighted(live, 1.0 - a, loop, a, 0.0)


def save_loop_to_file(loop_frames: list, codec, filepath: str, fps: float = 30.0) -> bool:
    """Save loop frames as an MP4 video file."""
    if not loop_frames:
        return False
    parent = os.path.dirname(os.path.abspath(filepath))
    if parent:
        os.makedirs(parent, exist_ok=True)
    first = codec.decode(loop_frames[0])
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(filepath, fourcc, fps, (w, h))
    if not writer.isOpened():
        return False
    for raw in loop_frames:
        writer.write(codec.decode(raw))
    writer.release()
    return True


def load_loop_from_file(filepath: str, codec) -> list:
    """Load loop frames from an MP4/video file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Loop file not found: {filepath}")
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {filepath}")
    frames = []
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(codec.encode(frame))
    cap.release()
    if not frames:
        raise ValueError(f"No frames read from {filepath}")
    return frames



class Player:
    def __init__(self, frames: list, codec, start: int = 0):
        if not frames:
            raise ValueError("empty loop")
        self._frames = frames
        self._codec = codec
        self._i = start % len(frames)

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def position(self) -> int:
        return self._i

    def next(self) -> np.ndarray:
        frame = self._codec.decode(self._frames[self._i])
        self._i = (self._i + 1) % len(self._frames)
        return frame


# --------------------------------------------------------------------------- #
# Studio post-processing (Background blur & Color grading LUTs)
# --------------------------------------------------------------------------- #

STUDIO_LUTS = ["normal", "warm", "cool", "vibrant", "noir", "soft"]
STUDIO_BLURS = ["off", "light", "studio", "strong"]


class StudioEffects:
    """Real-time color grading LUTs and portrait depth bokeh blur."""

    def __init__(self, lut_name: str = "normal", blur_mode: str = "off"):
        self.lut_name = lut_name if lut_name in STUDIO_LUTS else "normal"
        self.blur_mode = blur_mode if blur_mode in STUDIO_BLURS else "off"
        self._mask_cache: dict[tuple[int, int], np.ndarray] = {}
        self._warm_luts = None
        self._cool_luts = None
        self._noir_lut = None
        self._init_luts()

    def _init_luts(self):
        # Warm studio: boosted reds/yellows, flattering skin tones
        r_w = np.clip(np.linspace(0, 265, 256), 0, 255).astype(np.uint8)
        g_w = np.clip(np.linspace(0, 258, 256), 0, 255).astype(np.uint8)
        b_w = np.clip(np.linspace(0, 245, 256), 0, 255).astype(np.uint8)
        self._warm_luts = (b_w, g_w, r_w)

        # Cool cinematic: crisp teal-cyan tone curve
        b_c = np.clip(np.linspace(5, 265, 256), 0, 255).astype(np.uint8)
        g_c = np.clip(np.linspace(0, 252, 256), 0, 255).astype(np.uint8)
        r_c = np.clip(np.linspace(0, 245, 256), 0, 255).astype(np.uint8)
        self._cool_luts = (b_c, g_c, r_c)

        # Noir: S-curve contrast monochrome
        s_lut = np.clip(128.0 + 128.0 * np.sin((np.arange(256) - 128) / 128.0 * (np.pi / 2)), 0, 255).astype(np.uint8)
        self._noir_lut = s_lut

    def set_lut(self, name: str) -> None:
        if name in STUDIO_LUTS:
            self.lut_name = name

    def cycle_lut(self) -> str:
        idx = (STUDIO_LUTS.index(self.lut_name) + 1) % len(STUDIO_LUTS)
        self.lut_name = STUDIO_LUTS[idx]
        return self.lut_name

    def set_blur(self, mode: str) -> None:
        if mode in STUDIO_BLURS:
            self.blur_mode = mode

    def cycle_blur(self) -> str:
        idx = (STUDIO_BLURS.index(self.blur_mode) + 1) % len(STUDIO_BLURS)
        self.blur_mode = STUDIO_BLURS[idx]
        return self.blur_mode

    def _get_vignette_mask(self, h: int, w: int) -> np.ndarray:
        key = (w, h)
        if key in self._mask_cache:
            return self._mask_cache[key]
        y, x = np.ogrid[:h, :w]
        cx, cy = w / 2.0, h * 0.46
        rx, ry = w * 0.38, h * 0.46
        dist = ((x - cx) ** 2) / (rx ** 2) + ((y - cy) ** 2) / (ry ** 2)
        mask = np.clip(1.0 - dist, 0.0, 1.0).astype(np.float32)
        k_blur = max(11, (min(w, h) // 16) | 1)
        mask = cv2.GaussianBlur(mask, (k_blur, k_blur), 0)
        mask_3c = mask[:, :, None]
        self._mask_cache[key] = mask_3c
        return mask_3c

    def apply_lut(self, frame: np.ndarray) -> np.ndarray:
        if self.lut_name == "normal" or frame is None or len(frame.shape) < 3:
            return frame

        if self.lut_name == "warm":
            b, g, r = cv2.split(frame)
            b_l, g_l, r_l = self._warm_luts
            return cv2.merge([cv2.LUT(b, b_l), cv2.LUT(g, g_l), cv2.LUT(r, r_l)])

        if self.lut_name == "cool":
            b, g, r = cv2.split(frame)
            b_l, g_l, r_l = self._cool_luts
            return cv2.merge([cv2.LUT(b, b_l), cv2.LUT(g, g_l), cv2.LUT(r, r_l)])

        if self.lut_name == "vibrant":
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.25)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        if self.lut_name == "noir":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            contrasted = cv2.LUT(gray, self._noir_lut)
            return cv2.cvtColor(contrasted, cv2.COLOR_GRAY2BGR)

        if self.lut_name == "soft":
            small_w, small_h = max(16, frame.shape[1] // 2), max(16, frame.shape[0] // 2)
            small = cv2.resize(frame, (small_w, small_h))
            b_small = cv2.GaussianBlur(small, (15, 15), 0)
            bloom = cv2.resize(b_small, (frame.shape[1], frame.shape[0]))
            return cv2.addWeighted(frame, 0.78, bloom, 0.22, 0)

        return frame

    def apply_blur(self, frame: np.ndarray) -> np.ndarray:
        if self.blur_mode == "off" or frame is None or len(frame.shape) < 3:
            return frame

        ksize_map = {"light": 11, "studio": 21, "strong": 31}
        ksize = ksize_map.get(self.blur_mode, 21)
        h, w = frame.shape[:2]

        small_w, small_h = max(16, w // 4), max(16, h // 4)
        small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
        b_small = cv2.GaussianBlur(small, (ksize, ksize), 0)
        blurred = cv2.resize(b_small, (w, h), interpolation=cv2.INTER_LINEAR)

        mask = self._get_vignette_mask(h, w)
        return (frame.astype(np.float32) * mask + blurred.astype(np.float32) * (1.0 - mask)).astype(np.uint8)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if self.lut_name == "normal" and self.blur_mode == "off":
            return frame
        out = frame
        if self.blur_mode != "off":
            out = self.apply_blur(out)
        if self.lut_name != "normal":
            out = self.apply_lut(out)
        return out


# --------------------------------------------------------------------------- #
# State machine
# --------------------------------------------------------------------------- #

class LoopPedal:
    """LIVE --hold--> REC --release--> LOOP --live key--> LIVE. A too-short hold is ignored."""

    def __init__(self, codec, fps: float, max_seconds: float, min_seconds: float,
                 crossfade_seconds: float, smart_seam: bool = False,
                 ping_pong: bool = False, auto_cut: bool = True,
                 auto_stealth: bool = False, stealth_threshold: float = 0.65,
                 log=print):
        self.state = LIVE
        self.fps = fps
        self._codec = codec
        self._recorder = Recorder(codec, int(max_seconds * fps))
        self._min_frames = max(1, int(min_seconds * fps))
        self._crossfade = int(crossfade_seconds * fps)
        self.smart_seam = smart_seam
        self.ping_pong = ping_pong

        # Feature #1: AI Pose Auto-Cut
        self.auto_cut = auto_cut
        self.auto_cut_armed = False
        self.auto_cut_arm_time = 0.0
        self.auto_cut_threshold = 0.80
        self.auto_cut_timeout = 6.0

        # Feature #3: Presence Sensor Auto-Stealth
        self.auto_stealth = auto_stealth
        self.stealth_absence_frames = 0
        self.stealth_absence_threshold = max(5, int(1.2 * fps))
        self.stealth_engaged_by_sensor = False
        self.stealth_threshold = stealth_threshold
        self.stealth_return_threshold = min(0.85, max(0.68, stealth_threshold + 0.07))
        self.stealth_return_frames = 0
        self.stealth_return_required = max(3, int(0.4 * fps))
        self.last_presence_score: float | None = None

        # Feature #6: Studio Post-Processing
        self.studio = StudioEffects()

        self._player: Player | None = None
        self._fade_left = 0  # frames left in the loop-to-live dissolve
        self._frozen_frame: np.ndarray | None = None
        self._last_live_frame: np.ndarray | None = None
        self._slots: dict[int, list] = {1: [], 2: [], 3: [], 4: [], 5: []}
        self.active_slot: int = 1
        self._log = log

    # -- pedal events --------------------------------------------------------

    def pedal_down(self) -> None:
        if self.state == RECORDING:
            return
        if self._fade_left:
            self._fade_left = 0
            self._player = None
            self._frozen_frame = None
        self.auto_cut_armed = False
        self.stealth_engaged_by_sensor = False
        self._recorder.take()
        self.state = RECORDING
        self._log("REC   recording (live feed still going out) - release to loop")

    def pedal_up(self) -> None:
        if self.state != RECORDING:
            return
        frames = self._recorder.take()
        if len(frames) < self._min_frames:
            self.state = LOOPING if self._player is not None else LIVE
            self._log(f"{self.state:<5} tap ignored (hold at least {self._min_frames / self.fps:.1f}s to record)")
            return
        if self.ping_pong:
            loop = build_ping_pong_loop(frames, self._crossfade, self._codec)
        else:
            loop = build_loop(frames, self._crossfade, self._codec, smart_seam=self.smart_seam)
        self._slots[self.active_slot] = loop
        k = min(self._crossfade, len(frames) // 2)
        start = len(loop) - k - 1 if k > 0 else 0
        self._player = Player(loop, self._codec, start=start)
        self.state = LOOPING
        self._log(f"LOOP  playing {len(loop) / self.fps:.1f}s loop (slot {self.active_slot}) - hold pedal to re-record, live key to go live")

    def toggle_record(self) -> None:
        """For keyboards without press/release events (the preview window)."""
        if self.state == RECORDING:
            self.pedal_up()
        else:
            self.pedal_down()

    def freeze(self, live_frame: np.ndarray | None = None) -> None:
        """Freeze on current live frame or resume live if already frozen."""
        if self.state == FREEZE:
            self.go_live()
            return
        frame = live_frame if live_frame is not None else self._last_live_frame
        if frame is None:
            return
        self.auto_cut_armed = False
        self._frozen_frame = frame.copy()
        self._player = Player([self._codec.encode(self._frozen_frame)], self._codec)
        self.state = FREEZE
        self._log("FREEZE frozen on current frame - press 'f' or live key to go live")

    def toggle_freeze(self, live_frame: np.ndarray | None = None) -> None:
        self.freeze(live_frame)

    def toggle_ping_pong(self) -> None:
        self.ping_pong = not self.ping_pong
        self._log(f"LOOP  ping-pong mode {'ON' if self.ping_pong else 'OFF'}")

    def toggle_auto_cut(self) -> bool:
        self.auto_cut = not self.auto_cut
        if not self.auto_cut:
            self.auto_cut_armed = False
        self._log(f"AUTOCUT mode {'ON' if self.auto_cut else 'OFF'}")
        return self.auto_cut

    def toggle_auto_stealth(self) -> bool:
        self.auto_stealth = not self.auto_stealth
        self.stealth_absence_frames = 0
        self._log(f"STEALTH mode {'ON' if self.auto_stealth else 'OFF'}")
        return self.auto_stealth

    def set_stealth_threshold(self, threshold: float) -> float:
        self.stealth_threshold = round(max(0.30, min(0.90, threshold)), 2)
        self.stealth_return_threshold = min(0.95, max(0.60, self.stealth_threshold + 0.07))
        self._log(f"STEALTH threshold set to {int(self.stealth_threshold * 100)}% (return at {int(self.stealth_return_threshold * 100)}%)")
        return self.stealth_threshold

    def cycle_stealth_threshold(self) -> float:
        presets = [0.55, 0.60, 0.65, 0.70, 0.75]
        cur = round(self.stealth_threshold, 2)
        idx = 0
        min_diff = 999.0
        for i, p in enumerate(presets):
            if abs(cur - p) < min_diff:
                min_diff = abs(cur - p)
                idx = i
        nxt = presets[(idx + 1) % len(presets)]
        return self.set_stealth_threshold(nxt)

    def request_go_live(self) -> bool:
        """Triggers go_live with smart auto-cut pose alignment if armed."""
        if self.state not in (LOOPING, FREEZE):
            self.go_live()
            return False
        if not self.auto_cut:
            self.go_live()
            return False
        if self.auto_cut_armed:
            # Second tap/click = manual force override cut
            self.auto_cut_armed = False
            self.go_live()
            self._log("AUTOCUT manual override -> immediate live dissolve")
            return False

        self.auto_cut_armed = True
        self.auto_cut_arm_time = time.time()
        self._log("AUTOCUT armed: match posture to go live (or tap live key to force cut)")
        return True

    def check_auto_cut(self, align_score: float) -> str | None:
        """Checks if armed auto-cut posture threshold is satisfied or timed out."""
        if not self.auto_cut_armed:
            return None
        if align_score >= self.auto_cut_threshold:
            self.auto_cut_armed = False
            self.go_live()
            self._log(f"AUTOCUT matched ({align_score*100:.0f}%) -> cut to live")
            return "matched"
        if time.time() - self.auto_cut_arm_time >= self.auto_cut_timeout:
            self.auto_cut_armed = False
            self.go_live()
            self._log("AUTOCUT timeout fallback -> cut to live")
            return "timeout"
        return None

    def check_stealth_presence(self, live_frame: np.ndarray, align_score: float | None = None) -> str | None:
        """Detects desk absence to auto-loop, or desk return to resume live."""
        if not self.auto_stealth:
            self.last_presence_score = None
            return None

        ref_loop = self._slots.get(self.active_slot) or self._slots.get(1)
        if not ref_loop:
            self.last_presence_score = None
            return "no_loop"

        ref_frame = self._codec.decode(ref_loop[0])
        score = compute_presence_score(live_frame, ref_frame)
        self.last_presence_score = score

        if self.state == LIVE:
            if score < self.stealth_threshold:  # User stepped away from chair
                self.stealth_absence_frames += 1
                if self.stealth_absence_frames >= self.stealth_absence_threshold:
                    self.stealth_absence_frames = 0
                    self.stealth_return_frames = 0
                    self.stealth_engaged_by_sensor = True
                    chosen_slot = self.active_slot if self._slots.get(self.active_slot) else 1
                    self.select_slot(chosen_slot)
                    self._log(f"STEALTH absence detected ({score*100:.0f}%) -> auto-playing loop S{chosen_slot}")
                    return "stealth_on"
            else:
                self.stealth_absence_frames = max(0, self.stealth_absence_frames - 1)

        elif self.state == LOOPING and self.stealth_engaged_by_sensor:
            if score >= self.stealth_return_threshold or (align_score is not None and align_score >= self.auto_cut_threshold):
                self.stealth_return_frames += 1
                if self.stealth_return_frames >= self.stealth_return_required:
                    self.stealth_absence_frames = 0
                    self.stealth_return_frames = 0
                    self.stealth_engaged_by_sensor = False
                    self.go_live()
                    self._log(f"STEALTH user returned ({score*100:.0f}%) -> resumed live")
                    return "stealth_off"
            else:
                self.stealth_return_frames = max(0, self.stealth_return_frames - 1)

        return None

    def select_slot(self, slot_idx: int) -> None:
        if slot_idx not in self._slots:
            return
        self.active_slot = slot_idx
        slot_frames = self._slots[slot_idx]
        if slot_frames:
            self._player = Player(slot_frames, self._codec)
            self.state = LOOPING
            self._log(f"SLOT  slot {slot_idx} active ({len(slot_frames) / self.fps:.1f}s loop)")
        else:
            self._log(f"SLOT  slot {slot_idx} selected (empty - record to fill)")

    def save_slot(self, filepath: str, slot_idx: int | None = None) -> bool:
        idx = slot_idx if slot_idx is not None else self.active_slot
        frames = self._slots.get(idx)
        if not frames:
            self._log(f"SAVE  slot {idx} is empty, nothing to save")
            return False
        ok = save_loop_to_file(frames, self._codec, filepath, self.fps)
        if ok:
            self._log(f"SAVE  saved slot {idx} to {filepath}")
        return ok

    def load_slot(self, filepath: str, slot_idx: int | None = None) -> bool:
        idx = slot_idx if slot_idx is not None else self.active_slot
        try:
            raw_frames = load_loop_from_file(filepath, self._codec)
        except Exception as exc:
            self._log(f"LOAD  failed to load {filepath}: {exc}")
            return False
        loop = build_loop(raw_frames, self._crossfade, self._codec, smart_seam=self.smart_seam)
        self._slots[idx] = loop
        self.active_slot = idx
        self._player = Player(loop, self._codec)
        self.state = LOOPING
        self._log(f"LOAD  loaded {len(loop) / self.fps:.1f}s loop from {filepath} into slot {idx}")
        return True

    def go_live(self) -> None:
        self.auto_cut_armed = False
        if self.state == LIVE:
            return
        if self.state in (LOOPING, FREEZE) and self._fade_left > 0:
            return
        self._recorder.take()
        if self.state in (LOOPING, FREEZE) and self._crossfade > 0 and self._player is not None:
            self._fade_left = self._crossfade
            self._log(f"LIVE  dissolving into live over {self._crossfade / self.fps:.1f}s")
            return
        self._player = None
        self._fade_left = 0
        self._frozen_frame = None
        self.state = LIVE
        self._log("LIVE  live feed - hold the pedal key to record")

    # -- per-frame -----------------------------------------------------------

    def process(self, live_frame: np.ndarray) -> np.ndarray:
        """Given the newest camera frame, return the frame to send to the call."""
        self._last_live_frame = live_frame
        if self.state == RECORDING:
            self._recorder.push(live_frame)
            out = live_frame
        elif self.state == FREEZE and self._frozen_frame is not None:
            if self._fade_left == 0:
                out = self._frozen_frame
            else:
                a = (self._crossfade - self._fade_left + 1) / (self._crossfade + 1)
                self._fade_left -= 1
                frozen = self._frozen_frame
                if self._fade_left == 0:
                    self._player = None
                    self._frozen_frame = None
                    self.state = LIVE
                out = cv2.addWeighted(frozen, 1.0 - a, live_frame, a, 0.0)
        elif self.state == LOOPING and self._player is not None:
            loop_frame = self._player.next()
            if self._fade_left == 0:
                out = loop_frame
            else:
                a = (self._crossfade - self._fade_left + 1) / (self._crossfade + 1)
                self._fade_left -= 1
                if self._fade_left == 0:
                    self._player = None
                    self.state = LIVE
                out = cv2.addWeighted(loop_frame, 1.0 - a, live_frame, a, 0.0)
        else:
            out = live_frame

        if self.studio.lut_name != "normal" or self.studio.blur_mode != "off":
            out = self.studio.apply(out)
        return out

    def hud_info(self) -> dict:
        """What the preview overlay needs: state, seconds elapsed/total, 0..1 progress, slot info."""
        slots_info = {
            s: {
                "filled": bool(len(self._slots.get(s, []))),
                "duration": len(self._slots.get(s, [])) / self.fps if self._slots.get(s) else 0.0,
            }
            for s in sorted(self._slots.keys())
        }
        has_loop = bool(self._slots.get(self.active_slot) or self._slots.get(1))
        base = {
            "slot": self.active_slot,
            "ping_pong": self.ping_pong,
            "slots": slots_info,
            "auto_cut": self.auto_cut,
            "auto_cut_armed": self.auto_cut_armed,
            "auto_stealth": self.auto_stealth,
            "stealth_engaged": self.stealth_engaged_by_sensor,
            "stealth_score": self.last_presence_score,
            "stealth_absence_frames": self.stealth_absence_frames,
            "stealth_absence_max": self.stealth_absence_threshold,
            "stealth_threshold": self.stealth_threshold,
            "has_loop": has_loop,
            "studio_lut": self.studio.lut_name,
            "studio_blur": self.studio.blur_mode,
        }
        if self.state == RECORDING:
            elapsed = len(self._recorder) / self.fps
            total = self._recorder.capacity / self.fps
            return {**base, "state": RECORDING, "elapsed": elapsed, "total": total, "progress": min(1.0, elapsed / total)}
        if self.state == LOOPING and self._player is not None:
            p = self._player
            return {**base, "state": LOOPING, "elapsed": p.position / self.fps, "total": len(p) / self.fps,
                    "progress": p.position / len(p)}
        if self.state == FREEZE:
            return {**base, "state": FREEZE, "elapsed": 0.0, "total": 0.0, "progress": 1.0}
        return {**base, "state": LIVE, "elapsed": None, "total": None, "progress": None}





# --------------------------------------------------------------------------- #
# Global hotkey ("the pedal")
# --------------------------------------------------------------------------- #

def parse_key(keyboard, name: str):
    name = name.strip()
    if len(name) == 1:
        return keyboard.KeyCode.from_char(name.lower())
    try:
        return keyboard.Key[name.lower()]
    except KeyError:
        names = ", ".join(k.name for k in keyboard.Key)
        raise SystemExit(f"Unknown key {name!r}. Use a single character or one of: {names}")


def key_to_str(key) -> str:
    """Normalize a pynput Key or KeyCode object to a string identifier."""
    if hasattr(key, "name") and key.name:
        return key.name.lower()
    if hasattr(key, "char") and key.char:
        return key.char.lower()
    k_str = str(key).strip("'\"").lower()
    if k_str.startswith("key."):
        return k_str[4:]
    return k_str


class Pedal:
    """Listens system-wide for two keys, ignoring key-repeat.

    The pedal key queues 'down' / 'up'. The live key queues 'live' once per solo press:
    released with nothing else pressed in between, so a shortcut like right-Cmd+Tab in
    the focused app does not end your loop.
    """

    def __init__(self, key_name: str, events: queue.Queue, live_key_name: str | None = None):
        from pynput import keyboard  # imported lazily so --no-pedal works without it
        self._keyboard = keyboard
        self._key = parse_key(keyboard, key_name)
        self._live_key = parse_key(keyboard, live_key_name) if live_key_name else None
        if self._live_key is not None and self._live_key == self._key:
            raise SystemExit(f"--live-key must differ from --key (both are {key_name!r})")
        self._events = events
        self._held = False
        self._live_held = False
        self._live_solo = False
        self._capture_callback = None
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)

    def set_capture_callback(self, callback) -> None:
        self._capture_callback = callback

    def rebind(self, key_name: str, live_key_name: str | None = None) -> bool:
        self.stop()
        self._key = parse_key(self._keyboard, key_name)
        self._live_key = parse_key(self._keyboard, live_key_name) if live_key_name else None
        if self._live_key is not None and self._live_key == self._key:
            raise ValueError(f"Live key must differ from pedal key ({key_name!r})")
        self._held = False
        self._live_held = False
        self._live_solo = False
        self._listener = self._keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        return self.start()

    def start(self) -> bool:
        """Start listening. Returns False if macOS has not granted Input Monitoring."""
        self._listener.start()
        self._listener.wait()
        return bool(getattr(self._listener, "IS_TRUSTED", True))

    def stop(self) -> None:
        self._listener.stop()

    def _on_press(self, key) -> None:
        if self._capture_callback is not None:
            self._capture_callback(key_to_str(key))
            return
        if self._live_key is not None and key == self._live_key:
            if not self._live_held:
                self._live_held = True
                self._live_solo = True
            return
        if self._live_held:
            self._live_solo = False  # another key while the live key is down: a shortcut, not a tap
        if key == self._key and not self._held:
            self._held = True
            self._events.put("down")

    def _on_release(self, key) -> None:
        if self._capture_callback is not None:
            return
        if key == self._key and self._held:
            self._held = False
            self._events.put("up")
        elif self._live_key is not None and key == self._live_key and self._live_held:
            self._live_held = False
            if self._live_solo:
                self._events.put("live")


# --------------------------------------------------------------------------- #
# Devices
# --------------------------------------------------------------------------- #

def quiet_opencv() -> None:
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except AttributeError:
        pass


@contextlib.contextmanager
def stderr_silenced():
    """OpenCV's AVFoundation backend prints probe failures straight to fd 2; hide them."""
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


def list_cameras(max_index: int = 5) -> None:
    quiet_opencv()
    if sys.platform == "darwin":
        try:
            out = subprocess.run(["system_profiler", "SPCameraDataType"], capture_output=True,
                                 text=True, timeout=20).stdout
            names = [ln.strip().rstrip(":") for ln in out.splitlines()
                     if ln.startswith("    ") and not ln.startswith("      ")]
            if names:
                print("Cameras macOS knows about:", ", ".join(names))
        except (OSError, subprocess.SubprocessError):
            pass
    print("Probing OpenCV indexes (a static image is a virtual camera's placeholder, not a webcam):")
    for i in range(max_index + 1):
        with stderr_silenced():
            cap = open_capture(i)
            frames = read_frames(cap, 6) if cap.isOpened() else []
            cap.release()
        if frames:
            h, w = frames[-1].shape[:2]
            kind = "live camera" if looks_live(frames) else "STATIC image (skip)"
            print(f"  --camera {i}: {w}x{h}  {kind}")


def read_frames(cap: cv2.VideoCapture, count: int) -> list:
    frames = []
    for _ in range(count):
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    return frames


def looks_live(frames: list) -> bool:
    """A real sensor never produces two identical frames; a placeholder image does."""
    if len(frames) < 2:
        return False
    diffs = [cv2.absdiff(a, b).mean() for a, b in zip(frames, frames[1:])]
    return max(diffs) > 0.05


def open_capture(index: int) -> cv2.VideoCapture:
    if sys.platform == "darwin":
        return cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    return cv2.VideoCapture(index)


def find_live_camera(max_index: int = 5) -> int:
    """First index whose frames change over time, i.e. not a virtual camera's placeholder."""
    for i in range(max_index + 1):
        with stderr_silenced():
            cap = open_capture(i)
            frames = read_frames(cap, 6) if cap.isOpened() else []
            cap.release()
        if frames and looks_live(frames):
            return i
    sys.exit("No live camera found. Run --list-cameras, and check that this terminal is allowed "
             "to use the camera (System Settings > Privacy & Security > Camera).")


def open_camera(index: int | None, width: int, height: int, fps: float):
    quiet_opencv()
    if index is None:
        index = find_live_camera()
        print(f"Auto-picked camera {index} (first index that is a live sensor, not a placeholder)")
    with stderr_silenced():
        cap = open_capture(index)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {index}. Try --list-cameras, and check that this "
                 f"terminal is allowed to use the camera (System Settings > Privacy & Security > Camera).")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    ok, frame = cap.read()
    if not ok:
        sys.exit(f"Camera {index} opened but returned no frames.")
    h, w = frame.shape[:2]
    return cap, w, h


def open_vcam(width: int, height: int, fps: float):
    import pyvirtualcam
    try:
        cam = pyvirtualcam.Camera(width=width, height=height, fps=fps,
                                  fmt=pyvirtualcam.PixelFormat.BGR, print_fps=False)
    except RuntimeError as exc:
        sys.exit(f"Virtual camera unavailable: {exc}\n\n{VCAM_SETUP}\n"
                 "Then run this again. Use --no-vcam to try the preview without it.")
    return cam


VCAM_SETUP = {
    "darwin": (
        "One-time setup on macOS:\n"
        "  1. Install OBS Studio (brew install --cask obs) and open it.\n"
        "  2. Click 'Start Virtual Camera' (bottom right). macOS will ask you to allow the\n"
        "     OBS camera extension in System Settings > Privacy & Security. Allow it.\n"
        "  3. Click 'Stop Virtual Camera', quit OBS. Reboot if macOS asked you to."
    ),
    "win32": (
        "One-time setup on Windows:\n"
        "  Install OBS Studio (obsproject.com); its installer registers the virtual camera\n"
        "  driver. Keep OBS closed while this runs."
    ),
}.get(sys.platform, (
    "One-time setup on Linux (v4l2loopback):\n"
    "  sudo apt install v4l2loopback-dkms\n"
    "  sudo modprobe v4l2loopback video_nr=10 card_label=\"Video Pedal\" exclusive_caps=1"
))


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #

# What the HUD calls the modifier keys; the pynput names (alt_r, cmd_r, ...) are the same everywhere.
_MODIFIERS = {
    "darwin": {"alt": "Option", "cmd": "Command", "ctrl": "Control", "shift": "Shift"},
    "win32": {"alt": "Alt", "cmd": "Win", "ctrl": "Ctrl", "shift": "Shift"},
}.get(sys.platform, {"alt": "Alt", "cmd": "Super", "ctrl": "Ctrl", "shift": "Shift"})
KEY_LABELS = {"alt_gr": "AltGr"}
for _key, _label in _MODIFIERS.items():
    KEY_LABELS.update({_key: _label, f"{_key}_l": f"left {_label}", f"{_key}_r": f"right {_label}"})


def key_label(name: str) -> str:
    name = name.strip().lower()
    if len(name) == 1:
        return name.upper()
    return KEY_LABELS.get(name, name.upper() if name.startswith("f") and name[1:].isdigit() else name)


FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, GREY = (255, 255, 255), (185, 185, 185)
STATE_STYLE = {   # colour (BGR), title
    LIVE: ((70, 170, 60), "LIVE"),
    RECORDING: ((40, 40, 220), "REC"),
    LOOPING: ((200, 130, 30), "LOOP"),
    FREEZE: ((220, 140, 30), "FREEZE"),
}


def draw_text_shadow(img: np.ndarray, text: str, org: tuple[int, int],
                     font_face: int, font_scale: float,
                     color: tuple[int, int, int], thickness: int = 1,
                     shadow_color: tuple[int, int, int] = (10, 10, 10),
                     offset: tuple[int, int] = (1, 1)) -> None:
    """Renders anti-aliased text with drop shadow for crisp legibility over any video."""
    x, y = org
    dx, dy = offset
    cv2.putText(img, text, (x + dx, y + dy), font_face, font_scale, shadow_color, thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font_face, font_scale, color, thickness, cv2.LINE_AA)


def draw_vector_icon(img: np.ndarray, icon_type: str, center: tuple[int, int],
                     size: int, color: tuple[int, int, int]) -> None:
    """Draws crisp geometric icons for modern button glyphs."""
    cx, cy = center
    r = max(2, size // 2)
    if icon_type == "rec":
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
    elif icon_type == "stop":
        cv2.rectangle(img, (cx - r + 1, cy - r + 1), (cx + r - 1, cy + r - 1), color, -1)
    elif icon_type == "play":
        pts = np.array([[cx - r // 2, cy - r], [cx - r // 2, cy + r], [cx + r, cy]], np.int32)
        cv2.fillPoly(img, [pts], color, cv2.LINE_AA)
    elif icon_type in ("pause", "freeze"):
        bw = max(2, r // 2)
        cv2.rectangle(img, (cx - r + 1, cy - r), (cx - r + 1 + bw, cy + r), color, -1)
        cv2.rectangle(img, (cx + r - 1 - bw, cy - r), (cx + r - 1, cy + r), color, -1)
    elif icon_type in ("loop", "boom"):
        cv2.ellipse(img, (cx, cy), (r, r), 0, 45, 315, color, max(1, r // 4), cv2.LINE_AA)
        cv2.circle(img, (cx + r - 1, cy), max(2, r // 3), color, -1, cv2.LINE_AA)
    elif icon_type == "gear":
        cv2.circle(img, (cx, cy), r, color, max(1, r // 3), cv2.LINE_AA)
        cv2.circle(img, (cx, cy), max(1, r // 3), color, -1, cv2.LINE_AA)
    elif icon_type == "help":
        (tw, th), _ = cv2.getTextSize("?", FONT, 0.45, 1)
        cv2.putText(img, "?", (cx - tw // 2, cy + th // 2), FONT, 0.45, color, 1, cv2.LINE_AA)
    elif icon_type == "save":
        cv2.rectangle(img, (cx - r + 1, cy - r + 1), (cx + r - 1, cy + r - 1), color, 1, cv2.LINE_AA)
        cv2.line(img, (cx, cy - r + 2), (cx, cy + r - 2), color, 1, cv2.LINE_AA)
    elif icon_type == "expand":
        cv2.line(img, (cx - r, cy - r), (cx - r // 2, cy - r), color, 1, cv2.LINE_AA)
        cv2.line(img, (cx - r, cy - r), (cx - r, cy - r // 2), color, 1, cv2.LINE_AA)
        cv2.line(img, (cx + r, cy + r), (cx + r // 2, cy + r), color, 1, cv2.LINE_AA)
        cv2.line(img, (cx + r, cy + r), (cx + r, cy + r // 2), color, 1, cv2.LINE_AA)


def draw_rounded_rect(img: np.ndarray, pt1: tuple[int, int], pt2: tuple[int, int],
                      color: tuple[int, int, int], thickness: int = -1, radius: int = 8) -> None:
    """Draws a rounded rectangle with anti-aliasing."""
    x0, y0 = pt1
    x1, y1 = pt2
    if x1 <= x0 or y1 <= y0:
        return
    w, h = x1 - x0, y1 - y0
    r = max(1, min(radius, w // 2, h // 2))

    if thickness == -1:
        cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), color, -1)
        cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), color, -1)
        cv2.circle(img, (x0 + r, y0 + r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x1 - r, y0 + r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x0 + r, y1 - r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x1 - r, y1 - r), r, color, -1, cv2.LINE_AA)
    else:
        cv2.line(img, (x0 + r, y0), (x1 - r, y0), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x0 + r, y1), (x1 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x0, y0 + r), (x0, y1 - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y0 + r), (x1, y1 - r), color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x0 + r, y0 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 - r, y0 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 - r, y1 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x0 + r, y1 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)


def draw_glass_pill(img: np.ndarray, pt1: tuple[int, int], pt2: tuple[int, int],
                    bg_color: tuple[int, int, int] = (16, 20, 26),
                    border_color: tuple[int, int, int] | None = (55, 65, 75),
                    alpha: float = 0.78, radius: int = 8,
                    glow_color: tuple[int, int, int] | None = None) -> None:
    """Draws a translucent glassmorphic rounded pill background with subtle specular border."""
    x0, y0 = pt1
    x1, y1 = pt2
    H, W = img.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return

    roi = img[y0:y1, x0:x1]
    overlay = roi.copy()
    draw_rounded_rect(overlay, (0, 0), (x1 - x0, y1 - y0), bg_color, -1, radius)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, dst=roi)

    if border_color is not None:
        draw_rounded_rect(img, (x0, y0), (x1, y1), border_color, 1, radius)

    # Subtle top-edge specular highlight
    if x1 - x0 > 2 * radius + 4 and y1 - y0 > 8:
        hl_color = (min(255, bg_color[0] + 55), min(255, bg_color[1] + 55), min(255, bg_color[2] + 65))
        cv2.line(img, (x0 + radius, y0 + 1), (x1 - radius, y0 + 1), hl_color, 1, cv2.LINE_AA)


def draw_state_border(img: np.ndarray, state: str) -> None:
    """Draws a prominent perimeter border indicating state with modern broadcast reticle accents."""
    h, w = img.shape[:2]
    thick = max(4, int(min(w, h) * 0.015))

    if state == LIVE:
        color = (50, 205, 50)  # Solid rich emerald green
    elif state == LOOPING:
        pulse = 0.88 + 0.12 * math.sin(time.time() * 3.5)
        color = (int(30 * pulse), int(140 * pulse), int(240 * pulse))  # Glowing amber-orange
    elif state == RECORDING:
        flash = 1.0 if (time.time() % 0.5) < 0.28 else 0.60
        color = (int(30 * flash), int(30 * flash), int(235 * flash))  # High-alert flashing red
    elif state == FREEZE:
        color = (220, 160, 40)  # Ice-blue
    else:
        color = (100, 100, 100)

    # Perimeter border (ensures pixel [0, 0] is colored for test compatibility)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), color, thick)

    # High-tech studio viewfinder corner brackets
    if w >= 120 and h >= 90:
        b_len = max(14, min(w // 10, h // 10, 36))
        b_thick = max(thick + 1, 5)
        # Top-left
        cv2.line(img, (0, 0), (b_len, 0), color, b_thick, cv2.LINE_AA)
        cv2.line(img, (0, 0), (0, b_len), color, b_thick, cv2.LINE_AA)
        # Top-right
        cv2.line(img, (w - 1, 0), (w - 1 - b_len, 0), color, b_thick, cv2.LINE_AA)
        cv2.line(img, (w - 1, 0), (w - 1, b_len), color, b_thick, cv2.LINE_AA)
        # Bottom-left
        cv2.line(img, (0, h - 1), (b_len, h - 1), color, b_thick, cv2.LINE_AA)
        cv2.line(img, (0, h - 1), (0, h - 1 - b_len), color, b_thick, cv2.LINE_AA)
        # Bottom-right
        cv2.line(img, (w - 1, h - 1), (w - 1 - b_len, h - 1), color, b_thick, cv2.LINE_AA)
        cv2.line(img, (w - 1, h - 1), (w - 1, h - 1 - b_len), color, b_thick, cv2.LINE_AA)


class ToastNotifier:
    """Displays temporary high-visibility on-screen announcements with modern capsule styling."""

    def __init__(self, duration: float = 1.8):
        self.duration = duration
        self.text = ""
        self.color = (50, 220, 50)
        self.start_time = 0.0

    def show(self, text: str, color: tuple[int, int, int] = (50, 220, 50)) -> None:
        self.text = text
        self.color = color
        self.start_time = time.time()

    def draw(self, img: np.ndarray) -> None:
        elapsed = time.time() - self.start_time
        if elapsed >= self.duration or not self.text:
            return
        h, w = img.shape[:2]
        progress = elapsed / self.duration
        alpha = max(0.0, min(1.0, 1.0 - progress ** 2))
        scale = max(0.6, min(1.0, w / 900))
        (tw, th), _ = cv2.getTextSize(self.text, FONT, 0.65 * scale, 2)

        pad_x = int(24 * scale)
        pad_y = int(12 * scale)
        icon_space = int(20 * scale)
        bx0 = (w - (tw + icon_space)) // 2 - pad_x
        by0 = int(68 * scale)
        bx1 = (w + (tw + icon_space)) // 2 + pad_x
        by1 = by0 + th + 2 * pad_y

        draw_glass_pill(img, (bx0, by0), (bx1, by1),
                        bg_color=(15, 19, 26),
                        border_color=self.color,
                        alpha=0.90 * alpha,
                        radius=int(12 * scale))

        # Beacon dot
        dot_r = max(3, int(5 * scale))
        dot_cx = bx0 + pad_x + dot_r
        dot_cy = (by0 + by1) // 2
        col_alpha = tuple(int(c * alpha) for c in self.color)
        cv2.circle(img, (dot_cx, dot_cy), dot_r, col_alpha, -1, cv2.LINE_AA)

        text_col = tuple(int(c * alpha) for c in (255, 255, 255))
        tx = dot_cx + dot_r + int(10 * scale)
        ty = (by0 + by1 + th) // 2
        draw_text_shadow(img, self.text, (tx, ty), FONT, 0.65 * scale, text_col, 2)


def draw_key_config_modal(img: np.ndarray, step: int, current_pedal: str, current_live: str) -> None:
    """Interactive modal dialog for remapping keys directly in the preview UI with 3D keycaps."""
    h, w = img.shape[:2]
    dim = np.zeros_like(img)
    cv2.addWeighted(img, 0.22, dim, 0.78, 0, dst=img)

    mw, mh = min(w - 24, 560), min(h - 24, 230)
    mx0, my0 = (w - mw) // 2, (h - mh) // 2
    mx1, my1 = mx0 + mw, my0 + mh

    draw_glass_pill(img, (mx0, my0), (mx1, my1),
                    bg_color=(18, 22, 30), border_color=(0, 210, 255),
                    alpha=0.94, radius=14)

    # Header with icon
    cv2.putText(img, "CONFIGURE HARDWARE / PEDAL HOTKEYS", (mx0 + 24, my0 + 36),
                FONT, 0.60, (0, 215, 255), 2, cv2.LINE_AA)

    # Step progress pill
    if step == 1:
        step_txt = "Step 1 of 2: Hold pedal key (records loop while held)"
        prompt = "Press new PEDAL key on keyboard now:"
        curr_lbl = f"Current pedal key:  [{key_label(current_pedal)}]  ('{current_pedal}')"
        curr_col = (50, 215, 75)
    else:
        step_txt = "Step 2 of 2: Go-live tap key (ends loop on solo tap)"
        prompt = "Press new GO-LIVE key on keyboard now:"
        curr_lbl = f"Current live key:   [{key_label(current_live)}]  ('{current_live}')"
        curr_col = (0, 200, 255)

    cv2.putText(img, step_txt, (mx0 + 24, my0 + 68), FONT, 0.46, (170, 185, 200), 1, cv2.LINE_AA)
    cv2.putText(img, prompt, (mx0 + 24, my0 + 100), FONT, 0.54, (255, 255, 255), 2, cv2.LINE_AA)

    # Keycap badge
    (kw, kh), _ = cv2.getTextSize(curr_lbl, FONT, 0.46, 1)
    bx0, by0 = mx0 + 24, my0 + 118
    bx1, by1 = bx0 + kw + 20, by0 + kh + 16
    draw_glass_pill(img, (bx0, by0), (bx1, by1),
                    bg_color=(25, 36, 30), border_color=curr_col,
                    alpha=0.88, radius=8)
    cv2.putText(img, curr_lbl, (bx0 + 10, by0 + kh + 8), FONT, 0.46, curr_col, 1, cv2.LINE_AA)

    cv2.putText(img, "Listening for any keystroke... (press Esc to cancel)",
                (mx0 + 24, my1 - 20), FONT, 0.42, (140, 150, 165), 1, cv2.LINE_AA)


def draw_help_modal(img: np.ndarray, current_pedal: str, current_live: str,
                    mouse_pos: tuple[int, int] | None = None) -> None:
    """Renders a sleek frosted glass cheatsheet modal detailing all controls."""
    h, w = img.shape[:2]
    dim = np.zeros_like(img)
    cv2.addWeighted(img, 0.20, dim, 0.80, 0, dst=img)

    mw, mh = min(w - 32, 700), min(h - 32, 450)
    mx0, my0 = (w - mw) // 2, (h - mh) // 2
    mx1, my1 = mx0 + mw, my0 + mh

    draw_glass_pill(img, (mx0, my0), (mx1, my1),
                    bg_color=(16, 20, 28), border_color=(0, 210, 255),
                    alpha=0.95, radius=14)

    # Title
    cv2.putText(img, "LOOP PEDAL -- QUICK CONTROLS & SHORTCUTS",
                (mx0 + 24, my0 + 36), FONT, 0.62, (0, 215, 255), 2, cv2.LINE_AA)

    # Close button [X]
    cx0, cy0 = mx1 - 42, my0 + 16
    cx1, cy1 = mx1 - 16, my0 + 42
    hover_x = mouse_pos and (cx0 <= mouse_pos[0] <= cx1 and cy0 <= mouse_pos[1] <= cy1)
    x_border = (255, 255, 255) if hover_x else (70, 80, 95)
    draw_glass_pill(img, (cx0, cy0), (cx1, cy1), bg_color=(28, 34, 46),
                    border_color=x_border, alpha=0.9, radius=6)
    cv2.putText(img, "X", (cx0 + 8, cy0 + 18), FONT, 0.45, (220, 230, 240), 1, cv2.LINE_AA)

    # Section 1: Global Foot Pedal
    y = my0 + 72
    cv2.putText(img, "GLOBAL SYSTEM HOTKEYS (active while inside Zoom/Teams/Discord):",
                (mx0 + 24, y), FONT, 0.46, (60, 230, 80), 1, cv2.LINE_AA)
    y += 24
    pedal_lbl = key_label(current_pedal)
    live_lbl = key_label(current_live)
    cv2.putText(img, f"  Pedal Key  [{pedal_lbl}]: HOLD to record live feed; RELEASE to loop automatically",
                (mx0 + 24, y), FONT, 0.43, (240, 245, 250), 1, cv2.LINE_AA)
    y += 20
    cv2.putText(img, f"  Live Key   [{live_lbl}]: SOLO TAP to dissolve seamlessly back into live webcam",
                (mx0 + 24, y), FONT, 0.43, (240, 245, 250), 1, cv2.LINE_AA)

    # Section 2: In-App Keys & Dock Buttons
    y += 34
    cv2.putText(img, "PREVIEW WINDOW SHORTCUTS & CLICKABLE DOCK BUTTONS:",
                (mx0 + 24, y), FONT, 0.46, (255, 200, 40), 1, cv2.LINE_AA)
    y += 24

    col1 = [
        ("R", "Toggle Record / Loop (pedal simulator)"),
        ("L", "Go Live (smart dissolve / auto-cut)"),
        ("F", "Freeze frame snapshot / Unfreeze"),
        ("P", "Toggle Boomerang forward-reverse"),
        ("1 - 5", "Select loop slot (switch replays)"),
        ("A", "Toggle AI Pose Auto-Cut (alignment cut)"),
        ("[ / ]", "Adjust Stealth Threshold (-5% / +5%)"),
    ]
    col2 = [
        ("T", "Toggle Presence Auto-Stealth (desk sensor)"),
        ("C", "Cycle Studio Color LUT (Warm/Cool/Noir...)"),
        ("B", "Cycle Studio Blur (Light/Studio/Strong)"),
        ("S", "Save active loop to disk (saved_loops/)"),
        ("O / Tab", "Cycle alignment view (ghost/edge/diff)"),
        ("M / K", "Toggle Mini PiP / Remap keys in UI"),
    ]

    for (k_txt, desc) in col1:
        cv2.putText(img, f"  [{k_txt}]", (mx0 + 24, y), FONT, 0.40, (0, 210, 255), 1, cv2.LINE_AA)
        cv2.putText(img, desc, (mx0 + 95, y), FONT, 0.38, (200, 210, 220), 1, cv2.LINE_AA)
        y += 19

    y_col2 = my0 + 72 + 24 + 20 + 34 + 24
    for (k_txt, desc) in col2:
        cv2.putText(img, f"  [{k_txt}]", (mx0 + 350, y_col2), FONT, 0.40, (0, 210, 255), 1, cv2.LINE_AA)
        cv2.putText(img, desc, (mx0 + 440, y_col2), FONT, 0.38, (200, 210, 220), 1, cv2.LINE_AA)
        y_col2 += 19

    # Section 3: Alignment radar guidance
    y = max(y, y_col2) + 14
    cv2.putText(img, "ALIGNMENT RADAR HUD & POSTURE MATCHING:", (mx0 + 24, y), FONT, 0.44, (180, 140, 255), 1, cv2.LINE_AA)
    y += 20
    align_hint = "Ghost: 50% opacity | Edge: neon contour outline | Diff: heat map | Split: 50/50"
    cv2.putText(img, f"  {align_hint}", (mx0 + 24, y), FONT, 0.37, (190, 195, 205), 1, cv2.LINE_AA)
    y += 17
    match_hint = "Auto-Cut holds until posture >= 80% matches loop, then crossfades invisibly!"
    cv2.putText(img, f"  {match_hint}", (mx0 + 24, y), FONT, 0.37, (50, 220, 120), 1, cv2.LINE_AA)

    # Footer
    cv2.putText(img, "Tip: All bottom dock buttons and slots are directly CLICKABLE with mouse!",
                (mx0 + 24, my1 - 28), FONT, 0.42, (50, 220, 120), 1, cv2.LINE_AA)
    cv2.putText(img, "Press Esc, ?, or click anywhere outside to close",
                (mx0 + 24, my1 - 10), FONT, 0.38, (130, 140, 155), 1, cv2.LINE_AA)


def draw_control_dock(img: np.ndarray, info: dict, key: str, live_key: str,
                      align_mode: str = "ghost",
                      mouse_pos: tuple[int, int] | None = None,
                      clickable_buttons: list | None = None) -> None:
    """Renders the sleek floating interactive control dock at the bottom of the preview."""
    h, w = img.shape[:2]
    if w < 420 or h < 220:
        return

    scale = max(0.52, min(1.0, w / 1250))
    state = info["state"]
    active_slot = info.get("slot", 1)
    slots = info.get("slots", {})
    ping_pong = info.get("ping_pong", False)
    auto_cut_armed = info.get("auto_cut_armed", False)

    dock_h = int(42 * scale)
    dock_y1 = h - int(12 * scale)
    dock_y0 = dock_y1 - dock_h

    # Primary action button
    if state == RECORDING:
        prim_id, prim_txt, prim_col = "toggle_record", "LOOP", (200, 130, 30)
    elif state in (LOOPING, FREEZE):
        prim_id = "go_live"
        prim_txt = "MATCH..." if auto_cut_armed else "LIVE"
        prim_col = (0, 215, 255) if auto_cut_armed else (70, 170, 60)
    else:
        prim_id, prim_txt, prim_col = "toggle_record", "REC", (40, 40, 220)

    freeze_txt = "LIVE" if state == FREEZE else "FREEZE"
    freeze_col = (220, 140, 30) if state == FREEZE else (160, 170, 180)

    btn_specs = [
        {"id": prim_id, "text": prim_txt, "color": prim_col, "w": int(76 * scale), "glow": state == RECORDING or auto_cut_armed},
        {"id": "toggle_freeze", "text": freeze_txt, "color": freeze_col, "w": int(68 * scale)},
    ]

    # Slot chips S1..S5
    for s in range(1, 6):
        is_active = (s == active_slot)
        s_data = slots.get(s, {})
        has_loop = s_data.get("filled", False)
        s_col = (245, 175, 30) if is_active else ((60, 210, 80) if has_loop else (110, 120, 130))
        btn_specs.append({
            "id": f"slot_{s}",
            "text": f"S{s}",
            "color": s_col,
            "w": int(30 * scale),
            "active": is_active,
            "has_dot": has_loop and not is_active,
        })

    # Boomerang toggle
    boom_col = (30, 180, 245) if ping_pong else (130, 140, 150)
    btn_specs.append({"id": "toggle_ping_pong", "text": "BOOM", "color": boom_col, "w": int(62 * scale), "active": ping_pong})

    # Feature #1: Auto-cut toggle
    auto_cut = info.get("auto_cut", False)
    cut_txt = "MATCH..." if auto_cut_armed else "AUTOCUT"
    cut_col = (0, 215, 255) if auto_cut_armed else ((60, 220, 80) if auto_cut else (120, 130, 140))
    btn_specs.append({"id": "toggle_auto_cut", "text": cut_txt, "color": cut_col, "w": int(68 * scale), "active": auto_cut or auto_cut_armed, "glow": auto_cut_armed})

    # Feature #3: Auto-stealth toggle & threshold chip
    auto_stealth = info.get("auto_stealth", False)
    has_loop = info.get("has_loop", False)
    stealth_engaged = info.get("stealth_engaged", False)
    stl_th = info.get("stealth_threshold", 0.65)
    stl_th_pct = int(stl_th * 100)
    if auto_stealth:
        if not has_loop:
            stealth_txt = "STL:NO LOOP"
            stealth_col = (245, 160, 40)
        elif stealth_engaged:
            stealth_txt = "STL:LOOP"
            stealth_col = (210, 140, 255)
        else:
            stealth_txt = "STEALTH"
            stealth_col = (60, 220, 80)
    else:
        stealth_txt = "STEALTH"
        stealth_col = (120, 130, 140)
    btn_specs.append({
        "id": "toggle_auto_stealth",
        "text": stealth_txt,
        "color": stealth_col,
        "w": int(78 * scale) if not has_loop and auto_stealth else int(68 * scale),
        "active": auto_stealth,
        "glow": stealth_engaged,
    })
    btn_specs.append({
        "id": "cycle_stealth_th",
        "text": f"TH:{stl_th_pct}%",
        "color": (210, 140, 255) if auto_stealth else (140, 150, 165),
        "w": int(52 * scale),
        "active": auto_stealth,
    })

    # Feature #6: Studio LUT and Blur
    lut_name = info.get("studio_lut", "normal")
    lut_col = (255, 185, 45) if lut_name != "normal" else (140, 150, 165)
    btn_specs.append({"id": "cycle_lut", "text": f"LUT:{lut_name[:4].upper()}", "color": lut_col, "w": int(68 * scale), "active": lut_name != "normal"})

    blur_mode = info.get("studio_blur", "off")
    blur_col = (80, 210, 245) if blur_mode != "off" else (140, 150, 165)
    btn_specs.append({"id": "cycle_blur", "text": f"BLUR:{blur_mode[:4].upper()}", "color": blur_col, "w": int(70 * scale), "active": blur_mode != "off"})

    # Align mode cycle
    align_short = align_mode[:4].upper()
    btn_specs.append({"id": "cycle_align", "text": f"ALIGN:{align_short}", "color": (0, 200, 240), "w": int(82 * scale)})

    # Utilities: Save, Keys, Help, Mini
    btn_specs.append({"id": "save_loop", "text": "SAVE", "color": (150, 200, 230), "w": int(52 * scale)})
    btn_specs.append({"id": "config_keys", "text": "KEYS", "color": (170, 180, 195), "w": int(52 * scale)})
    btn_specs.append({"id": "toggle_help", "text": "? HELP", "color": (170, 180, 195), "w": int(54 * scale)})
    btn_specs.append({"id": "toggle_mini", "text": "MINI", "color": (170, 180, 195), "w": int(50 * scale)})

    pad = max(3, int(4 * scale))
    total_w = sum(b["w"] for b in btn_specs) + (len(btn_specs) - 1) * pad
    dock_pad_x = int(8 * scale)

    # Center the dock
    dx0 = (w - (total_w + 2 * dock_pad_x)) // 2
    dx1 = dx0 + total_w + 2 * dock_pad_x

    draw_glass_pill(img, (dx0, dock_y0), (dx1, dock_y1),
                    bg_color=(14, 18, 25), border_color=(50, 60, 75),
                    alpha=0.88, radius=int(12 * scale))

    cur_x = dx0 + dock_pad_x
    by0 = dock_y0 + int(4 * scale)
    by1 = dock_y1 - int(4 * scale)

    for b in btn_specs:
        bw = b["w"]
        bx0 = cur_x
        bx1 = bx0 + bw
        cur_x += bw + pad

        is_hovered = False
        if mouse_pos is not None:
            mx, my = mouse_pos
            if bx0 <= mx <= bx1 and by0 <= my <= by1:
                is_hovered = True

        if clickable_buttons is not None:
            clickable_buttons.append({"id": b["id"], "rect": (bx0, by0, bx1, by1), "hover": is_hovered})

        btn_bg = (32, 40, 52) if is_hovered else ((26, 32, 42) if b.get("active") else (20, 24, 32))
        btn_border = (255, 255, 255) if is_hovered else (b["color"] if b.get("active") or b.get("glow") else (55, 65, 80))
        btn_alpha = 0.95 if is_hovered else 0.85

        draw_glass_pill(img, (bx0, by0), (bx1, by1),
                        bg_color=btn_bg, border_color=btn_border,
                        alpha=btn_alpha, radius=int(6 * scale))

        if b.get("has_dot"):
            cv2.circle(img, (bx1 - int(5 * scale), by0 + int(5 * scale)), int(2.5 * scale), (60, 220, 80), -1, cv2.LINE_AA)

        fs = 0.38 * scale
        txt = b["text"]
        (tw, th), _ = cv2.getTextSize(txt, FONT, fs, 1)
        tx = bx0 + (bw - tw) // 2
        ty = (by0 + by1 + th) // 2
        txt_color = (255, 255, 255) if is_hovered else (b["color"] if b.get("active") else (220, 225, 235))
        draw_text_shadow(img, txt, (tx, ty), FONT, fs, txt_color, 1)


def draw_hud(img: np.ndarray, info: dict, key: str, live_key: str,
             align_mode: str = "ghost", align_score: float | None = None,
             mini: bool = False,
             clickable_buttons: list | None = None,
             mouse_pos: tuple[int, int] | None = None) -> None:
    """Status bar across the top of the preview and interactive control dock at the bottom."""
    h, w = img.shape[:2]
    state = info["state"]
    colour, title = STATE_STYLE.get(state, ((100, 100, 100), state))

    if mini:
        # Mini mode: completely omit the top bar so webcam feed is unobstructed!
        pill_h = 24
        pill_y1 = h - 8
        pill_y0 = pill_y1 - pill_h

        dot_color = colour
        if state == RECORDING:
            dot_color = (255, 255, 255) if (time.time() % 0.5) < 0.28 else colour
            badge_txt = f"REC  {info.get('elapsed', 0.0):.1f}s"
        elif state == LOOPING:
            badge_txt = f"LOOP S{info.get('slot', 1)}  {info.get('elapsed', 0.0):.0f}s"
        elif state == FREEZE:
            badge_txt = "FREEZE"
        else:
            badge_txt = "LIVE"

        fs = 0.42
        (tw, th), _ = cv2.getTextSize(badge_txt, FONT, fs, 1)
        pill_w = tw + 28
        pill_x0 = 8
        pill_x1 = pill_x0 + pill_w

        draw_glass_pill(img, (pill_x0, pill_y0), (pill_x1, pill_y1),
                        bg_color=(16, 20, 26), border_color=(60, 70, 85),
                        alpha=0.82, radius=12)

        dot_cx = pill_x0 + 10
        dot_cy = (pill_y0 + pill_y1) // 2
        cv2.circle(img, (dot_cx, dot_cy), 4, dot_color, -1, cv2.LINE_AA)

        cv2.putText(img, badge_txt, (pill_x0 + 19, (pill_y0 + pill_y1 + th) // 2),
                    FONT, fs, WHITE, 1, cv2.LINE_AA)

        if state == LOOPING and info.get("progress") is not None:
            prog_val = max(0.0, min(1.0, info["progress"]))
            cv2.line(img, (pill_x0 + 8, pill_y1 - 2),
                     (pill_x0 + 8 + int((pill_w - 16) * prog_val), pill_y1 - 2),
                     colour, 2, cv2.LINE_AA)

        hint_txt = "m:expand"
        (hw, hh), _ = cv2.getTextSize(hint_txt, FONT, 0.35, 1)
        ex_x0 = w - hw - 18
        ex_x1 = w - 8
        draw_glass_pill(img, (ex_x0, pill_y0), (ex_x1, pill_y1),
                        bg_color=(16, 20, 26), border_color=(50, 60, 70),
                        alpha=0.70, radius=12)
        cv2.putText(img, hint_txt, (w - hw - 13, (pill_y0 + pill_y1 + hh) // 2),
                    FONT, 0.35, (175, 185, 195), 1, cv2.LINE_AA)

        if clickable_buttons is not None:
            clickable_buttons.append({"id": "toggle_mini", "rect": (ex_x0, pill_y0, ex_x1, pill_y1)})
        return

    scale = max(0.65, min(1.0, w / 1280))
    margin_x = int(14 * scale)
    margin_y = int(8 * scale)
    island_h = int(48 * scale)
    island_x0 = margin_x
    island_y0 = margin_y
    island_x1 = w - margin_x
    island_y1 = island_y0 + island_h

    # Floating glass island at top
    draw_glass_pill(img, (island_x0, island_y0), (island_x1, island_y1),
                    bg_color=(15, 19, 26), border_color=(60, 72, 88),
                    alpha=0.85, radius=int(10 * scale))

    badge_margin = int(6 * scale)
    badge_y0 = island_y0 + badge_margin
    badge_y1 = island_y1 - badge_margin - int(3 * scale)
    badge_x0 = island_x0 + int(8 * scale)
    fs_badge = 0.68 * scale

    if state == RECORDING:
        badge_txt = "REC"
    elif state == LIVE:
        badge_txt = "LIVE"
    elif state == LOOPING:
        badge_txt = f"LOOP S{info.get('slot', 1)}"
    elif state == FREEZE:
        badge_txt = "FREEZE"
    else:
        badge_txt = title

    (tw, th), _ = cv2.getTextSize(badge_txt, FONT, fs_badge, 2)
    has_dot = state in (RECORDING, LIVE)
    dot_space = int(16 * scale) if has_dot else 0
    badge_w = tw + int(20 * scale) + dot_space

    draw_rounded_rect(img, (badge_x0, badge_y0), (badge_x0 + badge_w, badge_y1),
                      colour, -1, radius=int(6 * scale))

    tx = badge_x0 + int(10 * scale)
    if has_dot:
        if state == RECORDING:
            dot_on = (time.time() % 1.0) < 0.6
            dot_col = WHITE if dot_on else colour
            halo_r = int((4 + 2 * math.sin(time.time() * 6.0)) * scale)
            cv2.circle(img, (tx + int(4 * scale), (badge_y0 + badge_y1) // 2),
                       max(1, halo_r), (80, 80, 255), 1, cv2.LINE_AA)
        else:
            dot_col = WHITE
        cv2.circle(img, (tx + int(4 * scale), (badge_y0 + badge_y1) // 2),
                   int(4 * scale), dot_col, -1, cv2.LINE_AA)
        tx += dot_space

    draw_text_shadow(img, badge_txt, (tx, (badge_y0 + badge_y1 + th) // 2),
                     FONT, fs_badge, WHITE, 2)

    x = badge_x0 + badge_w + int(14 * scale)
    fs = 0.58 * scale
    timer = ""
    if state == RECORDING:
        timer = f"{info['elapsed']:.1f}s / {info['total']:.0f}s"
    elif state == LOOPING:
        pp_badge = " [BOOM]" if info.get("ping_pong") else ""
        timer = f"{info['elapsed']:4.1f}s / {info['total']:.1f}s{pp_badge}"
    elif state == FREEZE:
        timer = "STILL SNAPSHOT"
    elif state == LIVE:
        timer = "LIVE CAMERA FEED"

    if timer:
        (tmw, tmh), _ = cv2.getTextSize(timer, FONT, fs, 1)
        draw_text_shadow(img, timer, (x, (badge_y0 + badge_y1 + tmh) // 2),
                         FONT, fs, (240, 245, 250), 1)
        x += tmw + int(14 * scale)

    # Precision Alignment Match Reticle
    if state in (LOOPING, FREEZE) and align_score is not None:
        pct = int(align_score * 100)
        if pct >= 80:
            col = (60, 220, 60)
            status_desc = "MATCH READY"
        elif pct >= 60:
            col = (50, 205, 245)
            status_desc = "APPROACHING"
        else:
            col = (100, 110, 240)
            status_desc = "ALIGNING"

        align_txt = f"{pct}% {status_desc} [{align_mode.upper()}]"
        fs_a = 0.46 * scale
        (aw, ah), _ = cv2.getTextSize(align_txt, FONT, fs_a, 1)
        al_w = aw + int(16 * scale)
        draw_glass_pill(img, (x, badge_y0), (x + al_w, badge_y1),
                        bg_color=(22, 28, 38), border_color=col, alpha=0.92,
                        radius=int(6 * scale))
        draw_text_shadow(img, align_txt, (x + int(8 * scale), (badge_y0 + badge_y1 + ah) // 2),
                         FONT, fs_a, col, 1)
        if clickable_buttons is not None:
            clickable_buttons.append({"id": "cycle_align", "rect": (x, badge_y0, x + al_w, badge_y1)})
        x += al_w + int(14 * scale)

    # Auto-cut armed indicator
    if info.get("auto_cut_armed"):
        arm_txt = "[🎯 WAITING FOR POSTURE MATCH]"
        pulse_col = (0, 215, 255) if (time.time() % 0.8) < 0.4 else (255, 200, 40)
        (atw, ath), _ = cv2.getTextSize(arm_txt, FONT, 0.46 * scale, 1)
        draw_text_shadow(img, arm_txt, (x, (badge_y0 + badge_y1 + ath) // 2), FONT, 0.46 * scale, pulse_col, 1)
        x += atw + int(14 * scale)

    # Stealth status indicator in top bar
    if info.get("auto_stealth"):
        stl_txt = ""
        stl_col = (180, 180, 180)
        if not info.get("has_loop"):
            stl_txt = "[STEALTH: RECORD A LOOP FIRST]"
            stl_col = (245, 160, 40)
        elif state == LIVE:
            abs_frames = info.get("stealth_absence_frames", 0)
            score = info.get("stealth_score")
            score_str = f"{int(score * 100)}%" if score is not None else "--%"
            stl_th = info.get("stealth_threshold", 0.65)
            stl_th_str = f"{int(stl_th * 100)}%"
            if abs_frames > 0:
                abs_max = max(1, info.get("stealth_absence_max", 30))
                sec_left = max(0.0, (abs_max - abs_frames) / 30.0)
                stl_txt = f"[STEALTH: ABSENT ({score_str} < {stl_th_str}) -> LOOP IN {sec_left:.1f}s]"
                stl_col = (0, 215, 255) if (time.time() % 0.5) < 0.25 else (255, 120, 40)
            else:
                stl_txt = f"[STEALTH: ARMED (OCCUPIED {score_str} / CUT < {stl_th_str})]"
                stl_col = (60, 220, 80)
        elif state == LOOPING and info.get("stealth_engaged"):
            stl_txt = "[STEALTH: AUTO-LOOP (SIT TO RETURN)]"
            stl_col = (210, 140, 255)

        if stl_txt:
            (stlw, stlh), _ = cv2.getTextSize(stl_txt, FONT, 0.46 * scale, 1)
            draw_text_shadow(img, stl_txt, (x, (badge_y0 + badge_y1 + stlh) // 2), FONT, 0.46 * scale, stl_col, 1)
            x += stlw + int(14 * scale)

    # Keyboard shortcut hint pills on the right side
    pedal_str = key_label(key)
    live_str = key_label(live_key)
    hints = {
        LIVE: f"Hold [{pedal_str}]: Rec  |  A: Cut  |  T: Stealth  |  ?: Help",
        RECORDING: f"Release [{pedal_str}]: Loop  |  [{live_str}]: Cancel",
        LOOPING: f"[{live_str}]: Live  |  A: Auto-Cut  |  C: LUT  |  B: Blur",
        FREEZE: f"[{live_str}] or [F]: Go Live  |  ?: Help",
    }.get(state, "")
    fs_h = 0.44 * scale
    (hw, hh), _ = cv2.getTextSize(hints, FONT, fs_h, 1)
    hx = max(x, island_x1 - int(12 * scale) - hw)
    draw_text_shadow(img, hints, (hx, (badge_y0 + badge_y1 + hh) // 2),
                     FONT, fs_h, (185, 195, 205), 1)

    # Sleek embedded progress track & luminous progress bar
    if info.get("progress") is not None:
        p_y = island_y1 - int(4 * scale)
        p_x0 = island_x0 + int(8 * scale)
        p_x1 = island_x1 - int(8 * scale)
        track_w = p_x1 - p_x0
        prog_w = int(track_w * max(0.0, min(1.0, info["progress"])))
        cv2.line(img, (p_x0, p_y), (p_x1, p_y), (40, 48, 60), max(2, int(3 * scale)), cv2.LINE_AA)
        if prog_w > 0:
            cv2.line(img, (p_x0, p_y), (p_x0 + prog_w, p_y), colour, max(2, int(3 * scale)), cv2.LINE_AA)
            cv2.circle(img, (p_x0 + prog_w, p_y), max(2, int(3 * scale)), (255, 255, 255), -1, cv2.LINE_AA)

    # Render floating control dock at bottom
    draw_control_dock(img, info, key, live_key, align_mode=align_mode,
                      mouse_pos=mouse_pos, clickable_buttons=clickable_buttons)




_ORIGINAL_WINDOW_STYLES: dict[str, int] = {}


def set_window_borderless(window_title: str, borderless: bool) -> bool:
    """Removes or restores OS window chrome (title bar, close/min/max buttons) on Windows."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, window_title)
        if not hwnd:
            return False

        GWL_STYLE = -16
        WS_CAPTION = 0x00C00000
        WS_THICKFRAME = 0x00040000
        WS_MINIMIZEBOX = 0x00020000
        WS_MAXIMIZEBOX = 0x00010000
        WS_SYSMENU = 0x00080000
        SWP_FRAMECHANGED = 0x0020
        SWP_NOMOVE = 0x0002
        SWP_NOSIZE = 0x0001
        SWP_NOZORDER = 0x0004

        if borderless:
            style = user32.GetWindowLongW(hwnd, GWL_STYLE)
            if window_title not in _ORIGINAL_WINDOW_STYLES:
                _ORIGINAL_WINDOW_STYLES[window_title] = style
            new_style = style & ~WS_CAPTION & ~WS_THICKFRAME & ~WS_MINIMIZEBOX & ~WS_MAXIMIZEBOX & ~WS_SYSMENU
            user32.SetWindowLongW(hwnd, GWL_STYLE, new_style)
            user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0,
                               SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER)
        else:
            old_style = _ORIGINAL_WINDOW_STYLES.get(window_title)
            if old_style is not None:
                user32.SetWindowLongW(hwnd, GWL_STYLE, old_style)
                user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0,
                                   SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER)
        return True
    except Exception:
        return False


def parse_size(text: str) -> tuple[int, int]:
    try:
        w, h = text.lower().split("x")
        return int(w), int(h)
    except ValueError:
        raise argparse.ArgumentTypeError("size must look like 1280x720")


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cfg = load_config()
    sys_default_live = "ctrl_r" if sys.platform == "win32" else "cmd_r"
    default_key = cfg.get("key", "alt_r")
    default_live_key = cfg.get("live_key", sys_default_live)
    default_stealth_th = float(cfg.get("stealth_threshold", 0.65))

    ap.add_argument("--camera", type=int, default=None,
                    help="OpenCV camera index of your real webcam (default: first index that is a live sensor)")
    ap.add_argument("--list-cameras", action="store_true", help="probe camera indexes and exit")
    ap.add_argument("--size", type=parse_size, default=(1280, 720), help="requested capture size (default 1280x720)")
    ap.add_argument("--fps", type=float, default=30.0, help="output frame rate (default 30)")
    ap.add_argument("--key", default=default_key,
                    help=f"pedal key, held to record (default {default_key} = {key_label(default_key)}; try f13, or a letter)")
    ap.add_argument("--live-key", default=default_live_key,
                    help=f"key that ends the loop (or cancels a recording) with one press "
                         f"(default {default_live_key} = {key_label(default_live_key)}; try f14)")
    ap.add_argument("--no-pedal", action="store_true", help="no global hotkey; control from the preview window only")
    ap.add_argument("--no-vcam", action="store_true", help="preview only; don't publish the virtual camera")
    ap.add_argument("--no-preview", action="store_true", help="don't open the preview window (Ctrl+C to quit)")
    ap.add_argument("--topmost", action="store_true", help="keep preview window always on top of other apps")
    ap.add_argument("--mini", action="store_true", help="start in compact floating window mode")
    ap.add_argument("--frameless", action=argparse.BooleanOptionalAction, default=True,
                    help="remove OS window title bar and buttons in mini mode (default: enabled)")
    ap.add_argument("--no-border", action="store_true", help="disable state color perimeter border")
    ap.add_argument("--max-seconds", type=float, default=30.0, help="longest recording kept (default 30)")
    ap.add_argument("--min-seconds", type=float, default=1.0,
                    help="holds shorter than this are ignored (default 1.0)")
    ap.add_argument("--crossfade", type=float, default=0.5,
                    help="seconds of dissolve at the loop seam and when the loop ends (default 0.5, 0 = hard cut)")
    ap.add_argument("--overlay", type=float, default=0.5,
                    help="preview only: opacity of the loop ghosted over your live camera while looping, "
                         "so you can line up before going live (default 0.5, 0 = off)")
    ap.add_argument("--quality", type=int, default=90, help="JPEG quality for frames held in RAM (default 90)")
    ap.add_argument("--smart-seam", action=argparse.BooleanOptionalAction, default=True,
                    help="find minimal-difference cut points to eliminate ghosting at the seam (default: enabled)")
    ap.add_argument("--ping-pong", action="store_true",
                    help="loop in forward-reverse boomerang mode")
    ap.add_argument("--load-loop", type=str, default=None,
                    help="path to video file to preload into slot 1")
    ap.add_argument("--save-dir", type=str, default="./saved_loops",
                    help="directory to save loop recordings (default: ./saved_loops)")
    ap.add_argument("--auto-cut", action=argparse.BooleanOptionalAction, default=True,
                    help="AI pose alignment auto-cut: wait for posture match before going live (default: enabled)")
    ap.add_argument("--auto-stealth", action="store_true",
                    help="auto-stealth presence sensor: auto-switch to loop when leaving desk")
    ap.add_argument("--stealth-threshold", type=float, default=default_stealth_th,
                    help=f"absence presence score threshold for auto-stealth (default: {default_stealth_th:g})")
    ap.add_argument("--lut", choices=STUDIO_LUTS, default="normal",
                    help="studio color grading LUT (default: normal)")
    ap.add_argument("--blur", choices=STUDIO_BLURS, default="off",
                    help="portrait bokeh depth blur (default: off)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.list_cameras:
        list_cameras()
        return 0

    cap, width, height = open_camera(args.camera, *args.size, args.fps)
    print(f"Camera {args.camera}: {width}x{height} @ {args.fps:g} fps")

    pedal = LoopPedal(JpegCodec(args.quality), args.fps, args.max_seconds, args.min_seconds,
                      args.crossfade, smart_seam=args.smart_seam, ping_pong=args.ping_pong,
                      auto_cut=args.auto_cut, auto_stealth=args.auto_stealth,
                      stealth_threshold=args.stealth_threshold,
                      log=lambda m: print(time.strftime("%H:%M:%S"), m))
    pedal.studio.set_lut(args.lut)
    pedal.studio.set_blur(args.blur)

    if args.load_loop:
        pedal.load_slot(args.load_loop, slot_idx=1)

    events: queue.Queue = queue.Queue()
    hotkey = None
    pedal_on = False
    current_key = args.key
    current_live_key = args.live_key
    if not args.no_pedal:
        hotkey = Pedal(current_key, events, current_live_key)
        pedal_on = hotkey.start()
        if pedal_on:
            print(f"Pedal key: '{current_key}'  hold = record, release = loop  (works from any app)")
            print(f"Live key:  '{current_live_key}'  press once = end the loop / cancel a recording, go live")
        else:
            print("!! macOS is not letting this terminal watch the keyboard, so the global pedal and live keys are off.\n"
                  "   System Settings > Privacy & Security > Input Monitoring: enable your terminal app, restart it.\n"
                  "   Until then use the preview window keys.")
    pedal_label, live_label = (key_label(current_key), key_label(current_live_key)) if pedal_on else ("r", "l")

    vcam = None
    if not args.no_vcam:
        vcam = open_vcam(width, height, args.fps)
        print(f"Virtual camera: '{vcam.device}'  <- pick this camera in Zoom / Meet / Teams")
    else:
        print("Virtual camera: off (--no-vcam)")

    if not args.no_preview:
        print("Preview keys:\n"
              "  r = start/stop recording   l = go live / auto-cut\n"
              "  f = freeze frame           p = toggle ping-pong\n"
              "  a = toggle auto-cut        t = toggle auto-stealth\n"
              "  c = cycle studio LUT       b = cycle portrait blur\n"
              "  1-5 = select slot          s = save active loop\n"
              "  o / Tab = cycle alignment mode (ghost/edge/diff/split)\n"
              "  m = toggle small window mode (always-on-top PiP)\n"
              "  [ / ] = adjust stealth threshold (-5% / +5%)\n"
              "  k = configure hotkeys in UI   ? / h = cheat sheet\n"
              "  q = quit")
    print("Ctrl+C to quit.")

    window = "loop pedal"
    failures = 0
    align_modes = ["ghost", "edge", "diff", "split"]
    align_idx = 0
    mini_mode = args.mini
    topmost = args.topmost or mini_mode
    show_border = not args.no_border

    toast = ToastNotifier()
    prev_state = pedal.state

    key_config_step = 0
    cfg_pedal = current_key
    cfg_live = current_live_key
    captured_keys: queue.Queue = queue.Queue()

    mouse_pos = [0, 0]
    clickable_buttons: list[dict] = []
    clicked_action: list[str | None] = [None]
    show_help = [False]

    def on_capture(name: str):
        captured_keys.put(name)

    if not args.no_preview:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        if topmost:
            try:
                cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
            except Exception:
                pass
        if mini_mode:
            if args.frameless:
                set_window_borderless(window, True)
            cv2.resizeWindow(window, 360, 202)
        else:
            cv2.resizeWindow(window, width, height)

        def on_mouse(event, x, y, flags, param):
            mouse_pos[0] = x
            mouse_pos[1] = y
            if event == cv2.EVENT_LBUTTONDOWN:
                if show_help[0]:
                    show_help[0] = False
                    return
                # Check if a dock or mini button was clicked
                for btn in clickable_buttons:
                    bx0, by0, bx1, by1 = btn["rect"]
                    if bx0 <= x <= bx1 and by0 <= y <= by1:
                        clicked_action[0] = btn["id"]
                        return
                # In mini mode on Windows, clicking background drags the window
                if mini_mode and sys.platform == "win32":
                    try:
                        import ctypes
                        user32 = ctypes.windll.user32
                        hwnd = user32.FindWindowW(None, window)
                        if hwnd:
                            user32.ReleaseCapture()
                            user32.SendMessageW(hwnd, 0x00A1, 2, 0)  # WM_NCLBUTTONDOWN, HTCAPTION
                    except Exception:
                        pass

        cv2.setMouseCallback(window, on_mouse)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                failures += 1
                if failures % 30 == 0:
                    print(f"Camera frame read failure ({failures}), attempting reconnect...")
                    with stderr_silenced():
                        cap.release()
                        time.sleep(0.5)
                        cap = open_capture(args.camera if args.camera is not None else 0)
                card = np.zeros((height, width, 3), dtype=np.uint8)
                msg = f"CAMERA RECONNECTING ({failures})..."
                cv2.putText(card, msg, (width // 6, height // 2), FONT, 0.8, (0, 165, 255), 2)
                if vcam is not None:
                    vcam.send(card)
                    vcam.sleep_until_next_frame()
                if not args.no_preview:
                    cv2.imshow(window, card)
                    if (cv2.waitKey(30) & 0xFF) == ord("q"):
                        break
                time.sleep(0.03)
                continue
            failures = 0

            while True:
                try:
                    ev = events.get_nowait()
                except queue.Empty:
                    break
                if ev == "down":
                    pedal.pedal_down()
                elif ev == "up":
                    pedal.pedal_up()
                elif ev == "live":
                    if pedal.request_go_live():
                        toast.show("AUTOCUT: MATCH POSTURE TO CUT", (0, 215, 255))

            out = pedal.process(frame)

            # State transition notification toast
            if pedal.state != prev_state:
                if pedal.state == LIVE:
                    toast.show(">>> YOU ARE LIVE <<<", (50, 220, 50))
                elif pedal.state == LOOPING:
                    toast.show(">>> LOOPING STARTED <<<", (30, 150, 255))
                elif pedal.state == RECORDING:
                    toast.show(">>> RECORDING (LIVE OUT) <<<", (40, 40, 240))
                elif pedal.state == FREEZE:
                    toast.show(">>> STILL FRAME FROZEN <<<", (220, 160, 40))
                prev_state = pedal.state

            if vcam is not None:
                vcam.send(out)
                vcam.sleep_until_next_frame()

            if not args.no_preview:
                shown = out
                align_score = None
                mode = align_modes[align_idx]
                if pedal.state in (LOOPING, FREEZE):
                    align_score = compute_alignment_score(frame, out)
                    if pedal.auto_cut_armed and align_score is not None:
                        cut_res = pedal.check_auto_cut(align_score)
                        if cut_res == "matched":
                            toast.show(">>> AUTOCUT MATCHED -> LIVE <<<", (60, 220, 60))
                        elif cut_res == "timeout":
                            toast.show(">>> AUTOCUT TIMEOUT -> LIVE <<<", (240, 160, 40))
                    if args.overlay > 0:
                        shown = blend_overlay(frame, out, args.overlay, mode=mode)

                stealth_res = pedal.check_stealth_presence(frame, align_score=align_score)
                if stealth_res == "stealth_on":
                    toast.show(">>> STEALTH: DESK ABSENCE DETECTED <<<", (210, 140, 255))
                elif stealth_res == "stealth_off":
                    toast.show(">>> STEALTH: RESUMED LIVE <<<", (60, 220, 60))

                preview = cv2.flip(shown, 1)  # mirror the self-view only; the call gets it unmirrored

                if show_border:
                    draw_state_border(preview, pedal.state)

                toast.draw(preview)

                clickable_buttons.clear()
                draw_hud(preview, pedal.hud_info(), pedal_label, live_label,
                         align_mode=mode, align_score=align_score, mini=mini_mode,
                         clickable_buttons=clickable_buttons, mouse_pos=tuple(mouse_pos))

                if key_config_step > 0:
                    draw_key_config_modal(preview, key_config_step, cfg_pedal, cfg_live)
                elif show_help[0]:
                    draw_help_modal(preview, current_key, current_live_key, mouse_pos=tuple(mouse_pos))

                cv2.imshow(window, preview)
                key = cv2.waitKey(1) & 0xFF

                # Process mouse action if clicked
                action = clicked_action[0]
                clicked_action[0] = None
                if action is not None:
                    if action == "toggle_record":
                        pedal.toggle_record()
                    elif action == "go_live":
                        if pedal.request_go_live():
                            toast.show("AUTOCUT: MATCH POSTURE TO CUT", (0, 215, 255))
                    elif action == "toggle_freeze":
                        pedal.freeze(frame)
                    elif action.startswith("slot_"):
                        s_idx = int(action.split("_")[1])
                        pedal.select_slot(s_idx)
                    elif action == "toggle_ping_pong":
                        pedal.toggle_ping_pong()
                        toast.show(f"BOOMERANG: {'ON' if pedal.ping_pong else 'OFF'}", (30, 180, 245))
                    elif action == "toggle_auto_cut":
                        ac = pedal.toggle_auto_cut()
                        toast.show(f"AUTOCUT: {'ON' if ac else 'OFF'}", (0, 215, 255) if ac else (180, 180, 180))
                    elif action == "toggle_auto_stealth":
                        ast = pedal.toggle_auto_stealth()
                        has_loop = bool(pedal._slots.get(pedal.active_slot) or pedal._slots.get(1))
                        if ast:
                            if has_loop:
                                toast.show("STEALTH ON: WATCHING DESK", (210, 140, 255))
                            else:
                                toast.show("STEALTH ON: RECORD A LOOP FIRST!", (245, 160, 40))
                        else:
                            toast.show("STEALTH: OFF", (180, 180, 180))
                    elif action == "cycle_stealth_th":
                        new_th = pedal.cycle_stealth_threshold()
                        save_config({"stealth_threshold": new_th})
                        toast.show(f"STEALTH THRESHOLD: {int(new_th * 100)}% (SAVED)", (210, 140, 255))
                    elif action == "cycle_lut":
                        lut = pedal.studio.cycle_lut()
                        toast.show(f"LUT: {lut.upper()}", (255, 185, 45))
                    elif action == "cycle_blur":
                        b_mode = pedal.studio.cycle_blur()
                        toast.show(f"PORTRAIT BLUR: {b_mode.upper()}", (80, 210, 245))
                    elif action == "cycle_align":
                        align_idx = (align_idx + 1) % len(align_modes)
                        toast.show(f"ALIGN: {align_modes[align_idx].upper()}", (0, 200, 255))
                    elif action == "save_loop":
                        os.makedirs(args.save_dir, exist_ok=True)
                        fname = f"loop_slot_{pedal.active_slot}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                        if pedal.save_slot(os.path.join(args.save_dir, fname)):
                            toast.show(f"Saved Slot {pedal.active_slot} to MP4", (50, 220, 50))
                        else:
                            toast.show(f"Slot {pedal.active_slot} is empty!", (40, 40, 230))
                    elif action == "config_keys":
                        key_config_step = 1
                        cfg_pedal = current_key
                        cfg_live = current_live_key
                        while not captured_keys.empty():
                            captured_keys.get_nowait()
                        if hotkey is not None:
                            hotkey.set_capture_callback(on_capture)
                    elif action == "toggle_help":
                        show_help[0] = not show_help[0]
                    elif action == "toggle_mini":
                        mini_mode = not mini_mode
                        if mini_mode:
                            try:
                                cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
                            except Exception:
                                pass
                            if args.frameless:
                                set_window_borderless(window, True)
                            cv2.resizeWindow(window, 360, 202)
                            toast.show("MINI WINDOW (FRAMELESS PIP)", (0, 200, 255))
                        else:
                            if args.frameless:
                                set_window_borderless(window, False)
                            if not args.topmost:
                                try:
                                    cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 0)
                                except Exception:
                                    pass
                            cv2.resizeWindow(window, width, height)
                            toast.show("FULL PREVIEW", (0, 200, 255))

                # Key configuration modal handling
                if key_config_step > 0:
                    cap_k = None
                    try:
                        cap_k = captured_keys.get_nowait()
                    except queue.Empty:
                        pass
                    if key != 255:
                        if key == 27:
                            cap_k = "esc"
                        elif ord("a") <= key <= ord("z") or ord("0") <= key <= ord("9"):
                            cap_k = chr(key)

                    if cap_k is not None:
                        if cap_k == "esc":
                            key_config_step = 0
                            if hotkey is not None:
                                hotkey.set_capture_callback(None)
                            toast.show("Key config cancelled", (180, 180, 180))
                        elif key_config_step == 1:
                            cfg_pedal = cap_k
                            key_config_step = 2
                            toast.show(f"Pedal: {key_label(cfg_pedal)}! Now press Live key", (0, 200, 255))
                        elif key_config_step == 2:
                            if cap_k == cfg_pedal:
                                toast.show("Pedal and Live keys must differ!", (40, 40, 230))
                            else:
                                cfg_live = cap_k
                                current_key = cfg_pedal
                                current_live_key = cfg_live
                                if hotkey is not None:
                                    try:
                                        hotkey.rebind(current_key, current_live_key)
                                    except Exception as exc:
                                        print(f"Rebind error: {exc}")
                                pedal_label = key_label(current_key)
                                live_label = key_label(current_live_key)
                                save_config({"key": current_key, "live_key": current_live_key})
                                toast.show(f"Saved! Pedal: {pedal_label}, Live: {live_label}", (50, 220, 50))
                            key_config_step = 0
                            if hotkey is not None:
                                hotkey.set_capture_callback(None)
                    continue

                if key == ord("q"):
                    break
                if key in (ord("h"), ord("?")):
                    show_help[0] = not show_help[0]
                elif key == 27:  # Esc
                    if show_help[0]:
                        show_help[0] = False
                elif key == ord("r"):
                    pedal.toggle_record()
                elif key == ord("l"):
                    if pedal.request_go_live():
                        toast.show("AUTOCUT: MATCH POSTURE TO CUT", (0, 215, 255))
                elif key == ord("a"):
                    ac = pedal.toggle_auto_cut()
                    toast.show(f"AUTOCUT: {'ON' if ac else 'OFF'}", (0, 215, 255) if ac else (180, 180, 180))
                elif key == ord("t"):
                    ast = pedal.toggle_auto_stealth()
                    has_loop = bool(pedal._slots.get(pedal.active_slot) or pedal._slots.get(1))
                    if ast:
                        if has_loop:
                            toast.show("STEALTH ON: WATCHING DESK", (210, 140, 255))
                        else:
                            toast.show("STEALTH ON: RECORD A LOOP FIRST!", (245, 160, 40))
                    else:
                        toast.show("STEALTH: OFF", (180, 180, 180))
                elif key in (ord("["), ord("-")):
                    new_th = pedal.set_stealth_threshold(pedal.stealth_threshold - 0.05)
                    save_config({"stealth_threshold": new_th})
                    toast.show(f"STEALTH THRESHOLD: {int(new_th * 100)}% (SAVED)", (210, 140, 255))
                elif key in (ord("]"), ord("="), ord("+")):
                    new_th = pedal.set_stealth_threshold(pedal.stealth_threshold + 0.05)
                    save_config({"stealth_threshold": new_th})
                    toast.show(f"STEALTH THRESHOLD: {int(new_th * 100)}% (SAVED)", (210, 140, 255))
                elif key == ord("c"):
                    lut = pedal.studio.cycle_lut()
                    toast.show(f"LUT: {lut.upper()}", (255, 185, 45))
                elif key == ord("b"):
                    b_mode = pedal.studio.cycle_blur()
                    toast.show(f"PORTRAIT BLUR: {b_mode.upper()}", (80, 210, 245))
                elif key == ord("f"):
                    pedal.freeze(frame)
                elif key == ord("p"):
                    pedal.toggle_ping_pong()
                elif key == ord("s"):
                    os.makedirs(args.save_dir, exist_ok=True)
                    fname = f"loop_slot_{pedal.active_slot}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                    pedal.save_slot(os.path.join(args.save_dir, fname))
                elif key == ord("o") or key == 9:  # 'o' or Tab
                    align_idx = (align_idx + 1) % len(align_modes)
                elif ord("1") <= key <= ord("5"):
                    pedal.select_slot(int(chr(key)))
                elif key == ord("m"):
                    mini_mode = not mini_mode
                    if mini_mode:
                        try:
                            cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
                        except Exception:
                            pass
                        if args.frameless:
                            set_window_borderless(window, True)
                        cv2.resizeWindow(window, 360, 202)
                        toast.show("MINI WINDOW (FRAMELESS PIP)", (0, 200, 255))
                    else:
                        if args.frameless:
                            set_window_borderless(window, False)
                        if not args.topmost:
                            try:
                                cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 0)
                            except Exception:
                                pass
                        cv2.resizeWindow(window, width, height)
                        toast.show("FULL PREVIEW", (0, 200, 255))
                elif key == ord("k"):
                    key_config_step = 1
                    cfg_pedal = current_key
                    cfg_live = current_live_key
                    while not captured_keys.empty():
                        captured_keys.get_nowait()
                    if hotkey is not None:
                        hotkey.set_capture_callback(on_capture)
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break

    except KeyboardInterrupt:
        pass
    finally:
        if hotkey is not None:
            hotkey.stop()
        if vcam is not None:
            vcam.close()
        cap.release()
        cv2.destroyAllWindows()
    return 0



if __name__ == "__main__":
    sys.exit(main())
