# video pedal

<img width="640" height="480" alt="video-pedal" src="https://github.com/user-attachments/assets/7d989573-c523-494c-8923-651b8b29c1d9" />

A loop pedal for your webcam, in the same sense as a guitar looper: hold a key to
record what the camera sees, let go and the recording plays on repeat as your camera
output, press a second key to dissolve back to live. The output is a virtual camera,
so anything that reads a webcam (Zoom, Meet, Teams, Discord, QuickTime, OBS itself)
sees an ordinary camera device.

About 650 lines of Python on top of OpenCV, pyvirtualcam and pynput, with the
interesting parts (loop builder, ring buffer, state machine, hotkey logic) tested
without any hardware.

## What it does

- **Hold right Option (⌥)** to record. The live feed keeps going out while you hold;
  the switch to the loop happens on release, not on press.
- **Release** and the recording plays on a loop. The seam is crossfaded, and playback
  starts just before the seam, so both the loop's own wrap-around and the cut from
  live to loop are half-second dissolves rather than jump cuts.
- **Press right Command (⌘)** once and the loop dissolves back into the live feed.
- While a loop plays, the preview window ghosts it over the live camera at half
  opacity, so you can line yourself up with the loop before ending it.

Both keys are global hotkeys and work while any other app has focus. On Windows and
Linux the same defaults are right Alt and the right Windows / Super key (see
[Keys on Windows and Linux](#keys-on-windows-and-linux)). Change them with `--key`
and `--live-key`.

## How it works

Per frame: camera → `LoopPedal.process()` → virtual camera + preview window.

- **Ring buffer of JPEGs.** While recording, each frame is JPEG-encoded and pushed
  onto a `collections.deque` capped at `--max-seconds × fps`. The newest N seconds are
  always kept, so a long hold just slides the window. About 2 MB/s at 720p;
  `--quality` trades RAM for artifacts.
- **Seamless loop.** `build_loop()` dissolves the last *k* frames of the recording
  into the first *k* with `cv2.addWeighted` and drops the overlap. The result plays
  end-to-start with a short fade where the cut would be.
- **Fade in, fade out.** On release, the player starts *k* frames before the seam, so
  the first thing that goes out is the recording's tail dissolving into its head. The
  tail *was* the live feed a moment ago, so visually that's a dissolve from live into
  the loop. On go-live the same ramp runs in the other direction: each loop frame is
  blended with the current live frame until it's all live.
- **State machine.** `LIVE → REC → LOOP → LIVE`, plus the edges: holds shorter than
  `--min-seconds` are ignored, holding the pedal while looping re-records, the live
  key mid-recording cancels it, and the live key during a dissolve is a no-op.
- **Global hotkeys.** `pynput` listens system-wide for press/release on the pedal key
  and ignores key-repeat. The live key only counts as a solo press: if any other key
  goes down while it's held (right-⌘+Tab, say), it's treated as a shortcut and
  ignored, so normal use of the focused app doesn't end the loop.
- **Ghost preview.** `blend_overlay()` alpha-blends the loop frame over the live
  frame for the preview only; the virtual camera gets the plain loop.
- **Camera discovery.** Once the OBS extension is installed, the virtual camera shows
  up as a capture index too, and reading it would loop its placeholder image back into
  itself. So discovery reads a few frames from each index and picks the first one
  whose frames actually change over time.
- **Virtual camera.** `pyvirtualcam` publishes BGR frames to OBS's virtual camera
  device (a v4l2loopback device on Linux), which the OS exposes as a normal camera
  to every app.
- **Tests.** 31 tests cover the loop builder, ring buffer, player, state machine
  (including the dissolve to live), the overlay blend and the two-key hotkey mapping.
  They swap in a raw codec for JPEG and never touch a camera.

## Standalone executables (Linux and Windows)

Each build produces **one executable per OS**, with Python and the Python/native
dependencies included. End users do not need Python. Builds retain the console
for logs and Ctrl+C. Virtual-camera **drivers are not included or installed**:
install OBS Studio on Windows or install/load v4l2loopback on Linux as described
under [Virtual camera device](#2-virtual-camera-device-once).

### Download and run

Open this repository's [Actions page](https://github.com/uwbfritz/video-pedal/actions/workflows/build.yml),
select a successful **Build executables** run for the revision you trust, and
download the appropriate artifact from its **Artifacts** section (GitHub sign-in
may be required):

| Artifact download | Contents after unzipping the Actions download |
|---|---|
| `video-pedal-linux-x86_64` | `video-pedal-linux-x86_64.tar.gz`; extract it to get `video-pedal` |
| `video-pedal-windows-x86_64` | `video-pedal.exe` |

Linux, after unzipping the artifact:

```bash
tar -xzf video-pedal-linux-x86_64.tar.gz   # preserves executable permission
./video-pedal --help
./video-pedal --no-vcam                  # preview without a virtual-camera driver
./video-pedal                            # preview + virtual camera
```

Windows PowerShell, after **Extract All** on the artifact ZIP:

```powershell
.\video-pedal.exe --help
.\video-pedal.exe --no-vcam
.\video-pedal.exe
```

`--no-vcam` still needs a real webcam and GUI session. Add `--no-pedal` to use
preview-window controls instead of global hotkeys.

These are **x86_64** builds, not ARM or universal binaries. Linux CI builds on
**Ubuntu 22.04 (glibc 2.35)**; target a compatible glibc-based desktop with glibc
2.35 or newer, not all Linux distributions (in particular, not musl/Alpine).
Newer glibc alone does not guarantee all system libraries are present. The GUI
needs OpenGL/EGL, GLib, X11/XCB and related libraries; on Debian/Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libegl1 libglib2.0-0 libx11-6 libxext6 \
  libxrender1 libsm6 libice6 libxcb1 libxcb-xinerama0 libxcb-cursor0 \
  libxkbcommon-x11-0 libdbus-1-3
```

Global hotkeys use **X11/Xorg**. Wayland restricts observing keys in other windows;
use an Xorg session, or `--no-pedal` (the GUI may still need XWayland).
Windows CI uses **Windows Server 2022**; this is not a guarantee of compatibility
with every Windows version. Camera permissions and driver requirements from the
setup section still apply. One-file builds unpack libraries into a temporary
directory at startup; that directory must permit execution.

### Settings and saved loops

Packaged settings (hotkeys and thresholds) persist across launches and executable
updates in:

- Linux: `$XDG_CONFIG_HOME/video-pedal/config.json` when `XDG_CONFIG_HOME` is an
  absolute path, otherwise `~/.config/video-pedal/config.json`.
- Windows: `%APPDATA%\video-pedal\config.json`, falling back to
  `~\AppData\Roaming\video-pedal\config.json` if `APPDATA` is unset or not absolute.

The directory is created on first save, not in the temporary extraction directory
or beside the executable. Source execution continues using `config.json` beside
`loop_pedal.py`. Existing settings are not automatically moved; to reuse them,
copy that file to the packaged settings location.

Saved videos are **unchanged**: `./saved_loops` relative to the working directory,
not the settings or extraction directory. Launch from a writable directory or set
`--save-dir` explicitly, e.g. `./video-pedal --save-dir "$HOME/Videos/saved_loops"`
or `.\video-pedal.exe --save-dir "$env:USERPROFILE\Videos\saved_loops"`.
Existing video files are never migrated.

### Build locally

Use **64-bit CPython 3.12** and build natively on each target OS; PyInstaller is
not a Linux-to-Windows cross-compiler. To match the Linux CI baseline, build on
Ubuntu 22.04; building on a newer distribution can raise the required glibc version.
Keep the GUI-enabled `opencv-python` dependency, not `opencv-python-headless`.

Linux (install Python 3.12 first, plus the GUI libraries above):

```bash
sudo apt-get install -y build-essential xvfb xauth
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt -r requirements-build.txt
xvfb-run -a python -m pytest -q
python -m PyInstaller --clean --noconfirm video-pedal.spec
./dist/video-pedal --help
VIDEO_PEDAL_EXECUTABLE="$PWD/dist/video-pedal" xvfb-run -a python -m pytest tests/test_frozen.py -q
```

Windows PowerShell (install 64-bit Python 3.12 first):

```powershell
py -3.12 -m venv .venv
# No activation or execution-policy change is needed.
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt -r requirements-build.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m PyInstaller --clean --noconfirm video-pedal.spec
.\dist\video-pedal.exe --help
$env:VIDEO_PEDAL_EXECUTABLE = "$PWD\dist\video-pedal.exe"
.\.venv\Scripts\python.exe -m pytest tests/test_frozen.py -q
```

The common build command is `python -m PyInstaller --clean --noconfirm video-pedal.spec`.
Outputs are `dist/video-pedal` or `dist/video-pedal.exe`. Build tools are isolated
in `requirements-build.txt`; runtime requirements remain in `requirements.txt`.
The standard OpenCV/NumPy hooks collect native libraries and GUI plugins. Our
explicit pynput hook collects platform keyboard, mouse and utility backends
without importing display-sensitive modules during the build.

CI runs the source tests and frozen `--help`, then the packaging-only
`--check-packaged-dependencies` diagnostic: native pyvirtualcam backend imports,
pynput backend imports (no listeners), NumPy/OpenCV JPEG round-trip and a tiny GUI
window (under Xvfb on Linux). Neither smoke check opens a camera or validates
physical camera capture, installed virtual-camera drivers, or real global hotkey
delivery. Those need manual testing on the target machine.

## Setup (from source)

macOS, Windows and Linux. The one-time part is getting a virtual camera device onto
the system; after that it's `python loop_pedal.py`. I've only run this on a Mac; the
Windows and Linux paths are what the libraries document, not something I've
exercised, so reports welcome.

### 1. Python

```
git clone https://github.com/haxybaxy/video-pedal
cd video-pedal
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.9 or newer. Everything below assumes the venv is active.

### 2. Virtual camera device (once)

`pyvirtualcam` publishes frames to whatever virtual camera the OS has; it doesn't
create one itself.

| OS | do this |
|---|---|
| macOS | install [OBS Studio](https://obsproject.com) 30 or newer (`brew install --cask obs`), then follow the five steps below: macOS only creates the device after OBS's camera extension has been allowed once |
| Windows | install [OBS Studio](https://obsproject.com). The installer registers the virtual camera driver; you never need to open OBS |
| Linux | `sudo apt install v4l2loopback-dkms` (Debian/Ubuntu; same package name on Arch; `v4l2loopback` from RPM Fusion on Fedora), then `sudo modprobe v4l2loopback video_nr=10 card_label="Video Pedal" exclusive_caps=1` |

**macOS, the five steps:**

1. Open OBS. If it shows a setup wizard, cancel it.
2. Click **Start Virtual Camera** (bottom right, under Controls).
3. macOS pops up a message about a system extension. Open
   **System Settings > Privacy & Security**, scroll down, and **Allow** the OBS
   camera extension. Reboot if it asks you to.
4. Back in OBS click **Stop Virtual Camera**, then quit OBS.
5. Done. You never need to open OBS again.

**macOS and Windows: keep OBS closed while using the script.** If OBS is open with
its own virtual camera started, OBS owns the device and pushes its (empty, black)
scene out instead of your feed.

**Linux notes.** `video_nr=10` puts the device at `/dev/video10`, safely above the
indexes camera discovery probes, so it can't be mistaken for the webcam.
`exclusive_caps=1` is what makes Chrome, Zoom and friends list it as a camera at
all. `card_label` is the name they show. The module is gone after a reboot; to make
it stick:

```
echo v4l2loopback | sudo tee /etc/modules-load.d/v4l2loopback.conf
echo 'options v4l2loopback video_nr=10 card_label="Video Pedal" exclusive_caps=1' | sudo tee /etc/modprobe.d/v4l2loopback.conf
```

### 3. Permissions (once)

| OS | camera | global hotkeys |
|---|---|---|
| macOS | **System Settings > Privacy & Security > Camera**: macOS asks the first time you run it | **System Settings > Privacy & Security > Input Monitoring**: enable your terminal app, then restart it. Without this the script prints a warning at startup and the two keys do nothing; the preview-window keys still work |
| Windows | **Settings > Privacy & security > Camera**: "Let desktop apps access your camera" on | nothing to grant. One catch: if the app that has focus is running as administrator, a non-elevated script can't see its keys |
| Linux | your user needs to be in the `video` group (`groups` to check; usually already the case) | needs X11. Under a Wayland session `pynput` can't see keys pressed in other windows; log in to an X11/Xorg session, or run with `--no-pedal` and use the preview-window keys |

### 4. Run

```
python loop_pedal.py
```

You should see:

```
Camera 0: 1280x720 @ 30 fps
Pedal key: 'alt_r'  hold = record, release = loop  (works from any app)
Live key:  'cmd_r'  press once = end the loop / cancel a recording, go live
Virtual camera: 'OBS Virtual Camera'  <- pick this camera in Zoom / Meet / Teams
Preview keys: r = start/stop recording, l = go live, q = quit
```

and a preview window showing the output, with a status line at the top (LIVE, REC
with a red dot, or LOOP). While a loop plays, the preview ghosts the loop over your
live camera at half opacity; the virtual camera still gets the plain loop.

Start the script **before** opening the app you want to feed; most apps enumerate
cameras once at launch. Then pick the virtual camera in its video settings:
**OBS Virtual Camera** on macOS and Windows, **Video Pedal** on Linux (the script
prints it as `/dev/video10`).

### Keys on Windows and Linux

The key names are `pynput`'s and are the same everywhere; only the physical keys
differ. `alt_r` is right Alt and `cmd_r` is the right Windows key (right Super on
Linux). The HUD and `--help` show the local names.

Two things to watch for:

- **Tapping the Windows / Super key on its own opens the Start menu or the GNOME
  overview**, which is exactly what the live key does. Use a different one, e.g.
  `--live-key ctrl_r`.
- On keyboard layouts with **AltGr**, right Alt *is* AltGr and `pynput` reports it as
  `alt_gr`, so `--key alt_r` never fires. Use `--key alt_gr`, or move the pedal to
  `--key ctrl_r` and the live key to something else.

## Controls

| key | does |
|---|---|
| hold right Option | record. Live feed keeps going out while you hold. |
| release right Option | play the recording on a loop, with a short dissolve at the seam. |
| hold right Option again | replace the loop with a new recording. |
| tap right Option (under 1 s) | ignored; nothing changes. |
| press right Command (or right Ctrl on Windows) | go live: the loop dissolves into the live feed. Also cancels a recording in progress. |

Both keys work from any app. Right Command / right Ctrl only counts when pressed on its own; as
part of a shortcut it is ignored.

### Preview Window & Mouse Dock Controls

The interface features a studio-grade floating glassmorphic dock. All buttons can be clicked with your mouse, or triggered via keyboard shortcuts:

| Button / Key | Does |
|---|---|
| `[● REC]` / `r` | Start / stop recording (pedal simulator) |
| `[▲ LIVE]` / `l` | Go live with seamless crossfade dissolve / trigger smart auto-cut |
| `[❚❚ FREEZE]` / `f` | Freeze frame (instant still snapshot with smooth dissolve) |
| `[⇄ BOOM]` / `p` | Toggle ping-pong (forward-reverse boomerang) loop mode |
| `[AUTOCUT]` / `a` | Toggle AI pose auto-alignment cut (waits for posture match before going live) |
| `[STEALTH]` / `t` | Toggle auto-stealth presence sensor (loops when leaving desk; returns when seated) |
| `[TH:65%]` / `[` or `]` | Adjust absence presence threshold down/up by 5% (also `-` / `+`, saves to config) |
| `[LUT]` / `c` | Cycle color grading LUT (`normal` &rarr; `warm` &rarr; `cool` &rarr; `vibrant` &rarr; `noir` &rarr; `soft`) |
| `[BLUR]` / `b` | Cycle portrait bokeh background blur (`off` &rarr; `light` &rarr; `studio` &rarr; `strong`) |
| `[S1]..[S5]` / `1`–`5` | Switch between 5 loop preset slots (displays filled status dot & duration) |
| `[💾 SAVE]` / `s` | Save active loop to disk as an MP4 in `saved_loops/` |
| `[◉ ALIGN]` / `o` or `Tab` | Cycle alignment overlay mode: Ghost &rarr; Edge silhouette &rarr; Diff heat-map &rarr; Split screen |
| `[⚙ KEYS]` / `k` | Configure global pedal & live keys interactively with 3D keycap UI |
| `[? HELP]` / `h` or `?` | Toggle quick interactive shortcuts cheat sheet modal overlay |
| `[⤢ MINI]` / `m` | Toggle compact floating PiP mode (frameless; click & drag anywhere to move) |
| `q` or `Esc` | Quit app (or close active modal dialog) |

Ctrl+C in the terminal also quits.

### Advanced Studio Features

1. **AI Pose Auto-Alignment & Auto-Cut (#1)**:
   - When enabled, pressing the live key (`l` or `[▲ LIVE]`) arms the cut reticle (`[🎯 WAITING FOR POSTURE MATCH]`).
   - The app waits until your current posture matches the recorded loop (&ge; 80% match) before initiating the crossfade dissolve, eliminating telltale jump cuts.
   - Double-tapping `l` or clicking `[▲ LIVE]` forces an immediate override dissolve.
   - Built-in 6.0-second timeout fallback ensures you never get locked out.

2. **Desk Presence Sensor / Auto-Stealth (#3)**:
   - Real-time computer vision senses when you step away from your chair (posture correlation drops under empty-desk threshold).
   - Automatically switches your camera output to your saved loop so you never appear absent.
   - When you sit back down and align with your loop, it seamlessly dissolves you back to live.

3. **Studio Post-Processing: Background Blur & Color LUTs (#6)**:
   - **Portrait Bokeh Vignette Blur**: Softens distracting background clutter around your head/torso while keeping you crisp (`light`, `studio`, `strong`).
   - **Color Grading LUTs**: Professional real-time color palettes including `warm` (flattering skin tone warmth), `cool` (crisp modern cinematic), `vibrant` (rich saturation), `noir` (dramatic high-contrast black & white), and `soft` (filmic bloom).
   - Applied to both live feed and virtual camera output.

### Visual State Feedback & Alignment Radar

- **Studio Viewfinder Framing**:
  - **LIVE**: Solid rich emerald green border with broadcast viewfinder corner reticles.
  - **LOOPING**: Warm amber-orange border with gentle breathing pulse.
  - **RECORDING**: High-alert flashing red border with pulsing halo beacon.
  - **FREEZE**: Frost cyan border with still frame indicator.
- **Precision Alignment Radar**:
  - Shows real-time alignment match percentage and readiness tier (`MATCH READY` in green &ge; 80%, `APPROACHING` in yellow 60–79%, `ALIGNING` in red < 60%).
- **Interactive Floating Dock**:
  - Hover highlights, tactile active states, and slot status indicators.
- **On-Screen State Transition Capsules**: Instant high-visibility frosted glass toast notification whenever state changes.


## Handy variants

```
python loop_pedal.py --no-vcam            # no virtual camera needed: preview only
python loop_pedal.py --mini               # start in compact floating window mode
python loop_pedal.py --auto-stealth       # enable auto-stealth presence sensor
python loop_pedal.py --lut warm           # warm skin-tone studio color grading
python loop_pedal.py --blur studio        # portrait bokeh background blur
python loop_pedal.py --topmost            # keep window always pinned on top of Zoom/other apps
python loop_pedal.py --list-cameras       # which index is the real webcam?
python loop_pedal.py --camera 1           # use that index
python loop_pedal.py --ping-pong          # smooth boomerang looping
python loop_pedal.py --load-loop clip.mp4 # preload pre-recorded loop
python loop_pedal.py --no-smart-seam      # disable automatic pose-matching seam search
python loop_pedal.py --no-auto-cut        # disable auto-alignment cut gate
python loop_pedal.py --key f13            # different pedal key
python loop_pedal.py --live-key f14       # different go-live key
python loop_pedal.py --overlay 0          # no ghost: preview shows exactly what goes out
python loop_pedal.py --crossfade 0        # hard cuts at the seam and when going live
```

## All options

| flag | default | |
|---|---|---|
| `--key` | `alt_r` | pedal key. Modifier names like `alt_r`, `ctrl_r`, `shift_r`, function keys like `f13`, or a single letter. |
| `--live-key` | `ctrl_r` (Win) / `cmd_r` (macOS) | go-live key, pressed once on its own to end the loop or cancel a recording. |
| `--auto-cut` | `True` | wait for posture match (&ge; 80%) before cutting to live (`--no-auto-cut` to disable) |
| `--auto-stealth` | `False` | auto-stealth presence sensor: auto-switch to loop when leaving desk, resume when returning |
| `--stealth-threshold` | `0.65` | presence sensitivity score threshold below which absence is detected |
| `--lut` | `normal` | studio color grading profile (`normal`, `warm`, `cool`, `vibrant`, `noir`, `soft`) |
| `--blur` | `off` | portrait bokeh depth blur (`off`, `light`, `studio`, `strong`) |
| `--camera` | auto | OpenCV index of the real webcam; default is the first index that is a live sensor |
| `--size` | `1280x720` | requested capture size |
| `--fps` | `30` | output frame rate |
| `--max-seconds` | `30` | longest recording kept; older frames drop off. About 2 MB of RAM per second at 720p. |
| `--min-seconds` | `1.0` | holds shorter than this are ignored (too short to loop) |
| `--crossfade` | `0.5` | seconds of dissolve at the loop seam and when the loop dissolves into live, `0` for hard cuts |
| `--overlay` | `0.5` | preview only: opacity of the loop ghosted over the live camera while looping, `0` for no ghost |
| `--quality` | `90` | JPEG quality of frames held in RAM |
| `--smart-seam` | `True` | automatically searches candidate head/tail frames for minimal visual difference to prevent ghosting |
| `--ping-pong` | `False` | loop in forward-reverse boomerang mode |
| `--load-loop` | `None` | preload an MP4 video clip directly into slot 1 |
| `--save-dir` | `./saved_loops` | directory where loop recordings are saved |
| `--topmost` | `False` | keep preview window always on top of other apps |
| `--mini` | `False` | start in small floating window mode |
| `--frameless` | `True` | remove OS window title bar and buttons in mini mode (click & drag anywhere to move) |
| `--no-border` | `False` | disable state color perimeter border |
| `--no-pedal` | | skip the global hotkey |
| `--no-vcam` | | preview only, no virtual camera |
| `--no-preview` | | no window; Ctrl+C to quit |

## Tests

```
python -m pytest
```
