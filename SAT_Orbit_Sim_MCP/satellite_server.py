from __future__ import annotations

import inspect
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TextIO

from fastmcp import FastMCP

# Initialize the MCP Server
mcp = FastMCP("Satellite Sim")

_LOG_LOCK = threading.Lock()
_LOG_HANDLE: TextIO | None = None
_LOG_PATH: Path | None = None


def _orbit_run_dir() -> Path | None:
    raw = os.environ.get("SAT_ORBIT_RUN_DIR")
    if not raw:
        return None
    return Path(raw).expanduser()


def _orbit_role() -> str:
    return (os.environ.get("SAT_ORBIT_ROLE") or "server").strip().lower()


def _orbit_log_path() -> Path | None:
    run_dir = _orbit_run_dir()
    if run_dir is None:
        return None
    filename = "sim.log" if _orbit_role() == "sim" else "server.log"
    return run_dir / filename


def _write_log_file(line: str) -> None:
    global _LOG_HANDLE, _LOG_PATH
    path = _orbit_log_path()
    if path is None:
        return

    with _LOG_LOCK:
        try:
            if _LOG_HANDLE is None or _LOG_PATH != path:
                path.parent.mkdir(parents=True, exist_ok=True)
                _LOG_HANDLE = open(path, "a", encoding="utf-8")
                _LOG_PATH = path
            assert _LOG_HANDLE is not None
            _LOG_HANDLE.write(line + "\n")
            _LOG_HANDLE.flush()
        except Exception:
            return


def _log(message: str) -> None:
    role = _orbit_role()
    prefix = f"[{role} pid={os.getpid()}]"
    text = f"{prefix} {message}"
    print(text, file=sys.stderr, flush=True)
    _write_log_file(text)


def _write_run_artifact(name: str, content: str) -> None:
    run_dir = _orbit_run_dir()
    if run_dir is None:
        return
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / name).write_text(content, encoding="utf-8")
    except Exception:
        return


def _windows_session_id(pid: int) -> int | None:
    if not sys.platform.startswith("win"):
        return None
    try:
        import ctypes

        session_id = ctypes.c_uint()
        ok = ctypes.windll.kernel32.ProcessIdToSessionId(int(pid), ctypes.byref(session_id))
        if ok == 0:
            return None
        return int(session_id.value)
    except Exception:
        return None


def _windows_list_top_level_windows(pid: int) -> list[dict[str, object]]:
    if not sys.platform.startswith("win"):
        return []
    try:
        import ctypes

        user32 = ctypes.windll.user32
        windows: list[dict[str, object]] = []

        EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

        def callback(hwnd, lparam):  # noqa: ANN001
            try:
                window_pid = ctypes.c_uint()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(window_pid))
                if int(window_pid.value) != int(pid):
                    return True

                length = user32.GetWindowTextLengthW(hwnd)
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                title = buf.value
                visible = bool(user32.IsWindowVisible(hwnd))
                windows.append({"hwnd": hex(int(hwnd)), "visible": visible, "title": title})
            except Exception:
                return True
            return True

        user32.EnumWindows(EnumWindowsProc(callback), 0)
        return windows
    except Exception:
        return []

@mcp.tool()
def visualize_orbit(latitude: float, longitude: float) -> str:
    """
    Runs a 3D visualization of the ISS orbiting and tracking a specific 
    latitude/longitude target on Earth.
    
    Args:
        latitude: The target latitude (e.g., 40.7128 for NYC)
        longitude: The target longitude (e.g., -74.0060 for NYC)
    """
    os.environ.setdefault("SAT_ORBIT_ROLE", "server")
    _log(f"Tool call: visualize_orbit(latitude={latitude}, longitude={longitude})")
    run_dir = _orbit_run_dir()
    if run_dir is not None:
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        _log(f"Run directory: {run_dir}")
        _write_run_artifact(
            "request.txt",
            f"latitude={latitude}\nlongitude={longitude}\nserver_pid={os.getpid()}\nsession_id={_windows_session_id(os.getpid())}\n",
        )
    _log("Launching simulation in a separate process (avoids GUI/event-loop issues in MCP server process).")

    base_dir = Path(__file__).resolve().parent
    child_code = "\n".join(
        [
            "import sys, os",
            f"sys.path.insert(0, {repr(str(base_dir))})",
            "import satellite_server as s",
            "s.run_simulation_logic(float(sys.argv[1]), float(sys.argv[2]))",
        ]
    )
    cmd = [sys.executable, "-u", "-c", child_code, str(latitude), str(longitude)]
    _log(f"Simulation subprocess command: {cmd!r}")

    child_env = os.environ.copy()
    child_env["SAT_ORBIT_ROLE"] = "sim"
    child_env.setdefault("PYTHONUNBUFFERED", "1")
    child_env.setdefault("PYTHONIOENCODING", "utf-8")

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=None,
        env=child_env,
    )
    _log(f"Simulation subprocess started: pid={proc.pid} session_id={_windows_session_id(proc.pid or -1)}")

    start = time.monotonic()
    last_status = start
    next_window_check = start + 2.0
    window_checks = 0
    while True:
        exit_code = proc.poll()
        if exit_code is not None:
            break

        now = time.monotonic()
        if next_window_check is not None and now >= next_window_check:
            window_checks += 1
            windows = _windows_list_top_level_windows(proc.pid or -1)
            if windows:
                _log(f"Detected {len(windows)} top-level window(s) for sim pid={proc.pid}: {windows[:6]}")
                next_window_check = None
            else:
                _log(f"No top-level windows detected yet for sim pid={proc.pid}")
                next_window_check = (start + 10.0) if window_checks < 2 else None

        if now - last_status >= 30.0:
            last_status = now
            _log(f"Simulation still running (pid={proc.pid}, elapsed={now - start:.1f}s). Close the Matplotlib window to finish.")

        time.sleep(0.5)

    exit_code = int(exit_code)
    _log(f"Simulation subprocess exit code: {exit_code}")
    if exit_code != 0:
        raise RuntimeError(f"Simulation subprocess failed with exit code {exit_code}")

    # Note: exceptions are surfaced to the MCP client.
    return f"Simulation completed for Target ({latitude}, {longitude})"

def run_simulation_logic(target_lat, target_lon):
    _log("Entered run_simulation_logic()")
    _write_run_artifact("sim_started.txt", f"pid={os.getpid()}\nsession_id={_windows_session_id(os.getpid())}\n")
    import numpy as np
    import matplotlib
    import threading

    _log(f"PID: {os.getpid()}")
    _log(f"Python: {sys.version.splitlines()[0]}")
    _log(f"Platform: {sys.platform}")
    _log(f"Windows session id: {_windows_session_id(os.getpid())}")
    try:
        _log(f"stdio isatty: stdin={sys.stdin.isatty()} stdout={sys.stdout.isatty()} stderr={sys.stderr.isatty()}")
    except Exception:
        pass
    _log(
        "Thread: "
        f"name={threading.current_thread().name} "
        f"is_main={threading.current_thread() is threading.main_thread()}"
    )
    if threading.current_thread() is not threading.main_thread():
        _log("WARNING: visualize_orbit is not running on the main thread; Matplotlib GUI windows may not appear.")
    _log(f"Env DISPLAY={os.environ.get('DISPLAY')!r} WAYLAND_DISPLAY={os.environ.get('WAYLAND_DISPLAY')!r}")
    _log(f"Env MPLBACKEND={os.environ.get('MPLBACKEND')!r} SAT_ORBIT_MPL_BACKEND={os.environ.get('SAT_ORBIT_MPL_BACKEND')!r}")

    display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    if sys.platform.startswith("linux") and not display and not os.environ.get("MPLBACKEND"):
        raise RuntimeError(
            "No GUI display detected (missing DISPLAY/WAYLAND_DISPLAY). "
            "If you're running under WSL, enable WSLg or an X server, or set SAT_ORBIT_MPL_BACKEND=Agg for headless output."
        )

    forced_backend = (os.environ.get("SAT_ORBIT_MPL_BACKEND") or "").strip()
    if forced_backend:
        matplotlib.use(forced_backend, force=True)
    else:
        backend = str(matplotlib.get_backend() or "").lower()
        _log(f"Initial Matplotlib backend: {backend}")
        if backend in {"agg", "template"} or "inline" in backend:
            try:
                import tkinter  # noqa: F401

                matplotlib.use("TkAgg", force=True)
            except Exception:  # noqa: BLE001
                raise RuntimeError(
                    f"Matplotlib is using a non-interactive backend ({matplotlib.get_backend()!r}). "
                    "Install a GUI backend (e.g. Tk) or set SAT_ORBIT_MPL_BACKEND to an interactive backend."
                )

    import matplotlib.pyplot as plt
    from skyfield.api import load, wgs84
    from matplotlib.animation import FuncAnimation

    # Constants
    SIMULATION_DURATION_MINS = 95
    FRAMES = 200

    _log(f"Starting simulation for {target_lat}, {target_lon}...")
    _log(f"Matplotlib backend: {matplotlib.get_backend()}")
    try:
        _log(f"Matplotlib interactive: {plt.isinteractive()}")
    except Exception:
        pass

    # 1. Load Data
    # Prefer Skyfield's built-in timescale to avoid network downloads on first run.
    try:
        ts = load.timescale(builtin=True)
    except TypeError:  # pragma: no cover - older Skyfield versions
        ts = load.timescale()
    _log("Loaded Skyfield timescale.")

    local_tle = Path(__file__).with_name("stations.txt")
    tle_source = os.getenv("SAT_ORBIT_TLE_SOURCE")
    if not tle_source:
        tle_source = str(local_tle) if local_tle.exists() else "http://celestrak.org/NORAD/elements/stations.txt"
    _log(f"TLE source: {tle_source}")

    # Use Skyfield caching when a URL is provided.
    satellites_list = load.tle_file(tle_source)
    _log(f"Loaded TLE entries: {len(satellites_list)}")
    satellites = {sat.name: sat for sat in satellites_list}
    
    # Select Satellite
    if 'ISS (ZARYA)' in satellites:
        satellite = satellites['ISS (ZARYA)']
    elif 'ISS' in satellites:
        satellite = satellites['ISS']
    else:
        satellite = satellites_list[0]
    _log(f"Selected satellite: {getattr(satellite, 'name', '<unknown>')}")

    # 2. Time Setup
    t0 = ts.now()
    minutes = np.linspace(0, SIMULATION_DURATION_MINS, FRAMES)
    times = ts.utc(t0.utc_datetime().year, t0.utc_datetime().month, t0.utc_datetime().day, 
                   t0.utc_datetime().hour, t0.utc_datetime().minute + minutes)
    _log("Built simulation times array.")

    # 3. Calculations
    geocentric = satellite.at(times)
    x_sat, y_sat, z_sat = geocentric.position.km

    target_loc = wgs84.latlon(target_lat, target_lon)
    target_positions = target_loc.at(times).position.km
    x_targ, y_targ, z_targ = target_positions
    _log("Computed satellite + target positions.")

    # 4. Visualization
    # Note: When running as a server, we must be careful with GUI blocking.
    # This will open a window on the host machine.
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    _log("Created Matplotlib figure + axes.")
    run_dir = _orbit_run_dir()
    if run_dir is not None:
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            snapshot = run_dir / "figure_before_show.png"
            fig.savefig(snapshot)
            _log(f"Saved debug snapshot: {snapshot}")
        except Exception as exc:  # noqa: BLE001
            _log(f"Failed to save debug snapshot: {exc}")
    
    # Earth (Wireframe sphere)
    R_earth = 6371.0
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x_earth = R_earth * np.cos(u) * np.sin(v)
    y_earth = R_earth * np.sin(u) * np.sin(v)
    z_earth = R_earth * np.cos(v)
    ax.plot_wireframe(x_earth, y_earth, z_earth, color='gray', alpha=0.3)

    # Elements
    orbit_line, = ax.plot([], [], [], 'b--', label='Orbit', alpha=0.5)
    sat_point, = ax.plot([], [], [], 'ro', label='ISS')
    target_point, = ax.plot([], [], [], 'g^', label='Target')
    pointing_vector, = ax.plot([], [], [], 'r-', lw=2)

    status_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

    max_range = R_earth + 2000
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    ax.legend()
    ax.set_title(f"Targeting: {target_lat}, {target_lon}")

    def update(frame):
        cx, cy, cz = x_sat[frame], y_sat[frame], z_sat[frame]
        tx, ty, tz = x_targ[frame], y_targ[frame], z_targ[frame]

        orbit_line.set_data(x_sat[:frame], y_sat[:frame])
        orbit_line.set_3d_properties(z_sat[:frame])

        sat_point.set_data([cx], [cy])
        sat_point.set_3d_properties([cz])

        target_point.set_data([tx], [ty])
        target_point.set_3d_properties([tz])

        pointing_vector.set_data([cx, tx], [cy, ty])
        pointing_vector.set_3d_properties([cz, tz])
        
        dist = np.sqrt((tx-cx)**2 + (ty-cy)**2 + (tz-cz)**2)
        status_text.set_text(f"Dist: {dist:.1f} km")

        return orbit_line, sat_point, target_point, pointing_vector, status_text

    ani = FuncAnimation(fig, update, frames=FRAMES, interval=50, blit=False)
    _log("Animation configured; calling plt.show() (blocking).")

    try:
        manager = plt.get_current_fig_manager()
        window = getattr(manager, "window", None)
        if window is not None:
            try:
                window.attributes("-topmost", True)
                window.attributes("-topmost", False)
            except Exception:
                pass
            try:
                window.lift()
                window.focus_force()
            except Exception:
                pass
            _log("Requested GUI window focus/lift.")
    except Exception:
        pass
    
    # Block until window is closed
    plt.show()
    _log("plt.show() returned; simulation ending.")

if __name__ == "__main__":
    # This allows you to run the server
    try:
        signature = inspect.signature(mcp.run)
    except (TypeError, ValueError):  # pragma: no cover
        signature = None

    if signature is not None and "transport" in signature.parameters:
        mcp.run(transport="stdio")
    else:
        mcp.run()
