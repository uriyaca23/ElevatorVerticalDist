"""Tkinter GUI for *viewing* the accelerometer + gyroscope of USC-HAD segments.

A deliberately small, read-only companion to ``gt_editor.py``: it shows the
accelerometer and gyroscope traces of one USC-HAD recording at a time and
lets you step between segments. There is no editing — USC-HAD files are
already pre-segmented single-activity clips (one ~15-35 s trial per ``.mat``),
so there is nothing to slice.

The "View" dropdown chooses what the top panel shows: ``Raw`` (the body-frame
accelerometer, the default) or one of the accel+gyro orientation filters in
``src/physics/reconstruct_az`` (Complementary, Mahony, Madgwick, Valenti,
ESKF), which reconstruct the *real* world-frame vertical acceleration ``a_z``
by tracking device orientation and removing gravity. The "vs" dropdown adds a
second view side-by-side, so you can see how a reconstruction sharpens the
ride trapezoid relative to the raw trace (or to another algorithm).

Usage (note the hyphen in the name → run as a script, not ``-m``):

    venv/bin/python src/data/usc-had_gt_viewer.py

Navigation:
    ◀ Prev / Next ▶      step between segments
    ← / →                same, from the keyboard
    Home / End           jump to first / last segment
    click a row          jump to that segment in the list on the right

The "Activity" dropdown filters which segments you step through; it defaults
to the elevator rides (Elevator Up + Down), which is what this dataset was
pulled in for.

Data is read from the sibling ``USC-HAD/`` folder (``Subject*/aMtN.mat``).
Each file stores ``sensor_readings`` as an (N, 6) array — columns
[acc_x, acc_y, acc_z (g), gyro_x, gyro_y, gyro_z (dps)] — sampled at 100 Hz.
The accelerometer (or a reconstructed a_z) is shown in the top panel(s); the
gyroscope is always shown in the bottom panel.
"""

from __future__ import annotations

import re
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.figure import Figure

try:
    from scipy.io import loadmat
except ImportError:  # pragma: no cover - environment guard
    loadmat = None

# Run as a script (``python src/data/usc-had_gt_viewer.py``): put the project
# root on sys.path so the ``src`` package — and the reconstructors below — are
# importable. (parents[2] of src/data/<file>.py is the repo root.)
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.physics.reconstruct_az import (  # noqa: E402
    RECONSTRUCTORS,
    SensorChannel,
    build,
)

# USC-HAD is recorded on a MotionNode IMU at a fixed rate (see Readme.txt).
SAMPLE_RATE_HZ = 100.0
G_MS2 = 9.80665           # standard gravity, for the USC-HAD g → m/s² conversion

# View-dropdown sentinels: raw trace, and "no comparison".
_RAW = "Raw"
_NONE = "—"

# USC-HAD activity-number → name (Readme.txt, Section 3).
ACTIVITY_NAMES = {
    1: "Walking Forward",   2: "Walking Left",      3: "Walking Right",
    4: "Walking Upstairs",  5: "Walking Downstairs", 6: "Running Forward",
    7: "Jumping Up",        8: "Sitting",           9: "Standing",
    10: "Sleeping",         11: "Elevator Up",      12: "Elevator Down",
}

# Folder holding the dataset, sibling to this file: src/data/USC-HAD/.
DATA_ROOT = Path(__file__).resolve().parent / "USC-HAD"

# Filename like ``a11t3.mat`` → (activity_number, trial_number).
_FNAME_RE = re.compile(r"^a(\d+)t(\d+)\.mat$", re.IGNORECASE)


class Segment:
    """One USC-HAD recording, described from its path (no I/O until shown)."""

    __slots__ = ("path", "subject", "activity", "trial")

    def __init__(self, path: Path, subject: int, activity: int, trial: int):
        self.path = path
        self.subject = subject
        self.activity = activity
        self.trial = trial

    @property
    def activity_name(self) -> str:
        return ACTIVITY_NAMES.get(self.activity, f"activity {self.activity}")

    def label(self) -> str:
        return (f"S{self.subject:>2}  {self.activity_name:<17} "
                f"trial {self.trial}")


def _scan_segments(root: Path) -> list[Segment]:
    """Build the segment list from the folder layout (cheap, no .mat reads)."""
    segments: list[Segment] = []
    for subj_dir in sorted(root.glob("Subject*")):
        m = re.search(r"Subject(\d+)", subj_dir.name)
        if not m:
            continue
        subject = int(m.group(1))
        for mat in subj_dir.glob("*.mat"):
            fm = _FNAME_RE.match(mat.name)
            if not fm:
                continue
            segments.append(
                Segment(mat, subject, int(fm.group(1)), int(fm.group(2)))
            )
    # Group by activity, then subject, then trial — so the elevator rides
    # (and any other filtered activity) read in a sensible order.
    segments.sort(key=lambda s: (s.activity, s.subject, s.trial))
    return segments


class UscHadViewer(tk.Tk):
    """Read-only accelerometer viewer with segment-to-segment navigation."""

    # Filter presets shown in the Activity dropdown.
    _ELEVATOR = "Elevator (Up + Down)"
    _ALL = "All activities"

    def __init__(self) -> None:
        super().__init__()
        self.title("USC-HAD — Accel/Gyro + a_z Reconstruction Viewer")
        self.geometry("1100x680")

        if loadmat is None:
            messagebox.showerror(
                "Missing dependency",
                "scipy is required (pip install scipy). Use the project venv.",
            )
            self.destroy()
            return
        if not DATA_ROOT.is_dir():
            messagebox.showerror(
                "Dataset not found",
                f"Expected USC-HAD at:\n{DATA_ROOT}",
            )
            self.destroy()
            return

        self._all_segments = _scan_segments(DATA_ROOT)
        if not self._all_segments:
            messagebox.showerror("No data", f"No .mat files under {DATA_ROOT}")
            self.destroy()
            return

        self._segments: list[Segment] = []   # current (filtered) view
        self._cur = 0
        self._show_mag = tk.BooleanVar(value=True)

        # One reconstructor instance per algorithm, reused across renders.
        self._recons = {name: build(name) for name in RECONSTRUCTORS}

        self._build_controls()
        self._build_plot_and_list()
        self._bind_keys()

        self._filter.set(self._ELEVATOR)
        self._apply_filter()

    # ----------------------------------------------------------------- UI ---
    def _build_controls(self) -> None:
        bar = ttk.Frame(self, padding=(8, 6))
        bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(bar, text="Activity:").pack(side=tk.LEFT)
        names = [ACTIVITY_NAMES[k] for k in sorted(ACTIVITY_NAMES)]
        self._filter = ttk.Combobox(
            bar, width=22, state="readonly",
            values=[self._ELEVATOR, self._ALL, *names],
        )
        self._filter.pack(side=tk.LEFT, padx=(4, 12))
        self._filter.bind("<<ComboboxSelected>>", lambda _e: self._apply_filter())

        ttk.Button(bar, text="◀ Prev", command=self._prev).pack(side=tk.LEFT)
        ttk.Button(bar, text="Next ▶", command=self._next).pack(
            side=tk.LEFT, padx=(4, 12))

        self._counter = ttk.Label(bar, text="0 / 0", width=12)
        self._counter.pack(side=tk.LEFT)

        # Reconstruction view selectors: pick an algorithm (default Raw), and
        # optionally a second one to compare side-by-side.
        ttk.Label(bar, text="View:").pack(side=tk.LEFT, padx=(12, 2))
        self._viewA = ttk.Combobox(
            bar, width=13, state="readonly", values=[_RAW, *RECONSTRUCTORS],
        )
        self._viewA.set(_RAW)
        self._viewA.pack(side=tk.LEFT)
        self._viewA.bind("<<ComboboxSelected>>", lambda _e: self._render())

        ttk.Label(bar, text="vs").pack(side=tk.LEFT, padx=(8, 2))
        self._viewB = ttk.Combobox(
            bar, width=13, state="readonly", values=[_NONE, *RECONSTRUCTORS],
        )
        self._viewB.set(_NONE)
        self._viewB.pack(side=tk.LEFT)
        self._viewB.bind("<<ComboboxSelected>>", lambda _e: self._render())

        ttk.Checkbutton(
            bar, text="magnitude (|a|, |ω|)", variable=self._show_mag,
            command=self._render,
        ).pack(side=tk.RIGHT)

    def _build_plot_and_list(self) -> None:
        main = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        # Left: matplotlib canvas + standard navigation toolbar (pan/zoom/save).
        left = ttk.Frame(main)
        main.add(left, weight=5)
        self.fig = Figure(figsize=(9, 6.5))
        # Subplots are (re)built per render in _render(): the layout differs
        # between single-view and side-by-side compare modes.
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, left).update()

        # Right: clickable list of segments in the current filter.
        right = ttk.Frame(main, padding=(6, 0))
        main.add(right, weight=1)
        ttk.Label(right, text="Segments").pack(anchor=tk.W)
        list_frame = ttk.Frame(right)
        list_frame.pack(fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self._listbox = tk.Listbox(
            list_frame, activestyle="dotbox", exportselection=False,
            font=("Menlo", 11), yscrollcommand=sb.set,
        )
        sb.config(command=self._listbox.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._listbox.bind("<<ListboxSelect>>", self._on_list_select)

        # Bottom status line.
        self._status = ttk.Label(self, anchor=tk.W, padding=(8, 2),
                                 relief=tk.SUNKEN)
        self._status.pack(side=tk.BOTTOM, fill=tk.X)

    def _bind_keys(self) -> None:
        self.bind("<Left>", lambda _e: self._prev())
        self.bind("<Right>", lambda _e: self._next())
        self.bind("<Home>", lambda _e: self._goto(0))
        self.bind("<End>", lambda _e: self._goto(len(self._segments) - 1))

    # ------------------------------------------------------------- actions ---
    def _apply_filter(self) -> None:
        choice = self._filter.get()
        if choice == self._ALL:
            keep = lambda s: True
        elif choice == self._ELEVATOR:
            keep = lambda s: s.activity in (11, 12)
        else:
            wanted = next((k for k, v in ACTIVITY_NAMES.items()
                           if v == choice), None)
            keep = lambda s: s.activity == wanted

        self._segments = [s for s in self._all_segments if keep(s)]
        self._listbox.delete(0, tk.END)
        for seg in self._segments:
            self._listbox.insert(tk.END, "  " + seg.label())
        self._goto(0)

    def _goto(self, idx: int) -> None:
        if not self._segments:
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.set_title("No segments for this filter")
            ax.axis("off")
            self.canvas.draw_idle()
            self._counter.config(text="0 / 0")
            self._status.config(text="")
            return
        self._cur = max(0, min(idx, len(self._segments) - 1))
        self._listbox.selection_clear(0, tk.END)
        self._listbox.selection_set(self._cur)
        self._listbox.see(self._cur)
        self._render()

    def _prev(self) -> None:
        self._goto(self._cur - 1)

    def _next(self) -> None:
        self._goto(self._cur + 1)

    def _on_list_select(self, _event) -> None:
        sel = self._listbox.curselection()
        if sel and sel[0] != self._cur:
            self._goto(sel[0])

    # ------------------------------------------------------------- drawing ---
    def _render(self) -> None:
        if not self._segments:
            return
        seg = self._segments[self._cur]
        try:
            mat = loadmat(seg.path)
            sr = np.asarray(mat["sensor_readings"], dtype=float)
        except Exception as exc:  # noqa: BLE001 - surface any load issue
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.set_title(f"Could not load {seg.path.name}: {exc}")
            ax.axis("off")
            self.canvas.draw_idle()
            return

        acc_g = sr[:, 0:3]                                # acc_x, acc_y, acc_z (g)
        gyro_dps = sr[:, 3:6]                             # gyro_x, gyro_y, gyro_z (dps)
        t = np.arange(len(sr)) / SAMPLE_RATE_HZ           # seconds

        # SI bundle for the reconstructors (USC-HAD ships g / dps; the
        # reconstruct interface expects m/s² / rad/s). Both streams share the
        # same synthesised 100 Hz clock, so alignment inside is a no-op.
        ts_ms = t * 1000.0
        acc_df = pd.DataFrame({
            "timestamp_ms": ts_ms,
            "x": acc_g[:, 0] * G_MS2, "y": acc_g[:, 1] * G_MS2, "z": acc_g[:, 2] * G_MS2,
        })
        gyr_df = pd.DataFrame({
            "timestamp_ms": ts_ms,
            "x": np.deg2rad(gyro_dps[:, 0]),
            "y": np.deg2rad(gyro_dps[:, 1]),
            "z": np.deg2rad(gyro_dps[:, 2]),
        })
        sensors = {SensorChannel.ACC: acc_df, SensorChannel.GYR: gyr_df}

        view_a = self._viewA.get()
        view_b = self._viewB.get()
        compare = view_b not in ("", _NONE)

        self.fig.clear()
        if compare:
            # Two accel panels side-by-side over one shared gyro panel.
            gs = self.fig.add_gridspec(2, 2, height_ratios=[2.0, 1.0])
            ax_a = self.fig.add_subplot(gs[0, 0])
            share_y = (view_a != _RAW) == (view_b != _RAW)  # same units → shared y
            ax_b = self.fig.add_subplot(
                gs[0, 1], sharex=ax_a, sharey=ax_a if share_y else None)
            ax_g = self.fig.add_subplot(gs[1, :], sharex=ax_a)
            self._plot_view(ax_a, view_a, t, acc_g, sensors)
            self._plot_view(ax_b, view_b, t, acc_g, sensors)
            ax_a.tick_params(labelbottom=False)
            ax_b.tick_params(labelbottom=False)
        else:
            ax_a = self.fig.add_subplot(2, 1, 1)
            ax_g = self.fig.add_subplot(2, 1, 2, sharex=ax_a)
            self._plot_view(ax_a, view_a, t, acc_g, sensors)
            ax_a.tick_params(labelbottom=False)

        self._plot_gyro(ax_g, t, gyro_dps)

        # Dataset's own label/subject for the figure title (ground truth).
        act = str(mat.get("activity", [seg.activity_name])[0]).strip()
        self.fig.suptitle(
            f"Subject {seg.subject} · {act} · trial {seg.trial} "
            f"· {len(sr) / SAMPLE_RATE_HZ:.1f}s @ {SAMPLE_RATE_HZ:.0f}Hz",
            fontsize=10)
        self.fig.tight_layout(rect=(0, 0, 1, 0.97))
        self.canvas.draw_idle()

        self._counter.config(text=f"{self._cur + 1} / {len(self._segments)}")
        self._status.config(text=f"{seg.path}   |   {len(sr)} samples")

    def _plot_view(self, ax, view, t, acc_g, sensors) -> None:
        """Draw one panel: raw body-frame accel, or a reconstructed world az."""
        if view == _RAW:
            for col, name, color in zip(
                range(3), ("acc_x", "acc_y", "acc_z"),
                ("tab:red", "tab:green", "tab:blue"),
            ):
                ax.plot(t, acc_g[:, col], color=color, lw=1.0, label=name)
            if self._show_mag.get():
                ax.plot(t, np.linalg.norm(acc_g, axis=1),
                        color="black", lw=1.4, label="|a|")
                ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.6)
            ax.set_ylabel("accel (g)")
            ax.set_title("Raw (body frame)")
        else:
            try:
                rec = self._recons[view].reconstruct(sensors)
            except Exception as exc:  # noqa: BLE001 - surface filter failures
                ax.set_title(f"{view}: failed ({exc})")
                ax.axis("off")
                return
            # a_z is the reconstructed real vertical accel (the trapezoid);
            # a_x / a_y should collapse toward zero as tilt is removed.
            ax.plot(t, rec.az, color="tab:purple", lw=1.7, label="a_z (vertical)")
            ax.plot(t, rec.ax, color="tab:red", lw=0.8, alpha=0.5, label="a_x")
            ax.plot(t, rec.ay, color="tab:green", lw=0.8, alpha=0.5, label="a_y")
            if self._show_mag.get():
                mag = np.sqrt(rec.ax ** 2 + rec.ay ** 2 + rec.az ** 2)
                ax.plot(t, mag, color="black", lw=1.0, alpha=0.7, label="|a|")
            ax.axhline(0.0, color="gray", ls="--", lw=0.8, alpha=0.6)
            ax.set_ylabel("world accel (m/s²)")
            ax.set_title(f"{view} — reconstructed")
        ax.margins(x=0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=4, fontsize=7)

    def _plot_gyro(self, ax, t, gyro_dps) -> None:
        """Draw the shared gyroscope panel (raw dps) at the bottom."""
        for col, name, color in zip(
            range(3), ("gyro_x", "gyro_y", "gyro_z"),
            ("tab:red", "tab:green", "tab:blue"),
        ):
            ax.plot(t, gyro_dps[:, col], color=color, lw=1.0, label=name)
        if self._show_mag.get():
            ax.plot(t, np.linalg.norm(gyro_dps, axis=1),
                    color="black", lw=1.3, label="|ω|")
        ax.set_ylabel("angular rate (dps)")
        ax.set_xlabel("time (s)")
        ax.margins(x=0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=4, fontsize=7)


def main() -> None:
    app = UscHadViewer()
    # If __init__ aborted (missing data/deps) the window is already destroyed.
    if app.winfo_exists():
        app.mainloop()


if __name__ == "__main__":
    sys.exit(main())
