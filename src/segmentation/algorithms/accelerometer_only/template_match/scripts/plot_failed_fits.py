"""Per-ride PNGs for rides that have a failed trapezoid fit.

For every variant under ``template_match/labels/fit_elevator_paramater/``,
read each experiment's ``parameters.json`` and, for every ride whose fit
has ``t_c is None`` on either lobe, load the raw ACC/PRS stream, slice
the ride window, and render the same three-panel figure used in
``_all_rides.png`` (``a_vert`` with fitted trapezoids overlaid, ``vz``,
barometer height). Per-ride PNGs land at

    <variant>/_failed_fits/<exp>/ride_NN_<type>.png

so each failure can be eyeballed in isolation. A ride that happened to
recover one of its two lobes still shows up here — the successful lobe
is overlaid on top of the signal so the asymmetry is visible at a glance.

Run:
    venv/bin/python src/segmentation/algorithms/accelerometer_only/\
template_match/scripts/plot_failed_fits.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = Path(__file__).resolve().parent
_COMMON_PATH = _HERE.parent / "fit_elevator_parameters" / "common.py"
_COMMON_MOD_NAME = "_fit_ep_common"
if _COMMON_MOD_NAME in sys.modules:
    _common = sys.modules[_COMMON_MOD_NAME]
else:
    _spec = importlib.util.spec_from_file_location(_COMMON_MOD_NAME, _COMMON_PATH)
    _common = importlib.util.module_from_spec(_spec)
    assert _spec.loader is not None
    sys.modules[_COMMON_MOD_NAME] = _common
    _spec.loader.exec_module(_common)

LobeFit = _common.LobeFit
RideFit = _common.RideFit
build_ride_slices = _common.build_ride_slices
_draw_ride_panel = _common._draw_ride_panel
getExperimentData = _common.getExperimentData

VARIANTS_ROOT = _HERE.parent / "labels" / "fit_elevator_paramater"


def _lobe_from_dict(d: dict | None) -> LobeFit:
    if d is None:
        return LobeFit()
    return LobeFit(
        t_c=d.get("t_c"),
        a_peak=d.get("a_peak"),
        half_width_s=d.get("half_width_s"),
        frac_flat=d.get("frac_flat"),
        r2_local=d.get("r2_local"),
    )


def _ride_failed(ride_dict: dict) -> bool:
    l1 = ride_dict.get("lobe1") or {}
    l2 = ride_dict.get("lobe2") or {}
    return l1.get("t_c") is None or l2.get("t_c") is None


def _make_ride_panel_fig(ride) -> plt.Figure:
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 7.5), sharex=True)
    ax_top, ax_mid, ax_bot = axes
    title = f"#{ride.index:02d} {ride.ride_type} — {ride.gt_t1 - ride.gt_t0:.1f}s"
    _draw_ride_panel(ax_top, ax_mid, ax_bot, ride, title=title)
    fig.tight_layout()
    return fig


def _process_variant(variant_dir: Path) -> int:
    exp_dirs = sorted(
        p for p in variant_dir.iterdir()
        if p.is_dir() and not p.name.startswith("_")
    )
    out_root = variant_dir / "_failed_fits"
    n_written = 0
    for d in exp_dirs:
        jf = d / "parameters.json"
        if not jf.exists():
            continue
        data = json.loads(jf.read_text())
        failures_by_idx = {
            int(rd["index"]): rd for rd in data if _ride_failed(rd)
        }
        if not failures_by_idx:
            continue
        try:
            sensors, gt, _meta = getExperimentData(d.name)
        except Exception as exc:
            print(f"[error] {variant_dir.name}/{d.name}: {type(exc).__name__}: {exc}")
            continue
        if "ACC" not in sensors or sensors["ACC"].empty:
            continue
        slices = build_ride_slices(sensors["ACC"], gt, prs=sensors.get("PRS"))
        out_dir = out_root / d.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for s in slices:
            if s.index not in failures_by_idx:
                continue
            rd = failures_by_idx[s.index]
            s.fit = RideFit(
                index=s.index, ride_type=s.ride_type,
                duration_s=float(rd.get("duration_s") or 0.0),
                lobe1=_lobe_from_dict(rd.get("lobe1")),
                lobe2=_lobe_from_dict(rd.get("lobe2")),
                lobe_centroid_spacing_s=rd.get("lobe_centroid_spacing_s"),
            )
            fig = _make_ride_panel_fig(s)
            fname = f"ride_{s.index:02d}_{s.ride_type}.png"
            fig.savefig(out_dir / fname, dpi=110, bbox_inches="tight")
            plt.close(fig)
            n_written += 1
    return n_written


def main() -> int:
    if not VARIANTS_ROOT.exists():
        print(f"no variants root at {VARIANTS_ROOT}", file=sys.stderr)
        return 1
    variants = sorted(
        p for p in VARIANTS_ROOT.iterdir()
        if p.is_dir() and not p.name.startswith("_")
    )
    total = 0
    for vd in variants:
        n = _process_variant(vd)
        print(f"[{vd.name}] wrote {n} failed-ride PNGs under {vd / '_failed_fits'}")
        total += n
    if total == 0:
        print("no failures found in any variant")
    else:
        print(f"\n{total} total failed-ride PNGs written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
