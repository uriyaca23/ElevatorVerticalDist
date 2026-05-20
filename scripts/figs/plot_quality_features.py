"""Generate the Appendix-E per-quality-feature example figures for the IMWUT paper.

Nine PNGs land in ``paper_phd/figures/``, one per discriminating quality
feature. Each is a 1x2 good-vs-bad panel rendered from a real recording,
with the feature value and its threshold annotated:

Shared (gravity / orientation) features --
* ``qf_gravity_stability.png``  -- pre-window stationarity ``s_pre``.
* ``qf_pre_post_angle.png``     -- pre/post gravity-vector angle.
* ``qf_ride_drift.png``         -- in-ride gravity drift.

ZUPT-specific --
* ``qf_impact.png``             -- impact-peak magnitude.

Trapezoid-specific --
* ``qf_joint_r2.png``           -- matched-filter joint R^2.
* ``qf_residual_acf.png``       -- lag-1 residual autocorrelation.
* ``qf_anchor_ratio.png``       -- velocity-anchor amplitude ratio.
* ``qf_out_of_lobe.png``        -- out-of-lobe residual concentration.
* ``qf_cruise_cv.png``          -- cruise-velocity coefficient of variation.

The ``active_fraction`` and ``end_vel_ratio`` checks get no figure: every
labelled ride clears the active-motion gate, and the end-velocity ratio is
~0 by construction (the magnitude-detrended signal has zero mean), so
neither yields an honest good-vs-bad contrast.

The script flattens every experiment into labelled segments, runs both
accelerometer estimators on each, ranks candidate segments per feature, and
auto-picks a good and a bad example. Pass ``--scan-only`` to print the
shortlists without rendering; ``--limit N`` caps how many experiments are
scanned. ``OVERRIDES`` pins specific picks once chosen.

Run:  venv/bin/python -m scripts.figs.plot_quality_features
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.prediction.algorithms.accelerometer_only.zupt_accel.estimator import (  # noqa: E402
    ZuptAccelEstimator,
)
from src.prediction.algorithms.accelerometer_only.trapezoid_accel.estimator import (  # noqa: E402
    TrapezoidAccelEstimator,
)
from src.prediction.algorithms.accelerometer_only.trapezoid_accel.pulse_pair import (  # noqa: E402
    trapezoid_kernel,
)
from src.prediction.evaluation.dataset import load_all_segments  # noqa: E402
from src.utils.accelerometer_utils import zupt_integrate  # noqa: E402

PAPER_FIG = REPO / "paper_phd" / "figures"

# Pin chosen picks here once --scan-only has been inspected. Each value is a
# dict with "good" and/or "bad" keys; the value is "<exp_name>#<seg_idx>".
OVERRIDES: dict[str, dict[str, str]] = {}

_ZUPT = ZuptAccelEstimator()
_TRAP = TrapezoidAccelEstimator()

GOOD_GREEN = "#1a7a3a"
BAD_RED = "#b3261e"
RAW_GREY = "#b9c2cc"
SIG_DARK = "#2c3e50"
TPL_RED = "#d62728"


# ---------------------------------------------------------------------------
# Segment-level numeric helpers
# ---------------------------------------------------------------------------

def _axes(df) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (df["x"].to_numpy(float), df["y"].to_numpy(float),
            df["z"].to_numpy(float))


def _t_sec(df) -> np.ndarray:
    t = df["timestamp_ms"].to_numpy(float)
    return (t - t[0]) / 1000.0 if t.size else t


def _fs(df) -> float:
    t = df["timestamp_ms"].to_numpy(float)
    if t.size < 2:
        return 50.0
    dt = float(np.median(np.diff(t))) / 1000.0
    return 1.0 / dt if dt > 0 else 50.0


def _mag_detrended(df) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (t_sec, magnitude minus its mean, mean magnitude) -- the
    rotation-blind signal the ZUPT quality filter scores impacts and
    active-motion fraction against."""
    ax, ay, az = _axes(df)
    mag = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
    return _t_sec(df), mag - float(np.mean(mag)), float(np.mean(mag))


def _angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return float("nan")
    cos = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, cos)))))


def _drift_series(df, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-1 s-chunk gravity-direction angle relative to the first chunk --
    the quantity the ride-drift gate maxes over."""
    ax, ay, az = _axes(df)
    n = len(ax)
    chunk = max(10, int(fs * 1.0))
    n_chunks = max(1, n // chunk)
    if n_chunks < 2:
        return np.array([0.0]), np.array([0.0])
    gvecs, centers = [], []
    for i in range(n_chunks):
        s, e = i * chunk, min((i + 1) * chunk, n)
        gvecs.append(np.array([ax[s:e].mean(), ay[s:e].mean(), az[s:e].mean()]))
        centers.append((s + e) / 2.0 / fs)
    g0 = gvecs[0]
    angles = np.array([_angle_deg(g0, g) for g in gvecs])
    return np.array(centers), angles


def run_zupt(rec):
    try:
        return _ZUPT.predict_segment(rec.acc, rec.phone, rec.pre_acc, rec.post_acc)
    except Exception:
        return None


def run_trap(rec):
    try:
        return _TRAP.predict_segment(rec.acc, rec.phone, rec.pre_acc, rec.post_acc)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Shared rendering helpers
# ---------------------------------------------------------------------------

def _verdict_box(ax, lines: list[str], ok: bool) -> None:
    col = GOOD_GREEN if ok else BAD_RED
    verdict = "ACCEPTED" if ok else "REJECTED"
    ax.text(0.025, 0.045, "\n".join(lines + [verdict]),
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5,
            bbox=dict(facecolor="white", alpha=0.9, edgecolor=col,
                      boxstyle="round,pad=0.35"))


def good_bad_fig(out: Path, render, good, bad, suptitle: str) -> None:
    """1x2 good/bad figure. ``render(ax, rec, pred, ok)`` draws one panel."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    render(axes[0], good["rec"], good["pred"], ok=True)
    render(axes[1], bad["rec"], bad["pred"], ok=False)
    axes[0].set_title("good example -- feature in band", fontsize=9.5)
    axes[1].set_title("bad example -- feature out of band", fontsize=9.5)
    fig.suptitle(suptitle, fontsize=10.5, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


# ---------------------------------------------------------------------------
# Per-feature panel renderers
# ---------------------------------------------------------------------------

def render_gravity_stability(ax, rec, pred, ok: bool) -> None:
    """Pre-window 3-axis accelerometer trace. A stationary phone holds three
    flat lines; handling smears them."""
    df = rec.pre_acc if len(rec.pre_acc) else rec.acc
    t = _t_sec(df)
    axx, ayy, azz = _axes(df)
    for arr, col, lab in ((axx, "#1f77b4", "x"), (ayy, "#ff7f0e", "y"),
                          (azz, "#2ca02c", "z")):
        ax.plot(t, arr, color=col, lw=1.0, label=lab)
    s_pre = pred.meta.get("features", {}).get("pre_stability", float("nan"))
    ax.set_xlabel("t (s, pre-window)")
    ax.set_ylabel(r"$a$ (m/s$^2$)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="upper right", ncol=3)
    rel = r"$\leq$" if ok else ">"
    _verdict_box(ax, [f"stationarity $s_{{pre}}={s_pre:.2f}$ {rel} 1.0"], ok)


def render_pre_post_angle(ax, rec, pred, ok: bool) -> None:
    """Pre and post stationary windows side by side. A steady orientation
    keeps the three gravity components level across the gap."""
    pre, post = rec.pre_acc, rec.post_acc
    pax, pay, paz = _axes(pre)
    qax, qay, qaz = _axes(post)
    n_pre = len(pax)
    gap = max(5, n_pre // 12)
    x_pre = np.arange(n_pre)
    x_post = np.arange(len(qax)) + n_pre + gap
    for pa, qa, col, lab in ((pax, qax, "#1f77b4", "x"),
                             (pay, qay, "#ff7f0e", "y"),
                             (paz, qaz, "#2ca02c", "z")):
        ax.plot(x_pre, pa, color=col, lw=1.0, label=lab)
        ax.plot(x_post, qa, color=col, lw=1.0)
    ax.axvspan(n_pre, n_pre + gap, color="0.85", alpha=0.7)
    ax.text(n_pre - 1, ax.get_ylim()[1], "pre", ha="right", va="top", fontsize=7)
    ax.text(n_pre + gap + 1, ax.get_ylim()[1], "post", ha="left", va="top",
            fontsize=7)
    angle = pred.meta.get("features", {}).get("pre_post_angle_deg", float("nan"))
    ax.set_xlabel("sample (pre window | post window)")
    ax.set_ylabel(r"$a$ (m/s$^2$)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="upper right", ncol=3)
    rel = r"$\leq$" if ok else ">"
    _verdict_box(ax, [rf"pre/post angle $={angle:.0f}^\circ$ {rel} $25^\circ$"], ok)


def render_ride_drift(ax, rec, pred, ok: bool) -> None:
    """Per-second gravity-direction angle across the ride. A held phone stays
    near zero; reorientation climbs past the gate."""
    fs = _fs(rec.acc)
    centers, angles = _drift_series(rec.acc, fs)
    ax.plot(centers, angles, color=SIG_DARK, lw=1.6, marker="o", ms=4)
    ax.axhline(15.0, color=BAD_RED, lw=1.0, ls="--", label=r"gate $15^\circ$")
    ax.fill_between(centers, 0, angles, color="#8fa8c0", alpha=0.3)
    max_drift = pred.meta.get("features", {}).get("max_gravity_drift_deg",
                                                  float("nan"))
    ax.set_xlabel("t (s, ride)")
    ax.set_ylabel(r"gravity drift ($^\circ$)")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="upper left")
    rel = r"$\leq$" if ok else ">"
    _verdict_box(ax, [rf"max drift $={max_drift:.0f}^\circ$ {rel} $15^\circ$"], ok)


def render_impact(ax, rec, pred, ok: bool) -> None:
    """Magnitude-detrended acceleration with the dominant peak marked. A tap
    or hand-off spikes well past the 8 m/s^2 gate."""
    t, a_quick, _ = _mag_detrended(rec.acc)
    ax.plot(t, a_quick, color=SIG_DARK, lw=0.9)
    k = int(np.argmax(np.abs(a_quick)))
    ax.plot(t[k], a_quick[k], "o", color=BAD_RED, ms=7)
    for s in (8.0, -8.0):
        ax.axhline(s, color=BAD_RED, lw=0.9, ls="--")
    ax.axhline(0, color="gray", lw=0.4, ls=":")
    peak = pred.meta.get("features", {}).get("max_peak_m_s2", float("nan"))
    ax.set_xlabel("t (s, ride)")
    ax.set_ylabel(r"$|a|-\overline{|a|}$ (m/s$^2$)")
    ax.grid(True, alpha=0.25)
    rel = r"$\leq$" if ok else ">"
    _verdict_box(ax, [f"impact peak $={peak:.1f}$ m/s$^2$ {rel} 8.0"], ok)


def _trap_fit(pred):
    """Pull (t, a_smooth, a_template, params) from a trapezoid prediction."""
    m = pred.meta
    return (np.asarray(m["t_sec"]), np.asarray(m["a_smooth"]),
            np.asarray(m["a_template"]), m["params"])


def render_joint_r2(ax, rec, pred, ok: bool) -> None:
    """Smoothed vertical acceleration with the fitted trapezoid pulse pair.
    A genuine ride is explained by the template; a poor fit is not."""
    t, a_s, a_t, p = _trap_fit(pred)
    ax.plot(t, a_s, color=SIG_DARK, lw=1.3, label="smoothed $a$")
    ax.plot(t, a_t, color=TPL_RED, lw=1.8, label="fitted template")
    for tc in (p["t_c1"], p["t_c2"]):
        ax.axvspan(tc - p["W"], tc + p["W"], color="#1f77b4", alpha=0.10)
    ax.axhline(0, color="gray", lw=0.4, ls=":")
    qf = pred.meta["quality_features"]
    r2, r2min = qf["joint_r2"], qf["min_r2_effective"]
    ax.set_xlabel("t (s, ride)")
    ax.set_ylabel(r"$a$ (m/s$^2$)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="upper right")
    rel = r"$\geq$" if r2 >= r2min else "<"
    _verdict_box(ax, [f"joint $R^2={r2:.2f}$ {rel} min ${r2min:.2f}$"], ok)


def render_residual_acf(ax, rec, pred, ok: bool) -> None:
    """Fit residual over time. White residuals scatter around zero; a
    structured miss leaves slow, correlated excursions."""
    t, a_s, a_t, _ = _trap_fit(pred)
    res = a_s - a_t
    ax.plot(t, res, color="#7a4ea3", lw=1.0)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.fill_between(t, 0, res, color="#b79fd4", alpha=0.35)
    acf1 = pred.meta["quality_features"]["residual_acf1"]
    ax.set_xlabel("t (s, ride)")
    ax.set_ylabel(r"residual (m/s$^2$)")
    ax.grid(True, alpha=0.25)
    _verdict_box(ax, [f"lag-1 residual ACF $={acf1:.2f}$",
                      "score knee at 0.6"], ok)


def render_anchor_ratio(ax, rec, pred, ok: bool) -> None:
    """Matched-filter template amplitude vs the velocity-anchored amplitude.
    A trustworthy fit needs little rescaling; a large ratio flags a misfit."""
    t, a_s, a_t, p = _trap_fit(pred)
    ax.plot(t, a_s, color=SIG_DARK, lw=1.2, label="smoothed $a$")
    ax.plot(t, a_t, color=TPL_RED, lw=1.8, label=r"anchored ($A_{used}$)")
    sgn, W, f = p["sign"], p["W"], p["f"]
    a_fit = (sgn * p["A_fit"] * trapezoid_kernel(t, p["t_c1"], W, f)
             - sgn * p["A_fit"] * trapezoid_kernel(t, p["t_c2"], W, f))
    ax.plot(t, a_fit, color="#1f77b4", lw=1.3, ls="--",
            label=r"matched-filter ($A_{fit}$)")
    ax.axhline(0, color="gray", lw=0.4, ls=":")
    ratio = pred.meta["quality_features"]["A_anchor_ratio"]
    ax.set_xlabel("t (s, ride)")
    ax.set_ylabel(r"$a$ (m/s$^2$)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="upper right")
    band = "in [0.5, 2.0]" if ok else "outside [0.5, 2.0]"
    _verdict_box(ax, [f"anchor ratio $A_{{used}}/A_{{fit}}={ratio:.2f}$ {band}"],
                 ok)


def render_out_of_lobe(ax, rec, pred, ok: bool) -> None:
    """Fit residual with the two lobe-support windows shaded. A mid-ride
    disturbance concentrates residual energy outside the lobes."""
    t, a_s, a_t, p = _trap_fit(pred)
    res = a_s - a_t
    inside = ((np.abs(t - p["t_c1"]) <= p["W"])
              | (np.abs(t - p["t_c2"]) <= p["W"]))
    ax.plot(t, res, color="#7a4ea3", lw=1.0)
    ax.fill_between(t, 0, res, where=inside, color="#1f77b4", alpha=0.25,
                    label="inside lobes")
    ax.fill_between(t, 0, res, where=~inside, color="#ff7f0e", alpha=0.30,
                    label="outside lobes")
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    frac = pred.meta["quality_features"]["out_of_lobe_residual_frac"]
    ax.set_xlabel("t (s, ride)")
    ax.set_ylabel(r"residual (m/s$^2$)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="upper right")
    rel = r"$\leq$" if ok else ">"
    _verdict_box(ax, [f"out-of-lobe density ratio $={frac:.2f}$ {rel} 4.0"], ok)


def render_cruise_cv(ax, rec, pred, ok: bool) -> None:
    """ZUPT-integrated velocity with the cruise window shaded. At steady
    cruise the velocity is a flat plateau; a wobbly plateau fails the gate."""
    t, a_s, _, p = _trap_fit(pred)
    _pos, vel = zupt_integrate(a_s, t)
    cruise = (t > p["t_c1"] + p["W"]) & (t < p["t_c2"] - p["W"])
    ax.plot(t, vel, color=SIG_DARK, lw=1.5)
    if cruise.any():
        ax.fill_between(t, vel.min(), vel.max(), where=cruise,
                        color="#1f77b4", alpha=0.15, label="cruise window")
        ax.plot(t[cruise], vel[cruise], color=TPL_RED, lw=2.2)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    cv = pred.meta["quality_features"]["cruise_v_cv"]
    ax.set_xlabel("t (s, ride)")
    ax.set_ylabel(r"$v$ (m/s)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="upper right")
    rel = r"$\leq$" if ok else ">"
    _verdict_box(ax, [f"cruise velocity CV $={cv:.2f}$ {rel} 0.6"], ok)


# ---------------------------------------------------------------------------
# Feature specs -- one entry per output figure
# ---------------------------------------------------------------------------
# src        : "zupt" or "trap"  -- which estimator's features to read.
# feat       : feature key.
# reject_sub : substring of reject_reason that isolates a true bad example.
# bad_high   : True  -> bad is the largest value, good the smallest;
#              False -> bad is the smallest value, good the largest;
#              "two_sided" -> bad maximises |value - 1| (anchor ratio).
# good_band  : (lo, hi) the good value must fall inside.
SPECS = [
    dict(key="gravity_stability", src="zupt", feat="pre_stability",
         reject_sub="no_gravity_calibration", bad_high=True,
         good_band=(0.02, 0.6), render=render_gravity_stability,
         title="Pre-window stationarity gates the gravity calibration"),
    dict(key="pre_post_angle", src="zupt", feat="pre_post_angle_deg",
         reject_sub="orientation_changed", bad_high=True,
         good_band=(0.0, 12.0), render=render_pre_post_angle,
         title="Pre/post gravity-vector angle gates orientation stability"),
    dict(key="ride_drift", src="zupt", feat="max_gravity_drift_deg",
         reject_sub="ride_drift", bad_high=True,
         good_band=(0.0, 8.0), render=render_ride_drift,
         title="In-ride gravity drift gates the vertical-axis estimate"),
    dict(key="impact", src="zupt", feat="max_peak_m_s2",
         reject_sub="impact", bad_high=True,
         good_band=(0.0, 4.0), render=render_impact,
         title="Impact-peak magnitude rejects taps and hand-offs"),
    dict(key="joint_r2", src="trap", feat="joint_r2",
         reject_sub="low_r2", bad_high=False,
         good_band=(0.80, 1.0), render=render_joint_r2,
         title="Matched-filter joint $R^2$ gates trapezoid fit quality"),
    dict(key="residual_acf", src="trap", feat="residual_acf1",
         reject_sub=None, bad_high=True,
         good_band=(-1.0, 1.0), render=render_residual_acf,
         title="Residual autocorrelation scores unmodelled fit structure"),
    dict(key="anchor_ratio", src="trap", feat="A_anchor_ratio",
         reject_sub="A_anchor_mismatch", bad_high="two_sided",
         good_band=(0.8, 1.25), render=render_anchor_ratio,
         title="Velocity-anchor amplitude ratio rejects matched-filter misfits"),
    dict(key="out_of_lobe", src="trap", feat="out_of_lobe_residual_frac",
         reject_sub="out_of_lobe", bad_high=True,
         good_band=(0.0, 2.0), render=render_out_of_lobe,
         title="Out-of-lobe residual concentration rejects mid-ride disturbances"),
    dict(key="cruise_cv", src="trap", feat="cruise_v_cv",
         reject_sub="cruise_v_cv", bad_high=True,
         good_band=(0.05, 0.35), render=render_cruise_cv,
         title="Cruise-velocity coefficient of variation gates steady cruise"),
]


# ---------------------------------------------------------------------------
# Scan + pick
# ---------------------------------------------------------------------------

def _feat_dict(pred, src: str) -> dict | None:
    if pred is None:
        return None
    key = "features" if src == "zupt" else "quality_features"
    feats = pred.meta.get(key)
    return feats if isinstance(feats, dict) else None


def collect(spec, scan: list[dict]) -> list[dict]:
    """All segments carrying a finite value for ``spec['feat']``."""
    out = []
    for row in scan:
        pred = row["zout"] if spec["src"] == "zupt" else row["tout"]
        feats = _feat_dict(pred, spec["src"])
        if feats is None or spec["feat"] not in feats:
            continue
        v = feats[spec["feat"]]
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(v):
            continue
        out.append(dict(rec=row["rec"], pred=pred, value=v,
                        accepted=bool(pred.accepted),
                        reason=pred.reject_reason,
                        tag=f"{row['rec'].exp_name}#{row['rec'].seg_idx}"))
    return out


def pick(spec, cands: list[dict]) -> tuple[dict | None, dict | None]:
    if not cands:
        return None, None
    ov = OVERRIDES.get(spec["key"], {})
    by_tag = {c["tag"]: c for c in cands}
    bad_high = spec["bad_high"]

    # --- bad example ---
    if "bad" in ov and ov["bad"] in by_tag:
        bad = by_tag[ov["bad"]]
    else:
        rejects = []
        if spec["reject_sub"]:
            rejects = [c for c in cands if spec["reject_sub"] in c["reason"]]
        pool = rejects or cands
        if bad_high == "two_sided":
            bad = max(pool, key=lambda c: abs(c["value"] - 1.0))
        elif bad_high:
            bad = max(pool, key=lambda c: c["value"])
        else:
            bad = min(pool, key=lambda c: c["value"])

    # --- good example ---
    if "good" in ov and ov["good"] in by_tag:
        good = by_tag[ov["good"]]
    else:
        lo, hi = spec["good_band"]
        in_band = [c for c in cands
                   if c["accepted"] and lo <= c["value"] <= hi
                   and c["tag"] != bad["tag"]]
        pool = in_band or [c for c in cands if c["accepted"]
                           and c["tag"] != bad["tag"]] or \
            [c for c in cands if c["tag"] != bad["tag"]] or cands
        if bad_high == "two_sided":
            good = min(pool, key=lambda c: abs(c["value"] - 1.0))
        elif bad_high:
            good = min(pool, key=lambda c: c["value"])
        else:
            good = max(pool, key=lambda c: c["value"])
    return good, bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap how many experiments are scanned")
    ap.add_argument("--scan-only", action="store_true",
                    help="print candidate shortlists, render nothing")
    ap.add_argument("--only", nargs="*", default=None,
                    help="render only these feature keys")
    args = ap.parse_args()

    PAPER_FIG.mkdir(parents=True, exist_ok=True)

    print("loading segments ...")
    records = load_all_segments(verbose=False)
    if args.limit:
        records = records[:args.limit]
    print(f"running both estimators on {len(records)} segments ...")

    scan: list[dict] = []
    for k, rec in enumerate(records):
        scan.append(dict(rec=rec, zout=run_zupt(rec), tout=run_trap(rec)))
        if (k + 1) % 50 == 0:
            print(f"  ... {k + 1}/{len(records)}")

    specs = SPECS
    if args.only:
        specs = [s for s in SPECS if s["key"] in set(args.only)]

    n_ok = 0
    for spec in specs:
        cands = collect(spec, scan)
        good, bad = pick(spec, cands)
        print(f"\n[{spec['key']}] {len(cands)} candidates")
        if good:
            print(f"   good: {spec['feat']}={good['value']:.3f}  "
                  f"acc={good['accepted']}  {good['tag']}")
        if bad:
            print(f"   bad : {spec['feat']}={bad['value']:.3f}  "
                  f"acc={bad['accepted']}  reason='{bad['reason']}'  {bad['tag']}")
        if good is None or bad is None or good["tag"] == bad["tag"]:
            print(f"   SKIP qf_{spec['key']}.png -- no distinct good/bad pair")
            continue
        if args.scan_only:
            continue
        try:
            good_bad_fig(PAPER_FIG / f"qf_{spec['key']}.png",
                         spec["render"], good, bad, spec["title"])
            n_ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"   ERROR rendering qf_{spec['key']}.png: "
                  f"{type(exc).__name__}: {exc}")

    if not args.scan_only:
        print(f"\nrendered {n_ok}/{len(specs)} figures into {PAPER_FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
