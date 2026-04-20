"""Plot fitted trapezoids per experiment and report within-segment similarity.

Runs over **every** variant folder under
``template_match/labels/fit_elevator_paramater/`` (currently
``basicTrapezoidGrid`` and ``basicTrapezoidGridWithConstraint``) and writes
two PNGs directly inside each variant folder:

* ``_trapezoid_similarity_grid.png`` — per experiment overlay every
  fitted lobe on a shape-centered (``t_c = 0``) time axis; annotate each
  panel with the average pairwise RMS distance between its trapezoid
  waveforms. ``up`` and ``down`` lobes use the usual green/red. Here a
  "segment" = one experiment's JSON (same elevator + phone); low within-
  segment avg pair RMS ⇒ trapezoids are consistent across rides.

* ``_trapezoid_lobe_symmetry_grid.png`` — for every ride that has both
  lobes fit, overlay ``|lobe1|`` (blue) vs ``|lobe2|`` (orange) on the
  centered axis. The two lobes of one ride have opposite signs by
  construction (take-off vs landing); taking ``abs`` should make them
  coincide. Each panel annotates the per-ride RMS between ``|lobe1|``
  and ``|lobe2|`` (median + mean over that experiment's rides). For the
  ``basicTrapezoidGridWithConstraint`` variant the fitter already forces
  the two lobes to share ``|A|, W, f``, so the per-ride RMS collapses
  to zero — the plot is still useful as a visual sanity check.

Run:
    venv/bin/python src/segmentation/algorithms/accelerometer_only/\
template_match/scripts/plot_trapezoid_similarity.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

VARIANTS_ROOT = (
    Path(__file__).resolve().parents[1]
    / "labels" / "fit_elevator_paramater"
)

TYPE_COLORS = {"up": "#27ae60", "down": "#e74c3c"}
LOBE_COLORS = {"lobe1": "#2980b9", "lobe2": "#e67e22"}

# Shared centered time axis so different (A, W, f) shapes can be compared
# sample-wise. ±3.5 s comfortably covers the 0.4–3.0 s half-width grid.
T_GRID = np.linspace(-3.5, 3.5, 701)


def _trapezoid(t: np.ndarray, t_c: float, W: float, frac_flat: float) -> np.ndarray:
    frac_flat = max(0.0, min(1.0, float(frac_flat)))
    W = max(1e-6, float(W))
    flat_half = frac_flat * W
    ramp_width = W - flat_half + 1e-9
    dt = np.abs(t - t_c)
    return np.where(
        dt <= flat_half, 1.0,
        np.where(dt < W, (W - dt) / ramp_width, 0.0),
    )


def _waveform(lobe: dict) -> np.ndarray | None:
    if lobe is None or lobe.get("t_c") is None:
        return None
    return float(lobe["a_peak"]) * _trapezoid(
        T_GRID, 0.0, float(lobe["half_width_s"]), float(lobe["frac_flat"]),
    )


def _collect(exp_dir: Path) -> dict[str, list[np.ndarray]]:
    jf = exp_dir / "parameters.json"
    if not jf.exists():
        return {"up": [], "down": []}
    data = json.loads(jf.read_text())
    out: dict[str, list[np.ndarray]] = {"up": [], "down": []}
    for ride in data:
        rt = ride.get("ride_type")
        if rt not in out:
            continue
        for k in ("lobe1", "lobe2"):
            w = _waveform(ride.get(k))
            if w is not None:
                out[rt].append(w)
    return out


def _collect_lobe_pairs(
    exp_dir: Path,
) -> list[tuple[int, str, np.ndarray, np.ndarray]]:
    """Rides whose lobe1 and lobe2 are both fit — return (|l1|, |l2|) pairs.

    Each entry is ``(ride_index, ride_type, abs_lobe1_wave, abs_lobe2_wave)``
    on the common centered time axis ``T_GRID``.
    """
    jf = exp_dir / "parameters.json"
    if not jf.exists():
        return []
    data = json.loads(jf.read_text())
    pairs: list[tuple[int, str, np.ndarray, np.ndarray]] = []
    for ride in data:
        w1 = _waveform(ride.get("lobe1"))
        w2 = _waveform(ride.get("lobe2"))
        if w1 is None or w2 is None:
            continue
        pairs.append((
            int(ride.get("index", -1)),
            str(ride.get("ride_type", "?")),
            np.abs(w1),
            np.abs(w2),
        ))
    return pairs


def _rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _avg_pair_rms(waves: list[np.ndarray]) -> float | None:
    if len(waves) < 2:
        return None
    total = 0.0
    cnt = 0
    for i in range(len(waves)):
        for j in range(i + 1, len(waves)):
            total += _rms(waves[i], waves[j])
            cnt += 1
    return total / cnt


def _draw_panel(ax: plt.Axes, name: str, coll: dict[str, list[np.ndarray]]) -> None:
    ax.axhline(0.0, color="gray", lw=0.4, ls="--", alpha=0.5)
    lines: list[str] = []
    for rt in ("up", "down"):
        waves = coll[rt]
        if not waves:
            continue
        color = TYPE_COLORS[rt]
        for w in waves:
            ax.plot(T_GRID, w, color=color, lw=0.9, alpha=0.55)
        d = _avg_pair_rms(waves)
        if d is None:
            lines.append(f"{rt}: n={len(waves)}  (need ≥2 for RMS)")
        else:
            lines.append(f"{rt}: n={len(waves)}  avg pair RMS={d:.2f} m/s²")
    ax.set_title(name, fontsize=7.5)
    ax.set_xlabel("t (s, trapezoid-centered)")
    ax.set_ylabel("a (m/s²)")
    ax.grid(True, alpha=0.25)
    if lines:
        ax.text(
            0.02, 0.97, "\n".join(lines),
            transform=ax.transAxes, ha="left", va="top",
            fontsize=7, family="monospace",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
        )


def _draw_symmetry_panel(
    ax: plt.Axes, name: str,
    pairs: list[tuple[int, str, np.ndarray, np.ndarray]],
) -> float | None:
    """Overlay every ride's |lobe1| vs |lobe2|; return mean per-ride RMS."""
    ax.axhline(0.0, color="gray", lw=0.4, ls="--", alpha=0.5)
    if not pairs:
        ax.text(
            0.5, 0.5, "no rides with both lobes",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=9, color="#7f8c8d", style="italic",
        )
        ax.set_title(name, fontsize=7.5)
        ax.set_xlabel("t (s, trapezoid-centered)")
        ax.set_ylabel("|a| (m/s²)")
        ax.grid(True, alpha=0.25)
        return None

    rms_values: list[float] = []
    for _idx, _rt, abs_w1, abs_w2 in pairs:
        ax.plot(T_GRID, abs_w1, color=LOBE_COLORS["lobe1"], lw=0.9, alpha=0.55)
        ax.plot(T_GRID, abs_w2, color=LOBE_COLORS["lobe2"], lw=0.9, alpha=0.55)
        rms_values.append(_rms(abs_w1, abs_w2))

    arr = np.array(rms_values)
    lines = [
        f"n_rides={len(pairs)}",
        f"median per-ride RMS = {np.median(arr):.2f} m/s²",
        f"mean   per-ride RMS = {arr.mean():.2f} m/s²",
    ]
    ax.text(
        0.02, 0.97, "\n".join(lines),
        transform=ax.transAxes, ha="left", va="top",
        fontsize=7, family="monospace",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
    )
    ax.set_title(name, fontsize=7.5)
    ax.set_xlabel("t (s, trapezoid-centered)")
    ax.set_ylabel("|a| (m/s²)")
    ax.grid(True, alpha=0.25)
    return float(arr.mean())


def _render_similarity_grid(
    entries: list[tuple[str, dict[str, list[np.ndarray]]]],
    out_path: Path, variant_name: str,
) -> None:
    cols = 3
    rows = math.ceil(len(entries) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 3.5 * rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (name, coll) in zip(axes, entries):
        _draw_panel(ax, name, coll)
    for ax in axes[len(entries):]:
        ax.axis("off")

    per_segment_rms: list[float] = []
    for _, coll in entries:
        for rt in ("up", "down"):
            d = _avg_pair_rms(coll[rt])
            if d is not None:
                per_segment_rms.append(d)
    if per_segment_rms:
        arr = np.array(per_segment_rms)
        summary = (
            f"{len(per_segment_rms)} segment-type groups with ≥2 trapezoids — "
            f"median avg-pair RMS = {np.median(arr):.2f} m/s², "
            f"p25={np.percentile(arr, 25):.2f}, p75={np.percentile(arr, 75):.2f}"
        )
    else:
        summary = "no segment had ≥2 trapezoids for pairwise RMS."

    fig.suptitle(
        f"[{variant_name}] Fitted trapezoids per experiment — "
        f"within-segment shape consistency\n" + summary,
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    print(f"  {summary}")


def _render_symmetry_grid(
    pairs_by_exp: list[tuple[str, list[tuple[int, str, np.ndarray, np.ndarray]]]],
    out_path: Path, variant_name: str,
) -> None:
    cols = 3
    rows = math.ceil(len(pairs_by_exp) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 3.5 * rows))
    axes = np.atleast_1d(axes).flatten()

    all_per_ride_rms: list[float] = []
    for ax, (name, pairs) in zip(axes, pairs_by_exp):
        _draw_symmetry_panel(ax, name, pairs)
        for _i, _rt, a1, a2 in pairs:
            all_per_ride_rms.append(_rms(a1, a2))
    for ax in axes[len(pairs_by_exp):]:
        ax.axis("off")

    legend_handles = [
        plt.Line2D([0], [0], color=LOBE_COLORS["lobe1"], lw=2, label="|lobe1| (take-off)"),
        plt.Line2D([0], [0], color=LOBE_COLORS["lobe2"], lw=2, label="|lobe2| (landing)"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", ncol=2, fontsize=9,
               frameon=False, bbox_to_anchor=(0.995, 0.995))

    if all_per_ride_rms:
        arr = np.array(all_per_ride_rms)
        summary = (
            f"{len(arr)} rides with both lobes — "
            f"median per-ride |l1|vs|l2| RMS = {np.median(arr):.2f} m/s², "
            f"mean={arr.mean():.2f}, "
            f"p25={np.percentile(arr, 25):.2f}, p75={np.percentile(arr, 75):.2f}"
        )
    else:
        summary = "no rides had both lobes fit; nothing to compare."

    fig.suptitle(
        f"[{variant_name}] Per-ride lobe symmetry — |lobe1| vs |lobe2| after abs\n"
        + summary,
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    print(f"  {summary}")


def _process_variant(variant_dir: Path) -> bool:
    entries: list[tuple[str, dict[str, list[np.ndarray]]]] = []
    pairs_by_exp: list[tuple[str, list[tuple[int, str, np.ndarray, np.ndarray]]]] = []
    for d in sorted(p for p in variant_dir.iterdir() if p.is_dir()):
        coll = _collect(d)
        pairs = _collect_lobe_pairs(d)
        if any(coll.values()):
            entries.append((d.name, coll))
        if pairs or coll.values():
            pairs_by_exp.append((d.name, pairs))

    if not entries:
        print(f"[skip] {variant_dir.name}: no parameters.json files")
        return False

    print(f"[{variant_dir.name}]")
    _render_similarity_grid(
        entries, variant_dir / "_trapezoid_similarity_grid.png", variant_dir.name,
    )
    _render_symmetry_grid(
        pairs_by_exp, variant_dir / "_trapezoid_lobe_symmetry_grid.png", variant_dir.name,
    )
    return True


def main() -> int:
    if not VARIANTS_ROOT.exists():
        print(f"no fits root at {VARIANTS_ROOT}", file=sys.stderr)
        return 1

    variant_dirs = sorted(
        p for p in VARIANTS_ROOT.iterdir()
        if p.is_dir() and not p.name.startswith("_")
    )
    if not variant_dirs:
        print(f"no variant subfolders under {VARIANTS_ROOT}", file=sys.stderr)
        return 1

    any_ok = False
    for vd in variant_dirs:
        any_ok |= _process_variant(vd)
    return 0 if any_ok else 1


if __name__ == "__main__":
    sys.exit(main())
