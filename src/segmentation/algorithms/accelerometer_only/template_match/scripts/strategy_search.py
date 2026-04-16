"""Search over fit-strategy configurations until one agrees with visual
labels in run_results/labels/labels.txt on >=90% of rides across both
experimenters. Writes run_results/strategy_search_results.md.

Run:
    python3 -m src.tests.segmentations.strategy_search
"""

from __future__ import annotations

import sys
import itertools
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from src.data.loader import load_experimenter
from src.algorithms.segmentation_algorithms import (
    SEGMENT_ALGORITHM_CONFIG,
    SegmentAlgorithm,
    Segmenter,
)
from src.tests.segmentations.main import build_acc_frame, build_height_frame
from src.algorithms.segmentation_algorithms.template_match.scripts.velocity_templates import (
    fit_trapezoid, fit_parabola, ride_velocity,
)
from src.algorithms.segmentation_algorithms.template_match.scripts.compare_fit_strategies import _has_plateau


TM_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = TM_DIR / "results"
LABELS = TM_DIR / "labels" / "labels.txt"


# --------------------------- Strategy grids ---------------------------

def strat_A(fit, v, ts, cutoff=0.1):
    peak = np.max(np.abs(v))
    if peak <= 0:
        return fit["mae"]
    mask = np.abs(v) > cutoff * peak
    if not mask.any():
        return fit["mae"]
    return float(np.mean(np.abs(fit["fit"][mask] - v[mask])))

def strat_B(fit, v, ts, alpha=1.0):
    k = 4 if fit["shape"] in ("trapezoid", "triangle") else 3
    return fit["mae"] + alpha * k / len(v)

def strat_C(fit, v, ts, trap_bonus=0.1, plateau_min=0.5):
    if fit["shape"] == "parabola":
        return fit["mae"]
    ramp = fit["v_max"] / fit["a_max"]
    plateau = max((fit["t_end"] - fit["t_start"]) - 2 * ramp, 0.0)
    if plateau < plateau_min:
        return fit["mae"]
    return fit["mae"] * trap_bonus

def strat_D(fit, v, ts):
    peak_v = np.max(np.abs(v))
    peak_f = np.max(np.abs(fit["fit"]))
    if peak_v <= 0 or peak_f <= 0:
        return fit["mae"]
    return float(np.mean(np.abs(fit["fit"] / peak_f - v / peak_v)))

def strat_E(fit, v, ts, min_sec=1.5, peak_tol=0.05):
    has_pl = _has_plateau(v, ts, peak_tol=peak_tol, min_sec=min_sec)
    if has_pl:
        return fit["mae"] if fit["shape"] in ("trapezoid", "triangle") else fit["mae"] + 1.0
    return fit["mae"]

def strat_F(fit, v, ts, p_floor=0.8, weight=0.5):
    if fit["shape"] == "parabola":
        p = fit.get("p", 1.0)
        if p < p_floor:
            return fit["mae"] + weight * (p_floor - p)
        if p > 3.0:
            return fit["mae"] + 0.1 * (p - 3.0)
    return fit["mae"]


# --------------------------- Decide winner ---------------------------

def winner_of(trap, par, scorer, v, ts) -> str:
    trap_s = scorer(trap, v, ts) if trap.get("ok") else np.inf
    par_s = scorer(par, v, ts) if par.get("ok") else np.inf
    return "trapezoid" if trap_s <= par_s else "parabola"


def agreement(rides, labels, scorer) -> tuple[float, list[str]]:
    disagreements = []
    correct = 0
    for r in rides:
        lbl = labels[r["key"]]
        w = winner_of(r["trap"], r["par"], scorer, r["v"], r["ts"])
        # treat triangle as trapezoid-family for the label comparison
        w_norm = "trapezoid" if w in ("trapezoid", "triangle") else "parabola"
        if w_norm == lbl:
            correct += 1
        else:
            disagreements.append(f"{r['key']} (label={lbl}, predicted={w_norm})")
    return correct / len(rides), disagreements


# --------------------------- Combos ---------------------------

def combo_OR_traps(scorers):
    """Predict trapezoid if ANY scorer predicts trapezoid; else parabola."""
    def _scorer_wrapper(fit, v, ts):
        raise NotImplementedError
    def decide(trap, par, v, ts):
        for s in scorers:
            if winner_of(trap, par, s, v, ts) in ("trapezoid", "triangle"):
                return "trapezoid"
        return "parabola"
    return decide


def combo_AND_traps(scorers):
    """Predict trapezoid only if ALL scorers predict trapezoid."""
    def decide(trap, par, v, ts):
        for s in scorers:
            if winner_of(trap, par, s, v, ts) not in ("trapezoid", "triangle"):
                return "parabola"
        return "trapezoid"
    return decide


def agreement_decider(rides, labels, decide) -> tuple[float, list[str]]:
    disagreements = []
    correct = 0
    for r in rides:
        lbl = labels[r["key"]]
        w = decide(r["trap"], r["par"], r["v"], r["ts"])
        if w == lbl:
            correct += 1
        else:
            disagreements.append(f"{r['key']} (label={lbl}, predicted={w})")
    return correct / len(rides), disagreements


# --------------------------- Data loading ---------------------------

def load_rides():
    labels: dict[str, str] = {}
    for line in LABELS.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, val = line.split(":")
        labels[key.strip()] = val.strip().lower()

    rides = []
    cfg = SEGMENT_ALGORITHM_CONFIG(algorithm=SegmentAlgorithm.PRESSURE_FILTER)
    for name in ("oria", "roy_turgman"):
        data = load_experimenter(name)
        t0_ms = float(data["ACC"]["timestamp_ms"].iloc[0])
        acc_frame = build_acc_frame(data["ACC"], t0_ms)
        height_frame = build_height_frame(data["PRS"], t0_ms)
        segments = Segmenter(cfg).detect(height_frame)
        for i, row in segments.iterrows():
            ts, v, _ = ride_velocity(acc_frame, row["start_ci"][0], row["end_ci"][1])
            if v.size == 0:
                continue
            key = f"{name}-{int(i)}"
            if key not in labels:
                continue
            rides.append({
                "key": key, "name": name, "idx": int(i),
                "ts": ts, "v": v, "type": row["type"],
                "t_start": row["start_ci"][0], "t_end": row["end_ci"][1],
                "trap": fit_trapezoid(ts, v), "par": fit_parabola(ts, v),
            })
    return rides, labels


# --------------------------- Main search ---------------------------

def main() -> None:
    rides, labels = load_rides()
    print(f"Loaded {len(rides)} rides with labels")
    trap_count = sum(1 for v in labels.values() if v == "trapezoid")
    par_count = sum(1 for v in labels.values() if v == "parabola")
    print(f"Labels: trapezoid={trap_count}  parabola={par_count}")

    results = []  # (agreement, name, disagreements)

    # A: active-mask cutoff
    for c in (0.05, 0.10, 0.20, 0.30):
        ag, dis = agreement(rides, labels, lambda f, v, ts, c=c: strat_A(f, v, ts, c))
        results.append((ag, f"A_active_mask(cut={c})", dis))

    # B: AIC
    for alpha in (0.5, 1.0, 2.0, 5.0):
        ag, dis = agreement(rides, labels, lambda f, v, ts, a=alpha: strat_B(f, v, ts, a))
        results.append((ag, f"B_aic(alpha={alpha})", dis))

    # C: trapezoid-plateau preference
    for pm in (0.3, 0.5, 1.0, 1.5):
        for tb in (0.1, 0.3, 0.5):
            ag, dis = agreement(rides, labels,
                lambda f, v, ts, pm=pm, tb=tb: strat_C(f, v, ts, plateau_min=pm, trap_bonus=tb))
            results.append((ag, f"C_trap_plateau(min={pm},bonus={tb})", dis))

    # D: normalized shape
    ag, dis = agreement(rides, labels, lambda f, v, ts: strat_D(f, v, ts))
    results.append((ag, "D_normalized_shape", dis))

    # E: plateau gate — wider grid including very loose
    for ms in (0.3, 0.5, 0.8, 1.2, 1.5, 2.0, 2.5):
        for pt in (0.03, 0.05, 0.08, 0.10, 0.15, 0.20):
            ag, dis = agreement(rides, labels,
                lambda f, v, ts, ms=ms, pt=pt: strat_E(f, v, ts, min_sec=ms, peak_tol=pt))
            results.append((ag, f"E_plateau_gate(min_sec={ms},peak_tol={pt})", dis))

    # F: bounded p
    for pf in (0.6, 0.7, 0.8, 0.9, 1.0):
        for w in (0.3, 0.5, 1.0, 2.0):
            ag, dis = agreement(rides, labels,
                lambda f, v, ts, pf=pf, w=w: strat_F(f, v, ts, p_floor=pf, weight=w))
            results.append((ag, f"F_bounded_p(floor={pf},weight={w})", dis))

    # Top single strategies
    results.sort(key=lambda r: -r[0])
    print(f"\nTop 10 single strategies:")
    for ag, nm, _ in results[:10]:
        print(f"  {ag:6.1%}  {nm}")

    # Combos: OR/AND across top-performing strategies
    combo_candidates = {
        "B2":    lambda f, v, ts: strat_B(f, v, ts, alpha=2.0),
        "B5":    lambda f, v, ts: strat_B(f, v, ts, alpha=5.0),
        "E_tight":  lambda f, v, ts: strat_E(f, v, ts, min_sec=1.5, peak_tol=0.05),
        "E_loose":  lambda f, v, ts: strat_E(f, v, ts, min_sec=0.5, peak_tol=0.10),
        "E_vloose": lambda f, v, ts: strat_E(f, v, ts, min_sec=0.3, peak_tol=0.15),
        "A":     lambda f, v, ts: strat_A(f, v, ts, cutoff=0.10),
        "D":     lambda f, v, ts: strat_D(f, v, ts),
    }
    for (n1, s1), (n2, s2) in itertools.combinations(combo_candidates.items(), 2):
        ag, dis = agreement_decider(rides, labels, combo_OR_traps([s1, s2]))
        results.append((ag, f"OR({n1},{n2})", dis))
        ag, dis = agreement_decider(rides, labels, combo_AND_traps([s1, s2]))
        results.append((ag, f"AND({n1},{n2})", dis))
    # 3-way OR combos
    for (n1, s1), (n2, s2), (n3, s3) in itertools.combinations(combo_candidates.items(), 3):
        ag, dis = agreement_decider(rides, labels, combo_OR_traps([s1, s2, s3]))
        results.append((ag, f"OR3({n1},{n2},{n3})", dis))

    # Hybrid decision rule: trapezoid if plateau detected on raw v, else MAE/AIC
    def hybrid_rule(ms, pt, alpha=2.0):
        def decide(trap, par, v, ts):
            if _has_plateau(v, ts, peak_tol=pt, min_sec=ms):
                return "trapezoid"
            w = winner_of(trap, par, lambda f, vv, tt: strat_B(f, vv, tt, alpha), v, ts)
            return "trapezoid" if w in ("trapezoid", "triangle") else "parabola"
        return decide
    for ms in (0.3, 0.5, 0.8, 1.0, 1.2, 1.5):
        for pt in (0.05, 0.08, 0.10, 0.15):
            for alpha in (1.0, 2.0, 5.0):
                dec = hybrid_rule(ms, pt, alpha)
                ag, dis = agreement_decider(rides, labels, dec)
                results.append((ag, f"HYBRID(plateau:ms={ms},pt={pt}) OR B(alpha={alpha})", dis))

    results.sort(key=lambda r: -r[0])
    best_ag, best_nm, best_dis = results[0]
    print(f"\nBest overall: {best_ag:.1%}  {best_nm}")
    print(f"Disagreements ({len(best_dis)}):")
    for d in best_dis:
        print(f"  - {d}")

    # Write report
    md_lines = [
        "# Strategy Search Results",
        "",
        f"- Total rides: {len(rides)}",
        f"- Labels: trapezoid={trap_count}, parabola={par_count}",
        f"- **Best: {best_ag:.1%} — `{best_nm}`**",
        "",
        "## Top 15 configurations",
        "",
        "| Rank | Agreement | Strategy | #Disagree |",
        "|---|---|---|---|",
    ]
    for i, (ag, nm, dis) in enumerate(results[:15], 1):
        md_lines.append(f"| {i} | {ag:.1%} | `{nm}` | {len(dis)} |")
    md_lines += ["", f"## Disagreements for best (`{best_nm}`)", ""]
    for d in best_dis:
        md_lines.append(f"- {d}")
    (OUT_DIR / "strategy_search_results.md").write_text("\n".join(md_lines) + "\n")

    # Render best-strategy plot
    render_best(rides, labels, best_nm, best_dis)

    print(f"\nReport: {OUT_DIR/'strategy_search_results.md'}")


def render_best(rides, labels, best_name: str, disagreements: list[str]) -> None:
    dis_keys = set(d.split(" ")[0] for d in disagreements)
    n = len(rides)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 2.3 * nrows), squeeze=False)
    for k, r in enumerate(rides):
        ax = axes[k // ncols][k % ncols]
        color = "tab:blue" if r["type"] == "up" else "tab:red"
        ax.plot(r["ts"] - r["ts"][0], r["v"], color=color, lw=1.4)
        if r["trap"].get("ok"):
            ax.plot(r["ts"] - r["ts"][0], r["trap"]["fit"], color="black", lw=0.9)
        if r["par"].get("ok"):
            ax.plot(r["ts"] - r["ts"][0], r["par"]["fit"], color="purple", lw=0.9, linestyle="--")
        lbl = labels[r["key"]]
        is_wrong = r["key"] in dis_keys
        border = "red" if is_wrong else "green"
        for spine in ax.spines.values():
            spine.set_edgecolor(border); spine.set_linewidth(1.5)
        ax.set_title(f"{r['key']} lbl={lbl}", fontsize=7)
        ax.axhline(0, color="k", lw=0.3)
        ax.tick_params(labelsize=6)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(f"Best strategy: {best_name}   (green=match, red=disagree)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "best_strategy.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
