"""Per-bin conditional-coverage analysis for the prediction CI model.

Split-conformal guarantees *marginal* coverage: ``Pr(|err| <= k*sigma) >=
1 - alpha`` over the exchangeable test distribution. It does NOT
guarantee *conditional* coverage, e.g. ``Pr(covered | |Δh| in bin) >=
1 - alpha``. Bin-by-bin failures (classically the 0-3 m bin) show up
here.

For each |Δh| bin this tool reports:

* ``n``         — number of quality-filter-accepted segments.
* ``cov``       — empirical coverage at the reported CI half-width.
* ``mae``       — mean absolute error.
* ``med_w``     — median CI half-width.
* ``q_bin``     — empirical ``(1 - alpha)`` quantile of
  ``|err|/sigma`` *inside the bin*.
* ``q_global``  — the global conformal multiplier (one scalar).
* ``ratio``     — ``q_bin / q_global``. Values within ~[0.8, 1.25]
  mean the σ model's scale is consistent across this bin;
  ``ratio > 1.25`` signals the σ model under-counts in this bin and
  conditional coverage is being dragged up only by the margin from
  other bins.

Two output modes:
* ``--latex`` emits a table in ``docs/latex/figures/prediction/
  bin_conditional_<mode>.tex``.
* default text mode prints the table to stdout.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BIN_EDGES = [0.0, 3.0, 6.0, 12.0, 24.0, 60.0]
DEFAULT_ALPHA = 0.10

REPO = Path(__file__).resolve().parents[3]
DEFAULT_INPUTS = {
    "train": REPO / "src/data/structuredData/test_results/prediction/train/predictions_trapezoid_train.csv",
    "test":  REPO / "src/data/structuredData/test_results/prediction/test/predictions_trapezoid_test.csv",
}
DEFAULT_LATEX_OUT = REPO / "docs/latex/figures/prediction"


def per_bin_conformal_scores(df: pd.DataFrame, bin_edges=BIN_EDGES,
                              alpha: float = DEFAULT_ALPHA) -> pd.DataFrame:
    accepted = df[df["accepted"].astype(bool)].copy()
    if accepted.empty:
        return pd.DataFrame()
    accepted["bin_lo"] = pd.cut(
        accepted["true_dh"].abs(), bins=bin_edges,
        labels=[f"{lo:.0f}-{hi:.0f}" for lo, hi in zip(bin_edges, bin_edges[1:])],
        include_lowest=True,
    )
    # Non-conformity per segment
    sigma = accepted["theoretical_sigma"].replace(0.0, np.nan)
    score = (accepted["pred_dh"] - accepted["true_dh"]).abs() / sigma
    accepted["score"] = score
    q_global = float(np.quantile(score.dropna(), 1.0 - alpha))
    has_mode = "mode" in accepted.columns
    rows = []
    for bin_label, sub in accepted.groupby("bin_lo", observed=True):
        n = len(sub)
        cov = float(sub["covered"].mean()) if "covered" in sub.columns else float("nan")
        mae = float((sub["pred_dh"] - sub["true_dh"]).abs().mean())
        med_w = float(sub["ci_half_width"].median())
        bin_scores = sub["score"].dropna()
        q_bin = float(np.quantile(bin_scores, 1.0 - alpha)) if len(bin_scores) >= 5 else float("nan")
        ratio = q_bin / q_global if q_global > 0 else float("nan")
        row = {
            "bin": str(bin_label),
            "n": int(n),
            "cov": cov,
            "mae": mae,
            "med_w": med_w,
            "q_bin": q_bin,
            "q_global": q_global,
            "ratio": ratio,
        }
        if has_mode:
            row["n_joined"] = int((sub["mode"] == "joined").sum())
            row["n_pair"] = int((sub["mode"] == "pair").sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _format_latex(df: pd.DataFrame, mode: str) -> str:
    """Return a compilable LaTeX tabular block (no table env)."""
    lines = [
        "% Auto-generated bin-conditional coverage (" + mode + ")",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"\textbf{Bin (m)} & $n$ & Cov. & MAE (m) & Med $\tilde{w}$ (m) & $q_\mathrm{bin}$ & $q_\mathrm{global}$ & Ratio \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        cov = "--" if pd.isna(r["cov"]) else f"{100*r['cov']:.1f}\\%"
        q_bin = "--" if pd.isna(r["q_bin"]) else f"{r['q_bin']:.2f}"
        ratio = "--" if pd.isna(r["ratio"]) else f"{r['ratio']:.2f}"
        lines.append(
            f"{r['bin']} & {r['n']} & {cov} & {r['mae']:.2f} & "
            f"{r['med_w']:.2f} & {q_bin} & {r['q_global']:.2f} & {ratio} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["train", "test"], default="test")
    ap.add_argument("--predictions", type=Path, default=None,
                    help="Override path to predictions_trapezoid_*.csv")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                    help="Coverage target 1 - alpha (default: 0.10)")
    ap.add_argument("--latex", action="store_true",
                    help="Write a LaTeX tabular to docs/latex/figures/prediction/")
    args = ap.parse_args()

    src = args.predictions or DEFAULT_INPUTS[args.mode]
    df = pd.read_csv(src)
    out = per_bin_conformal_scores(df, alpha=args.alpha)
    if out.empty:
        print("No accepted segments, nothing to report.")
        return

    print(f"\nBin-conditional analysis ({args.mode})  α={args.alpha}:")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    if args.latex:
        DEFAULT_LATEX_OUT.mkdir(parents=True, exist_ok=True)
        tex_path = DEFAULT_LATEX_OUT / f"bin_conditional_{args.mode}.tex"
        tex_path.write_text(_format_latex(out, args.mode), encoding="utf-8")
        print(f"\nLaTeX table written to {tex_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
