"""Tests for the new structuredData loader entrypoints.

Run as a standalone script (no pytest required):
    venv/bin/python -m src.tests.data.test_loader

Or with pytest if installed:
    venv/bin/python -m pytest src/tests/data/test_loader.py -v
"""

from __future__ import annotations

import pickle
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import pandas as pd

# Allow running as a script
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data.loader import (
    ExperimentPipeline,
    FOR_BAROMETER_PLOT_FILENAME,
    PIPELINE_CACHE_FILENAME,
    STRUCTURED_ROOT,
    _find_sensor_log,
    _parse_sensor_log,
    getExperimentPipelineData,
    getExperimentRawParsed,
)

EXP_NO_SECONDARY = STRUCTURED_ROOT / "eyalyakir_milleniumHotel_SamsungSM-S911B_15-04-2026_exp1"
EXP_WITH_SECONDARY_A = STRUCTURED_ROOT / "UriyaCohenEliya_milleniumHotel_Xiaomi22101320I_15-04-2026_exp1"
EXP_WITH_SECONDARY_B = STRUCTURED_ROOT / "UriyaCohenEliya_acroBuilding_SamsungSM-A235F_15-04-2026_exp4"

ALL_EXPS = [EXP_NO_SECONDARY, EXP_WITH_SECONDARY_A, EXP_WITH_SECONDARY_B]
EXPS_WITH_SECONDARY = [EXP_WITH_SECONDARY_A, EXP_WITH_SECONDARY_B]


# --------------------------------------------------------------------------
# Assertions
# --------------------------------------------------------------------------

def check_raw_parsed_has_required_sensors(exp_path: Path) -> None:
    data = getExperimentRawParsed(exp_path)
    assert "ACC" in data, f"{exp_path.name}: ACC missing"
    assert "PRS" in data, f"{exp_path.name}: PRS missing"
    assert not data["PRS"].empty, f"{exp_path.name}: PRS is empty"
    assert "timestamp_ms" in data["PRS"].columns
    assert "GT_height_m" in data["PRS"].columns
    assert "timestamp_ms" in data["ACC"].columns


def check_raw_parsed_swaps_prs_when_forBarometer_exists(exp_path: Path) -> None:
    primary_frames = _parse_sensor_log(_find_sensor_log(exp_path))
    merged = getExperimentRawParsed(exp_path)
    assert "PRS" in merged and not merged["PRS"].empty, (
        f"{exp_path.name}: merged PRS empty"
    )
    primary_prs_len = len(primary_frames.get("PRS", pd.DataFrame()))
    merged_prs_len = len(merged["PRS"])
    assert (primary_prs_len == 0) or (primary_prs_len != merged_prs_len), (
        f"{exp_path.name}: expected PRS to come from secondary "
        f"(primary={primary_prs_len}, merged={merged_prs_len})"
    )


def check_forBarometer_plot_saved(exp_path: Path) -> None:
    plot_path = exp_path / FOR_BAROMETER_PLOT_FILENAME
    if plot_path.exists():
        plot_path.unlink()
    getExperimentRawParsed(exp_path)
    assert plot_path.exists(), f"{exp_path.name}: {FOR_BAROMETER_PLOT_FILENAME} not created"
    assert plot_path.stat().st_size > 1000, (
        f"{exp_path.name}: plot file suspiciously small ({plot_path.stat().st_size} bytes)"
    )


def check_pipeline_contract(exp_path: Path) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="pipeline_test_"))
    try:
        tmp_exp = tmp / exp_path.name
        shutil.copytree(exp_path, tmp_exp)
        p = getExperimentPipelineData(tmp_exp, use_cache=False)
        assert isinstance(p, ExperimentPipeline)
        assert hasattr(p, "data") and hasattr(p, "gt") and hasattr(p, "metaData")
        assert set(p.gt.columns) == {"start_ms", "end_ms", "type"}, (
            f"{exp_path.name}: gt columns = {list(p.gt.columns)}"
        )
        assert p.gt["type"].isin({"up", "down", "outside"}).all()
        assert len(p.gt) >= 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_pipeline_gt_is_full_timeline(exp_path: Path) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="pipeline_test_"))
    try:
        tmp_exp = tmp / exp_path.name
        shutil.copytree(exp_path, tmp_exp)
        p = getExperimentPipelineData(tmp_exp, use_cache=False)

        gt = p.gt.sort_values("start_ms").reset_index(drop=True)
        # Monotonic, no gaps
        for i in range(1, len(gt)):
            assert int(gt.loc[i, "start_ms"]) == int(gt.loc[i - 1, "end_ms"]), (
                f"{exp_path.name}: gap between row {i - 1} and {i}"
            )
        # No consecutive outside rows
        types = gt["type"].tolist()
        for i in range(1, len(types)):
            assert not (types[i] == "outside" and types[i - 1] == "outside"), (
                f"{exp_path.name}: consecutive 'outside' at index {i}"
            )
        # Covers the PRS window
        prs = p.data["PRS"]
        assert int(gt.iloc[0]["start_ms"]) == int(prs["timestamp_ms"].iloc[0])
        assert int(gt.iloc[-1]["end_ms"]) == int(prs["timestamp_ms"].iloc[-1])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_pipeline_iteration(exp_path: Path) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="pipeline_test_"))
    try:
        tmp_exp = tmp / exp_path.name
        shutil.copytree(exp_path, tmp_exp)
        p = getExperimentPipelineData(tmp_exp, use_cache=False)

        count = 0
        for slice_dict, row, meta in p:
            count += 1
            assert "ACC" in slice_dict, f"{exp_path.name}: ACC missing from slice"
            assert "PRS" in slice_dict, f"{exp_path.name}: PRS missing from slice"
            assert row["type"] in {"up", "down", "outside"}
            assert meta is p.metaData
            s, e = int(row["start_ms"]), int(row["end_ms"])
            acc = slice_dict["ACC"]
            if len(acc):
                ts = acc["timestamp_ms"]
                assert int(ts.min()) >= s
                assert int(ts.max()) < e
        assert count == len(p.gt), f"{exp_path.name}: iterated {count}, expected {len(p.gt)}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_metadata_parsed(exp_path: Path) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="pipeline_test_"))
    try:
        tmp_exp = tmp / exp_path.name
        shutil.copytree(exp_path, tmp_exp)
        p = getExperimentPipelineData(tmp_exp, use_cache=False)
        for key in ("Name", "Phone", "Date", "Location"):
            assert key in p.metaData, f"{exp_path.name}: missing metadata key {key}"
        assert "Description" in p.metaData, (
            f"{exp_path.name}: Description missing from metaData"
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_pipeline_cache_roundtrip(exp_path: Path) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="pipeline_test_"))
    try:
        tmp_exp = tmp / exp_path.name
        shutil.copytree(exp_path, tmp_exp)
        cache_file = tmp_exp / PIPELINE_CACHE_FILENAME
        if cache_file.exists():
            cache_file.unlink()

        p1 = getExperimentPipelineData(tmp_exp, use_cache=True)
        assert cache_file.exists(), "cache file not created"
        mtime1 = cache_file.stat().st_mtime

        p2 = getExperimentPipelineData(tmp_exp, use_cache=True)
        assert isinstance(p2, ExperimentPipeline)
        pd.testing.assert_frame_equal(p1.gt, p2.gt)
        # Second call should not rewrite the cache (cache hit path)
        mtime2 = cache_file.stat().st_mtime
        assert mtime1 == mtime2, "cache was rewritten on hit"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_pipeline_cache_corrupt_falls_back(exp_path: Path) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="pipeline_test_"))
    try:
        tmp_exp = tmp / exp_path.name
        shutil.copytree(exp_path, tmp_exp)
        cache_file = tmp_exp / PIPELINE_CACHE_FILENAME
        cache_file.write_bytes(b"this is not a valid pickle file \x00\xff")

        p = getExperimentPipelineData(tmp_exp, use_cache=True)
        assert isinstance(p, ExperimentPipeline)
        # After rebuild, the cache should be a valid pickle again
        with cache_file.open("rb") as f:
            reloaded = pickle.load(f)
        assert isinstance(reloaded, ExperimentPipeline)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------
# Pytest entry points (if pytest is installed)
# --------------------------------------------------------------------------

try:
    import pytest  # type: ignore

    @pytest.mark.parametrize("exp_path", ALL_EXPS, ids=lambda p: p.name)
    def test_raw_parsed_has_required_sensors(exp_path):
        check_raw_parsed_has_required_sensors(exp_path)

    @pytest.mark.parametrize("exp_path", EXPS_WITH_SECONDARY, ids=lambda p: p.name)
    def test_raw_parsed_swaps_prs(exp_path):
        check_raw_parsed_swaps_prs_when_forBarometer_exists(exp_path)

    @pytest.mark.parametrize("exp_path", EXPS_WITH_SECONDARY, ids=lambda p: p.name)
    def test_forBarometer_plot_saved(exp_path):
        check_forBarometer_plot_saved(exp_path)

    @pytest.mark.parametrize("exp_path", ALL_EXPS, ids=lambda p: p.name)
    def test_pipeline_contract(exp_path):
        check_pipeline_contract(exp_path)

    @pytest.mark.parametrize("exp_path", ALL_EXPS, ids=lambda p: p.name)
    def test_pipeline_gt_is_full_timeline(exp_path):
        check_pipeline_gt_is_full_timeline(exp_path)

    @pytest.mark.parametrize("exp_path", ALL_EXPS, ids=lambda p: p.name)
    def test_pipeline_iteration(exp_path):
        check_pipeline_iteration(exp_path)

    @pytest.mark.parametrize("exp_path", ALL_EXPS, ids=lambda p: p.name)
    def test_metadata_parsed(exp_path):
        check_metadata_parsed(exp_path)

    @pytest.mark.parametrize("exp_path", [EXP_NO_SECONDARY], ids=lambda p: p.name)
    def test_pipeline_cache_roundtrip(exp_path):
        check_pipeline_cache_roundtrip(exp_path)

    @pytest.mark.parametrize("exp_path", [EXP_NO_SECONDARY], ids=lambda p: p.name)
    def test_pipeline_cache_corrupt_falls_back(exp_path):
        check_pipeline_cache_corrupt_falls_back(exp_path)

except ImportError:
    pass


# --------------------------------------------------------------------------
# Standalone runner (no pytest needed)
# --------------------------------------------------------------------------

def _run_case(label: str, fn, *args) -> bool:
    try:
        fn(*args)
        print(f"  PASS  {label}")
        return True
    except AssertionError as e:
        print(f"  FAIL  {label}: {e}")
        return False
    except Exception as e:
        print(f"  ERROR {label}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def main() -> int:
    cases: list[tuple[str, object, tuple]] = []

    for exp in ALL_EXPS:
        cases.append((f"raw_parsed_sensors [{exp.name}]",
                      check_raw_parsed_has_required_sensors, (exp,)))

    for exp in EXPS_WITH_SECONDARY:
        cases.append((f"prs_swapped [{exp.name}]",
                      check_raw_parsed_swaps_prs_when_forBarometer_exists, (exp,)))
        cases.append((f"forBarometer_plot [{exp.name}]",
                      check_forBarometer_plot_saved, (exp,)))

    for exp in ALL_EXPS:
        cases.append((f"pipeline_contract [{exp.name}]",
                      check_pipeline_contract, (exp,)))
        cases.append((f"pipeline_full_timeline [{exp.name}]",
                      check_pipeline_gt_is_full_timeline, (exp,)))
        cases.append((f"pipeline_iteration [{exp.name}]",
                      check_pipeline_iteration, (exp,)))
        cases.append((f"metadata_parsed [{exp.name}]",
                      check_metadata_parsed, (exp,)))

    cases.append((f"cache_roundtrip [{EXP_NO_SECONDARY.name}]",
                  check_pipeline_cache_roundtrip, (EXP_NO_SECONDARY,)))
    cases.append((f"cache_corrupt_fallback [{EXP_NO_SECONDARY.name}]",
                  check_pipeline_cache_corrupt_falls_back, (EXP_NO_SECONDARY,)))

    passed = 0
    for label, fn, args in cases:
        if _run_case(label, fn, *args):
            passed += 1
    total = len(cases)
    print(f"\n{passed}/{total} tests passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
