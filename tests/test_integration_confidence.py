"""
Integration tests for the full ZUPT confidence interval pipeline.
Tests the end-to-end flow: dataset generation -> ZUPT -> conformal calibration -> coverage validation.
"""
import os, sys, json, numpy as np, pytest
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from algorithms.algo2_zupt import estimate_height_zupt
from algorithms.zupt_confidence import ZuptConfidenceAnalyzer
from dataset.synthetic_work_dataset import create_dataset
import pandas as pd


def run_full_pipeline(tmp_path, n_train=80, n_test=30, seed=123):
    """Helper: run the full pipeline and return coverage stats."""
    np.random.seed(seed)
    import random; random.seed(seed)
    
    base = tmp_path / "dataset"
    create_dataset(str(base), n_train=n_train, n_test=n_test)
    
    analyzer = ZuptConfidenceAnalyzer(dt=0.01)
    train_dir = str(base / "train")
    
    # Process train
    results = []
    for s in sorted(os.listdir(train_dir)):
        sp = os.path.join(train_dir, s)
        if not os.path.isdir(sp): continue
        df = pd.read_csv(os.path.join(sp, 'accel.csv'))
        with open(os.path.join(sp, 'metadata.json')) as f: meta = json.load(f)
        t, az = df['time'].values, df['az'].values
        pos = estimate_height_zupt(t, az, gravity=9.81)
        
        # Active window detection
        ws = 50
        az_s = np.convolve(np.abs(az - 9.81), np.ones(ws)/ws, mode='same')
        dt_arr = np.diff(t); dt_arr = np.insert(dt_arr, 0, 0)
        idx = np.where(az_s > 0.05)[0]
        if len(idx) == 0: continue
        margin_n = int(1.0 / np.mean(dt_arr[1:])) if np.mean(dt_arr[1:]) > 0 else 100
        si = max(0, idx[0] - margin_n)
        ei = min(len(t)-1, idx[-1] + margin_n)
        
        rej, _ = analyzer.evaluate_rejection(az, si, ei, meta['phone_model'])
        if rej: continue
        
        ns = ei - si
        err = abs(pos[-1] - meta['gt_height_meters'])
        theo = analyzer.compute_theoretical_confidence(ns, meta['phone_model'])
        results.append({'error': err, 'theo': theo, 'ns': ns, 'phone': meta['phone_model']})
    
    errors = [r['error'] for r in results]
    theos = [r['theo'] for r in results]
    analyzer.fit_conformal(errors, theos, alpha=0.1)
    
    # Check coverage
    margins = [analyzer.get_confidence_interval(r['ns'], r['phone']) for r in results]
    covered = sum(1 for e, m in zip(errors, margins) if e <= m)
    coverage = covered / len(errors) * 100
    
    return coverage, analyzer, len(results)


class TestIntegration:
    def test_coverage_above_85(self, tmp_path):
        """The calibrated CI should achieve at least 85% coverage (allowing some variance)."""
        coverage, _, n = run_full_pipeline(tmp_path, n_train=80)
        assert coverage >= 85.0, f"Coverage {coverage:.1f}% is too low (n={n})"
    
    def test_coverage_below_100(self, tmp_path):
        """Coverage should not be 100% — that would mean CIs are too wide."""
        coverage, _, n = run_full_pipeline(tmp_path, n_train=80)
        assert coverage < 100.0, f"Coverage is 100% — CIs may be trivially wide"
    
    def test_multiplier_above_theoretical(self, tmp_path):
        """The calibrated multiplier should be larger than the theoretical 1.645."""
        _, analyzer, _ = run_full_pipeline(tmp_path, n_train=80)
        assert analyzer.calibrated_multiplier > 1.645, \
            f"Multiplier {analyzer.calibrated_multiplier} should exceed theoretical 1.645"
    
    def test_different_seeds_stable(self, tmp_path):
        """Coverage should be reasonably stable across different random seeds."""
        coverages = []
        for seed in [42, 99, 7]:
            sub = tmp_path / f"seed_{seed}"
            sub.mkdir()
            cov, _, _ = run_full_pipeline(sub, n_train=60, seed=seed)
            coverages.append(cov)
        # All should be between 80% and 100%
        for c in coverages:
            assert 80.0 <= c <= 100.0, f"Coverage {c:.1f}% is out of expected range"
    
    def test_rejection_filters_anomalies(self, tmp_path):
        """At least some samples should be rejected (we inject anomalies)."""
        np.random.seed(42)
        import random; random.seed(42)
        
        base = tmp_path / "rej_test"
        create_dataset(str(base), n_train=100, n_test=20)
        
        analyzer = ZuptConfidenceAnalyzer(dt=0.01)
        rejected = 0
        train_dir = str(base / "train")
        for s in sorted(os.listdir(train_dir)):
            sp = os.path.join(train_dir, s)
            if not os.path.isdir(sp): continue
            df = pd.read_csv(os.path.join(sp, 'accel.csv'))
            with open(os.path.join(sp, 'metadata.json')) as f: meta = json.load(f)
            t, az = df['time'].values, df['az'].values
            ws = 50
            az_s = np.convolve(np.abs(az - 9.81), np.ones(ws)/ws, mode='same')
            dt_arr = np.diff(t); dt_arr = np.insert(dt_arr, 0, 0)
            idx = np.where(az_s > 0.05)[0]
            if len(idx) == 0: continue
            margin_n = int(1.0 / np.mean(dt_arr[1:])) if np.mean(dt_arr[1:]) > 0 else 100
            si = max(0, idx[0] - margin_n)
            ei = min(len(t)-1, idx[-1] + margin_n)
            rej, _ = analyzer.evaluate_rejection(az, si, ei, meta['phone_model'])
            if rej: rejected += 1
        
        assert rejected >= 1, "Expected at least 1 rejected sample from 100 with anomaly injection"

    def test_test_set_has_no_gt(self, tmp_path):
        """Test set metadata should NOT contain gt_height_meters."""
        np.random.seed(42)
        import random; random.seed(42)
        
        base = tmp_path / "gt_test"
        create_dataset(str(base), n_train=5, n_test=5)
        
        test_dir = str(base / "test")
        for s in sorted(os.listdir(test_dir)):
            sp = os.path.join(test_dir, s)
            if not os.path.isdir(sp): continue
            with open(os.path.join(sp, 'metadata.json')) as f: meta = json.load(f)
            assert 'gt_height_meters' not in meta, f"Test sample {s} should not have GT"
            assert 'phone_model' in meta, f"Test sample {s} must have phone_model"
