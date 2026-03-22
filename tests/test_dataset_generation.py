import os
import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# Add src to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from dataset.synthetic_work_dataset import generate_sample, create_dataset

def test_generate_sample(tmp_path):
    output_dir = tmp_path / "dataset"
    os.makedirs(output_dir)
    
    meta = generate_sample(output_dir, "sample_0001", include_gt=True, phone_models=["generic_premium"])
    assert "phone_model" in meta
    assert "gt_height_meters" in meta
    assert "anomaly" in meta
    
    sample_dir = output_dir / "sample_0001"
    assert os.path.exists(sample_dir / "metadata.json")
    assert os.path.exists(sample_dir / "accel.csv")
    
    df = pd.read_csv(sample_dir / "accel.csv")
    assert "az" in df.columns
    assert "time" in df.columns
    assert len(df) > 100 # At least 1 second

def test_create_dataset(tmp_path):
    base_dir = tmp_path / "work_dataset"
    create_dataset(base_dir, n_train=2, n_test=2)
    
    assert os.path.exists(base_dir / "train")
    assert os.path.exists(base_dir / "test")
    
    train_samples = os.listdir(base_dir / "train")
    assert len(train_samples) == 2
    
    test_samples = os.listdir(base_dir / "test")
    assert len(test_samples) == 2
