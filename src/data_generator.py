"""
data_generator.py
Synthetic hospital emergency department data generation.
Made by: Krishna Mathur | AS25DXB018 | MAIB September
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path

SEED = 42


def generate_patients(n=500, seed=SEED):
    """Generate n synthetic emergency department patient records."""
    np.random.seed(seed)
    urgency_levels = np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.10, 0.20, 0.30, 0.25, 0.15])

    arrival_gaps = np.random.exponential(scale=5, size=n)
    arrival_times = np.cumsum(arrival_gaps).astype(int)

    severity = np.clip(
        urgency_levels * 15 + np.random.normal(0, 5, n), 10, 100
    ).astype(int)

    max_wait = np.array([max(10, 120 - u * 20 + np.random.randint(-5, 5))
                         for u in urgency_levels])

    df = pd.DataFrame({
        "patient_id": [f"P{str(i+1).zfill(4)}" for i in range(n)],
        "arrival_time": arrival_times,
        "urgency_level": urgency_levels,
        "severity_score": severity,
        "age_group": np.random.choice(["Child", "Adult", "Senior"], size=n, p=[0.15, 0.55, 0.30]),
        "required_bed": np.random.choice([1, 0], size=n, p=[0.85, 0.15]),
        "required_doctor": np.ones(n, dtype=int),
        "required_nurse": np.random.choice([1, 2], size=n, p=[0.60, 0.40]),
        "required_equipment": np.random.choice([0, 1, 2], size=n, p=[0.40, 0.40, 0.20]),
        "max_safe_wait_time": max_wait,
        "fairness_group": np.random.choice(["Group_A", "Group_B", "Group_C"], size=n, p=[0.40, 0.35, 0.25]),
        "expected_treatment_time": np.random.randint(20, 120, size=n),
    })
    return df


def generate_resources():
    """Generate hospital resource capacity table."""
    return pd.DataFrame({
        "resource_name": ["Beds", "Doctors", "Nurses", "Equipment"],
        "available_count": [30, 10, 20, 15],
        "emergency_buffer": [5, 2, 3, 2],
        "unit_cost": [500, 1200, 400, 800],
        "max_capacity": [35, 12, 25, 18],
    })


def save_data(output_dir="data"):
    """Generate and persist both CSVs; returns (patients_df, resources_df)."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    patients_df = generate_patients(500)
    resources_df = generate_resources()
    patients_df.to_csv(os.path.join(output_dir, "patients.csv"), index=False)
    resources_df.to_csv(os.path.join(output_dir, "resources.csv"), index=False)
    return patients_df, resources_df


def load_data(data_dir="data"):
    """Load CSVs; generate them first if they don't exist."""
    patients_path = os.path.join(data_dir, "patients.csv")
    resources_path = os.path.join(data_dir, "resources.csv")
    if not os.path.exists(patients_path) or not os.path.exists(resources_path):
        return save_data(data_dir)
    return pd.read_csv(patients_path), pd.read_csv(resources_path)
