"""
main.py — Pipeline Orchestrator
================================
Runs the complete visual analytics pipeline:
  1. Data Ingestion (real Yahoo Finance + synthetic)
  2. Time Alignment & Imputation
  3. Feature Engineering
  4. Visualization Generation
"""
import os
import sys

# Ensure src/ is importable
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import generate_synthetic_data
from preprocess import align_and_resample
from features import engineer_features
from visualizations import generate_all_visuals


def main():
    print("=====================================================")
    print("Multimodal Visual Analytics of Geopolitical Conflict")
    print("=====================================================")

    # --- 1. Data Ingestion ---
    print("[1/4] Fetching data (Yahoo Finance + synthetic)...")
    raw_data = generate_synthetic_data(start_date='2024-01-01', end_date='2026-06-30')
    raw_data.to_csv('data/raw/raw_data.csv', index=False)

    # --- 2. Time Alignment ---
    print("[2/4] Aligning and resampling data (Daily Frequency)...")
    resampled_data = align_and_resample(raw_data, freq='D')

    # --- 3. Feature Engineering ---
    print("[3/4] Engineering features (Lags, rolling volatilities, indices)...")
    df_features = engineer_features(resampled_data)

    unified_csv_path = 'data/processed/unified_dataset.csv'
    df_features.to_csv(unified_csv_path)
    print(f"      Unified dataset saved to: {unified_csv_path}")

    # --- 4. Visualization ---
    print("[4/4] Generating visualizations...")
    generate_all_visuals(df_features)

    print("=====================================================")
    print("Pipeline Execution Complete!")
    print("Check 'outputs/charts/' for visualizations.")
    print("Check 'data/processed/' for the unified dataset.")
    print("=====================================================")


if __name__ == "__main__":
    main()
