#!/usr/bin/env python3
"""
Main script for Experiment 1: Accountable CLIP-based Classification

This is the simplified entry point using modular pipeline stages:
1. Data Preparation (model, dataset, embeddings)
2. Calibration (temperature, questions, conformal)
3. Uncertainty Estimation (ensemble, entropy)
4. Policy & Decisions (optimization, final predictions)
5. Evaluation (metrics, plots, audit logs)

For detailed implementation, see src/pipeline/ modules.
"""

import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import configuration
from configs import config

# Import pipeline stages
from src.pipeline import (
    setup_model_and_data,
    run_calibration_pipeline,
    run_uncertainty_estimation,
    run_policy_pipeline,
    run_evaluation
)


def main():
    """Main execution function."""
    
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: ACCOUNTABLE CLIP-BASED CLASSIFICATION")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Model: {config.MODEL_NAME} ({config.MODEL_PRETRAINED})")
    print(f"  Dataset: {config.DATASET_NAME}")
    print(f"  Device: {config.DEVICE}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Optimize thresholds: {config.OPTIMIZE_THRESHOLDS}")
    print(f"  Random seed: {config.RANDOM_SEED}")
    print("=" * 70)
    
    # Setup device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    
    # Create output directories
    config.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    config.LOG_DIR.mkdir(exist_ok=True, parents=True)
    
    # Run pipeline stages
    try:
        # Stage 1-3: Model & Data Preparation
        data = setup_model_and_data(config, device)
        
        # Stage 4-8: Calibration Pipeline
        calib = run_calibration_pipeline(data, config, device)
        
        # Stage 9-11: Uncertainty Estimation
        uncertainty = run_uncertainty_estimation(data, calib, config)
        
        # Stage 12-14: Policy & Decision Making
        policy_results = run_policy_pipeline(data, calib, uncertainty, config, device)
        
        # Stage 15-17: Evaluation & Reporting
        final_metrics = run_evaluation(data, calib, uncertainty, policy_results, config)
        
        print("\n✓ Experiment completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
