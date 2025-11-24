"""
Evaluation and reporting pipeline module.
Handles final metrics, visualization, and audit logging.
"""

import numpy as np
from sklearn.metrics import accuracy_score

from src.evaluation.metrics import (
    expected_calibration_error, 
    compute_intervention_metrics,
    plot_reliability_diagram,
    plot_coverage_vs_error,
    save_audit_log,
    print_results
)


def run_evaluation(data, calib, uncertainty, policy_results, config):
    """
    Evaluate results, generate visualizations, and save audit logs.
    
    Args:
        data: dict from data_preparation step
        calib: dict from calibration step
        uncertainty: dict from uncertainty estimation step
        policy_results: dict from policy step
        config: Configuration object/module
        
    Returns:
        dict containing final metrics
    """
    print("\n" + "=" * 70)
    print("STEP 15-17: EVALUATION & REPORTING")
    print("=" * 70)
    
    test_labels = data['test_labels']
    test_preds_cal = calib['test_preds_cal']
    test_probs_cal = calib['test_probs_cal']
    baseline_acc = calib['baseline_acc']
    
    decisions = policy_results['decisions']
    final_preds = policy_results['final_preds']
    action_confidence = policy_results['action_confidence']
    
    # 15. Evaluate Results
    print("\n--- Step 15: Computing evaluation metrics ---")
    post_acc = accuracy_score(test_labels, final_preds)
    errors_baseline = (test_preds_cal != test_labels)
    errors_after = (final_preds != test_labels)
    precision, recall = compute_intervention_metrics(decisions, errors_baseline)
    
    ece = expected_calibration_error(test_probs_cal, test_labels, n_bins=config.ECE_BINS)
    
    print(f"✓ Metrics computed:")
    print(f"    Baseline accuracy: {baseline_acc:.4f}")
    print(f"    Post-policy accuracy: {post_acc:.4f}")
    print(f"    ECE (calibrated): {ece:.6f}")
    print(f"    Intervention precision: {precision:.4f}")
    print(f"    Intervention recall: {recall:.4f}")
    
    # Print detailed results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print_results(baseline_acc, post_acc, decisions, errors_baseline, errors_after, precision, recall)
    
    # 16. Save Visualizations
    if config.SAVE_PLOTS:
        print("\n--- Step 16: Generating visualizations ---")
        
        plot_reliability_diagram(
            test_labels, test_preds_cal, test_probs_cal,
            save_path=config.OUTPUT_DIR / 'reliability_diagram.png',
            n_bins=config.RELIABILITY_BINS, dpi=config.FIGURE_DPI
        )
        print(f"✓ Saved: reliability_diagram.png")
        
        plot_coverage_vs_error(
            action_confidence, test_preds_cal, test_labels,
            save_path=config.OUTPUT_DIR / 'coverage_vs_error.png',
            dpi=config.FIGURE_DPI
        )
        print(f"✓ Saved: coverage_vs_error.png")
    else:
        print("\n--- Step 16: Visualizations (DISABLED) ---")
    
    # 17. Save Audit Log
    print("\n--- Step 17: Saving audit log ---")
    save_audit_log(
        test_labels, test_preds_cal, final_preds, decisions,
        action_confidence, calib['test_q_prob_yes'], 
        calib['question_engine'].question_keys,
        calib['test_set_sizes'], uncertainty['class_entropy'], 
        uncertainty['ensemble_conf'],
        save_path=config.OUTPUT_DIR / 'audit_log_test.jsonl'
    )
    print(f"✓ Saved: audit_log_test.jsonl ({len(test_labels)} samples)")
    
    print(f"\n{'=' * 70}")
    print(f"All outputs saved to: {config.OUTPUT_DIR}")
    print(f"{'=' * 70}\n")
    
    return {
        'baseline_acc': baseline_acc,
        'post_acc': post_acc,
        'ece': ece,
        'precision': precision,
        'recall': recall,
        'errors_baseline': errors_baseline,
        'errors_after': errors_after
    }
