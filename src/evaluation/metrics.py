"""
Evaluation metrics for calibration and policy performance.
"""
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from pathlib import Path


def expected_calibration_error(probs, labels, n_bins=15):
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        probs: Predicted probabilities (N x K)
        labels: True labels (N,)
        n_bins: Number of bins
    
    Returns:
        ECE value
    """
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    bins = np.linspace(0, 1, n_bins + 1)
    
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = (preds[mask] == labels[mask]).mean()
        conf = confidences[mask].mean()
        ece += (mask.sum() / len(labels)) * abs(acc - conf)
    
    return ece


def compute_intervention_metrics(decisions, errors_baseline):
    """
    Compute intervention precision and recall.
    
    Args:
        decisions: Policy decisions array
        errors_baseline: Boolean array of baseline errors
    
    Returns:
        precision, recall
    """
    interventions_mask = decisions != 'auto'
    intervene_n = interventions_mask.sum()
    
    if intervene_n > 0:
        tp = np.logical_and(interventions_mask, errors_baseline).sum()
        precision = tp / intervene_n
        recall = tp / errors_baseline.sum() if errors_baseline.sum() > 0 else 0.0
    else:
        precision = recall = 0.0
    
    return precision, recall


def plot_reliability_diagram(test_labels, test_preds_cal, test_probs_cal, 
                            save_path=None, n_bins=10, dpi=150):
    """
    Plot reliability diagram.
    
    Args:
        test_labels: True labels
        test_preds_cal: Calibrated predictions
        test_probs_cal: Calibrated probabilities
        save_path: Path to save figure (None to skip)
        n_bins: Number of bins
        dpi: Figure DPI
    """
    prob_true, prob_pred = calibration_curve(
        test_labels == test_preds_cal, 
        test_probs_cal.max(axis=1), 
        n_bins=n_bins
    )
    
    plt.figure(figsize=(5, 5))
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect')
    plt.xlabel('Predicted confidence')
    plt.ylabel('Observed accuracy')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved reliability diagram to {save_path}")
    
    plt.close()


def plot_coverage_vs_error(action_confidence, test_preds_cal, test_labels,
                           save_path=None, n_points=50, dpi=150):
    """
    Plot coverage vs error rate curve.
    
    Args:
        action_confidence: Action confidence scores
        test_preds_cal: Calibrated predictions
        test_labels: True labels
        save_path: Path to save figure
        n_points: Number of threshold points
        dpi: Figure DPI
    """
    ths = np.linspace(0.0, 1.0, n_points)
    coverages = []
    error_rates = []
    
    for th in ths:
        preds = np.where(action_confidence >= th, test_preds_cal, test_labels)
        coverages.append((action_confidence >= th).mean())
        error_rates.append(1.0 - accuracy_score(test_labels, preds))
    
    plt.figure(figsize=(6, 4))
    plt.plot(coverages, error_rates, marker='o', markersize=3)
    plt.xlabel('Auto-execute coverage')
    plt.ylabel('Error rate after policy')
    plt.title('Coverage vs Error')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved coverage vs error plot to {save_path}")
    
    plt.close()


def save_audit_log(test_labels, test_preds_cal, final_preds, decisions, 
                  action_confidence, question_confidences, question_keys,
                  conformal_set_sizes, class_entropy, ensemble_conf,
                  save_path):
    """
    Save per-sample audit log.
    
    Args:
        test_labels: Ground truth labels
        test_preds_cal: Baseline calibrated predictions
        final_preds: Final policy predictions
        decisions: Policy decisions
        action_confidence: Action confidence scores
        question_confidences: Question confidence matrix
        question_keys: List of question keys
        conformal_set_sizes: Conformal set sizes
        class_entropy: Class entropy values
        ensemble_conf: Ensemble confidence scores
        save_path: Path to save JSONL file
    """
    import json
    
    with open(save_path, 'w') as f:
        for i in range(len(test_labels)):
            rec = {
                'idx': int(i),
                'gt': int(test_labels[i]),
                'baseline_pred': int(test_preds_cal[i]),
                'final_pred': int(final_preds[i]),
                'decision': decisions[i],
                'action_conf': float(action_confidence[i]),
                'q_confidences': {question_keys[j]: float(question_confidences[i, j]) 
                                 for j in range(len(question_keys))},
                'conformal_set_size': int(conformal_set_sizes[i]),
                'ensemble_conf': float(ensemble_conf[i]),
                'class_entropy': float(class_entropy[i])
            }
            f.write(json.dumps(rec) + "\n")
    
    print(f"Saved audit log to {save_path}")


def print_results(baseline_acc, post_acc, decisions, errors_baseline, errors_after,
                 precision, recall):
    """Print formatted results."""
    print("\n" + "=" * 50)
    print("=== RESULTS ===")
    print("=" * 50)
    print(f"Baseline (CLIP calibrated) accuracy: {baseline_acc:.4f}")
    print(f"Post-pipeline accuracy (with policy): {post_acc:.4f}")
    print(f"\nAction coverage:")
    print(f"  Auto-execute:  {np.mean(decisions=='auto'):.1%}")
    print(f"  Clarify:       {np.mean(decisions=='clarify'):.1%}")
    print(f"  Human:         {np.mean(decisions=='human'):.1%}")
    print(f"\nError analysis:")
    print(f"  Baseline errors: {errors_baseline.sum()}")
    print(f"  Errors after policy: {errors_after.sum()}")
    print(f"  Errors prevented: {errors_baseline.sum() - errors_after.sum()}")
    print(f"\nIntervention metrics:")
    print(f"  Precision: {precision:.4f} (% of interventions that caught real errors)")
    print(f"  Recall: {recall:.4f} (% of baseline errors that were caught)")
    print("=" * 50)
