"""
Threshold optimization using validation set.
"""
import numpy as np
from tqdm.auto import tqdm


def optimize_policy_thresholds(policy, action_conf, min_crit_vals, true_labels, baseline_preds,
                                conformal_sets, conformal_set_sizes, aug_probs,
                                c_human=1.0, c_error=10.0, n_grid=10):
    """
    Grid-search over policy thresholds to minimize operational cost.
    
    Cost = c_human * (# interventions) + c_error * (# errors)
    
    Args:
        policy: AccountablePolicy instance
        action_conf: Action confidence scores
        min_crit_vals: Minimum critical signals
        true_labels: Ground truth labels
        baseline_preds: Baseline predictions
        conformal_sets: Conformal prediction sets
        conformal_set_sizes: Conformal set sizes
        aug_probs: Augmentation ensemble probabilities
        c_human: Cost per human intervention
        c_error: Cost per error
        n_grid: Grid resolution
    
    Returns:
        best_params: Dictionary with best thresholds and metrics
        results: List of all evaluated configurations
    """
    tau_crit_vals = np.linspace(0.2, 0.7, n_grid)
    auto_vals = np.linspace(0.70, 0.95, n_grid)
    clarify_vals = np.linspace(0.40, 0.80, n_grid)
    
    best_cost = float('inf')
    best_params = None
    results = []
    
    print(f"Grid-searching {n_grid}^3 = {n_grid**3} threshold combinations...")
    
    for tau_c in tau_crit_vals:
        for auto_th in auto_vals:
            for clarify_th in clarify_vals:
                if clarify_th >= auto_th:
                    continue  # Must have clarify < auto
                
                # Temporarily update policy thresholds
                orig_tau = policy.tau_critical_low
                orig_auto = policy.action_auto
                orig_clarify = policy.action_clarify
                
                policy.tau_critical_low = tau_c
                policy.action_auto = auto_th
                policy.action_clarify = clarify_th
                
                # Simulate policy
                decisions, preds = policy.make_decisions(
                    action_conf, min_crit_vals, baseline_preds, true_labels,
                    conformal_sets, conformal_set_sizes, aug_probs
                )
                
                # Restore original thresholds
                policy.tau_critical_low = orig_tau
                policy.action_auto = orig_auto
                policy.action_clarify = orig_clarify
                
                # Compute metrics
                n_interventions = np.sum((decisions == 'clarify') | (decisions == 'human'))
                errors = np.sum(preds != true_labels)
                cost = c_human * n_interventions + c_error * errors
                
                results.append({
                    'tau_critical': tau_c,
                    'auto_th': auto_th,
                    'clarify_th': clarify_th,
                    'cost': cost,
                    'interventions': n_interventions,
                    'errors': errors,
                    'accuracy': np.mean(preds == true_labels)
                })
                
                if cost < best_cost:
                    best_cost = cost
                    best_params = {
                        'tau_critical_low': tau_c,
                        'ACTION_AUTO': auto_th,
                        'ACTION_CLARIFY': clarify_th,
                        'cost': cost,
                        'interventions': n_interventions,
                        'errors': errors,
                        'accuracy': np.mean(preds == true_labels)
                    }
    
    return best_params, results


def apply_optimized_thresholds(policy, best_params):
    """Apply optimized thresholds to policy."""
    policy.tau_critical_low = best_params['tau_critical_low']
    policy.action_auto = best_params['ACTION_AUTO']
    policy.action_clarify = best_params['ACTION_CLARIFY']
    
    print(f"Applied optimized thresholds:")
    print(f"  tau_critical_low: {policy.tau_critical_low:.4f}")
    print(f"  ACTION_AUTO: {policy.action_auto:.4f}")
    print(f"  ACTION_CLARIFY: {policy.action_clarify:.4f}")
