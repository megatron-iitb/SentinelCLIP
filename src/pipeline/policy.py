"""
Policy optimization and decision-making pipeline module.
Handles threshold optimization and final decisions.
"""

import numpy as np

from src.policy.decision_policy import AccountablePolicy
from src.policy.threshold_optimizer import optimize_policy_thresholds, apply_optimized_thresholds
from src.calibration.temperature import apply_temperature
from src.calibration.conformal import compute_conformal_sets, conformal_confidence
from src.policy.decision_policy import compute_ood_confidence
from src.evaluation.ensemble import compute_augmentation_ensemble


def run_policy_pipeline(data, calib, uncertainty, config, device):
    """
    Initialize policy, optimize thresholds, and make final decisions.
    
    Args:
        data: dict from data_preparation step
        calib: dict from calibration step
        uncertainty: dict from uncertainty estimation step
        config: Configuration object/module
        device: torch.device
        
    Returns:
        dict containing policy, decisions, and predictions
    """
    print("\n" + "=" * 70)
    print("STEP 12-14: POLICY & DECISION MAKING")
    print("=" * 70)
    
    # 12. Initialize Policy
    print("\n--- Step 12: Initialize accountable policy ---")
    policy = AccountablePolicy(
        tau_critical_low=config.TAU_CRITICAL_LOW,
        action_auto=config.ACTION_AUTO_DEFAULT,
        action_clarify=config.ACTION_CLARIFY_DEFAULT,
        sim_human_accuracy=config.SIM_HUMAN_ACCURACY,
        rng_seed=config.RANDOM_SEED
    )
    print(f"✓ Policy initialized:")
    print(f"  tau_critical_low: {policy.tau_critical_low:.4f}")
    print(f"  ACTION_AUTO:      {policy.action_auto:.4f}")
    print(f"  ACTION_CLARIFY:   {policy.action_clarify:.4f}")
    print(f"  Simulated human accuracy: {policy.sim_human_accuracy:.2%}")
    
    # Get question indices
    q_idx = calib['question_engine'].get_question_index()
    noncrit_idx = [q_idx[k] for k in config.NONCRITICAL_QUESTIONS if k in q_idx]
    print(f"  Non-critical questions: {len(noncrit_idx)}/{len(config.QUESTION_PROMPTS)}")
    
    # Compute action confidence for test set
    test_labels = data['test_labels']
    test_probs_cal = calib['test_probs_cal']
    test_q_prob_yes = calib['test_q_prob_yes']
    ensemble_conf = uncertainty['ensemble_conf']
    conformal_conf = calib['conformal_conf']
    ood_conf = uncertainty['ood_conf']
    
    action_confidence, min_crit, crit_stack = policy.compute_action_confidence(
        test_probs_cal, ensemble_conf, conformal_conf, ood_conf,
        test_q_prob_yes, noncrit_idx
    )
    print(f"✓ Action confidence computed: mean={action_confidence.mean():.4f}")
    
    # 13. Optimize Thresholds (if enabled)
    if config.OPTIMIZE_THRESHOLDS:
        print("\n--- Step 13: Threshold optimization on validation set ---")
        
        # Prepare validation data
        val_labels = data['val_labels']
        val_logits = calib['val_logits']
        val_probs_cal = calib['val_probs_cal']
        val_preds_cal = val_probs_cal.argmax(axis=1)
        val_q_prob_yes = calib['val_q_prob_yes']
        
        val_conformal_sets, val_set_sizes = compute_conformal_sets(
            val_probs_cal, calib['conformal_threshold']
        )
        val_conformal_conf = conformal_confidence(val_set_sizes)
        val_ood_conf, _ = compute_ood_confidence(val_probs_cal)
        
        # Compute val augmentation ensemble (subset for speed)
        print(f"Computing validation ensemble (subset: {config.VAL_AUGMENTATION_SUBSET} samples)...")
        val_imgs = data['val_imgs']
        val_subset_size = min(len(val_imgs), config.VAL_AUGMENTATION_SUBSET)
        
        val_aug_probs_list, val_aug_mean_probs, val_ensemble_conf = compute_augmentation_ensemble(
            val_imgs[:val_subset_size], data['clip_model'], data['class_text_embs'], 
            calib['temperature'], config.AUGMENTATION_SCALES
        )
        
        # Pad if subset
        if val_subset_size < len(val_imgs):
            for idx in range(val_subset_size, len(val_imgs)):
                val_aug_probs_list.append(val_probs_cal[idx:idx+1])
            val_aug_mean_probs = np.vstack([val_aug_mean_probs, val_probs_cal[val_subset_size:]])
            val_ensemble_conf = np.append(val_ensemble_conf, 
                                         np.ones(len(val_imgs) - val_subset_size))
        
        # Compute val action confidence
        val_action_confidence, val_min_crit, _ = policy.compute_action_confidence(
            val_probs_cal, val_ensemble_conf, val_conformal_conf, val_ood_conf,
            val_q_prob_yes, noncrit_idx
        )
        
        # Optimize
        print(f"Running grid search ({config.THRESHOLD_GRID_SIZE}^3 = {config.THRESHOLD_GRID_SIZE**3} combinations)...")
        best_thresholds, threshold_results = optimize_policy_thresholds(
            policy, val_action_confidence, val_min_crit, val_labels, val_preds_cal,
            val_conformal_sets, val_set_sizes, val_aug_mean_probs,
            c_human=config.THRESHOLD_COST_HUMAN, 
            c_error=config.THRESHOLD_COST_ERROR, 
            n_grid=config.THRESHOLD_GRID_SIZE
        )
        
        print(f"\n✓ Optimization complete!")
        print(f"  Best thresholds:")
        print(f"    tau_critical_low: {best_thresholds['tau_critical_low']:.4f}")
        print(f"    ACTION_AUTO:      {best_thresholds['ACTION_AUTO']:.4f}")
        print(f"    ACTION_CLARIFY:   {best_thresholds['ACTION_CLARIFY']:.4f}")
        print(f"  Validation metrics:")
        print(f"    Cost:         {best_thresholds['cost']:.4f}")
        print(f"    Interventions: {best_thresholds['interventions']:.4f}")
        print(f"    Errors:       {best_thresholds['errors']:.4f}")
        print(f"    Accuracy:     {best_thresholds['accuracy']:.4f}")
        
        # Apply optimized thresholds
        apply_optimized_thresholds(policy, best_thresholds)
        print("\n✓ Optimized thresholds applied to policy")
    else:
        print("\n--- Step 13: Threshold optimization (DISABLED) ---")
        best_thresholds = None
    
    # 14. Make Final Decisions
    print("\n--- Step 14: Making final decisions ---")
    test_preds_cal = calib['test_preds_cal']
    test_conformal_sets = calib['test_conformal_sets']
    test_set_sizes = calib['test_set_sizes']
    aug_mean_probs = uncertainty['aug_mean_probs']
    
    decisions, final_preds = policy.make_decisions(
        action_confidence, min_crit, test_preds_cal, test_labels,
        test_conformal_sets, test_set_sizes, aug_mean_probs
    )
    
    # Count decisions
    unique, counts = np.unique(decisions, return_counts=True)
    decision_counts = dict(zip(unique, counts))
    print(f"✓ Decisions made:")
    for decision in ['auto', 'clarify', 'human']:
        count = decision_counts.get(decision, 0)
        pct = 100 * count / len(decisions)
        print(f"    {decision:8s}: {count:5d} ({pct:5.2f}%)")
    
    return {
        'policy': policy,
        'action_confidence': action_confidence,
        'min_crit': min_crit,
        'decisions': decisions,
        'final_preds': final_preds,
        'best_thresholds': best_thresholds
    }
