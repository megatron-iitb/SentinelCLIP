"""
Ensemble and uncertainty estimation pipeline module.
Handles augmentation ensemble and entropy-based confidence.
"""

import numpy as np

from src.evaluation.ensemble import (compute_augmentation_ensemble, compute_entropy_ensemble_conf, 
                                     combine_ensemble_confidences)
from src.policy.decision_policy import compute_ood_confidence


def run_uncertainty_estimation(data, calib, config):
    """
    Compute ensemble-based and entropy-based uncertainty estimates.
    
    Args:
        data: dict from data_preparation step
        calib: dict from calibration step
        config: Configuration object/module
        
    Returns:
        dict containing ensemble confidences and OOD scores
    """
    print("\n" + "=" * 70)
    print("STEP 9-11: UNCERTAINTY ESTIMATION")
    print("=" * 70)
    
    clip_model = data['clip_model']
    class_text_embs = data['class_text_embs']
    test_imgs = data['test_imgs']
    test_probs_cal = calib['test_probs_cal']
    temperature = calib['temperature']
    
    # 9. Augmentation Ensemble
    print("\n--- Step 9: Augmentation ensemble ---")
    aug_probs_list, aug_mean_probs, ensemble_conf_std = compute_augmentation_ensemble(
        test_imgs, clip_model, class_text_embs, temperature, config.AUGMENTATION_SCALES
    )
    print(f"✓ Ensemble computed with {len(aug_probs_list)} augmentations per sample")
    print(f"  Std-based confidence: mean={ensemble_conf_std.mean():.4f}, std={ensemble_conf_std.std():.4f}")
    
    # 10. Entropy-based Ensemble Confidence
    print("\n--- Step 10: Entropy-based ensemble confidence ---")
    ensemble_conf_entropy, mi_proxy = compute_entropy_ensemble_conf(aug_probs_list)
    print(f"✓ Entropy-based confidence: mean={ensemble_conf_entropy.mean():.4f}, std={ensemble_conf_entropy.std():.4f}")
    print(f"  Correlation (std vs entropy): {np.corrcoef(ensemble_conf_std, ensemble_conf_entropy)[0,1]:.4f}")
    
    # Combine ensemble confidences
    ensemble_conf = combine_ensemble_confidences(
        ensemble_conf_std, ensemble_conf_entropy, strategy=config.ENSEMBLE_CONF_STRATEGY
    )
    print(f"✓ Using '{config.ENSEMBLE_CONF_STRATEGY}' strategy: mean={ensemble_conf.mean():.4f}")
    
    # 11. OOD Confidence
    print("\n--- Step 11: Out-of-distribution confidence ---")
    ood_conf, class_entropy = compute_ood_confidence(test_probs_cal)
    print(f"✓ OOD confidence computed: mean={ood_conf.mean():.4f}")
    print(f"  Class entropy: mean={class_entropy.mean():.4f}")
    
    return {
        'aug_probs_list': aug_probs_list,
        'aug_mean_probs': aug_mean_probs,
        'ensemble_conf_std': ensemble_conf_std,
        'ensemble_conf_entropy': ensemble_conf_entropy,
        'ensemble_conf': ensemble_conf,
        'ood_conf': ood_conf,
        'class_entropy': class_entropy
    }
