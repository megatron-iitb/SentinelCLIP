"""
Accountable decision policy with human-in-the-loop.
"""
import numpy as np
from tqdm.auto import tqdm
from src.utils.math_utils import geometric_mean_stable, compute_entropy


class AccountablePolicy:
    """Human-in-the-loop decision policy with explainable confidence scores."""
    
    def __init__(self, tau_critical_low=0.45, action_auto=0.90, action_clarify=0.60,
                 sim_human_accuracy=1.0, rng_seed=0):
        """
        Initialize policy.
        
        Args:
            tau_critical_low: Minimum critical signal threshold
            action_auto: Threshold for automatic execution
            action_clarify: Threshold for clarification
            sim_human_accuracy: Simulated human accuracy (1.0 = perfect)
            rng_seed: Random seed for human simulation
        """
        self.tau_critical_low = tau_critical_low
        self.action_auto = action_auto
        self.action_clarify = action_clarify
        self.sim_human_accuracy = sim_human_accuracy
        self.rng = np.random.default_rng(rng_seed)
    
    def compute_action_confidence(self, probs_cal, ensemble_conf, conformal_conf, 
                                  ood_conf, question_confidences, noncrit_indices):
        """
        Compute action confidence from multiple signals.
        
        Args:
            probs_cal: Calibrated class probabilities (N x K)
            ensemble_conf: Ensemble confidence scores (N,)
            conformal_conf: Conformal confidence scores (N,)
            ood_conf: Out-of-distribution confidence scores (N,)
            question_confidences: Question confidence matrix (N x Q)
            noncrit_indices: Indices of non-critical questions
        
        Returns:
            action_confidence: Final action confidence (N,)
            min_crit: Minimum critical signal (N,)
            crit_stack: All critical signals (N x 4)
        """
        # Critical signals
        q_primary = probs_cal.max(axis=1)
        ensemble_conf_clipped = np.clip(ensemble_conf, 0.0, 1.0)
        
        crit_stack = np.vstack([
            q_primary, 
            ensemble_conf_clipped, 
            conformal_conf, 
            ood_conf
        ]).T
        
        min_crit = crit_stack.min(axis=1)
        
        # Geometric mean of critical signals (log-space for stability)
        geom_mean = geometric_mean_stable(crit_stack, axis=1)
        
        # Non-critical questions contribution
        if len(noncrit_indices) > 0:
            noncrit_conf = question_confidences[:, noncrit_indices].mean(axis=1)
        else:
            noncrit_conf = np.ones_like(geom_mean)
        
        # Combine: geometric mean × mixture with non-critical
        action_confidence = geom_mean * (0.5 * noncrit_conf + 0.5)
        
        return action_confidence, min_crit, crit_stack
    
    def make_decisions(self, action_confidence, min_crit, baseline_preds, 
                      true_labels, conformal_sets, conformal_set_sizes, aug_mean_probs):
        """
        Make policy decisions for each sample.
        
        Args:
            action_confidence: Action confidence scores (N,)
            min_crit: Minimum critical signals (N,)
            baseline_preds: Baseline model predictions (N,)
            true_labels: Ground truth labels (N,) - for human simulation
            conformal_sets: Conformal prediction sets (N x K)
            conformal_set_sizes: Conformal set sizes (N,)
            aug_mean_probs: Ensemble mean probabilities (N x K)
        
        Returns:
            decisions: Decision types (N,) - "auto", "clarify", or "human"
            final_preds: Final predictions (N,)
        """
        decisions = []
        final_preds = []
        num_classes = len(aug_mean_probs[0])
        
        for i in range(len(action_confidence)):
            # Hard safety gate: critical signals too low
            if min_crit[i] <= self.tau_critical_low:
                decisions.append('human')
                final_preds.append(self._simulate_human(true_labels[i], num_classes))
            else:
                ac = action_confidence[i]
                
                if ac >= self.action_auto:
                    # High confidence: auto-execute
                    decisions.append('auto')
                    final_preds.append(int(baseline_preds[i]))
                    
                elif ac >= self.action_clarify:
                    # Medium confidence: use clarification strategy
                    decisions.append('clarify')
                    if conformal_set_sizes[i] == 1:
                        # Single-element conformal set: use it
                        lab = int(np.where(conformal_sets[i] == 1)[0][0])
                        final_preds.append(lab)
                    else:
                        # Use ensemble prediction
                        final_preds.append(int(aug_mean_probs[i].argmax()))
                        
                else:
                    # Low confidence: defer to human
                    decisions.append('human')
                    final_preds.append(self._simulate_human(true_labels[i], num_classes))
        
        return np.array(decisions), np.array(final_preds)
    
    def _simulate_human(self, true_label, num_classes):
        """Simulate human decision with configurable accuracy."""
        if self.rng.random() < self.sim_human_accuracy:
            return int(true_label)
        else:
            # Pick incorrect label at random
            choices = [c for c in range(num_classes) if c != int(true_label)]
            return int(self.rng.choice(choices))


def compute_ood_confidence(probs_cal):
    """
    Compute OOD confidence from normalized entropy.
    
    Args:
        probs_cal: Calibrated probabilities (N x K)
    
    Returns:
        ood_conf: OOD confidence scores (N,)
        class_entropy: Raw entropy values (N,)
    """
    import math
    
    class_entropy = compute_entropy(probs_cal)
    max_ent = math.log(probs_cal.shape[1])
    ood_conf = 1.0 - (class_entropy / max_ent)
    
    return ood_conf, class_entropy
