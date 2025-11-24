"""
Isotonic regression calibration for question prompts.
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression


def calibrate_questions_isotonic(val_q_probs, val_labels, question_keys_list, question_gt_map):
    """
    Train isotonic regression calibrators for each question.
    
    Args:
        val_q_probs: Validation question probabilities (N x Q)
        val_labels: Validation labels (N,)
        question_keys_list: List of question keys
        question_gt_map: Dictionary mapping questions to positive class indices
    
    Returns:
        List of fitted IsotonicRegression models (one per question)
    """
    calibrators = []
    
    for q_idx, q_key in enumerate(question_keys_list):
        if q_key not in question_gt_map:
            # No mapping - skip calibration (use identity)
            calibrators.append(None)
            continue
        
        # Create binary ground truth for this question
        positive_classes = question_gt_map[q_key]
        y_true = np.isin(val_labels, positive_classes).astype(int)
        
        # Get predicted probabilities for "yes" answer
        y_pred = val_q_probs[:, q_idx]
        
        # Fit isotonic regression
        iso = IsotonicRegression(out_of_bounds='clip')
        try:
            iso.fit(y_pred, y_true)
            calibrators.append(iso)
        except Exception as e:
            print(f"Warning: Could not fit isotonic for {q_key}: {e}")
            calibrators.append(None)
    
    return calibrators


def apply_question_calibration(q_probs, calibrators):
    """
    Apply isotonic calibration to question probabilities.
    
    Args:
        q_probs: Question probabilities (N x Q)
        calibrators: List of fitted calibrators
    
    Returns:
        Calibrated question probabilities
    """
    calibrated = q_probs.copy()
    for q_idx, cal in enumerate(calibrators):
        if cal is not None:
            calibrated[:, q_idx] = cal.predict(q_probs[:, q_idx])
    return calibrated
