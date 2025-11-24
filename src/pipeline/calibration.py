"""
Calibration pipeline module.
Handles temperature scaling, question calibration, and conformal prediction.
"""

import numpy as np

from src.calibration.temperature import fit_temperature, apply_temperature
from src.calibration.isotonic import calibrate_questions_isotonic, apply_question_calibration
from src.calibration.conformal import compute_conformal_threshold, compute_conformal_sets, conformal_confidence
from src.evaluation.questions import QuestionEngine
from src.utils.math_utils import softmax
from sklearn.metrics import accuracy_score


def run_calibration_pipeline(data, config, device):
    """
    Run full calibration pipeline: temperature, questions, conformal.
    
    Args:
        data: dict from data_preparation step
        config: Configuration object/module
        device: torch.device
        
    Returns:
        dict containing calibrated predictions, confidences, etc.
    """
    print("\n" + "=" * 70)
    print("STEP 4-8: CALIBRATION PIPELINE")
    print("=" * 70)
    
    clip_model = data['clip_model']
    class_text_embs = data['class_text_embs']
    val_embs = data['val_embs']
    val_labels = data['val_labels']
    test_embs = data['test_embs']
    test_labels = data['test_labels']
    
    # 4. Zero-shot Predictions
    print("\n--- Step 4: Zero-shot predictions ---")
    val_logits = clip_model.compute_similarities(val_embs, class_text_embs)
    test_logits = clip_model.compute_similarities(test_embs, class_text_embs)
    
    val_probs = softmax(val_logits, axis=1)
    test_probs = softmax(test_logits, axis=1)
    test_preds = test_probs.argmax(axis=1)
    
    baseline_acc_raw = accuracy_score(test_labels, test_preds)
    print(f"✓ Baseline zero-shot test accuracy: {baseline_acc_raw:.4f}")
    
    # 5. Temperature Scaling
    print("\n--- Step 5: Temperature scaling ---")
    temperature = fit_temperature(val_logits, val_labels, device=device)
    
    test_probs_cal = apply_temperature(test_logits, temperature)
    test_preds_cal = test_probs_cal.argmax(axis=1)
    
    baseline_acc = accuracy_score(test_labels, test_preds_cal)
    print(f"✓ Test accuracy after temperature scaling: {baseline_acc:.4f}")
    
    # 6. Question Confidences
    print("\n--- Step 6: Computing question confidences ---")
    question_engine = QuestionEngine(config.QUESTION_PROMPTS, clip_model)
    
    test_q_prob_yes, test_q_entropy, test_q_probs = question_engine.get_question_confidences(test_embs)
    val_q_prob_yes, val_q_entropy, _ = question_engine.get_question_confidences(val_embs)
    
    print(f"✓ Question confidences computed for {len(config.QUESTION_PROMPTS)} questions")
    print(f"  Mean confidence (raw): {test_q_prob_yes.mean():.4f}")
    
    # 7. Question Calibration (Isotonic)
    print("\n--- Step 7: Isotonic calibration of questions ---")
    question_calibrators = calibrate_questions_isotonic(
        val_q_prob_yes, val_labels, 
        question_engine.question_keys, config.QUESTION_GT_MAP
    )
    
    test_q_prob_yes_cal = apply_question_calibration(test_q_prob_yes, question_calibrators)
    val_q_prob_yes_cal = apply_question_calibration(val_q_prob_yes, question_calibrators)
    
    n_calibrated = sum(c is not None for c in question_calibrators)
    print(f"✓ Calibrated {n_calibrated}/{len(question_calibrators)} questions")
    print(f"  Mean confidence before: {test_q_prob_yes.mean():.4f}")
    print(f"  Mean confidence after:  {test_q_prob_yes_cal.mean():.4f}")
    
    # 8. Conformal Prediction
    print("\n--- Step 8: Conformal prediction ---")
    val_probs_cal = apply_temperature(val_logits, temperature)
    threshold, quantile = compute_conformal_threshold(val_probs_cal, val_labels, alpha=config.CONFORMAL_ALPHA)
    print(f"✓ Conformal threshold: keep labels with p >= {threshold:.4f}")
    
    test_conformal_sets, test_set_sizes = compute_conformal_sets(test_probs_cal, threshold)
    print(f"  Mean conformal set size: {test_set_sizes.mean():.4f}")
    print(f"  Coverage target: {1 - config.CONFORMAL_ALPHA:.2%}")
    
    conformal_conf = conformal_confidence(test_set_sizes)
    
    return {
        'baseline_acc_raw': baseline_acc_raw,
        'baseline_acc': baseline_acc,
        'temperature': temperature,
        'val_logits': val_logits,
        'val_probs_cal': val_probs_cal,
        'test_logits': test_logits,
        'test_probs_cal': test_probs_cal,
        'test_preds_cal': test_preds_cal,
        'question_engine': question_engine,
        'test_q_prob_yes': test_q_prob_yes_cal,
        'val_q_prob_yes': val_q_prob_yes_cal,
        'question_calibrators': question_calibrators,
        'conformal_threshold': threshold,
        'test_conformal_sets': test_conformal_sets,
        'test_set_sizes': test_set_sizes,
        'conformal_conf': conformal_conf
    }
