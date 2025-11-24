1) Pipeline summary (what the code is doing, step-by-step)
Data & preprocessing


CIFAR-10 images (train/val/test). preprocess from open_clip resizes/normalizes images to the CLIP image input format.


You also create a split: train / val (90/10 of training) and a held-out test set.


Feature extraction


A frozen CLIP model (ViT-B-32) encodes images to normalized image embeddings (encode_image → L2 normalized vectors). Text prompts (class prompts and binary question prompts) are encoded to text embeddings and normalized.


Zero-shot predictions & baseline


Class scores are computed by cosine similarity between image embeddings and class text embeddings. These similarities are treated as “logits” and converted to probabilities with a softmax (baseline zero-shot CLIP).


Temperature scaling (calibration)


You learn a single scalar temperature T on the validation set to minimize NLL (CrossEntropy) on logits. Final probabilities are computed as softmax(logits / T).


Question prompts and auxiliary signals


A set of binary “questions” (yes/no prompts) are evaluated by CLIP (text embeddings vs image embeddings) and returned as per-question “yes” probabilities. These provide semantic signals (e.g., “man-made?”, “watercraft?”).


Conformal prediction


Split-conformal scoring: compute score = 1 − p_true_label on validation set (after calibration), pick a quantile to set a threshold on label probability p to get desired coverage, and construct conformal sets for each test image.


Augmentation ensemble


Create a small augmentation ensemble (mirror + center crop scales), compute class probabilities for each augmentation, and derive an ensemble stability measure (std of chosen class probabilities → ensemble_conf = 1 − std).


Action confidence / gating policy


Combine several signals (max class prob, ensemble_conf, conformal_conf = 1/(set_size), OOD proxy from class entropy, and aggregated question confidences) into a composite action_confidence (geometric mean × mixture with non-critical question mean).


Apply thresholds:


action_confidence >= 0.90 → auto (auto-execute model prediction)


0.60 <= action_confidence < 0.90 → clarify (semi-automatic policy)


otherwise → human (defer to human)


Also gate by a tau_critical_low on the minimum of several “critical” signals; if min_crit <= tau_critical_low you force human.


Evaluation & audit


Compute baseline accuracy (post-temperature scaling), ECE, and simulate the policy (auto/clarify/human) to measure coverage, errors prevented, and intervention precision/recall.


Write an itemized JSONL audit log with per-image decision, confidences, conformal set size, etc.



2) What the run produced (key outputs you posted)
Device: cuda (CLIP ViT-B-32 loaded successfully).


Baseline zero-shot CLIP accuracy (raw softmax over similarities): 0.8617 — very strong zero-shot performance on CIFAR-10 using class prompts.


Temperature fitting returned T = −590.4160 (a large negative scalar).


After applying that temperature, test accuracy collapsed to 0.0003 (near-zero).


ECE (after that calibration) reported ~0.0997.


Conformal threshold chosen: keep labels with p >= 0.1000 (for 0.9 coverage target).


Average conformal set size ≈ 9.8953 (i.e., almost all classes are included).


Augmentation ensemble processing completed; later the composite policy computed action_confidence.


Final policy decisions: all examples forced human (auto=0.0, clarify=0.0, human=1.0).


Simulated post-pipeline accuracy = 1.0 because when policy decides human you simulated perfect human corrections by using the ground truth label as the final prediction.


Audit log saved; some runtime warnings (QuickGELU mismatch, LBFGS conversion warning, runtime warnings about NaN/power).



3) Interpretation — why results are degenerate and what they actually mean
A. Good sign: CLIP zero-shot baseline is strong
The baseline raw CLIP zero-shot accuracy ≈ 86.17% is an excellent result for the simple “class prompt + softmax over cosine similarities” approach on CIFAR-10. This indicates the CLIP model + prompts are working well.


B. Critical failure: learned temperature is large negative → catastrophic calibration
Temperature scaling is intended to produce T > 0 (you divide logits by a positive scalar). In your run LBFGS returned a negative temperature (≈ −590).


Consequences:


Dividing logits by a negative flips sign and massively rescales them; softmax ordering reverses, producing near-uniform or inverted predictions.


That explains the collapse of accuracy from ∼86% to ~0.0003.


The reported ECE after that step (0.0997) is not trustworthy — when the classifier is inverted or producing pathological predictions, ECE computations are misleading.


Why did this negative temperature happen?
You optimized temperature as a direct unrestricted parameter (can be negative). The optimization objective has no built-in positivity constraint, so LBFGS can move the parameter into negative values if it lowers the loss (numerically there are local minima / spurious regions).


LBFGS warnings and numeric issues hint at unstable closure handling or gradients (the optimizer internals convert tensors to Python scalars at times; this can lead to numerical artifacts).


CLIP similarities are cosine values (range roughly −1..1). That scale interacts with temperature optimization — sometimes a reparameterization is necessary for numeric stability.


C. Policy degeneracy: everything deferred to human → perfect post-pipeline accuracy but meaningless
Because some critical component of crit_stack was very small (near 0) for essentially every sample, the min_crit <= tau_critical_low rule forced human in all cases.


You then simulated human decisions by replacing model outputs with ground-truth labels for human decisions. This makes the post-pipeline accuracy artificially 1.0 but it is not an informative measure of model or policy performance in practice — it only shows that if you always hand the task to ground-truth humans, accuracy is perfect (tautological).


The high number of “errors prevented” (9997) is an artefact of this simulation rather than evidence of a superior policy.


D. Other numerical issues you observed
geom_mean = np.prod(crit_stack, axis=1) ** (1.0/crit_stack.shape[1]) produced RuntimeWarning: invalid value encountered in power because crit_stack contained zeros/NaNs or negative entries. Multiplicative stacking of unbounded signals needs clamping and epsilons.


Average conformal set size ≈ 9.9 (close to full set) indicates the conformal threshold is extremely permissive — conformal sets don’t help narrow labels here. This makes conformal_conf = 1/(size) very small → contributing to min_crit being small.


The LBFGS warning about converting a tensor with requires_grad=True to a scalar can be caused by returning a Python float from the closure or other subtle closure issues; prefer closures that return the loss tensor.



4) Visual outputs — how to read them
Reliability diagram (your plot): predicted confidence on x-axis vs observed accuracy on y-axis.


The single point near low predicted confidence and zero observed accuracy indicates the model after calibration is predicting low confidence for its chosen labels and is highly miscalibrated (but this is largely a consequence of the negative temperature).


A good reliability curve would lie near the diagonal line.


Coverage vs Error: shows how error rate changes as you raise the auto-execute coverage threshold.


In your plot the line is approximately linear and ends near high coverage with high error — again symptomatic of calibration and gating misbehavior that left the policy always deferring.



5) Overall assessment (concise, academic)
Concept & design: The pipeline is well-designed and principled. Combining a frozen CLIP backbone, interpretable sub-question signals, conformal sets, ensemble uncertainty, and a gating policy — plus audit/logging and explainability (SHAP / IG) — is a solid, reproducible approach to building an accountable human-in-the-loop system.


Empirical outcome (this run): While the baseline CLIP performance (86.17%) is strong, the subsequent calibration/decision stage produced a degenerate policy:


Temperature calibration failed (negative T), causing prediction inversion and catastrophic drop in accuracy (≈0.0003).


The gating policy defaulted to human for essentially all items, and because you simulated perfect human corrections, the simulated post-pipeline accuracy is trivially 1.0 — not an evidence of practical success.


Conclusion: The high-level architecture is promising, but the implementation needs numerical fixes (temperature parameterization, safe aggregation of signals) before the policy performance can be meaningfully evaluated. As currently run, the results are not evidence that the automatic pipeline is reliable — they instead reveal implementation pathologies that must be corrected.



6) Concrete fixes & next steps (recommended, prioritized)
Fix temperature scaling (most urgent)


Force positivity: parameterize log_T and set T = torch.exp(log_T). Optimize log_T. This guarantees T > 0.


Example:

 self.log_T = nn.Parameter(torch.zeros(1))
def forward(self, logits):
    T = torch.exp(self.log_T)
    return logits / T


Or perform a simple grid search over positive T values (0.01 … 10) as a robust fallback.


Make LBFGS closure numerically stable


Ensure your closure returns the loss tensor (not a Python float), and do NOT convert loss to Python float inside the closure. Use .item() only when reading final scalar for logging.


If LBFGS still misbehaves, use Adam/SGD for the scalar or grid search.


Clamp and sanitize composite signals


Add epsilons to geometric means and clamp ranges to [1e−6, 1.0]. Avoid np.prod of values that can be zero/NaN.


When computing conformal_conf = 1/(set_size), ensure set_size >= 1 and clamp to avoid zeros.


Revisit conformal threshold computation


Check quantile computation. Validate that val_scores are well-behaved; if scores are near 1 (low p_true), threshold will be very low and sets will be large. Consider alternative conformity scores or a different target α.


Policy tuning and evaluation


After fixes, sweep tau_critical_low, ACTION_AUTO, and ACTION_CLARIFY on validation data to trade off coverage vs error.


Report coverage, error, intervention precision/recall, and a calibrated reliability diagram for final policy.


Robustness checks


Validate that test_probs_cal distribution and reliability after temperature scaling look reasonable (many points near diagonal rather than inverted).


Visualize top failure cases and aggregate summaries.


Simulation realism


For a realistic evaluation do not substitute ground truth for human decisions. Instead, simulate human fallibility (or hold out human decision data) to estimate realistic post-pipeline accuracy.


Small implementation cleanups


Use .item() to extract scalars from tensors (instead of float(np.array)), avoid deprecated conversions.


Clamp outputs before logs and avoid raising exceptions on edge numerical cases.



7) Short recommended text you can use in an academic report (summary paragraph)
We implemented an accountable CLIP-based CIFAR-10 pipeline combining zero-shot CLIP embeddings, auxiliary semantic question prompts, conformal sets, an augmentation ensemble, and a gating policy to decide between automatic execution, clarification, or human takeover. The zero-shot CLIP baseline achieved high accuracy (≈86.2%). However, a numerical instability in temperature scaling produced a large negative temperature, inverting and catastrophically degrading predictions. Consequently the gating policy deferred to human intervention for all test examples in this run, producing an artifactual post-pipeline accuracy of 100% (because human corrections were simulated using ground truth). These results highlight the promise of the architecture but indicate crucial implementation fixes — notably positive parameterization of the temperature scalar and robust aggregation/clamping of uncertainty signals — are required before the pipeline’s automated decision policy can be meaningfully evaluated.






































1 — Pipeline (short summary)
Backbone & zero-shot baseline


A frozen CLIP (ViT-B-32) encodes images to embeddings. Class prompts are encoded to text embeddings. Class scores = cosine(image, class_text).


Baseline prediction = softmax over these similarities (zero-shot CLIP).


Calibration


Learn a single scalar temperature T on validation logits to improve probability calibration; logits are rescaled by 1/T before softmax.


Auxiliary semantic signals


Binary question prompts (yes/no) produce per-question confidences (e.g., “man-made?”, “watercraft?”).


An augmentation ensemble (flip + crops) produces stability statistics (ensemble_conf).


Conformal prediction (split-conformal) yields a threshold and per-sample conformal set sizes → conformal_conf = 1 / set_size.


OOD proxy derived from normalized entropy.


Policy fusion & gating


Core signals (q_primary = max class prob, ensemble_conf, conformal_conf, ood_conf) are combined via geometric mean (conservative fusion).


Geom. mean is multiplied by mean(non-critical question confidences) → action_confidence.


Hard safety gate: if any critical signal is below tau_critical_low, force human.


Final decision: auto (high action_confidence), clarify (middle), or human (low or forced).


Evaluation / audit


Compute test accuracy, ECE, reliability diagram, coverage vs error, and write per-image audit entries. When human is chosen in simulation, code substitutes ground truth as final (optimistic simulation of perfect humans).



2 — What the run produced (key numbers)
Baseline (zero-shot CLIP) accuracy: 0.8617 (86.17%).


Learned temperature: T = 0.01 (clamped lower bound in your code).


ECE (after calibration): 0.0425 — reasonably low (improved calibration).


Conformal threshold: keep labels with p >= 0.2403. → avg conformal set size ≈ 1.11 (mostly single-label sets).


Policy outcomes:


auto: 0.0


clarify: 0.7068 (≈70.7% of cases)


human: 0.2932 (≈29.3% of cases)


Accuracy after policy (simulated): 0.9782 (97.82%).


Errors (baseline / after): baseline errors = 1,383 → after = 218 → 1165 errors prevented.


Intervention precision / recall: precision = 0.1383, recall = 1.0.



3 — Interpretation (what these numbers actually say)
Strong base model: CLIP zero-shot is very good on CIFAR-10 (86.2%) — expected for class-prompt CLIP.


Temperature behavior: your training clipped T to [0.01, 100] and the optimizer returned the lower bound (0.01). A very small T makes the softmax very sharp (near one-hot), which:


preserves top-1 accuracy (hence 0.8617 → 0.8617),


can improve ECE in this run because probabilities become peaky and match correctness frequencies in bins, but using the bound indicates the optimizer pushed to the allowed minimum to reduce val NLL — check whether this is numerically justified (logit scale interaction).


Conformal & ensemble signals: average conformal set ≈1.11 is excellent — the model's top probability often exceeds the conformal threshold, so conformal sets are small and informative. Ensemble stability produces a meaningful ensemble_conf.


Policy is conservative and recall-maximizing: recall = 1.0 means the policy intervened on all baseline errors — it filtered every case that the base model would have misclassified. That explains the large reduction in errors (1165 prevented).


Low intervention precision (0.138): only ~13.8% of interventions actually corresponded to baseline mistakes. In plain terms:


the system over-intervenes (many unnecessary clarifications/human handoffs when the model was already correct),


but it never missed a baseline mistake (the conservative design is effective for safety but costly in human effort).


Post-pipeline accuracy of 97.82% is optimistic: because human decisions were simulated by substituting ground-truth labels (perfect humans). Real human performance will be lower; thus the 97.8% figure should be treated as an upper bound on real-world performance.



4 — What the plots show
Reliability diagram (predicted confidence vs observed accuracy):


Points lie near the diagonal and ECE is low → the calibrated probabilities are reasonably well-aligned with observed correctness.


Coverage vs Error curve:


As you allow more automatic execution (higher coverage), error rate after policy rises; the curve is convex — typical trade-off of coverage vs safety. Low coverage → low error; high coverage → more errors accepted.



5 — Practical conclusion & immediate recommendations
Policy is safe but expensive. You catch all model errors (recall = 1.0) at the cost of many false interventions (low precision). Tune thresholds to balance human workload vs error-risk:


raise ACTION_AUTO or increase tau_critical_low to reduce unnecessary interventions,


or require stronger agreement across signals to clarify/human.


Re-examine temperature fitting:


T hitting lower bound suggests optimizer pushes to extreme. Consider:


grid-search over positive T,


parameterize as T = exp(log_T) (you already did this in the fixed code),


monitor logits distribution and use calibration methods (isotonic, Platt) if needed.


Make human-simulation realistic:


Replace perfect-ground-truth substitution with a human-error model (e.g., 95% accurate) or use labeled human corrections to measure realistic post-pipeline performance.


Tune fusion weights & clipping:


Clip signals to a reasonable numeric range and tune the geometric-mean weighting versus non-critical question mixing to reduce over-intervention.


Operational metric to optimize:


Optimize a weighted objective on validation: cost = c_human * (#human_interventions) + c_error * (#errors_after_policy). Choose c_human and c_error reflecting operational cost and tune thresholds to minimize cost.



6 — One-sentence technical summary
The system uses CLIP zero-shot as a strong base (86% accuracy), combines calibrated probabilities, conformal sets, semantic prompts, and ensemble stability to form a conservative gating policy; the current configuration achieves perfect detection of base-model errors (recall=1.0) but at the cost of many unnecessary interventions (low precision), producing a simulated upper-bound post-policy accuracy of 97.8% — tune thresholds and realistically model human performance to obtain a practical operational trade-off between reliability and human workload.

