# Experiment 1: SentinelCLIP - Detailed Technical Explanations

## 1. Core Intuition - Why Accountable AI?

### The Restaurant Review Analogy

**Scenario**: You're choosing a restaurant based on AI recommendations.

**Bad AI (Black-Box Confidence)**:
```
AI: "This restaurant is 95% good."
You: "Why?"
AI: "Trust me."
Reality: Food poisoning because AI was overconfident.
```

**Good AI (SentinelCLIP Approach)**:
```
AI: "This restaurant has 85% positive reviews."
AI: "Let me check critical factors:"
  - Hygiene rating? → A grade ✅ (95% confidence)
  - Recent complaints? → None ✅ (90% confidence)
  - Price matches quality? → Yes ✅ (80% confidence)
  - Chef consistency? → Medium ⚠️ (60% confidence) ← Critical!
  
AI: "Overall 85% positive, BUT chef consistency is uncertain (60%)."
AI: "Recommendation: CLARIFY - Check recent customer photos of food."
OR
AI: "Recommendation: DEFER - Too risky, choose safer option."

You: Make informed decision with transparent reasoning ✅
```

**Key insight**: It's not just about "how confident" but **"what makes you confident"** and **"what could go wrong"**.

---

## 2. Temperature Scaling - Mathematical Deep Dive

### The Problem: Overconfident Neural Networks

**Why it happens**:
1. **Cross-entropy loss** encourages extreme probabilities (0 or 1)
2. **Model capacity**: Large networks can memorize training data
3. **Overfitting**: High training accuracy but poor calibration

**Formula**:
```
Before: P(class_i) = exp(z_i) / Σ_j exp(z_j)  [Standard softmax]
After:  P(class_i) = exp(z_i / T) / Σ_j exp(z_j / T)  [Temperature-scaled]

Where:
- z_i: Logit for class i
- T: Temperature (T > 1 → softer, T < 1 → sharper)
- Our learned value: T = 1.23
```

### Worked Example

**Raw CLIP logits**: `[3.0, 1.0, 0.5, ...]` (cat, dog, bird)

**Standard softmax (T=1.0)**:
```
P(cat) = exp(3.0) / [exp(3.0) + exp(1.0) + ...] = 0.804 (80.4%)
```

**Temperature-scaled (T=1.23)**:
```
P(cat) = exp(3.0/1.23) / [...] = 0.695 (69.5%)  ← Less confident!
```

### Expected Calibration Error (ECE)

**Formula**:
```
ECE = Σ_bins (weight × |confidence - accuracy|)

Where:
- Bin predictions by confidence [0-0.1, 0.1-0.2, ..., 0.9-1.0]
- weight = (# samples in bin) / (total samples)
```

**Our results**:
- Before: ECE = 0.156 (15.6% average gap)
- After: ECE = 0.042 (4.2% gap) → **73% reduction!**

---

## 3. Conformal Prediction - Coverage Guarantees

### The Mathematical Guarantee

**Theorem**:
```
Given significance level α (e.g., 0.10 for 90% coverage):
  P(Y_test ∈ C(X_test)) ≥ 1 - α

Where C(X_test) is the prediction set constructed as:
  C = {y : 1 - P(y) ≤ quantile(validation_scores, 1-α)}
```

**Key properties**:
1. **Finite-sample**: Works for any test size
2. **Model-agnostic**: Works with any classifier
3. **Distribution-free**: No distribution assumptions
4. **Guaranteed coverage**: Mathematical proof

### Worked Example

**Step 1**: Compute non-conformity scores on validation (5,000 samples)
```python
val_scores = [1 - probs[true_label] for sample in val_data]
# [0.05, 0.12, 0.02, ..., 0.18]
```

**Step 2**: Find 90th percentile
```python
threshold = np.quantile(val_scores, 0.90) = 0.24
```

**Step 3**: Apply to test sample (cat image)
```python
test_probs = [0.05, 0.08, 0.02, 0.82, 0.01, ...]  # Classes: [air, car, bird, cat, ...]
test_scores = [1 - p for p in test_probs]
# [0.95, 0.92, 0.98, 0.18, 0.99, ...]

prediction_set = [i for i, s in enumerate(test_scores) if s <= 0.24]
# [3]  (only cat, since 0.18 ≤ 0.24)
```

**Step 4**: Verify coverage on test set
```python
coverage = 9020 / 10000 = 0.902 (90.2%) ✅ Exceeds 90% target
```

### Why It Works

**Intuition**: Non-conformity scores from validation and test come from the same distribution (exchangeability).

**Analogy**: Drawing balls from an urn
```
Urn: 5,000 validation balls + 1 test ball (all i.i.d.)
Threshold: 90th percentile of validation colors
Test ball: By definition, 90% chance it's ≤ threshold
```

---

## 4. Semantic Questions - Design Philosophy

### Why Questions?

**Problem**: CLIP's reasoning is opaque
```
CLIP: "85% confident this is a cat"
You: "Why? Is it the fur? Face? Posture?"
CLIP: [512-dim embedding] (incomprehensible)
```

**Solution**: Force explicit reasoning via binary questions
```
Q1: "Does this have fur?" → 90% Yes ✅
Q2: "Does this have 4 legs?" → 85% Yes ✅
Q3: "Is this domesticated?" → 70% Yes ⚠️
Q4: "Does this have wheels?" → 5% No ✅
```

### Our 9 Questions (CIFAR-10)

```python
QUESTIONS = [
    "Is this a vehicle?",           # [airplane, car, ship, truck]
    "Is this an animal?",           # [bird, cat, deer, dog, frog, horse]
    "Does this have wheels?",       # [car, truck]
    "Does this fly?",               # [airplane, bird]
    "Does this move on water?",     # [ship]
    "Does this have four legs?",    # [cat, deer, dog, horse]
    "Is this domesticated?",        # [car, cat, dog, truck]
    "Does this have fur?",          # [cat, deer, dog, frog, horse]
    "Is this man-made?",            # [airplane, car, ship, truck]
]
```

**Design principles**:
1. **Binary or simple**: Easy to calibrate
2. **Complementary**: Cover different aspects (shape, function, habitat)
3. **Diverse granularity**: Coarse (animal/vehicle) + Fine (4 legs)
4. **Ground-truth mappable**: Known correct answer per class

### Example: Classifying a Horse

**Base CLIP**: 85% horse, 10% deer, 3% dog

**Question reasoning**:
```
Q1: "Is this a vehicle?" → 5% ✅ (Low, correct: No)
Q2: "Is this an animal?" → 95% ✅ (High, correct: Yes)
Q3: "Does this have wheels?" → 2% ✅ (Low, correct: No)
Q4: "Does this fly?" → 8% ✅ (Low, correct: No)
Q5: "Does this move on water?" → 3% ✅ (Low, correct: No)
Q6: "Does this have four legs?" → 88% ✅ (High, correct: Yes) ← CRITICAL
Q7: "Is this domesticated?" → 75% ⚠️ (Medium, could be wild)
Q8: "Does this have fur?" → 82% ✅ (High, correct: Yes)
Q9: "Is this man-made?" → 4% ✅ (Low, correct: No)

Minimum confidence: 4% (Q9)
Critical reasoning: NOT man-made, IS animal, HAS 4 legs, HAS fur
→ Narrows to: {horse, deer, dog, cat}
→ Combined with base (85% horse) → Final: Horse ✅
```

**If Q6 was LOW** (40% confidence):
```
Critical confidence = 40% < 0.65 threshold
→ Policy: DEFER TO HUMAN
→ Reason: Uncertain about fundamental attribute
```

---

## 5. Ensemble Uncertainty - Two Methods

### Standard Deviation-Based

**Concept**: Measure variation in predicted class across augmentations

**Algorithm**:
```python
# Generate 16 augmented versions
augmented_images = [original, flip_h, rotate_15, color_jitter, ...]

# Get predictions
predictions = [CLIP(aug) for aug in augmented_images]
# [[0.85, 0.10, ...], [0.82, 0.12, ...], [0.88, 0.08, ...], ...]

# Extract predicted class probabilities
pred_class_probs = [pred[argmax(pred)] for pred in predictions]
# [0.85, 0.82, 0.88, 0.80, 0.87, ...]

# Standard deviation
std = np.std(pred_class_probs) = 0.03

# Confidence = 1 - std
ensemble_conf_std = 1 - 0.03 = 0.97  # High confidence!
```

**Interpretation**:
- Low std (<0.05): Consistent → High confidence
- High std (>0.15): Varying → Low confidence

---

### Entropy-Based (Mutual Information Proxy)

**Formula**:
```
MI(Pred; Aug) ≈ H(Pred) - E[H(Pred|Aug)]

Where:
- H(Pred): Entropy of average prediction
- E[H(Pred|Aug)]: Expected entropy of individual predictions
- Higher MI → More disagreement → Lower confidence
```

**Algorithm**:
```python
# Average prediction
avg_pred = np.mean(predictions, axis=0)
# [0.85, 0.10, 0.03, ...]

# Entropy of average
H_avg = -sum(p * log(p) for p in avg_pred if p > 0) ≈ 0.65

# Entropy of each prediction
entropies = [-sum(p * log(p) for p in pred if p > 0) for pred in predictions]
# [0.58, 0.62, 0.53, 0.68, ...]

E_H_given_aug = np.mean(entropies) = 0.60

# Mutual information
MI = H_avg - E_H_given_aug = 0.05

# Confidence = 1 - normalized MI
ensemble_conf_entropy = 1 - (MI / H_avg) = 0.92
```

**Interpretation**:
- Low MI: Augmentations don't change prediction → Robust → High confidence
- High MI: Augmentations cause shifts → Sensitive → Low confidence

---

### Combined Strategy (Our Default)

**Weighted Average**:
```python
confidence = 0.6 * conf_std + 0.4 * conf_entropy
```

**Why**:
- Std captures class-level variation
- Entropy captures distribution-level variation
- Balances both perspectives

**Alternative**: Geometric mean (more conservative)
```python
confidence = (conf_std^0.6 × conf_entropy^0.4)
```

---

## 6. Threshold Optimization - Cost Function

### The Goal

**Balance**: Automation vs human intervention

**Formula**:
```
Cost = c_human × (% samples with human intervention)
     + c_error × (% samples with errors)
```

**Example weights**:
```
c_human = 1.0   (baseline: 1 hour of human time)
c_error = 10.0  (errors 10× more expensive)
```

### Grid Search Over 3 Thresholds

**Parameters** (5 values each = 125 combinations):

1. **τ_critical_low**: Minimum question confidence
   - Range: [0.1, 0.3, 0.5, 0.7, 0.9]
   - Role: Filter samples with low-confidence questions

2. **θ_auto**: Auto-execute threshold
   - Range: [0.6, 0.7, 0.8, 0.85, 0.95]
   - Role: High confidence → Auto

3. **θ_clarify**: Clarify threshold
   - Range: [0.4, 0.5, 0.6, 0.7, 0.8]
   - Role: Medium confidence → Use ensemble
   - Constraint: θ_clarify < θ_auto

### Optimization Results

**Best configuration**:
```
τ_critical_low = 0.35
θ_auto = 0.80
θ_clarify = 0.65

Distribution:
- AUTO: 45% (high confidence)
- CLARIFY: 35% (medium, use ensemble)
- HUMAN: 20% (low, defer to expert)

Cost: 0.20 × 1.0 + 0.001 × 10.0 = 0.21
      ^^^^^^^^^^   ^^^^^^^^^^^^
      Human cost   Error cost

Intervention recall: 100% (all errors caught!)
Intervention precision: 42% (42% of interventions were errors)
```

### Trade-offs

**Conservative (High intervention)**:
```
τ=0.70, θ_auto=0.95, θ_clarify=0.80
→ HUMAN: 60%, AUTO: 10%
→ Cost: 0.60 (high human cost, zero errors)
```

**Aggressive (Low intervention)**:
```
τ=0.10, θ_auto=0.60, θ_clarify=0.40
→ HUMAN: 5%, AUTO: 75%
→ Cost: 0.55 (many errors slip through)
```

**Optimized (Balanced)** ← Our result:
```
τ=0.35, θ_auto=0.80, θ_clarify=0.65
→ HUMAN: 20%, AUTO: 45%, CLARIFY: 35%
→ Cost: 0.21 (optimal!)
→ 100% error recall, reasonable human load
```

---

## 7. System Architecture - Modular Design

### 13 Modules Organized by Function

#### Data & Models
```
src/models/clip_model.py (150 lines)
- CLIPModel class wrapper
- Embedding extraction
- Zero-shot classification

src/data/dataset.py (200 lines)
- CIFAR10DataModule
- Train/val/test splits
- Augmentation pipelines (16 transforms)
```

#### Calibration Stack
```
src/calibration/temperature.py (100 lines)
- Temperature scaling with Adam
- Grid-search fallback
- Log-space parameterization

src/calibration/isotonic.py (120 lines)
- Per-question isotonic regression
- Monotonic mapping
- Validation-based fitting

src/calibration/conformal.py (180 lines)
- Split conformal prediction
- Non-conformity scores
- Prediction set construction
```

#### Uncertainty Estimation
```
src/evaluation/ensemble.py (150 lines)
- Test-time augmentation
- Std + entropy measures
- Combined confidence scoring

src/evaluation/questions.py (250 lines)
- Semantic question engine
- CLIP-based reasoning
- Per-question confidence
```

#### Policy & Decision
```
src/policy/decision_policy.py (200 lines)
- Three-tier decision logic
- AccountablePolicy class
- Simulated human oracle

src/policy/threshold_optimizer.py (180 lines)
- Grid-search over thresholds
- Cost function minimization
- Validation-based tuning
```

#### Utilities & Evaluation
```
src/evaluation/metrics.py (220 lines)
- ECE computation
- Reliability diagrams
- Coverage analysis

src/utils/math_utils.py (80 lines)
- Softmax (log-space stable)
- Geometric mean
- Entropy calculations
```

### Configuration Management

**Single source**: `configs/config.py`

```python
# Model
MODEL_NAME = "ViT-B-32"
BATCH_SIZE = 256

# Calibration
CONFORMAL_ALPHA = 0.10
TEMPERATURE_MAX_EPOCHS = 500

# Ensemble
N_AUGMENTATIONS = 16
ENSEMBLE_CONF_STRATEGY = "combined"

# Policy
SIM_HUMAN_ACCURACY = 1.0
THRESHOLD_COST_HUMAN = 1.0
THRESHOLD_COST_ERROR = 10.0
```

---

## 8. Practical Implementation Details

### Memory Usage

**Peak RAM**: ~1.0 GB
- Embeddings: 512 dim × 10k samples × 4 bytes = 20 MB
- Augmentations: 16× cache = 320 MB
- Model: ViT-B-32 = 350 MB
- Overhead: 300 MB

### Runtime Breakdown (L40 GPU)

| Stage | Time | % Total |
|-------|------|---------|
| Model loading | 30s | 14% |
| Embedding extraction | 60s | 29% |
| Calibration | 45s | 21% |
| Ensemble (16 augs) | 50s | 24% |
| Policy optimization | 20s | 10% |
| Evaluation | 5s | 2% |
| **Total** | **210s (3.5 min)** | **100%** |

### Hyperparameter Sensitivity

**Temperature T**:
```
T = 1.0: ECE = 0.156 (no scaling)
T = 1.2: ECE = 0.045
T = 1.23: ECE = 0.042 ← Optimal
T = 1.3: ECE = 0.048 (over-scaling)
```

**Conformal α**:
```
α = 0.05: Coverage = 95.2%, Set size = 1.8
α = 0.10: Coverage = 90.2%, Set size = 1.11 ← Optimal
α = 0.20: Coverage = 80.5%, Set size = 1.01
```

**Ensemble size**:
```
4 augs: Conf std = 0.12 (noisy)
8 augs: Conf std = 0.08
16 augs: Conf std = 0.05 ← Optimal
32 augs: Conf std = 0.04 (diminishing returns, 2× slower)
```

---

## 9. Common Pitfalls & Solutions

### Pitfall 1: Temperature Hits Bounds

**Symptom**: `T = 0.01` or `T = 100` after optimization

**Cause**: Optimizer diverging or poor initialization

**Solution**: Grid-search fallback
```python
if T < 0.1 or T > 10:
    # Try grid of values
    best_T = min([0.5, 1.0, 1.5, 2.0], key=lambda t: ece(t))
```

### Pitfall 2: All Decisions = 'human'

**Symptom**: No automation (100% human intervention)

**Cause**: Low question confidences or conservative thresholds

**Solutions**:
1. Adjust cost function weights
   ```python
   THRESHOLD_COST_ERROR = 5.0  # Less conservative
   ```
2. Retune questions (add easier ones)
3. Check isotonic calibration (may be over-regularized)

### Pitfall 3: Conformal Sets Too Large

**Symptom**: Avg set size > 3 classes

**Cause**: High α or poor base model calibration

**Solutions**:
1. Reduce α (e.g., 0.10 → 0.15 for 85% coverage)
2. Improve base model (fine-tuning)
3. Use adaptive conformal prediction (future work)

### Pitfall 4: Ensemble Disagreement

**Symptom**: High std (>0.20) for many samples

**Cause**: Model sensitive to augmentations

**Solutions**:
1. Reduce augmentation intensity
2. Use ensemble as signal (defer when high std)
3. Train model with augmentations (improve robustness)

### Pitfall 5: Questions Always High/Low Confidence

**Symptom**: All questions >0.95 or <0.20

**Cause**: Questions too easy/hard for CLIP

**Solutions**:
1. Redesign questions (balanced difficulty)
2. Check ground-truth mappings (may be incorrect)
3. Use isotonic calibration (corrects miscalibration)

---

## 10. Extensions & Future Directions

### Short-term (1-3 months)

**1. Adaptive Question Selection**
```python
# Learn which questions are informative per sample
def select_questions(image, base_pred):
    # Predict which questions will have low confidence
    critical_questions = uncertainty_predictor(image)
    # Only ask those (save compute)
    return critical_questions[:5]  # Top 5
```

**2. Real Human Interface**
```python
# Replace simulated human
def real_human_review(image, conformal_set, questions):
    # Show to expert via web UI
    human_label = get_label_from_ui()
    # Track human accuracy over time
    return human_label
```

### Medium-term (3-6 months)

**3. Hierarchical Conformal Prediction**
```
Level 1: Animal vs Vehicle (99% coverage, small sets)
Level 2: Cat vs Dog vs Horse (90% coverage, larger sets)
```

**4. Uncertainty Decomposition**
```
Total = Aleatoric + Epistemic

Aleatoric (data noise):
  - Inherently ambiguous images
  - Cannot reduce with more model capacity

Epistemic (model uncertainty):
  - Lack of training data
  - Can reduce with fine-tuning or active learning
```

### Long-term (6-12 months)

**5. Meta-Learning for Thresholds**
```python
# Learn to set thresholds per domain
meta_model = train_on_domains(['medical', 'autonomous', 'moderation'])
new_thresholds = meta_model.adapt(new_domain, K=10)
```

**6. Continuous Learning**
```python
# Update calibration online
for batch in production_stream:
    preds = model(batch)
    labels = get_feedback(low_confidence_samples)
    update_calibration(preds, labels)
```

---

## Summary: Key Takeaways

### Scientific Contributions

1. **Multi-stage calibration**: Temperature + Isotonic + Conformal
2. **Cost-based optimization**: Thresholds tuned to domain costs
3. **Semantic reasoning**: Questions for interpretability
4. **Ensemble uncertainty**: Std + Entropy combined

### Practical Impact

- **99.99% accuracy** with human-in-loop (vs 86.16% baseline)
- **100% error recall** (all mistakes caught)
- **80% automation** (45% auto + 35% clarify)
- **<4 min runtime** (10k samples, single GPU)

### Design Principles

1. **Calibration is essential**: Raw confidences mislead
2. **Multiple uncertainty signals**: No single method perfect
3. **Explainability builds trust**: Questions > black-box
4. **Thresholds must be optimized**: Not arbitrary
5. **Modularity enables innovation**: Easy to extend

---

**Repository**: https://github.com/megatron-iitb/SentinelCLIP  
**Contact**: anupam.rawat@iitb.ac.in  
**Institution**: IIT Bombay, MEDAL Lab
