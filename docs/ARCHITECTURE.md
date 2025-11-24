# Architecture Overview

**High-level design and module interactions for the Accountable CLIP Classification System.**

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     run_experiment.py                        │
│                   (Main Orchestrator)                         │
└────────────┬────────────────────────────────────────────────┘
             │
             ├──► 1. Load Model & Data
             │    ├── src/models/clip_model.py
             │    └── src/data/dataset.py
             │
             ├──► 2. Calibration Pipeline
             │    ├── src/calibration/temperature.py
             │    ├── src/calibration/isotonic.py
             │    └── src/calibration/conformal.py
             │
             ├──► 3. Uncertainty Estimation
             │    ├── src/evaluation/questions.py
             │    └── src/evaluation/ensemble.py
             │
             ├──► 4. Policy Optimization
             │    ├── src/policy/threshold_optimizer.py
             │    └── src/policy/decision_policy.py
             │
             └──► 5. Evaluation & Reporting
                  └── src/evaluation/metrics.py
```

---

## 📦 Module Hierarchy

### Core Modules

#### 1. **configs/config.py**
**Purpose**: Centralized configuration management  
**Exports**: All hyperparameters, paths, constants  
**Dependencies**: None (base module)

```python
# Key sections:
- Paths (DATA_DIR, OUTPUT_DIR, MODEL_CACHE)
- Model config (MODEL_NAME, BATCH_SIZE, DEVICE)
- Calibration (TEMPERATURE_*, CONFORMAL_*, ISOTONIC_*)
- Questions (QUESTION_PROMPTS, QUESTION_GT_MAP)
- Policy (THRESHOLD_*, SIM_HUMAN_ACCURACY)
```

---

#### 2. **src/models/clip_model.py**
**Purpose**: CLIP model wrapper with offline support  
**Exports**: `CLIPModel` class  
**Dependencies**: `open_clip`, `torch`

```python
class CLIPModel:
    def __init__(model_name, pretrained, device)
    def encode_images(images) -> embeddings
    def encode_text(texts) -> embeddings
    def compute_similarities(img_emb, txt_emb) -> similarities
```

**Key Features**:
- Automatic fallback for offline mode
- Cached model loading (~/.cache/huggingface/)
- Batch processing support

---

#### 3. **src/data/dataset.py**
**Purpose**: Dataset loading, preprocessing, augmentation  
**Exports**: `DatasetLoader`, augmentation utilities  
**Dependencies**: `torchvision`, `torch`, `CLIPModel`

```python
class DatasetLoader:
    def __init__(data_dir, batch_size, num_workers)
    def load_cifar10() -> (train, val, test)
    def extract_embeddings(dataset, model) -> (embeddings, labels)

def create_augmentations(image, n_augs) -> [augmented_images]
```

**Key Features**:
- Train/val/test split (40k/10k/10k)
- Embedding extraction with progress bars
- Test-time augmentation (rotation, flip, color jitter)

---

### Calibration Modules

#### 4. **src/calibration/temperature.py**
**Purpose**: Temperature scaling for probability calibration  
**Exports**: `fit_temperature()`, `TemperatureScaler`  
**Dependencies**: `torch`, `sklearn.metrics`

```python
def fit_temperature_optimizer(logits, labels, lr, max_epochs) -> T
def fit_temperature_grid(logits, labels, T_range) -> T
def fit_temperature(logits, labels, ...) -> T  # Auto-fallback
```

**Features**:
- Log-parameterization for numerical stability
- Adam optimizer with early stopping
- Grid-search fallback if bounds hit

---

#### 5. **src/calibration/isotonic.py**
**Purpose**: Per-question isotonic regression calibration  
**Exports**: `calibrate_questions_isotonic()`, `apply_question_calibration()`  
**Dependencies**: `sklearn.isotonic`, `CLIPModel`

```python
def calibrate_questions_isotonic(
    embeddings, labels, model, questions, gt_map
) -> [IsotonicRegression], stats

def apply_question_calibration(embeddings, model, questions, funcs) -> q_confs
```

**Features**:
- Non-parametric monotonic mapping
- Per-question correctness prediction
- Validation-based fitting

---

#### 6. **src/calibration/conformal.py**
**Purpose**: Split-conformal prediction for uncertainty sets  
**Exports**: `compute_conformal_threshold()`, `compute_conformal_sets()`  
**Dependencies**: `numpy`

```python
def compute_conformal_threshold(scores, alpha) -> threshold
def compute_conformal_sets(probs, threshold) -> [label_sets]
def conformal_confidence(probs, threshold) -> confidences
```

**Features**:
- Finite-sample coverage guarantees
- 1-α coverage on test set
- Efficient set size (near-singleton)

---

### Evaluation Modules

#### 7. **src/evaluation/questions.py**
**Purpose**: Question-based semantic reasoning engine  
**Exports**: `QuestionEngine` class  
**Dependencies**: `CLIPModel`

```python
class QuestionEngine:
    def __init__(model, questions)
    def get_question_confidences(embeddings) -> q_confs
```

**Features**:
- Pre-compute text embeddings for efficiency
- Batch similarity computation
- Semantic signal extraction

---

#### 8. **src/evaluation/ensemble.py**
**Purpose**: Augmentation ensemble for uncertainty  
**Exports**: `compute_augmentation_ensemble()`, `compute_entropy_ensemble_conf()`  
**Dependencies**: `torch`, `math_utils`

```python
def compute_augmentation_ensemble(
    images, model, text_emb, n_augs, preprocess
) -> pred_classes, probs, std_conf, entropy_conf

def combine_ensemble_confidences(std, entropy, strategy) -> combined
```

**Strategies**:
- `"std"`: 1 - (std / std_max)
- `"entropy"`: Normalized mutual information
- `"combined"`: Average of std and entropy
- `"geometric"`: Geometric mean (log-space stable)

---

#### 9. **src/evaluation/metrics.py**
**Purpose**: Evaluation metrics, visualization, audit logging  
**Exports**: `compute_*()`, `plot_*()`, `save_audit_log()`  
**Dependencies**: `sklearn.metrics`, `matplotlib`

```python
def expected_calibration_error(probs, labels, n_bins) -> ece
def compute_intervention_metrics(decisions, correct, errors) -> precision, recall
def plot_reliability_diagram(probs, labels, path)
def save_audit_log(samples, path)  # JSONL format
```

**Features**:
- ECE with adaptive binning
- Reliability diagrams (calibration plots)
- Coverage vs error curves
- Per-sample audit trails

---

### Policy Modules

#### 10. **src/policy/decision_policy.py**
**Purpose**: Accountable decision policy with human-in-the-loop  
**Exports**: `AccountablePolicy` class  
**Dependencies**: `numpy`, `random`

```python
class AccountablePolicy:
    def __init__(tau_low, tau_high, action_thresholds, sim_human_acc)
    
    def compute_action_confidence(
        baseline_conf, question_conf, conformal_conf, 
        ensemble_conf, ensemble_strategy
    ) -> action_conf
    
    def make_decisions(action_confs) -> decisions  # "auto"/"clarify"/"human"
    def _simulate_human(true_labels) -> predictions
```

**Decision Logic**:
```
if action_conf >= ACTION_AUTO:
    → auto-execute (use model prediction)
elif action_conf >= ACTION_CLARIFY:
    → clarify (use ensemble + conformal)
else:
    → defer to human (simulated or real)
```

---

#### 11. **src/policy/threshold_optimizer.py**
**Purpose**: Grid-search optimization of policy thresholds  
**Exports**: `optimize_policy_thresholds()`, `apply_optimized_thresholds()`  
**Dependencies**: `AccountablePolicy`, `tqdm`

```python
def optimize_policy_thresholds(
    action_confs, labels, policy_class, 
    tau_range, auto_range, clarify_range, 
    cost_human, cost_error, grid_size
) -> best_params, best_cost
```

**Cost Function**:
```python
cost = c_human × (% interventions) + c_error × (% errors)
```

**Search Space**: tau × auto × clarify = grid_size³ combinations

---

### Utility Modules

#### 12. **src/utils/math_utils.py**
**Purpose**: Numerical utilities and helpers  
**Exports**: Mathematical functions  
**Dependencies**: `numpy`, `scipy`

```python
def softmax(logits) -> probs
def geometric_mean_stable(values) -> mean  # Log-space
def compute_entropy(probs) -> entropy
def clip_and_log(values, min_val, max_val) -> clipped_values
```

**Features**:
- Numerically stable implementations
- Log-space operations to prevent overflow/underflow
- Vectorized operations for efficiency

---

## 🔄 Data Flow

### Phase 1: Initialization
```
configs/config.py
    ↓
run_experiment.py
    ↓
CLIPModel.load() + DatasetLoader.load_cifar10()
    ↓
Extract embeddings (train, val, test)
```

### Phase 2: Calibration
```
Train embeddings + Train labels
    ↓
fit_temperature() → T_opt
    ↓
calibrate_questions_isotonic() → [isotonic_funcs]
    ↓
compute_conformal_threshold() → conf_threshold
```

### Phase 3: Inference (per test sample)
```
Test image
    ↓
CLIP.encode_image() → embedding
    ↓
├─► Temperature-scaled logits → baseline_conf
├─► Question similarities → question_conf
├─► Conformal set size → conformal_conf
└─► Augmentation ensemble → ensemble_conf
    ↓
PolicyCompute_action_confidence() → action_conf
    ↓
Policy.make_decision() → "auto" / "clarify" / "human"
    ↓
Execute decision → final_prediction
```

### Phase 4: Evaluation
```
Final predictions + True labels
    ↓
compute_metrics() → accuracy, ECE, precision, recall
    ↓
plot_reliability_diagram() + save_audit_log()
```

---

## 🎯 Design Principles

### 1. **Modularity**
- Each module has single responsibility
- Clear interfaces (function signatures)
- Minimal cross-dependencies

### 2. **Configurability**
- All hyperparameters in config.py
- Easy to experiment with different settings
- No hardcoded magic numbers in logic

### 3. **Robustness**
- Automatic fallbacks (temperature grid-search)
- Offline mode support (model caching)
- Graceful degradation (isotonic regression failures)

### 4. **Reproducibility**
- Seeded RNG (torch.manual_seed, np.random.seed)
- Deterministic splits (fixed train/val/test)
- Version-controlled configs

### 5. **Auditability**
- Per-sample decision logs (JSONL)
- Calibration diagnostics
- Comprehensive evaluation metrics

---

## 🔌 Extension Points

### Adding New Calibration Methods
```python
# In src/calibration/new_method.py
def fit_new_calibration(logits, labels, **kwargs):
    # Your implementation
    return calibrated_model

# In run_experiment.py
from src.calibration.new_method import fit_new_calibration
calibrated = fit_new_calibration(logits, labels)
```

### Adding New Decision Criteria
```python
# In src/policy/decision_policy.py
def compute_action_confidence(self, ..., new_signal):
    # Incorporate new signal
    weighted_conf = (
        w1 * baseline_conf + 
        w2 * question_conf + 
        w3 * new_signal  # ← Add here
    ) / (w1 + w2 + w3)
    return weighted_conf
```

### Adding New Datasets
```python
# In src/data/dataset.py
def load_custom_dataset(data_dir):
    # Implement loader
    return train_dataset, val_dataset, test_dataset

# Update config.py
DATASET_NAME = "custom"  # Switch from "cifar10"
```

---

## 📊 Complexity Analysis

### Time Complexity (CIFAR-10, 10k test)

| Operation | Complexity | Runtime (L40) |
|-----------|------------|---------------|
| **Embedding extraction** | O(N × d) | ~46s |
| **Temperature scaling** | O(E × N) | ~15s |
| **Isotonic calibration** | O(Q × N log N) | ~1.4s |
| **Conformal prediction** | O(N) | <1s |
| **Ensemble inference** | O(N × A × d) | ~45s |
| **Threshold optimization** | O(G³ × N) | ~21s |
| **Total** | O(N × (d + E + A×d + G³)) | **~4.5 min** |

Where:
- N = test size (10k)
- d = embedding dim (512)
- Q = num questions (9)
- A = num augmentations (16)
- E = temperature epochs (500)
- G = grid size (5)

### Space Complexity

| Component | Size (CIFAR-10) |
|-----------|-----------------|
| **Model weights** | ~605 MB |
| **Train embeddings** | 40k × 512 × 4B = 78 MB |
| **Val embeddings** | 10k × 512 × 4B = 20 MB |
| **Test embeddings** | 10k × 512 × 4B = 20 MB |
| **Augmentation cache** | 10k × 16 × 512 × 4B = 313 MB |
| **Total (peak)** | **~1.0 GB** |

---

## 🧪 Testing Strategy

### Unit Tests (TODO)
```python
# tests/test_calibration.py
def test_temperature_scaling():
    # Test that T=1 returns original logits
    # Test that T>1 smooths distribution
    # Test that T<1 sharpens distribution
    
def test_isotonic_monotonicity():
    # Test that output is monotonic w.r.t. input
```

### Integration Tests (TODO)
```python
# tests/test_pipeline.py
def test_end_to_end():
    # Run full pipeline on small dataset
    # Verify output format and ranges
```

### Validation Tests (Current)
- Manual inspection of logs
- Reliability diagram visualization
- Coverage vs error analysis

---

## 📚 References

### Academic Papers
- **CLIP**: Radford et al. (2021) "Learning Transferable Visual Models From Natural Language Supervision"
- **Temperature Scaling**: Guo et al. (2017) "On Calibration of Modern Neural Networks"
- **Conformal Prediction**: Angelopoulos & Bates (2021) "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification"
- **Isotonic Regression**: Zadrozny & Elkan (2002) "Transforming Classifier Scores into Accurate Multiclass Probability Estimates"

### Code References
- **OpenCLIP**: https://github.com/mlfoundations/open_clip
- **PyTorch Calibration**: https://github.com/gpleiss/temperature_scaling
- **Sklearn Isotonic**: https://scikit-learn.org/stable/modules/isotonic.html

---

**Last Updated**: November 24, 2025  
**Version**: 1.0
