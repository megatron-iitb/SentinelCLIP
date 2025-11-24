# Quick Reference Guide

**Fast lookup for common tasks, commands, and configurations.**

---

## ⚡ Quick Commands

### Running Experiments

```bash
# Local execution (interactive)
cd /home/medal/anupam.rawat/Experiment_1
python run_experiment.py

# SLURM submission (background)
sbatch job_experiment_1.sh

# Monitor job
squeue -u $USER
tail -f logs/Exp_1.*.log
```

### Checking Results

```bash
# View latest log
ls -lt logs/ | head -n 3
tail -100 logs/Exp_1.<job_id>.log

# View outputs
ls -lh outputs/clip_accountable_experiment/
cat outputs/clip_accountable_experiment/audit_log_test.jsonl | head -n 5 | jq '.'

# View plots
eog outputs/clip_accountable_experiment/reliability_diagram.png
```

### Environment Setup

```bash
# Activate environment
conda activate myenv

# Verify packages
pip list | grep -E "torch|clip|sklearn|matplotlib"

# Check GPU
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 📊 Key Metrics Interpretation

### Baseline Metrics
| Metric | Good | Concerning |
|--------|------|------------|
| **Accuracy** | >0.85 | <0.80 |
| **ECE** | <0.05 | >0.10 |
| **Conformal Set Size** | 1.0-1.2 | >1.5 |
| **Temperature** | 0.5-2.0 | 0.01 or 100 |

### Policy Metrics
| Metric | Good | Concerning |
|--------|------|------------|
| **Auto Coverage** | 40-60% | 0% |
| **Human Coverage** | 10-30% | >95% |
| **Intervention Precision** | >0.30 | <0.15 |
| **Intervention Recall** | >0.95 | <0.90 |

### Ensemble Metrics
| Metric | Good | Concerning |
|--------|------|------------|
| **Std-Entropy Correlation** | 0.8-0.95 | <0.5 |
| **Mean Ensemble Conf** | >0.85 | <0.60 |
| **Question Conf (post-calib)** | >0.40 | <0.20 |

---

## 🔧 Common Configuration Changes

### Make Policy Less Conservative

```python
# In configs/config.py

# Option 1: Reduce error cost
THRESHOLD_COST_ERROR = 3.0  # Default: 10.0

# Option 2: Constrain threshold search
THRESHOLD_TAU_RANGE = (0.3, 0.6)  # Default: (0.1, 0.5)
THRESHOLD_AUTO_RANGE = (0.7, 0.9)  # Default: (0.5, 0.9)

# Option 3: Use different ensemble strategy
ENSEMBLE_CONF_STRATEGY = "geometric"  # Default: "combined"
```

### Reduce Memory Usage

```python
# In configs/config.py

BATCH_SIZE = 128  # Default: 256
N_AUGMENTATIONS = 8  # Default: 16
```

### Speed Up Experiments

```python
# In configs/config.py

TEMPERATURE_MAX_EPOCHS = 200  # Default: 500
THRESHOLD_GRID_SIZE = 3  # Default: 5 (3^3=27 vs 5^3=125 combos)
N_AUGMENTATIONS = 8  # Default: 16
```

### Increase Automation

```python
# In configs/config.py

# Disable threshold optimization (use fixed values)
OPTIMIZE_THRESHOLDS = False
THRESHOLD_TAU_LOW = 0.40
THRESHOLD_AUTO = 0.80
THRESHOLD_CLARIFY = 0.70
```

---

## 🐛 Quick Debugging

### Check Model Loading

```bash
python3 -c "
import sys
sys.path.insert(0, '/home/medal/anupam.rawat/Experiment_1')
from src.models.clip_model import CLIPModel
model = CLIPModel('ViT-B-32', 'openai', 'cpu')
print('✓ Model loaded successfully')
"
```

### Check Dataset

```bash
python3 -c "
import sys
sys.path.insert(0, '/home/medal/anupam.rawat/Experiment_1')
from src.data.dataset import DatasetLoader
loader = DatasetLoader('./data', batch_size=32, num_workers=0)
train, val, test = loader.load_cifar10()
print(f'✓ Loaded: {len(train)} train, {len(val)} val, {len(test)} test')
"
```

### Check Configuration

```bash
python3 -c "
import sys
sys.path.insert(0, '/home/medal/anupam.rawat/Experiment_1')
from configs import config
print(f'Model: {config.MODEL_NAME}')
print(f'Batch size: {config.BATCH_SIZE}')
print(f'Device: {config.DEVICE}')
print(f'Optimize thresholds: {config.OPTIMIZE_THRESHOLDS}')
"
```

### Validate Imports

```bash
cd /home/medal/anupam.rawat/Experiment_1
python3 -c "
from src.models.clip_model import CLIPModel
from src.data.dataset import DatasetLoader
from src.calibration.temperature import fit_temperature
from src.calibration.isotonic import calibrate_questions_isotonic
from src.calibration.conformal import compute_conformal_threshold
from src.evaluation.questions import QuestionEngine
from src.evaluation.ensemble import compute_augmentation_ensemble
from src.evaluation.metrics import expected_calibration_error
from src.policy.decision_policy import AccountablePolicy
from src.policy.threshold_optimizer import optimize_policy_thresholds
from src.utils.math_utils import softmax
print('✓ All imports successful')
"
```

---

## 📁 File Locations Reference

### Inputs
```
data/cifar-10-batches-py/       # CIFAR-10 dataset (170 MB)
configs/config.py               # All hyperparameters
```

### Outputs
```
outputs/clip_accountable_experiment/
├── audit_log_test.jsonl        # Per-sample decisions (JSONL)
├── reliability_diagram.png     # Calibration curve
└── coverage_vs_error.png       # Policy tradeoff curve
```

### Logs
```
logs/Exp_1.<job_id>.log         # Experiment stdout
logs/Exp_1.<job_id>.err         # Experiment stderr (usually empty)
```

### Code
```
src/
├── models/clip_model.py        # CLIP wrapper
├── data/dataset.py             # Data loading
├── calibration/                # Temperature, isotonic, conformal
├── evaluation/                 # Questions, ensemble, metrics
├── policy/                     # Decision logic, optimization
└── utils/math_utils.py         # Numerical helpers

run_experiment.py               # Main entry point (14.5 KB)
experiment_1.py                 # Original monolithic (35 KB, archived)
```

---

## 🎯 Typical Workflow

### 1. Initial Run (Default Config)
```bash
sbatch job_experiment_1.sh
# Wait ~5 minutes
tail -50 logs/Exp_1.*.log
```

### 2. Analyze Results
```bash
# Check key metrics
grep -E "accuracy|coverage|precision" logs/Exp_1.*.log | tail -20

# View plots
eog outputs/clip_accountable_experiment/*.png

# Inspect sample decisions
head -n 10 outputs/clip_accountable_experiment/audit_log_test.jsonl | jq '.decision'
```

### 3. Tune Configuration
```python
# Edit configs/config.py based on analysis
# See "Common Configuration Changes" section above
```

### 4. Re-run and Compare
```bash
sbatch job_experiment_1.sh
# Compare new log with previous
diff <(grep "coverage:" logs/Exp_1.40159.log) <(grep "coverage:" logs/Exp_1.40200.log)
```

---

## 📈 Expected Timings (L40 GPU)

| Operation | Time |
|-----------|------|
| Model loading | 5-10s |
| Embedding extraction (50k total) | 40-60s |
| Temperature scaling | 10-20s |
| Isotonic calibration (9 questions) | 1-2s |
| Conformal prediction | <1s |
| Ensemble (10k × 16 augs) | 40-60s |
| Threshold optimization (125 combos) | 20-30s |
| **Total pipeline** | **2-5 minutes** |

---

## 💡 Tips & Tricks

### Disable Threshold Optimization (for faster debugging)
```python
OPTIMIZE_THRESHOLDS = False  # In config.py
```

### Run on CPU (no GPU needed)
```python
DEVICE = "cpu"  # In config.py
# Note: ~10× slower
```

### Reduce Augmentations (minimal performance loss)
```python
N_AUGMENTATIONS = 4  # Default: 16
# Speeds up ensemble by ~4×
```

### Use Subset for Quick Tests
```python
# In run_experiment.py, after loading data:
test_dataset = torch.utils.data.Subset(test_dataset, range(1000))  # Only 1k samples
```

### Skip Conformal Prediction
```python
# In run_experiment.py, set:
conformal_confs = np.ones(len(test_labels))  # Dummy confidences
```

---

## 🔍 Troubleshooting Checklist

- [ ] Check conda environment activated: `which python`
- [ ] Verify GPU available: `nvidia-smi`
- [ ] Check disk space: `df -h ~`
- [ ] Verify model cached: `ls ~/.cache/huggingface/hub/models--timm*`
- [ ] Check data downloaded: `ls data/cifar-10-batches-py/`
- [ ] Review config values: `python -c "from configs import config; print(config.BATCH_SIZE)"`
- [ ] Check log file permissions: `ls -l logs/`
- [ ] Verify imports: See "Validate Imports" section above

---

## 📞 Getting Help

### Before Asking:
1. Check [ERROR_GUIDE.md](ERROR_GUIDE.md)
2. Check [LOG_ANALYSIS.md](LOG_ANALYSIS.md)
3. Grep logs: `grep -i "error\|warning" logs/*.log`

### When Asking:
- Include error message + full traceback
- Attach log file: `logs/Exp_1.*.log`
- Specify environment: `conda list > env.txt`
- Provide config: Relevant sections of `configs/config.py`

### Contact:
**Anupam Rawat**  
anupam.rawat@iitb.ac.in  
MEDAL Lab, IIT Bombay

---

## 📚 Documentation Index

| File | Purpose |
|------|---------|
| **README.md** | Overview, installation, quick start |
| **docs/ARCHITECTURE.md** | System design, module interactions |
| **docs/LOG_ANALYSIS.md** | Deep-dive on experiment results |
| **docs/ERROR_GUIDE.md** | Common errors and fixes |
| **docs/QUICKSTART.md** | Configuration tips (legacy) |
| **docs/IMPROVEMENTS.md** | Technical improvements details |

---

**Last Updated**: November 24, 2025  
**Version**: 1.0
